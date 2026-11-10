"""CPU CatBoost with the study's MAE loss and training-only stopping iteration."""

from catboost import CatBoostRegressor

from wdaoh.models.training import chronological_split, scale


def flatten(x):
    return x.reshape(len(x), -1)


def fit(train, config, seed, raw, threads, directory, floor=1e-6):
    """Choose tree count from the purged training period and refit on all eligible labels."""
    x, y = train["X"], train["y"]
    options = {k: v for k, v in config.items() if k not in ("id", "patience")}
    common = dict(
        loss_function="MAE",
        task_type="CPU",
        random_seed=seed,
        thread_count=threads,
        verbose=False,
        allow_writing_files=False,
    )
    stats = scale(x, y, raw, floor)
    split = {}
    if "patience" in config:
        fit_ids, val_ids, boundary = chronological_split(train)
        inner_scale = scale(x[fit_ids], y[fit_ids], raw, floor)
        inner = CatBoostRegressor(**options, **common)
        inner.fit(
            flatten(inner_scale.inputs(x[fit_ids])),
            inner_scale.labels(y[fit_ids]),
            eval_set=(
                flatten(inner_scale.inputs(x[val_ids])),
                inner_scale.labels(y[val_ids]),
            ),
            early_stopping_rounds=config["patience"],
            use_best_model=True,
        )
        options["iterations"] = int(inner.get_best_iteration() + 1)
        split = dict(
            boundary=boundary,
            fit_rows=len(fit_ids),
            stop_rows=len(val_ids),
            purge_hours=48,
        )
    final = CatBoostRegressor(**options, **common)
    final.fit(flatten(stats.inputs(x)), stats.labels(y))
    final.save_model(str(directory / "model.cbm"))
    return {"iterations": int(final.tree_count_), "inner_split": split}, stats


def load(directory, stats):
    """Load the refitted CatBoost model and return a predictor in native concentration units."""
    model = CatBoostRegressor()
    model.load_model(str(directory / "model.cbm"))

    def predict(query):
        return stats.restore(
            model.predict(flatten(stats.inputs(query["X"])), thread_count=2)
        )

    return predict
