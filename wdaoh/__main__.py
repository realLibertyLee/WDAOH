"""Command-line entry points for provider records, frozen samples, forecasts and evaluation."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m wdaoh", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in [
        ("model-sources", "Fetch pinned upstream architecture definitions"),
        ("download", "Download and verify provider observations"),
        ("prepare", "Construct station pools and matched forecast samples"),
        ("run", "Fit on training data, freeze on validation and forecast"),
        ("evaluate", "Measure errors of newly predicted targets"),
        ("study", "Write the complete study schedule or execute a selected stage"),
    ]:
        cmd = commands.add_parser(name, help=description)
        cmd.add_argument("--work", type=Path, default=Path("runs"))
        if name in ("download", "prepare", "run"):
            cmd.add_argument("--provider", choices=["EPA", "EEA"], default="EPA")
            cmd.add_argument(
                "--pollutant",
                choices=["CO", "NO2", "SO2", "PM25", "PM10", "O3", "all"]
                if name == "download"
                else ["CO", "NO2", "SO2", "PM25", "PM10", "O3"],
                default="CO",
            )
        if name in ("prepare", "run"):
            cmd.add_argument(
                "--seed", type=int, choices=[10, 20, 30, 40, 50], default=10
            )
            cmd.add_argument("--horizon", type=int, choices=[6, 12, 24], default=6)
            cmd.add_argument(
                "--collection",
                choices=["STANDARD", "MULTIYEAR_STABLE"],
                default="STANDARD",
            )
        if name == "model-sources":
            cmd.add_argument(
                "--model",
                choices=["DLinear+side", "iTransformer", "ModernTCN"],
                required=True,
            )
        if name == "prepare":
            cmd.add_argument(
                "--context",
                action="store_true",
                help="Include other pollutants and neighbors (EPA requires all six source series)",
            )
        if name == "run":
            cmd.add_argument("--model", default="CatBoost")
            cmd.add_argument(
                "--experiment",
                default="reference",
                help="Choose a protocol listed by the study command",
            )
            cmd.add_argument(
                "--weights",
                type=Path,
                help="Directory of provider-supplied weights for pretrained models",
            )
        if name == "evaluate":
            cmd.add_argument(
                "--forecast",
                type=Path,
                help="Forecast directory (default: EPA CO h6 seed10 CatBoost reference)",
            )
        if name == "study":
            cmd.add_argument(
                "--stage",
                choices=[
                    "all",
                    "acquire",
                    "prepare",
                    "validate",
                    "predict",
                    "evaluate",
                    "compare",
                ],
                default="all",
            )
            cmd.add_argument(
                "--execute",
                action="store_true",
                help="Execute the specified study protocol",
            )
            cmd.add_argument("--experiment")
            cmd.add_argument("--model")
            cmd.add_argument("--weights", type=Path)
    args = parser.parse_args()
    if args.command == "model-sources":
        from wdaoh.models.upstream import acquire_sources

        print(acquire_sources(args.work, args.model))
    elif args.command == "download":
        from wdaoh.data.download import download

        download(args.provider, args.pollutant, args.work / "raw")
    elif args.command == "prepare":
        from wdaoh.data.prepare import prepare

        prepare(
            args.work,
            args.provider,
            args.collection,
            args.pollutant,
            args.horizon,
            args.seed,
            args.context,
        )
    elif args.command == "run":
        from wdaoh.experiments.runner import run

        run(
            args.work,
            args.provider,
            args.collection,
            args.pollutant,
            args.horizon,
            args.seed,
            args.model,
            args.experiment,
            args.weights,
        )
    elif args.command == "evaluate":
        from wdaoh.evaluation.metrics import evaluate

        evaluate(
            args.forecast
            or args.work / "forecasts/EPA_PRIMARY_CO_h6_seed10/CatBoost/reference"
        )
    elif args.command == "study":
        from wdaoh.experiments.study import study

        study(
            args.work,
            args.execute,
            args.experiment,
            args.model,
            args.weights,
            args.stage,
        )


if __name__ == "__main__":
    main()
