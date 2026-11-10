"""Registered comparisons, information arms, frozen parameters and their statistical units."""

from wdaoh.protocol import settings

SEVEN = (
    "CatBoost",
    "DLinear+side",
    "GRU-D",
    "ModernTCN",
    "iTransformer",
    "TabPFN",
    "TabICL",
)
FIVE = ("CatBoost", "DLinear+side", "iTransformer", "TabPFN", "TabICL")
FOUR = ("CatBoost", "DLinear+side", "TabPFN", "TabICL")
REFERENCE = "S0_TARGET24"
CONTEXT = "S1_CONTEXT"
NINE = (
    "C",
    "C_T",
    "C_T_MT",
    "C_SV",
    "C_SV_SA",
    "C_CV",
    "C_CV_CA",
    "Extended_Values",
    "Full",
)
FACTORIAL = ("P1_00", "P1_10", "P1_01", "P1_11")
COMPONENTS = ("P2_V", "P2_VM", "P2_VC", "BASE10")
GRAPH = ("SELF_ONLY", "GRAPH_NEIGHBORS", "MASKED_NEIGHBOR_POOL")
# The reference example uses the actual reference-arm configuration locked by 2024 validation.
DESIGNS = {
    "reference": ("EPA", SEVEN, (REFERENCE,)),
    "joint": ("EPA", SEVEN, (REFERENCE, CONTEXT)),
    "sources": ("EPA", FIVE, (*FACTORIAL, "BASE10")),
    "components": ("EPA", FIVE, COMPONENTS),
    "compact": ("EPA", FIVE, ("P1_11", "P3_COMPACT11", "P1_01")),
    "graph": ("EPA", ("LocalSAGE-TCN",), GRAPH),
    "raw-selection": ("EPA", SEVEN, (REFERENCE, CONTEXT)),
    "nine-view": ("EPA", FOUR, NINE),
    "native-views": (
        "EPA",
        ("iTransformer",),
        (
            "TARGET24_VALUES",
            "TARGET24_MASK",
            "CROSS_VALUES",
            "CROSS_MASK",
            "SPATIAL_VALUES",
            "SPATIAL_MASK",
            "ALL_VALUES",
            "ALL_MASK",
            "TARGET168_MASK",
        ),
    ),
    "long-history": ("EPA", FOUR, ("A_S0", "B_S0_H", "C_S1", "D_S1_H")),
    "temporal": ("EPA", FOUR, ("C", "C_T", "C_T_MT")),
    "budget": (
        "EPA",
        FOUR,
        tuple(
            f"{arm}@{n}" for n in (1000, 5000, 10000) for arm in (REFERENCE, CONTEXT)
        ),
    ),
    "eea-selection": ("EEA", FIVE, (REFERENCE, CONTEXT)),
}
SELECTED = (
    "long-history",
    "raw-selection",
    "nine-view",
    "native-views",
    "eea-selection",
)


def parameter_set(model, pollutant, horizon, seed, arm, experiment, collection):
    """Resolve each setting's registered reference or transferred configuration independently of predictions."""
    if experiment == "budget":
        key = f"{model}/{pollutant}/{horizon}/{seed}/{arm}"
        transfer = settings("budget_choices").get(key)
        if transfer is not None:
            return settings("model_parameters")[model][transfer]
    if experiment == "graph":
        return settings("local_graph")
    if experiment in ("nine-view", "temporal") and model in (
        "CatBoost",
        "DLinear+side",
    ):
        return settings("structured_parameters")[f"{model}/{pollutant}/{horizon}"]
    if experiment == "long-history":
        override = settings("long_history_parameters").get(
            f"{model}/{pollutant}/{horizon}/{seed}/{arm}"
        )
        if override is not None:
            return override
    base_arm = arm.split("@")[0]
    if base_arm not in (REFERENCE, CONTEXT):
        base_arm = REFERENCE
    key = f"{model}/{pollutant}/{horizon}/{seed}/{base_arm}"
    if experiment == "eea-selection" and model not in ("TabPFN", "TabICL"):
        chosen = settings("eea_choices")[collection + "/" + key]
    else:
        chosen = settings("reference_choices")[key]
    return settings("model_parameters")[model][chosen]
