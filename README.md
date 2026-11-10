# WDAOH

**When Do Additional Observations Help? Support-Conditioned Evaluation in Air-Quality Forecasting**

This repository contains the code for evaluating how additional air-quality observations affect forecasting when the target station has different amounts of recent measurement history. It supports matched comparisons of target history, auxiliary pollutants, and observations from other monitoring stations across several forecasting models.

## Data

Our experiments use hourly monitoring records from 2021–2025:

- **U.S. EPA Air Quality System (AQS):** [AirData hourly downloads](https://aqs.epa.gov/aqsweb/airdata/download_files.html)
- **European Environment Agency (EEA):** [Air quality data service](https://www.eea.europa.eu/en/datahub/datahubitem-view/778ef9f5-6293-4846-badd-56a29c70880d)

The code downloads and checks the required official files. Training uses 2021–2023, validation uses 2024, and testing uses 2025. No copy of the official measurements is bundled with the repository.

## Installation

The tested CPU environment is **Python 3.12 on Linux x86-64**. In an activated Python environment, run:

```bash
python -m pip install -r requirements.txt
```

The same requirements file covers all models. For GPU runs, select a compatible build using the [PyTorch 2.7.1 installation instructions](https://pytorch.org/get-started/previous-versions/#v271).

## Quick start

From the repository root:

```bash
python -m wdaoh download
python -m wdaoh prepare
python -m wdaoh run
python -m wdaoh evaluate
```

These commands download official EPA CO measurements, prepare the registered station and query samples, fit the registered CatBoost reference configuration, evaluate it using 2024 validation queries, generate 2025 forecasts, and report evaluation metrics. The default example predicts **CO six hours ahead (seed 10)**. Results are saved under `runs/`.

To run a different verified EPA example, specify the target pollutant, forecast horizon, and seed in the preparation and execution commands:

```bash
python -m wdaoh download --pollutant PM10
python -m wdaoh prepare --pollutant PM10 --horizon 24 --seed 40
python -m wdaoh run --pollutant PM10 --horizon 24 --seed 40
python -m wdaoh evaluate --forecast runs/forecasts/EPA_PRIMARY_PM10_h24_seed40/CatBoost/reference
```

## European monitoring example

An independently exercised EEA example uses the source-consistent collection and two validation objectives:

```bash
python -m wdaoh download --provider EEA --pollutant all --work runs/eea
python -m wdaoh prepare --provider EEA --collection MULTIYEAR_STABLE --seed 30 --work runs/eea
python -m wdaoh run --provider EEA --collection MULTIYEAR_STABLE --seed 30 --experiment eea-selection --work runs/eea
python -m wdaoh evaluate --forecast runs/eea/forecasts/EEA_MULTIYEAR_STABLE_CO_h6_seed30/CatBoost/eea-selection
```

## Models and experiments

The study includes CatBoost, DLinear, GRU-D, ModernTCN, iTransformer, TabPFN, and TabICL, plus a separate graph-based sensitivity experiment. Input alternatives, model configurations, training and validation choices, and evaluation procedures are implemented in the `wdaoh/` package.

To inspect the available experiments and run options:

```bash
python -m wdaoh study
python -m wdaoh study --help
```

The default example has been independently exercised from official data. Larger model schedules require their corresponding software, hardware, and (for pretrained models) provider checkpoints.

## Tests and third-party notices

```bash
python -m unittest discover -s tests -v
```

## License

Original project code is licensed under the MIT License.
