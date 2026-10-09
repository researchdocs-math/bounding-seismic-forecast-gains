# Bounding Seismic Forecast Gains

Reproducible experiments for 30-day earthquake occurrence forecasts across Italy, Greece, Alaska, the Chile region, and the Japan region.

The benchmark contains 152 historically selected cells and 10,944 evaluation forecasts. Three successive geographic audits test spatial forecast-error borrowing, activity duration conditional on multiscale rates, and direct online calibration correction. Protocols preserve the distinction between development and new geographic evaluation. The primary result is a set of conditional upper bounds on added-model skill, rather than a claim of a newly discovered earthquake precursor.

## Installation

Use Python 3.12 and install the pinned dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

XGBoost requires a working native OpenMP runtime on the host platform. Optional graphics-processor verification uses `requirements-gpu.txt`. CPU computation is sufficient for all published results. The original calculations used up to eight CPU workers; double-precision CPU calculations determine the published uncertainty bounds.

## Reproduce

Recompute feature selection, forecasts, uncertainty intervals, numerical diagnostics, and synthetic controls from the included processed baseline matrices:

```sh
python reproduce.py
```

Rebuild catalog cohorts and ETAS baseline forecasts from the frozen raw responses as well:

```sh
python reproduce.py --full
```

Both commands run offline. They write analysis outputs to `results`, keep logs in `run_logs`, and compare numeric outputs and selected specifications against `expected/results`. A fresh full rebuild takes longer than replay from the baseline matrices. Eight parallel workers are used for independent fitting jobs. The included event catalogs and target matrices are checked for duplicate event IDs and recomputed occurrence labels.

## Files

- `data/raw/`: immutable annual USGS responses and the inherited INGV and NOA source snapshots.
- `data/processed/`: cleaned regional catalogs, eligible cells, ETAS forecast matrices, and constructed feature matrices.
- `data/*manifest.json`: acquisition queries, provider URLs, response sizes, and checksums.
- `experiments/etas.py`: marked temporal ETAS likelihood, analytic gradient, and branching occurrence probability.
- `experiments/build_forecasts.py` and `features.py`: annual ETAS fits, original catalog descriptors, and baseline probabilities.
- `experiments/borrowed_memory.py`: constructs the borrowed-error matrices from baseline forecasts, pools completed errors over space and time, and performs historical model selection.
- `experiments/evaluate_borrowed.py`: first-stage paired evaluation, annual updating, and temporal block bootstrap.
- `experiments/persistence_features.py` and `persistence_models.py`: multiscale counts, activity duration, spatial spread, magnitude tails, and matched learners.
- `experiments/evaluate_persistence.py`: second-stage evaluation with the frozen duration-family choice.
- `experiments/online_calibration.py` and `evaluate_online.py`: annually issued reference probabilities, regularized error correction, and the final Japan audit.
- `experiments/final_checks.py`: independently reconstructed labels, numerical gradient and quadrature checks, conditional skill bounds, and optional CPU/GPU comparison.
- `experiments/synthetic_control.py`: known-drift simulations and a future-label perturbation test of causal updates.
- `protocol/`: three analysis protocols, selection digests, and evaluation locks.
- `expected/results/`: reference numerical outputs for verification.
- `results/`: recorded forecasts, scores, uncertainty estimates, diagnostics, fitted model states, and the full-rebuild verification report.
- `SHA256SUMS`: checksums of the distributed inputs, programs, and numerical outputs.
- `results/etas_*.json`: fitted regional model states used by the quick reproduction mode.
- `environment.json`: versions used for the recorded calculations.
- `DATA_SOURCES.md`, `LICENSE`, and `CITATION.cff`: provenance, software reuse terms, and citation information.

## Reading the outputs

Positive paired `gain` means lower log loss for the additional model. Units are nats per cell–origin forecast. All cells at a given origin remain together in the bootstrap. The default 20,000 draws use blocks of 3, 6, and 12 origins. `skill_resolution.csv` applies a one-sided tail probability of 0.05/9 to the three primary geographic audits and three block lengths. These are conditional bounds for the fitted models and recorded catalogs.

The final model choices use only pre-2020 outcomes in each region. Previously examined audit regions become development evidence in later stages. The duration mechanism was the least unfavorable stage-two additional family on mean development loss and did not beat the multiscale-rate reference in that selection. The protocols and full score tables retain that result. Catalog snapshots are retrospective preferred solutions; the experiment does not reconstruct historical catalog publication latency.

The synthetic control holds its own six-origin regional correction and penalty of four fixed. It adds known autocorrelated log-odds drift to simulated Bernoulli occurrence probabilities; it checks the statistical procedure and is separate from empirical earthquake evidence.
