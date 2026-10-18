# Random Forest Ensemble Learning for Network Intrusion Detection

Code for the Assignment 4 ensemble-learning study on the course UNSW-NB15
traffic dataset (`networkTraffic.csv`). The pipeline tunes a random forest,
compares it against re-tuned decision-tree and k-nearest-neighbour baselines
on a locked held-out partition, runs controlled hyperparameter experiments,
and performs a duplicate-aware robustness check.

---

## 1. Requirements

### Install

Create and activate a virtual environment, then install the dependencies:

```bash
# Windows (Git Bash / PowerShell)
python -m venv .venv
source .venv/Scripts/activate      # PowerShell: .venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate

pip install "numpy>=1.21" "pandas>=1.3" "scikit-learn>=1.1" "matplotlib>=3.4"
```

## 2. Input data

Place the lecturer's **`networkTraffic.csv`** in the project folder (the same
folder as the scripts). It is not included in this submission. Every script
defaults to this filename and can be pointed elsewhere with `--data <path>`.

## 3. How to run

Run the four numbered scripts **in order** from the project folder. Each phase
reads the files written by the previous one, so the order matters. All results
are written to a `results/` folder, which is created automatically.
Be patient with her she needs a min to run.

```bash
python tune.py                # tuning + controlled experiments
python evaluate.py            # held-out comparison
python report_assets.py       # figures for the report
python group_robustness.py    # duplicate-aware robustness check
```

### Useful options

- `python 03_tune.py --quick` — fast smoke test on a small sample. **Not for
  report results.** If Phase 3 was run with `--quick`, Phase 4 refuses to run
  unless given `--allow-quick`.
- `--data <path>` — use a dataset in another location (all scripts).
- `--jobs <n>` — number of parallel workers (`03`, `04`, `06`).

## 4. What each file does

### Scripts

| File                  | Role                                                                                                                                                                                                                                                                                                           |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `common.py`           | Shared module (not run directly). Deterministic data cleaning, the leak-safe preprocessing pipelines, the three model definitions, the stratified 80:20 split, and helpers. Adapted from the Assignment 2 `prep.py`.                                                                                           |
| `tune.py`             | **Phase 3.** Randomised search for the random forest, re-tuning of the decision-tree and kNN baselines, five one-factor-at-a-time sweeps (trees, features, depth, leaf size, class weighting), the 1-standard-error ensemble-size rule, and a controlled per-class analysis of weighting, depth and leaf size. |
| `evaluate.py`         | **Phase 4.** Fits the selected models on the full development set, evaluates them on the locked held-out test set, records per-class metrics and timings, and runs a paired stratified bootstrap of the macro-F1 differences.                                                                                  |
| `report_assets.py`    | **Phase 5.** Generates the five controlled-sweep figures used in the report.                                                                                                                                                                                                                                   |
| `group_robustness.py` | **Phase 6.** Repeats the held-out comparison under a group-aware partition in which identical feature vectors cannot straddle the split, using the already-selected configurations (no re-tuning).                                                                                                             |

### Key outputs (written to `results/`)

| File                                                      | Written by | Report use                                |
| --------------------------------------------------------- | ---------- | ----------------------------------------- |
| `selected_params.json`                                    | Phase 3    | Table I (selected configurations)         |
| `rf_tuning.csv`, `rf_sweeps.csv`                          | Phase 3    | sweep numbers and Figures 1–5             |
| `tree_baseline_tuning.csv`, `knn_baseline_tuning.csv`     | Phase 3    | evidence baselines were re-tuned          |
| `rf_per_class_effects.csv`                                | Phase 3    | Table VI (controlled per-class weighting) |
| `phase3_metadata.json`                                    | Phase 3    | seed, sample sizes, selection rule        |
| `model_test_comparison.csv`                               | Phase 4    | Table II (held-out performance)           |
| `model_cv_summary.csv`                                    | Phase 4    | Table III (CV diagnostics)                |
| `model_per_class.csv`                                     | Phase 4    | Table IV (per-class held-out)             |
| `paired_bootstrap_macro_f1.csv`                           | Phase 4    | bootstrap intervals                       |
| `phase4_metadata.json`                                    | Phase 4    | duplicate audit, class labels             |
| `figures/rf_sweep_*.png`                                  | Phase 5    | Figures 1–5                               |
| `group_split_comparison.csv`, `group_split_per_class.csv` | Phase 6    | Table V (duplicate-aware)                 |
| `phase6_metadata.json`                                    | Phase 6    | group-split audit                         |

## 5. Notes

- A fixed random seed (42) is set throughout, so runs are reproducible.
- All data-dependent preprocessing (imputation, encoding, quantile scaling) is
  fitted inside scikit-learn pipelines on training folds only, so no test
  information leaks into tuning.
- On load, the data audit prints a warning reporting how many test rows share
  identical feature vectors with development rows. This is expected and is
  investigated directly in Phase 6

## 6. AI declaration

- Helped me Construct the figures for my report.
- Helped me break down all the content and generated me a step guide on the different phases on what each entailed.
- Used in conjuncture with the specified writing rules to ensure I adhered to them
