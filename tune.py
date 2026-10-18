import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import precision_recall_fscore_support

from common import (RESULTS, SEED, dataset_split, label_name, load_data,  make_models, safe_n_folds, save_json, subset_for_tuning)


def tune_baseline(name, model, X, y, cv, jobs):
    if name == "Decision Tree":
        grid = {
            "m__criterion": ["gini", "entropy"],
            "m__max_depth": [10, 20, 30, None],
            "m__min_samples_leaf": [1, 5, 20, 50],
            "m__class_weight": [None, "balanced"],
        }
        search = RandomizedSearchCV(model, grid, n_iter=20, scoring="f1_macro",
                                    cv=cv, n_jobs=jobs, random_state=SEED, refit=False)
    else:
        # These are the same k, weight and distance choices as the provided prep.py.
        grid = {"m__n_neighbors": [1, 3, 5, 9, 15, 25],
                "m__weights": ["uniform", "distance"],
                "m__metric": ["euclidean", "manhattan"]}
        search = RandomizedSearchCV(model, grid, n_iter=12, scoring="f1_macro",
                                    cv=cv, n_jobs=jobs, random_state=SEED, refit=False)
    print(f"   Tuning baseline {name} ...", flush=True)
    search.fit(X, y)
    slug = "tree" if name == "Decision Tree" else "knn"
    pd.DataFrame(search.cv_results_).to_csv(RESULTS / f"{slug}_baseline_tuning.csv", index=False)
    return search.best_params_, float(search.best_score_)


def main():
    p = argparse.ArgumentParser(description="Phase 3 tuning and RF hyperparameter experiments")
    p.add_argument("--data", default="networkTraffic.csv")
    p.add_argument("--tune-sample", type=int, default=20000)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--rf-trials", type=int, default=20)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--quick", action="store_true", help="Smoke-test settings, not final report settings")
    args = p.parse_args()
    RESULTS.mkdir(exist_ok=True)

    X, y, audit = load_data(args.data)
    Xdev, _, ydev, _, duplicate_test = dataset_split(X, y)
    sample = min(args.tune_sample, len(Xdev))
    if args.quick:
        # A 1,000-row proportional subsample may contain just one Worms case.
        # Choose a small but class-safe sample, preserving the natural prevalence.
        # Two extra examples give a margin for integer rounding in stratification.
        min_class = int(ydev.value_counts().min())
        sample_needed = int(np.ceil((args.folds + 2) * len(ydev) / min_class))
        sample = min(len(Xdev), max(1000, sample_needed))
        args.rf_trials = 4
        args.jobs = 1
        print(f"Quick test: using {sample:,} rows to retain at least {args.folds} "
              "examples of the rarest class for CV.", flush=True)
    Xsub, ysub = subset_for_tuning(Xdev, ydev, sample)
    folds = safe_n_folds(ysub, args.folds)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    print(f"Development rows: {len(Xdev)}, tuning rows: {len(Xsub)}, CV folds: {folds}", flush=True)
    models = make_models(Xsub)
    # Ensemble size held at this value for the initial search AND for every
    # sweep except the n_estimators sweep itself. Reported in the metadata so
    # the paper can state it instead of implying the sweeps used 100 trees.
    sweep_fixed_trees = int(models["Random Forest"].named_steps["m"].n_estimators)

    rf_distributions = {
        "m__max_features": ["sqrt", 0.5, None],
        "m__max_depth": [15, 30, None],
        "m__min_samples_leaf": [1, 3, 10],
        "m__class_weight": [None, "balanced_subsample"],
    }
    # Keep 150 trees fixed for initial multivariate tuning; later sweep
    search = RandomizedSearchCV(
        models["Random Forest"], rf_distributions,
        n_iter=args.rf_trials, cv=cv, scoring="f1_macro",
        n_jobs=args.jobs, random_state=SEED, refit=False, return_train_score=True
    )
    t0 = time.perf_counter()
    print("Tuning Random Forest...", flush=True)
    search.fit(Xsub, ysub)
    pd.DataFrame(search.cv_results_).to_csv(RESULTS / "rf_tuning.csv", index=False)
    print(f"RF best mean inner macro-F1: {search.best_score_:.4f} ({time.perf_counter()-t0:.1f}s)", flush=True)
    rf_best = dict(search.best_params_)

    # Baselines are always re-tuned under the revised preprocessing, as the
    # report states. Assignment 2's saved GridSearchCV results are NOT reused:
    # its preprocessing differs, so its parameters are not comparable here.
    baseline = {}
    baseline_method = {}
    for name in ["Decision Tree", "kNN"]:
        if args.quick:
            baseline[name] = {}
            baseline_method[name] = "untuned default, quick mode only"
        else:
            baseline[name], score = tune_baseline(name, models[name], Xsub, ysub, cv, args.jobs)
            baseline_method[name] = f"fresh {folds}-fold tuning on development subset (inner score {score:.4f})"

    print("Running controlled Random Forest sweeps...", flush=True)
    sweep_settings = {
        "n_estimators": [1, 5, 25, 50, 100, 200, 400],
        "max_features": ["sqrt", 0.5, None],
        "max_depth": [15, 30, None],
        "min_samples_leaf": [1, 3, 10],
        "class_weight": [None, "balanced_subsample"],
    }
    if args.quick:
        sweep_settings = {k: v[:2] for k, v in sweep_settings.items()}
    rows = []
    for param, vals in sweep_settings.items():
        for value in vals:
            # All OTHER parameters held constant at best selected RF config.
            chosen = dict(rf_best)
            chosen[f"m__{param}"] = value
            estimator = make_models(Xsub, {"Random Forest": chosen})["Random Forest"]
            tic = time.perf_counter()
            result = cross_validate(estimator, Xsub, ysub, scoring="f1_macro", cv=cv, n_jobs=args.jobs, return_train_score=False)
            rows.append({
                "parameter": param, "value": "None" if value is None else str(value),
                "macro_f1_mean": result["test_score"].mean(),
                "macro_f1_std": result["test_score"].std(ddof=1) if folds > 1 else 0,
                "fit_seconds_mean": result["fit_time"].mean(),
                "elapsed_seconds": time.perf_counter()-tic,
                "folds": folds, "tuning_rows": len(Xsub),
            })
            print(f"  {param}={value}: {rows[-1]['macro_f1_mean']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "rf_sweeps.csv", index=False)

    # Predeclared computational trade-off decision: choose the smallest tree count
    # with mean macro-F1 within one standard error of the best sweep outcome.
    # Restrict to at least 25 trees in full mode; 1 and 5 are diagnostic ablations.
    sizes = [r for r in rows if r["parameter"] == "n_estimators"]
    deployment_candidates = [r for r in sizes if int(r["value"]) >= 25]
    if not deployment_candidates:
        deployment_candidates = sizes
    best = max(deployment_candidates, key=lambda r: r["macro_f1_mean"])
    threshold = best["macro_f1_mean"] - best["macro_f1_std"] / np.sqrt(folds)
    eligible = [r for r in deployment_candidates if r["macro_f1_mean"] >= threshold]
    chosen_n_trees = min(int(r["value"]) for r in eligible)
    rf_best["m__n_estimators"] = chosen_n_trees
    print(f"Selected n_estimators={chosen_n_trees} using smallest-within-1-SE heuristic", flush=True)

    # RQ4: class weighting AND tree complexity need per-class evidence, not
    # macro-F1 alone. Macro-F1 can rise while individual minority categories get worse, so each regularisation setting is re-scored per class using
    # out-of-fold predictions, holding every other parameter at the selected configuration
    per_class_experiments = {
        "class_weight": [None, "balanced_subsample"],
        "max_depth": [15, 30, None],
        "min_samples_leaf": [1, 3, 10],
    }
    if args.quick:
        per_class_experiments = {k: v[:2] for k, v in per_class_experiments.items()}
    labels = sorted(ysub.unique().tolist())
    per_class_rows = []
    for param, values in per_class_experiments.items():
        for value in values:
            print(f"   Per-class out-of-fold scoring: {param}={value}", flush=True)
            candidate = dict(rf_best)
            candidate[f"m__{param}"] = value
            estimator = make_models(Xsub, {"Random Forest": candidate})["Random Forest"]
            pred = cross_val_predict(estimator, Xsub, ysub, cv=cv, n_jobs=args.jobs)
            precision, recall, f1, support = precision_recall_fscore_support(
                ysub, pred, labels=labels, zero_division=0
            )
            for label, pval, rval, fval, count in zip(labels, precision, recall, f1, support):
                per_class_rows.append({
                    "parameter": param,
                    "value": "None" if value is None else str(value),
                    "class": label_name(label), "label": label,
                    "precision": pval, "recall": rval, "f1": fval,
                    "tuning_support": int(count),
                    "n_estimators": chosen_n_trees,
                })
    pd.DataFrame(per_class_rows).to_csv(RESULTS / "rf_per_class_effects.csv", index=False)

    config = {
        "Random Forest": rf_best,
        "Decision Tree": baseline["Decision Tree"],
        "kNN": baseline["kNN"],
    }
    save_json("selected_params.json", config)
    save_json("phase3_metadata.json", {
        "dataset": audit, "seed": SEED, "train_test_ratio": [0.8, 0.2],
        "development_rows": len(Xdev), "tuning_rows": len(Xsub),
        "tuning_folds": folds, "rf_trials": args.rf_trials,
        "rf_best_inner_macro_f1": float(search.best_score_),
        "selected_n_estimators": chosen_n_trees,
        "sweep_fixed_n_estimators": sweep_fixed_trees,
        "ensemble_size_selection": "Smallest >=25 trees within one standard error of the best mean sweep macro-F1 (quick mode permits smaller candidates)",
        "baseline_parameter_source": baseline_method,
        "duplicates_across_dev_test_feature_hash": duplicate_test,
        "quick_mode": args.quick,
        "methodology_warning": "Inner CV and sweeps are exploratory; selected-configuration CV on these observations is not an independent performance estimate. Phase 4 reports held-out evaluation.",
    })


if __name__ == "__main__":
    main()