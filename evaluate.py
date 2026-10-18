import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_validate

from common import (RESULTS, SEED, dataset_split, label_name, load_data, make_models, safe_n_folds, save_json, subset_for_tuning)


def stratified_bootstrap_diff(y_true, pred_a, pred_b, labels, other_name, n_boot=250):
    rng = np.random.default_rng(SEED)
    y = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    per_class = [np.flatnonzero(y == label) for label in labels]
    values = []
    for _ in range(n_boot):
        ix = np.concatenate([rng.choice(indices, size=len(indices), replace=True)
                             for indices in per_class])
        values.append(f1_score(y[ix], a[ix], labels=labels, average="macro", zero_division=0)
                      - f1_score(y[ix], b[ix], labels=labels, average="macro", zero_division=0))
    return {
        "comparison": "Random Forest minus " + other_name,
        "observed_difference": f1_score(y, a, labels=labels, average="macro", zero_division=0)
                               - f1_score(y, b, labels=labels, average="macro", zero_division=0),
        "ci_95_low": float(np.percentile(values, 2.5)),
        "ci_95_high": float(np.percentile(values, 97.5)),
        "bootstrap_repeats": n_boot,
    }


def main():
    p = argparse.ArgumentParser(description="Phase 4 comparison and held-out testing")
    p.add_argument("--data", default="networkTraffic.csv")
    p.add_argument("--cv-sample", type=int, default=20000,
                   help="Development sample for descriptive CV (0 = full development)")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--bootstrap-repeats", type=int, default=250)
    p.add_argument("--allow-quick", action="store_true", help="Only for smoke tests, not publishable results")
    args = p.parse_args()
    if not (RESULTS / "selected_params.json").exists():
        raise FileNotFoundError("Missing results/selected_params.json. Run 03_tune.py first.")
    with open(RESULTS / "selected_params.json", encoding="utf-8") as f:
        params = json.load(f)
    meta_path = RESULTS / "phase3_metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        tune_meta = json.load(f)
    if tune_meta["quick_mode"] and not args.allow_quick:
        raise RuntimeError("Phase 3 used --quick; rerun its FULL settings for report-quality results (or --allow-quick for testing only).")

    X, y, audit = load_data(args.data)
    Xdev, Xtest, ydev, ytest, overlapping = dataset_split(X, y)
    labels = sorted(y.unique().tolist())
    if tune_meta["quick_mode"] and args.allow_quick and args.cv_sample > 0:
        # In quick mode, preserve the real class proportions while providing enough of the rarest class for the requested number of CV folds.
        rarest = int(ydev.value_counts().min())
        minimum = int(np.ceil((args.cv_folds + 2) * len(ydev) / rarest))
        if args.cv_sample < minimum:
            args.cv_sample = min(len(Xdev), minimum)
            print(f"Quick CV sample increased to {args.cv_sample:,} rows so the rarest class can be represented in {args.cv_folds} folds.", flush=True)
    Xcv, ycv = subset_for_tuning(Xdev, ydev, args.cv_sample)
    folds = safe_n_folds(ycv, args.cv_folds)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    models = make_models(Xdev, params)
    records, perclass, cv_summary = [], [], []
    preds = {}
    print(f"Development: {len(Xdev)}; locked final test: {len(Xtest)}; CV diagnostic sample: {len(Xcv)}", flush=True)

    for name, model in models.items():
        print(f"\nEvaluating {name}...", flush=True)
        scores = cross_validate(clone(model), Xcv, ycv, cv=cv,
                                scoring={"macro_f1": "f1_macro",
                                         "accuracy": "accuracy",
                                         "balanced_accuracy": "balanced_accuracy"},
                                return_train_score=False, n_jobs=args.jobs)
        cv_summary.append({
            "model": name, "folds": folds, "cv_sample_rows": len(Xcv),
            "macro_f1_mean": np.mean(scores["test_macro_f1"]),
            "macro_f1_sd": np.std(scores["test_macro_f1"], ddof=1),
            "accuracy_mean": np.mean(scores["test_accuracy"]),
            "accuracy_sd": np.std(scores["test_accuracy"], ddof=1),
            "balanced_accuracy_mean": np.mean(scores["test_balanced_accuracy"]),
            "balanced_accuracy_sd": np.std(scores["test_balanced_accuracy"], ddof=1),
        })

        start = time.perf_counter()
        model.fit(Xdev, ydev)
        fit_time = time.perf_counter() - start
        start = time.perf_counter()
        pred = model.predict(Xtest)
        prediction_time = time.perf_counter() - start
        preds[name] = pred
        mean_f1 = f1_score(ytest, pred, labels=labels, average="macro", zero_division=0)
        records.append({
            "model": name, "test_macro_f1": mean_f1,
            "test_accuracy": accuracy_score(ytest, pred),
            "test_balanced_accuracy": balanced_accuracy_score(ytest, pred),
            "training_seconds": fit_time,
            "prediction_seconds": prediction_time,
            "development_rows_used_for_final_fit": len(Xdev),
            "test_rows": len(Xtest)
        })
        precision, recall, f1, supports = precision_recall_fscore_support(
            ytest, pred, labels=labels, zero_division=0
        )
        for label, pp, rr, ff, ss in zip(labels, precision, recall, f1, supports):
            perclass.append({"model": name, "class": label_name(label), "label": label, "precision": pp, "recall": rr, "f1": ff, "test_support": int(ss)})

        print(f"   test macro-F1={mean_f1:.4f}  accuracy={records[-1]['test_accuracy']:.4f}  "
              f"fit={fit_time:.1f}s  predict={prediction_time:.1f}s", flush=True)

    pd.DataFrame(records).to_csv(RESULTS / "model_test_comparison.csv", index=False)
    pd.DataFrame(perclass).to_csv(RESULTS / "model_per_class.csv", index=False)
    pd.DataFrame(cv_summary).to_csv(RESULTS / "model_cv_summary.csv", index=False)

    ci_rows = []
    if args.bootstrap_repeats:
        for other in ("Decision Tree", "kNN"):
            result = stratified_bootstrap_diff(ytest, preds["Random Forest"], preds[other], labels, other_name=other, n_boot=args.bootstrap_repeats)
            result["comparison"] = f"Random Forest - {other}"
            ci_rows.append(result)
        pd.DataFrame(ci_rows).to_csv(RESULTS / "paired_bootstrap_macro_f1.csv", index=False)

    save_json("phase4_metadata.json", {
        "dataset": audit, "seed": SEED, "test_ratio": 0.2,
        "cv_rows": len(Xcv), "cv_folds": folds,
        "fit_development_rows": len(Xdev), "locked_test_rows": len(Xtest),
        "class_labels": [label_name(x) for x in labels],
        "overlap_test_rows_identical_features_in_dev": overlapping,
        "quick_mode": tune_meta["quick_mode"],
        "interpretation_warning": "CV of fixed chosen models reuses development data previously used to select hyperparameters; treat as diagnostics, not independent generalisation estimates. Interpret held-out test as the primary comparison. Bootstrap uncertainty is conditional on this test sample and fitted models."
    })


if __name__ == "__main__":
    main()