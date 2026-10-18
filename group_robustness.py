"""
The stratified 80:20 split lets observations with identical descriptive
features fall on both sides of the partition, which can flatter the test
score. This phase repeats the held-out comparison under a partition that
groups identical feature vectors, so no feature vector seen in training can
reappear in the test set. The SAME selected configurations are used; nothing
is re-tuned here, so any change in performance is attributable to the
partitioning rule rather than to a different model.

The comparison is descriptive. A group-aware split is not stratified, so the
class proportions shift slightly; the per-class supports are reported so the
change can be judged.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support)
from sklearn.model_selection import GroupShuffleSplit
from common import RESULTS, SEED, label_name, load_data, make_models, save_json


def main():
    p = argparse.ArgumentParser(description="Phase 6 duplicate-aware robustness check")
    p.add_argument("--data", default="networkTraffic.csv")
    p.add_argument("--jobs", type=int, default=2)
    args = p.parse_args()

    if not (RESULTS / "selected_params.json").exists():
        raise FileNotFoundError("Missing results/selected_params.json. Run 03_tune.py first.")
    with open(RESULTS / "selected_params.json", encoding="utf-8") as f:
        params = json.load(f)

    X, y, audit = load_data(args.data)
    labels = sorted(y.unique().tolist())

    # Group label = hash of the descriptive feature vector. Rows sharing a
    # feature vector share a group and therefore always land in the same partition, whatever their target label.
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    n_groups = len(np.unique(groups))
    print(f"{len(X):,} rows form {n_groups:,} distinct feature vectors "
          f"({len(X)-n_groups:,} rows are repeats of another row).", flush=True)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    dev_idx, test_idx = next(splitter.split(X, y, groups=groups))
    Xdev, Xtest = X.iloc[dev_idx], X.iloc[test_idx]
    ydev, ytest = y.iloc[dev_idx], y.iloc[test_idx]

    # Verify the partition actually achieved what it claims.
    leaked = len(set(groups[test_idx]) & set(groups[dev_idx]))
    if leaked:
        raise RuntimeError(f"Group split failed: {leaked} feature vectors on both sides.")
    missing = set(labels) - set(ytest.unique())
    if missing:
        print(f"WARNING: group-aware test partition contains no examples of {[label_name(m) for m in missing]}", flush=True)

    print(f"Group-aware split: {len(Xdev):,} development, {len(Xtest):,} test, 0 shared feature vectors.", flush=True)

    models = make_models(Xdev, params)
    records, perclass = [], []
    for name, model in models.items():
        print(f"\nEvaluating {name} on the group-aware partition...", flush=True)
        start = time.perf_counter()
        model.fit(Xdev, ydev)
        fit_time = time.perf_counter() - start
        pred = model.predict(Xtest)
        macro = f1_score(ytest, pred, labels=labels, average="macro", zero_division=0)
        records.append({
            "model": name,
            "group_split_macro_f1": macro,
            "group_split_accuracy": accuracy_score(ytest, pred),
            "group_split_balanced_accuracy": balanced_accuracy_score(ytest, pred),
            "training_seconds": fit_time,
            "development_rows": len(Xdev),
            "test_rows": len(Xtest),
        })
        precision, recall, f1, supports = precision_recall_fscore_support(
            ytest, pred, labels=labels, zero_division=0
        )
        for label, pp, rr, ff, ss in zip(labels, precision, recall, f1, supports):
            perclass.append({"model": name, "class": label_name(label), "label": label,
                             "precision": pp, "recall": rr, "f1": ff,
                             "group_test_support": int(ss)})
        print(f" group-aware macro-F1={macro:.4f} (fit {fit_time:.1f}s)", flush=True)

    comparison = pd.DataFrame(records)

    # Attach the stratified-split result so the report can quote the change
    # directly instead of the reader having to cross-reference two files.
    stratified_path = RESULTS / "model_test_comparison.csv"
    if stratified_path.exists():
        stratified = pd.read_csv(stratified_path)[["model", "test_macro_f1"]]
        stratified = stratified.rename(columns={"test_macro_f1": "stratified_split_macro_f1"})
        comparison = comparison.merge(stratified, on="model", how="left")
        comparison["macro_f1_change"] = (comparison["group_split_macro_f1"] - comparison["stratified_split_macro_f1"])
    else:
        print("Note: model_test_comparison.csv not found", flush=True)

    comparison.to_csv(RESULTS / "group_split_comparison.csv", index=False)
    pd.DataFrame(perclass).to_csv(RESULTS / "group_split_per_class.csv", index=False)

    save_json("phase6_metadata.json", {
        "dataset": audit, "seed": SEED, "test_ratio": 0.20,
        "split_rule": "GroupShuffleSplit on a hash of the descriptive feature vector; "
                      "rows with identical features always share a partition",
        "stratified": False,
        "distinct_feature_vectors": int(n_groups),
        "repeated_rows": int(len(X) - n_groups),
        "development_rows": int(len(Xdev)), "test_rows": int(len(Xtest)),
        "shared_feature_vectors_across_partitions": 0,
        "class_counts_group_test": {label_name(k): int(v) for k, v in ytest.value_counts().items()},
        "interpretation_warning": "Models were NOT re-tuned for this partition; the selected "
                                  "configurations come from the stratified development data, so "
                                  "this is a robustness check on the partitioning rule, not an "
                                  "independent model-selection experiment. The group-aware split "
                                  "is unstratified, so class proportions differ from the main split.",
    })

    if "macro_f1_change" in comparison.columns:
        print("\nMacro-F1 under each partitioning rule:")
        for _, r in comparison.iterrows():
            print(f"  {r['model']:<16} stratified {r['stratified_split_macro_f1']:.4f} -> "
                  f"group-aware {r['group_split_macro_f1']:.4f} "
                  f"({r['macro_f1_change']:+.4f})")



if __name__ == "__main__":
    main()
