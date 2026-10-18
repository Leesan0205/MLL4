from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TARGET = "attack_cat"
NOMINAL = ["proto", "state", "service"]
ATTACK_NAMES = {0: "Normal", 1: "Reconnaissance", 2: "Backdoor", 3: "DoS", 4: "Exploits", 5: "Analysis", 6: "Fuzzers", 7: "Worms", 8: "Shellcode", 9: "Generic"}
KNN_DROP = [
    "sloss", "dloss", "spkts", "ct_srv_dst", "ct_src_dport_ltm",
    "dwin", "ct_ftp_cmd", "stcpb", "dtcpb", "response_body_len"
]
RESULTS = Path("results")


def json_convert(value):
    if isinstance(value, dict):
        return {str(k): json_convert(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_convert(x) for x in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is pd.NA:
        return None
    return value


def save_json(filename, obj):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / filename, "w", encoding="utf-8") as handle:
        json.dump(json_convert(obj), handle, indent=2, allow_nan=False)


def label_name(label):
    try:
        if float(label).is_integer() and int(label) in ATTACK_NAMES:
            return ATTACK_NAMES[int(label)]
    except (ValueError, TypeError):
        pass
    return str(label)


def load_data(csv_path="networkTraffic.csv"):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset missing: {csv_path.resolve()}. Copy your lecturer's networkTraffic.csv here.")
    df = pd.read_csv(csv_path, na_values=["?"]).copy()
    missing = [c for c in NOMINAL + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing expected columns: {missing}")
    original_n = len(df)
    if df[TARGET].isna().any():
        warnings.warn(f"Dropping {df[TARGET].isna().sum()} rows without an attack-category target; do not impute targets.")
        df = df.dropna(subset=[TARGET]).copy()
    df = df.drop(columns=["id"], errors="ignore")
    for col in ("is_ftp_login", "ct_ftp_cmd"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(upper=1)
    df["service"] = df["service"].fillna("none")
    for col in df.columns.difference(NOMINAL + [TARGET]):
        if df[col].dtype == "object":
            try_numeric = pd.to_numeric(df[col], errors="coerce")
            if try_numeric.notna().sum() + df[col].isna().sum() >= len(df):
                df[col] = try_numeric
            else:
                raise ValueError(f"Unexpected nonnumeric values in numeric feature '{col}'. Revisit cleaning.")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    if y.nunique() < 2:
        raise ValueError("At least two target classes are needed.")
    if y.value_counts().min() < 2:
        raise ValueError("At least one class has <2 rows; stratified 80/20 split is not possible as configured.")
    audit = {
        "dataset_file": str(csv_path), "original_rows": original_n,
        "rows_after_target_cleaning": len(df), "descriptive_features": X.shape[1],
        "classes": {label_name(k): int(v) for k, v in y.value_counts().items()},
        "dropped_id": True, "service_null_filled_with": "none",
        "ftp_values_clipped_upper": 1,
        "numeric_missing_cells": int(X.select_dtypes(include="number").isna().sum().sum()),
        "nominal_missing_cells": int(X[NOMINAL].isna().sum().sum()),
    }
    return X, y, audit


def dataset_split(X, y):
    Xdev, Xtest, ydev, ytest = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    # Hash feature rows only; if hashes intersect, report possible duplicate leakage.
    train_hashes = set(pd.util.hash_pandas_object(Xdev, index=False).tolist())
    test_hashes = pd.util.hash_pandas_object(Xtest, index=False)
    duplicates = int(test_hashes.isin(train_hashes).sum())
    if duplicates:
        warnings.warn(f"{duplicates} test rows share identical descriptive features with development rows. Discuss duplicate/related traffic risks.")
    return Xdev, Xtest, ydev, ytest, duplicates


def subset_for_tuning(X, y, n=20000):
    if n <= 0 or n >= len(X):
        return X, y
    if n < y.nunique():
        raise ValueError("Tuning sample is smaller than the number of classes.")
    Xsub, _, ysub, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=SEED
    )
    if set(ysub.unique()) != set(y.unique()):
        raise ValueError("The tuning subset omitted one or more attack classes")
    return Xsub, ysub


def safe_n_folds(y, asked):
    low = int(y.value_counts().min())
    if low < 2:
        raise ValueError("Too few observations of at least one class for stratified CV.")
    if low < asked:
        warnings.warn(f"Smallest class contains {low} observations; reducing CV folds from {asked} to {low}.")
        return low
    return asked


def build_tree_preprocessor(X):
    num_cols = [c for c in X.select_dtypes(include="number") if c not in NOMINAL]
    if not num_cols:
        raise ValueError("No numerical columns detected.")
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=False))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.001))
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, num_cols), ("cat", cat_pipe, NOMINAL)
    ], sparse_threshold=0.3)


def build_knn_preprocessor(X):
    num_cols = [c for c in X.select_dtypes(include="number") if c not in NOMINAL + KNN_DROP]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("quantiles", QuantileTransformer(
            output_distribution="uniform", n_quantiles=1000,
            subsample=100000, random_state=SEED
        ))
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.001))
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, num_cols), ("cat", cat_pipe, NOMINAL)
    ], sparse_threshold=0.3)


def make_models(X, config=None):
    config = config or {}
    rf = Pipeline([
        ("pre", build_tree_preprocessor(X)),
        ("m", RandomForestClassifier(n_estimators=150, bootstrap=True, random_state=SEED, n_jobs=1))
    ])
    dt = Pipeline([
        ("pre", build_tree_preprocessor(X)),
        ("m", DecisionTreeClassifier(random_state=SEED))
    ])
    knn = Pipeline([
        ("pre", build_knn_preprocessor(X)),
        ("m", KNeighborsClassifier())
    ])
    models = {"Random Forest": rf, "Decision Tree": dt, "kNN": knn}
    for name, model in models.items():
        if config.get(name):
            model.set_params(**config[name])
    return models
