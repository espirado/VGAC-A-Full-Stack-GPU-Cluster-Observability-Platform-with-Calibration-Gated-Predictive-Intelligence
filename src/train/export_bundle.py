#!/usr/bin/env python3
"""
Train and export a calibrated submit-time classifier bundle for production use.

Outputs (by default to results/models/v2025/production/):
- model.joblib           (sklearn Pipeline incl. preprocessing and base model)
- calibrator.joblib      (IsotonicRegression on validation scores)
- thresholds.json        (selected operating_threshold and optional band cuts)
- feature_schema.json    (feature names by type used by the pipeline)
- manifest.json          (metadata: versions, metrics, columns, paths)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sqlite3

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None


ROOT = Path("/Users/andrewespira/Downloads/st_peters/research-fall2025")
DEFAULT_OUT = ROOT / "results" / "models" / "v2025" / "production"
LOG_DB_PATH = ROOT / "results" / "logs.db"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_feedback_logs(db_path: Path) -> pd.DataFrame:
    """
    Load completed jobs from the feedback loop (logs.db).
    Only selects records with 'outcome' populated.
    """
    if not db_path.exists():
        return pd.DataFrame()
    
    try:
        with sqlite3.connect(db_path) as conn:
            query = """
            SELECT 
                job_id, cpu_request, memory_request, gpu_request, max_instance_per_node,
                partition, app_name, role, timestamp as submit_ts,
                actual_wait_time, outcome
            FROM requests
            WHERE outcome IS NOT NULL AND outcome != 'cancelled'
            """
            df = pd.read_sql_query(query, conn)
            
            # Rename to match training schema if needed, or ensure columns align
            return df
    except Exception as e:
        print(f"Warning: Failed to read logs.db: {e}")
        return pd.DataFrame()

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if path.suffix.lower() in (".feather", ".ft"):
        import pyarrow.feather as feather  # type: ignore
        return feather.read_feather(path)
    if path.suffix.lower() in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def build_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
    ])
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=5)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    # Balanced LR for imbalanced labels; fast CPU inference
    clf = LogisticRegression(
        solver="lbfgs", max_iter=1000, n_jobs=None, class_weight="balanced"
    )
    pipe = Pipeline(steps=[
        ("pre", pre),
        ("clf", clf),
    ])
    return pipe


def pick_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    # Choose threshold maximizing macro-F1 on validation set
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 181):
        yhat = (p >= t).astype(int)
        f1 = f1_score(y_true, yhat, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export calibrated submit-time classifier bundle")
    ap.add_argument("--input", type=str, required=True, help="Input CSV/Parquet with submit-time features")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Output directory for artifacts")
    ap.add_argument("--label-col", type=str, default=None, help="Binary label column; if missing, derive from --wait-col and --pctl")
    ap.add_argument("--wait-col", type=str, default="wait_time", help="Wait-time seconds column (for label derivation)")
    ap.add_argument("--pctl", type=float, default=90.0, help="Percentile cutoff for long-wait label")
    ap.add_argument("--merge-feedback", action="store_true", help="Merge live feedback logs from logs.db")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    df = read_table(Path(args.input))
    
    # Merge live feedback if requested
    if args.merge_feedback:
        feedback_df = load_feedback_logs(LOG_DB_PATH)
        if not feedback_df.empty:
            print(f"Merging {len(feedback_df)} records from feedback loop...")
            
            # Align columns
            # Feedback table has 'actual_wait_time' which maps to 'wait_time' (or args.wait_col)
            if args.wait_col not in feedback_df.columns and "actual_wait_time" in feedback_df.columns:
                feedback_df[args.wait_col] = feedback_df["actual_wait_time"]
            
            # Concatenate
            df = pd.concat([df, feedback_df], ignore_index=True)
            print(f"Total training samples after merge: {len(df)}")
        else:
            print("No feedback logs found or merge skipped.")

    # Expected submit-time columns (align with app/service.py)
    expected_numeric = ["cpu_request", "memory_request", "gpu_request", "max_instance_per_node"]
    expected_categorical = ["partition", "app_name", "role"]
    optional_numeric = ["submit_ts"]  # treated as numeric if present

    missing_numeric = [c for c in expected_numeric if c not in df.columns]
    missing_categorical = [c for c in expected_categorical if c not in df.columns]
    if missing_numeric:
        print(f"Warning: missing numeric features {missing_numeric} - will be filled as NaN.")
        for c in missing_numeric:
            df[c] = np.nan
    if missing_categorical:
        print(f"Warning: missing categorical features {missing_categorical} - will be filled as empty.")
        for c in missing_categorical:
            df[c] = ""
    numeric_cols = [c for c in expected_numeric + optional_numeric if c in df.columns]
    categorical_cols = [c for c in expected_categorical if c in df.columns]

    # Label
    label_col = args.label_col
    if label_col and label_col in df.columns:
        y = df[label_col].astype(int).to_numpy()
    else:
        if args.wait_col not in df.columns:
            raise ValueError(f"Label column not provided and wait_col '{args.wait_col}' not found in input.")
        wt = pd.to_numeric(df[args.wait_col], errors="coerce")
        cutoff = float(np.nanpercentile(wt.values, args.pctl))
        y = (wt >= cutoff).astype(int).to_numpy()
        label_col = f"long_wait_ge_p{int(args.pctl)}"
        df[label_col] = y
        print(f"Derived label '{label_col}' at cutoff {cutoff:.3f} seconds.")

    # Feature matrix
    X = df[numeric_cols + categorical_cols].copy()

    # Train/cal/test split (time-aware if submit_ts present)
    if "submit_ts" in X.columns and pd.api.types.is_numeric_dtype(X["submit_ts"]):
        order = np.argsort(X["submit_ts"].to_numpy())
        X = X.iloc[order].reset_index(drop=True)
        y = y[order]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=args.seed)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=args.seed+1)

    pipe = build_pipeline(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    pipe.fit(X_train, y_train)

    # Raw probabilities
    if hasattr(pipe, "predict_proba"):
        p_cal_raw = pipe.predict_proba(X_cal)[:, 1]
        p_test_raw = pipe.predict_proba(X_test)[:, 1]
    else:
        p_cal_raw = pipe.decision_function(X_cal)  # type: ignore
        p_test_raw = pipe.decision_function(X_test)  # type: ignore
        # Map to [0,1] via logistic
        p_cal_raw = 1.0 / (1.0 + np.exp(-p_cal_raw))
        p_test_raw = 1.0 / (1.0 + np.exp(-p_test_raw))

    # Calibrator (isotonic on raw scores)
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    cal.fit(p_cal_raw, y_cal)
    p_cal = cal.transform(p_cal_raw)
    p_test = cal.transform(p_test_raw)

    # Metrics
    metrics = {
        "auc_raw": float(roc_auc_score(y_test, p_test_raw)) if len(np.unique(y_test)) > 1 else float("nan"),
        "auc_cal": float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else float("nan"),
        "auprc_cal": float(average_precision_score(y_test, p_test)),
        "brier_cal": float(brier_score_loss(y_test, p_test)),
        "n_train": int(len(X_train)),
        "n_cal": int(len(X_cal)),
        "n_test": int(len(X_test)),
        "label_col": label_col,
    }

    # Operating threshold on calibration set (maximize macro-F1)
    thr = pick_threshold(y_cal, p_cal)

    # Save artifacts
    if joblib is None:
        raise RuntimeError("joblib not available to save artifacts")
    joblib.dump(pipe, out_dir / "model.joblib")
    joblib.dump(cal, out_dir / "calibrator.joblib")

    # --- Compute Reference Distributions for Drift Detection ---
    reference_dist = {}
    
    # Numeric: compute histograms (20 bins)
    for col in numeric_cols:
        if col in X.columns:
            data_col = X[col].dropna().values
            if len(data_col) > 0:
                counts, bin_edges = np.histogram(data_col, bins=20, density=False)
                reference_dist[col] = {
                    "type": "numeric",
                    "counts": counts.tolist(),
                    "bin_edges": bin_edges.tolist(),
                    "mean": float(np.mean(data_col)),
                    "std": float(np.std(data_col)),
                    "q25": float(np.percentile(data_col, 25)),
                    "q50": float(np.percentile(data_col, 50)),
                    "q75": float(np.percentile(data_col, 75))
                }

    # Categorical: compute top 50 value counts
    for col in categorical_cols:
        if col in X.columns:
            vc = X[col].value_counts().head(50)
            reference_dist[col] = {
                "type": "categorical",
                "categories": vc.index.tolist(),
                "counts": vc.values.tolist(),
                "total_count": int(len(X))
            }
            
    (out_dir / "reference_dist.json").write_text(json.dumps(reference_dist, indent=2) + "\n")

    thresholds = {
        "operating_threshold": float(thr),
        "bands": {
            "very_low": 0.0,
            "low": 0.2,
            "medium": 0.4,
            "high": 0.6,
            "very_high": 0.8,
        },
    }
    (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")

    feature_schema = {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "label": label_col,
    }
    (out_dir / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2) + "\n")

    manifest = {
        "paths": {
            "model": str((out_dir / "model.joblib").resolve()),
            "calibrator": str((out_dir / "calibrator.joblib").resolve()),
            "thresholds": str((out_dir / "thresholds.json").resolve()),
            "feature_schema": str((out_dir / "feature_schema.json").resolve()),
            "reference_dist": str((out_dir / "reference_dist.json").resolve()),
        },
        "metrics": metrics,
        "columns": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "label": label_col,
        },
        "env": {
            "python": os.environ.get("PYTHONHASHSEED", "N/A"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps({
        "out_dir": str(out_dir.resolve()),
        "operating_threshold": thr,
        "metrics": metrics,
    }, indent=2))


if __name__ == "__main__":
    main()


