import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main() -> None:
    p = argparse.ArgumentParser(description="Train calibrated LR on k8s features")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True, help="Repo-local output directory for metrics/results")
    p.add_argument("--target", type=str, default="label_long_wait")
    args = p.parse_args()

    fp = str(args.features)
    if fp.endswith(".parquet"):
        df = pd.read_parquet(args.features)
    else:
        df = pd.read_csv(args.features)
    y = df[args.target].astype(int)

    num_cols = [
        "num_containers",
        "req_cpu_m",
        "req_mem_mb",
        "req_gpu",
        "has_node_selector",
        "has_affinity",
        "num_tolerations",
        "job_parallelism",
        "job_completions",
        "cluster_node_count",
        "cluster_gpu_capacity",
        "recent_failed_scheduling_ns",
    ]
    X = df[num_cols].fillna(0.0).astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y.values, test_size=0.2, random_state=42, stratify=y.values if y.nunique() == 2 else None
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base = LogisticRegression(max_iter=1000, n_jobs=1)
    # sklearn>=1.5 uses 'estimator' instead of 'base_estimator'
    try:
        clf = CalibratedClassifierCV(estimator=base, cv=5, method="sigmoid")
    except TypeError:
        clf = CalibratedClassifierCV(base_estimator=base, cv=5, method="sigmoid")
    clf.fit(X_train_s, y_train)

    proba = clf.predict_proba(X_test_s)[:, 1]
    auroc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) == 2 else float("nan")
    aupr = average_precision_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)

    args.outdir.mkdir(parents=True, exist_ok=True)
    metrics = {"auroc": float(auroc), "aupr": float(aupr), "brier": float(brier), "n_test": int(len(y_test))}
    (args.outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()


