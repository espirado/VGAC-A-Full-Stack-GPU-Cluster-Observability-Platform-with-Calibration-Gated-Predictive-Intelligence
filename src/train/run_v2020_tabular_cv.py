from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from src.data.alibaba_v2020 import discover_v2020_paths, build_modeling_table
from src.eval.metrics import evaluate


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    boots = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/alibaba_2020")
    ap.add_argument("--label", choices=["underutilized", "long_wait"], default="underutilized")
    ap.add_argument("--out", default="results/models/v2020")
    ap.add_argument("--sample", type=int, default=200000)
    ap.add_argument("--wait-threshold", type=float, default=3600.0, help="Seconds defining long_wait label")
    ap.add_argument("--wait-percentile", type=float, default=None, help="If set (e.g., 0.10), defines long_wait as top p percentile of wait_time")
    ap.add_argument("--repeats", type=int, default=1, help="Number of repeats for RepeatedStratifiedKFold (1 = single 5-fold CV)")
    ap.add_argument("--bootstrap-train", type=int, default=0, help="If >0, run B training bootstraps with OOB evaluation instead of CV")
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    ap.add_argument("--suffix", default="", help="Optional suffix for output filename, e.g., 'rcv' or 'boot'")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_v2020_paths(args.root)
    df = build_modeling_table(
        paths,
        sample_rows=args.sample,
        wait_threshold=args.wait_threshold,
        wait_percentile=args.wait_percentile,
    )

    y = df[args.label].astype(int).values
    num_cols = [c for c in ["plan_cpu", "plan_mem", "plan_gpu", "cap_cpu", "cap_mem", "cap_gpu"] if c in df.columns]
    cat_cols = [c for c in ["gpu_type_task"] if c in df.columns]
    X = df[num_cols + cat_cols]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    model = Pipeline([
        ("pre", pre),
        ("clf", LGBMClassifier(
            n_estimators=400, learning_rate=0.08, num_leaves=63,
            subsample=0.9, colsample_bytree=0.9, random_state=42,
            class_weight="balanced", n_jobs=-1
        )),
    ])

    ci_method = "cv-fold-bootstrap"
    scores = []
    reports = []
    eces = []
    briers = []

    if args.bootstrap_train and args.bootstrap_train > 0:
        # Training bootstrap with OOB evaluation
        ci_method = "train-bootstrap-oob"
        rng = np.random.default_rng(args.bootstrap_seed)
        idx_all = np.arange(len(X))
        B = int(args.bootstrap_train)
        for _ in range(B):
            idx = rng.integers(0, len(X), len(X))
            oob = np.setdiff1d(idx_all, np.unique(idx))
            if len(oob) == 0:
                continue
            model.fit(X.iloc[idx], y[idx])
            probas = model.predict_proba(X.iloc[oob])
            classes = ["0", "1"]
            y_true = y[oob].astype(str)
            res = evaluate(y_true, probas, classes)
            scores.append(res.report["macro avg"]["f1-score"]) 
            reports.append(res.report)
            eces.append(res.ece)
            briers.append(res.brier)
    else:
        # Repeated stratified CV (repeats=1 -> single CV)
        if args.repeats and args.repeats > 1:
            rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=int(args.repeats), random_state=42)
            splitter = rkf.split(X, y)
            ci_method = "repeated-cv-fold-bootstrap"
        else:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            splitter = skf.split(X, y)
            ci_method = "cv-fold-bootstrap"

        for tr, te in splitter:
            model.fit(X.iloc[tr], y[tr])
            probas = model.predict_proba(X.iloc[te])
            classes = ["0", "1"]
            y_true = y[te].astype(str)
            res = evaluate(y_true, probas, classes)
            scores.append(res.report["macro avg"]["f1-score"]) 
            reports.append(res.report)
            eces.append(res.ece)
            briers.append(res.brier)

    scores = np.array(scores, dtype=float)
    eces = np.array(eces, dtype=float)
    briers = np.array(briers, dtype=float)
    lo, hi = bootstrap_ci(scores, n_boot=2000)
    ece_lo, ece_hi = bootstrap_ci(eces, n_boot=2000)
    brier_lo, brier_hi = bootstrap_ci(briers, n_boot=2000)

    # Label prevalence sweeps (for transparency)
    label_balance = {}
    if "wait_time" in df.columns:
        fixed = []
        for thr in [300, 600, 1200, 1800, 3600, 7200]:
            fixed.append({"thr": float(thr), "rate": float((df["wait_time"] > thr).mean())})
        pct = []
        for p in [0.05, 0.10, 0.15, 0.20]:
            wt = df["wait_time"].dropna()
            if len(wt) > 0:
                thr = float(wt.quantile(p))
                rate = float((df["wait_time"] >= thr).mean())
                pct.append({"p": float(p), "thr": thr, "rate": rate})
        label_balance = {"fixed": fixed, "pct": pct}

    used_fallback = bool(df.attrs.get("long_wait_used_fallback_sojourn", False))
    label_source = ("sojourn_time" if used_fallback else "wait_time") if args.label == "long_wait" else None

    out = {
        "label": args.label,
        "wait_threshold": float(df.attrs.get("long_wait_threshold", args.wait_threshold)),
        "wait_percentile": args.wait_percentile,
        "num_rows": int(len(df)),
        "positive_rate": float(y.mean()),
        "folds": 5,
        "repeats": int(args.repeats),
        "ci_method": ci_method,
        "long_wait_used_fallback_sojourn": used_fallback if args.label == "long_wait" else None,
        "long_wait_label_source": label_source,
        "macro_f1_mean": float(scores.mean()),
        "macro_f1_ci": [lo, hi],
        "ece_mean": float(eces.mean()),
        "ece_ci": [float(ece_lo), float(ece_hi)],
        "brier_mean": float(briers.mean()),
        "brier_ci": [float(brier_lo), float(brier_hi)],
        "fold_reports": reports,
        "label_balance": label_balance,
    }
    suf = ("_" + args.suffix.strip()) if args.suffix and not args.suffix.strip().startswith("_") else (args.suffix or "")
    out_path = out_dir / f"v2020_cv_{args.label}{suf}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved CV summary to {out_path}")


if __name__ == "__main__":
    main()


