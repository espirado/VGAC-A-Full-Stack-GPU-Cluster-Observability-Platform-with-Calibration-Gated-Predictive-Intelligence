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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier

from src.data.gpu_v2025_dlrm import discover_v2025_paths, build_modeling_table
from src.eval.metrics import evaluate


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    boots = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/gpu_v2025")
    ap.add_argument("--out", default="results/models/v2025")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--wait-percentile", type=float, default=0.10)
    ap.add_argument("--underutil-percentile", type=float, default=0.10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--model", choices=["lgbm", "xgb", "catboost", "logreg", "rf", "gbdt", "mlp", "linsvc", "knn"], default="lgbm")
    ap.add_argument("--mode", choices=["tabular", "tfidf", "combined"], default="tabular",
                    help="tabular: numeric+onehot; tfidf: text-only (role/app); combined: tfidf + tabular")
    ap.add_argument("--label", choices=["long_wait", "underutilization"], default="long_wait")
    ap.add_argument("--suffix", default="rcv")
    ap.add_argument("--bootstrap-train", type=int, default=0, help="If >0, run B training bootstraps with OOB evaluation instead of CV")
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    ap.add_argument("--save-fold-preds", action="store_true", help="Save per-fold predictions for calibration plots")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_v2025_paths(args.root)
    df = build_modeling_table(paths, sample_rows=args.sample, wait_percentile=args.wait_percentile, underutil_percentile=args.underutil_percentile)

    # Choose label
    target_col = args.label
    if target_col not in df.columns:
        raise ValueError(f"Requested label '{target_col}' not available in DataFrame columns: {list(df.columns)}")

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).values
    num_cols = [
        c for c in [
            "cpu_request","memory_request","gpu_request","rdma_request","disk_request",
            "cpu_limit","memory_limit","gpu_limit","rdma_limit","disk_limit",
            "max_instance_per_node",
        ] if c in df.columns
    ]
    cat_cols = [c for c in ["role","app_name"] if c in df.columns]
    # Construct a simple text field from available categorical tokens
    text_series = None
    if len(cat_cols) > 0:
        text_series = df[cat_cols].fillna("").astype(str)
        try:
            text_series = text_series.agg(" ".join, axis=1)
        except Exception:
            text_series = (text_series[cat_cols[0]] if len(cat_cols) == 1 else text_series.apply(lambda r: " ".join(r.values.tolist()), axis=1))

    # Select X based on mode
    if getattr(ap.parse_args, "__name__", None) is None:
        pass
    if args.mode == "tabular":
        X = df[num_cols + cat_cols]
    elif args.mode == "tfidf":
        if text_series is None:
            raise ValueError("tfidf mode requires at least one text-like column (role/app_name).")
        X = pd.DataFrame({"text": text_series})
    else:
        if text_series is None:
            raise ValueError("combined mode requires at least one text-like column (role/app_name).")
        X = pd.concat([pd.DataFrame({"text": text_series}), df[num_cols + cat_cols]], axis=1)

    # We'll set the preprocessor based on mode and model
    pre = None

    # Build preprocessor according to mode
    if args.mode == "tabular":
        try:
            if args.model == "catboost":
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            else:
                ohe = OneHotEncoder(handle_unknown="ignore")
        except TypeError:
            if args.model == "catboost":
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            else:
                ohe = OneHotEncoder(handle_unknown="ignore")
        pre = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", ohe, cat_cols),
        ])
    elif args.mode == "tfidf":
        pre = ColumnTransformer([
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000), "text"),
        ], remainder="drop")
    else:
        try:
            if args.model == "catboost":
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            else:
                ohe = OneHotEncoder(handle_unknown="ignore")
        except TypeError:
            if args.model == "catboost":
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            else:
                ohe = OneHotEncoder(handle_unknown="ignore")
        pre = ColumnTransformer([
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000), "text"),
            ("num", StandardScaler(), num_cols),
            ("cat", ohe, cat_cols),
        ], remainder="drop")

    if args.mode in ("tfidf", "combined") and args.model not in ("logreg", "linsvc"):
        raise ValueError("For tfidf/combined modes, please use --model logreg or --model linsvc for compatibility.")

    if args.model == "lgbm":
        clf = LGBMClassifier(
            n_estimators=400, learning_rate=0.08, num_leaves=63,
            subsample=0.9, colsample_bytree=0.9, random_state=42,
            class_weight="balanced", n_jobs=-1
        )
    elif args.model == "xgb":
        from xgboost import XGBClassifier  # lazy import
        clf = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.9,
            colsample_bytree=0.9, reg_lambda=1.0, random_state=42, n_jobs=-1,
            objective="binary:logistic", eval_metric="logloss"
        )
    elif args.model == "catboost":
        from catboost import CatBoostClassifier  # lazy import
        clf = CatBoostClassifier(
            iterations=1000, learning_rate=0.05, depth=6, random_seed=42,
            loss_function="Logloss", eval_metric="AUC", verbose=False, thread_count=-1
        )
    elif args.model == "logreg":
        lr_n_jobs = 1 if args.mode in ("tfidf", "combined") else -1
        clf = LogisticRegression(
            solver="lbfgs", max_iter=2000, n_jobs=lr_n_jobs, class_weight="balanced"
        )
    elif args.model == "rf":
        clf = RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, class_weight="balanced_subsample", random_state=42
        )
    elif args.model == "gbdt":
        clf = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.08, max_depth=3, random_state=42
        )
    elif args.model == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 64), activation="relu", learning_rate_init=0.001,
            max_iter=200, random_state=42
        )
    elif args.model == "linsvc":
        base = LinearSVC(class_weight="balanced", random_state=42)
        # scikit-learn >= 1.4 uses 'estimator'; older versions use 'base_estimator'
        try:
            clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
        except TypeError:
            clf = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)
    elif args.model == "knn":
        clf = KNeighborsClassifier(n_neighbors=25, weights="distance")
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    model = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])

    scores, reports, eces, briers = [], [], [], []
    rocs, pras = [], []
    fold_preds = []
    ci_method = "cv-fold-bootstrap"
    if args.bootstrap_train and args.bootstrap_train > 0:
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
            if res.roc_auc is not None:
                rocs.append(res.roc_auc)
            if res.pr_auc is not None:
                pras.append(res.pr_auc)
            if args.save_fold_preds:
                fold_preds.append({
                    "indices": oob.tolist(),
                    "y_true": y_true.tolist(),
                    "probas": probas.tolist(),
                })
    else:
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
            if res.roc_auc is not None:
                rocs.append(res.roc_auc)
            if res.pr_auc is not None:
                pras.append(res.pr_auc)
            if args.save_fold_preds:
                try:
                    example_ids = df.iloc[te]["example_id"].astype(str).tolist()
                except Exception:
                    example_ids = None
                fold_preds.append({
                    "indices": te.tolist(),
                    "example_id": example_ids,
                    "y_true": y_true.tolist(),
                    "probas": probas.tolist(),
                })

    scores = np.array(scores, dtype=float)
    eces = np.array(eces, dtype=float)
    briers = np.array(briers, dtype=float)
    lo, hi = bootstrap_ci(scores, n_boot=2000)
    ece_lo, ece_hi = bootstrap_ci(eces, n_boot=2000)
    brier_lo, brier_hi = bootstrap_ci(briers, n_boot=2000)
    roc_mean = float(np.mean(rocs)) if len(rocs) > 0 else None
    pr_mean = float(np.mean(pras)) if len(pras) > 0 else None
    roc_ci = list(bootstrap_ci(np.array(rocs), n_boot=2000)) if len(rocs) > 0 else None
    pr_ci = list(bootstrap_ci(np.array(pras), n_boot=2000)) if len(pras) > 0 else None

    label_balance = {}
    for p in [0.05, 0.10, 0.15, 0.20]:
        base = None
        if args.label == "long_wait":
            base = df[df["wait_time"].notna()]["wait_time"] if df.attrs.get("long_wait_label_source") == "wait_time" else df[df["sojourn_time"].notna()]["sojourn_time"]
        else:
            base = pd.to_numeric(df["underutilization"], errors="coerce").dropna()
        if base is not None and len(base) > 0:
            thr = float(base.quantile(p)) if args.label == "long_wait" else float(p)
            rate = float((base >= thr).mean()) if args.label == "long_wait" else float((df["underutilization"] == 1).mean())
            label_balance.setdefault("pct", []).append({"p": p, "thr": thr, "rate": rate})

    out = {
        "label": args.label,
        "num_rows": int(len(df)),
        "positive_rate": float(pd.to_numeric(df[args.label], errors="coerce").fillna(0).astype(int).mean()),
        "folds": 5,
        "repeats": int(args.repeats),
        "ci_method": ci_method,
        "wait_percentile": float(args.wait_percentile),
        "underutil_percentile": float(args.underutil_percentile),
        "long_wait_label_source": df.attrs.get("long_wait_label_source"),
        "long_wait_threshold": float(df.attrs.get("long_wait_threshold", float("nan"))),
        "underutil_label_source": df.attrs.get("underutil_label_source"),
        "underutil_threshold": float(df.attrs.get("underutil_threshold", float("nan"))),
        "macro_f1_mean": float(scores.mean()),
        "macro_f1_ci": [lo, hi],
        "ece_mean": float(eces.mean()),
        "ece_ci": [float(ece_lo), float(ece_hi)],
        "brier_mean": float(briers.mean()),
        "brier_ci": [float(brier_lo), float(brier_hi)],
        "fold_reports": reports,
        "roc_auc_mean": roc_mean,
        "roc_auc_ci": roc_ci,
        "pr_auc_mean": pr_mean,
        "pr_auc_ci": pr_ci,
        "label_balance": label_balance,
    }
    suf = ("_" + args.suffix.strip()) if args.suffix and not args.suffix.strip().startswith("_") else (args.suffix or "")
    model_tag = args.model
    out_path = out_dir / f"v2025_cv_{args.label}_{model_tag}{suf}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    if args.save_fold_preds:
        preds_path = out_dir / f"v2025_cv_{args.label}_{model_tag}{suf}.fold_preds.json"
        with preds_path.open("w") as f:
            json.dump({"folds": fold_preds}, f)
        print(f"Saved fold predictions to {preds_path}")
    print(f"Saved CV summary to {out_path}")


if __name__ == "__main__":
    main()




