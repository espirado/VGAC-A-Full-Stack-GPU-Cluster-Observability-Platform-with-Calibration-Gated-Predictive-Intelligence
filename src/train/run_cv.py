from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data.logs_loader import load_logs
from src.models.baselines import build_tfidf_logreg
from src.models.lgbm import build_tfidf_lgbm
from src.eval.metrics import evaluate


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(np.mean(sample))
    lo = np.percentile(boots, 100 * (alpha / 2))
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/synthetic/logs.csv")
    ap.add_argument("--model", choices=["logreg", "lgbm"], default="logreg")
    ap.add_argument("--out", default="results/models/cv")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_logs(args.csv)
    X = df["message"].values
    y = df["category"].values

    if args.model == "logreg":
        build = build_tfidf_logreg
    else:
        build = build_tfidf_lgbm

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    macro_f1_scores = []
    all_reports = []
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        model = build()
        model.fit(X[tr], y[tr])
        probas = model.predict_proba(X[te])
        classes = list(model.classes_)
        res = evaluate(y[te], probas, classes)
        macro_f1 = res.report["macro avg"]["f1-score"]
        macro_f1_scores.append(macro_f1)
        all_reports.append(res.report)

    macro_f1_scores = np.array(macro_f1_scores)
    lo, hi = bootstrap_ci(macro_f1_scores, n_boot=2000)

    summary = {
        "model": args.model,
        "folds": args.folds,
        "macro_f1_mean": float(macro_f1_scores.mean()),
        "macro_f1_ci": [lo, hi],
        "fold_reports": all_reports,
    }

    out_path = out_dir / f"cv_{args.model}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved CV summary to {out_path}")


if __name__ == "__main__":
    main()


