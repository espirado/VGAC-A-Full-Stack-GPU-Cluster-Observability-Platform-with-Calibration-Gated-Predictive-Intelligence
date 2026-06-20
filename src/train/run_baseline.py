from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.logs_loader import load_logs, stratified_split
from src.models.baselines import build_tfidf_logreg
from src.eval.metrics import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/synthetic/logs.csv")
    ap.add_argument("--out", default="results/models/tfidf_logreg")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_logs(args.csv)
    split = stratified_split(df, label_col="category", seed=42)

    model = build_tfidf_logreg()
    model.fit(split.train["message"], split.train["category"])

    probas = model.predict_proba(split.test["message"])
    classes = list(model.classes_)
    res = evaluate(split.test["category"].values, probas, classes)

    # Save metrics
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump({
            "report": res.report,
            "ece": res.ece,
            "brier": res.brier,
            "classes": classes,
        }, f, indent=2)

    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()


