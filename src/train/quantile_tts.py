import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split


def main() -> None:
    p = argparse.ArgumentParser(description="Quantile regression for TTS using sklearn GradientBoostingRegressor")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--alphas", type=str, default="0.9,0.95", help="Comma-separated quantiles, e.g., 0.9,0.95")
    args = p.parse_args()

    fp = str(args.features)
    if fp.endswith(".parquet"):
        df = pd.read_parquet(args.features)
    else:
        df = pd.read_csv(args.features)

    # Target
    if "tts_seconds" not in df.columns:
        raise SystemExit("features file must include tts_seconds column for quantile regression")
    df_q = df[df["tts_seconds"].notna()].copy()
    y = df_q["tts_seconds"].astype(float).values

    feat_cols = [
        "num_containers","req_cpu_m","req_mem_mb","req_gpu",
        "has_node_selector","node_selector_keys","has_affinity",
        "na_required_terms","na_preferred_terms",
        "num_tolerations","toleration_keys_count","tolerations_effect_NoSchedule" if "tolerations_effect_NoSchedule" in df_q.columns else "tolerations_effect_noSchedule",
        "image_pull_always","num_images_ecr","num_images_dockerhub","num_images_other",
        "job_parallelism","job_completions",
        "cluster_node_count","cluster_gpu_capacity",
        "recent_failed_scheduling_ns","recent_image_pull_err_ns","recent_backoff_ns",
    ]
    feat_cols = [c for c in feat_cols if c in df_q.columns]
    X = df_q[feat_cols].fillna(0.0).astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}
    for a_str in args.alphas.split(","):
        a = float(a_str.strip())
        q = GradientBoostingRegressor(loss="quantile", alpha=a, random_state=42)
        q.fit(X_train, y_train)
        yq = q.predict(X_test)
        pin = mean_pinball_loss(y_test, yq, alpha=a)
        results[f"q{int(a*100)}"] = {
            "alpha": a,
            "mean_pinball_loss": float(pin),
            "n_test": int(len(y_test)),
        }

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "metrics_tts_quantiles.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
















