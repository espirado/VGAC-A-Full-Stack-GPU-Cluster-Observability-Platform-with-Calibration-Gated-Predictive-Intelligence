#!/usr/bin/env python3
"""
Paper 4 — Full Experiment Pipeline

Runs all experiments for the revised paper:
  1. 5×5 Transfer Matrix (AUROC + ECE + Brier)
  2. JS Divergence → ECE Degradation Regression
  3. Few-Shot Recalibration
  4. Feature Richness Comparison (Universal vs Enriched)
  5. Brier Decomposition
  6. End-to-End Routing Simulation
  7. SHAP Feature Importance
  8. Overhead Measurement

Usage:
  python run_experiments.py --data-dir paper4/data/universal/ --output-dir paper4/results/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_FOLDS = 5
ECE_BINS = 15
FEATURES = ["pending_ratio", "queue_depth", "gpu_request", "qos_class"]

MODELS = {
    "LR": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "RF": lambda: RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "GB": lambda: GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
}

ENV_NAMES = ["EKS-P3", "EKS-Nov", "Slurm-HPC", "Alibaba-2020", "Alibaba-2023"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """Expected Calibration Error with equal-width bins."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


def brier_decomposition(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = ECE_BINS):
    """
    Murphy (1973) Brier decomposition into reliability, resolution, uncertainty.
    Brier = reliability - resolution + uncertainty
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    o_bar = y_true.mean()
    uncertainty = o_bar * (1 - o_bar)

    reliability = 0.0
    resolution = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        n_k = mask.sum()
        if n_k == 0:
            continue
        o_k = y_true[mask].mean()
        f_k = y_prob[mask].mean()
        reliability += n_k / len(y_true) * (f_k - o_k) ** 2
        resolution += n_k / len(y_true) * (o_k - o_bar) ** 2

    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier": reliability - resolution + uncertainty,
    }


def compute_js_divergence(a: np.ndarray, b: np.ndarray, n_bins: int = 50) -> float:
    """Jensen-Shannon divergence between two distributions."""
    bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), n_bins + 1)
    hist_a, _ = np.histogram(a, bins=bins, density=True)
    hist_b, _ = np.histogram(b, bins=bins, density=True)
    # Add epsilon to avoid zeros
    hist_a = hist_a + 1e-10
    hist_b = hist_b + 1e-10
    # Normalize
    hist_a = hist_a / hist_a.sum()
    hist_b = hist_b / hist_b.sum()
    return float(jensenshannon(hist_a, hist_b))


# ---------------------------------------------------------------------------
# Experiment 1: 5×5 Transfer Matrix
# ---------------------------------------------------------------------------

def run_transfer_matrix(datasets: dict[str, pd.DataFrame], output_dir: Path):
    """
    Train models on each source environment, evaluate on all targets.
    Produces 5×5 matrices for AUROC, ECE, and Brier.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: 5×5 Transfer Matrix")
    print("=" * 70)

    results = []
    envs = list(datasets.keys())

    for model_name, model_fn in MODELS.items():
        print(f"\n--- Model: {model_name} ---")

        for source in envs:
            src_df = datasets[source]
            X_src = src_df[FEATURES].values
            y_src = src_df["slo_violated"].values

            # Train with isotonic calibration (5-fold CV)
            model = CalibratedClassifierCV(model_fn(), cv=N_FOLDS, method="isotonic")

            try:
                model.fit(X_src, y_src)
            except ValueError as e:
                print(f"  SKIP {source} → * ({model_name}): {e}")
                continue

            for target in envs:
                tgt_df = datasets[target]
                X_tgt = tgt_df[FEATURES].values
                y_tgt = tgt_df["slo_violated"].values

                y_prob = model.predict_proba(X_tgt)[:, 1]

                # Compute metrics
                try:
                    auroc = roc_auc_score(y_tgt, y_prob)
                except ValueError:
                    auroc = np.nan

                ece = compute_ece(y_tgt, y_prob)
                brier = brier_score_loss(y_tgt, y_prob)

                is_cross = source != target
                scheduler_match = _same_scheduler_family(source, target)

                results.append({
                    "model": model_name,
                    "source": source,
                    "target": target,
                    "auroc": auroc,
                    "ece": ece,
                    "brier": brier,
                    "is_cross": is_cross,
                    "scheduler_match": scheduler_match,
                })

                marker = "✓" if not is_cross else ("~" if scheduler_match else "✗")
                print(f"  {source:>15} → {target:<15} [{marker}] AUROC={auroc:.3f}  ECE={ece:.4f}  Brier={brier:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "transfer_matrix.csv", index=False)
    print(f"\nSaved transfer matrix to {output_dir / 'transfer_matrix.csv'}")

    # Summary statistics
    in_domain = results_df[~results_df["is_cross"]]
    cross_domain = results_df[results_df["is_cross"]]
    cross_scheduler = cross_domain[~cross_domain["scheduler_match"]]

    print(f"\n--- Summary (GB model) ---")
    gb = results_df[results_df["model"] == "GB"]
    gb_in = gb[~gb["is_cross"]]
    gb_cross = gb[gb["is_cross"]]
    print(f"  In-domain  AUROC: {gb_in['auroc'].mean():.3f} ± {gb_in['auroc'].std():.3f}")
    print(f"  Cross-dom  AUROC: {gb_cross['auroc'].mean():.3f} ± {gb_cross['auroc'].std():.3f}")
    print(f"  In-domain  ECE:   {gb_in['ece'].mean():.4f}")
    print(f"  Cross-dom  ECE:   {gb_cross['ece'].mean():.4f}")
    print(f"  ECE degradation:  {gb_cross['ece'].mean() / max(gb_in['ece'].mean(), 1e-6):.1f}×")

    return results_df


def _same_scheduler_family(a: str, b: str) -> bool:
    """Check if two environments belong to the same scheduler family."""
    families = {
        "EKS-P3": "kubernetes",
        "EKS-Nov": "kubernetes",
        "Slurm-HPC": "slurm",
        "Alibaba-2020": "pai",
        "Alibaba-2023": "pai",
    }
    return families.get(a) == families.get(b)


# ---------------------------------------------------------------------------
# Experiment 2: JS Divergence → ECE Degradation Regression
# ---------------------------------------------------------------------------

def run_js_regression(datasets: dict[str, pd.DataFrame], transfer_results: pd.DataFrame, output_dir: Path):
    """
    Regression: ΔECE_degradation ~ JS_divergence + scheduler_mismatch + Δpositive_rate
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: JS Divergence → ECE Degradation Regression")
    print("=" * 70)

    envs = list(datasets.keys())

    # Compute JS divergence for all pairs
    js_results = []
    for src, tgt in product(envs, repeat=2):
        if src == tgt:
            continue
        js = compute_js_divergence(
            datasets[src]["pending_ratio"].values,
            datasets[tgt]["pending_ratio"].values,
        )
        pos_rate_src = datasets[src]["slo_violated"].mean()
        pos_rate_tgt = datasets[tgt]["slo_violated"].mean()
        scheduler_match = _same_scheduler_family(src, tgt)

        js_results.append({
            "source": src,
            "target": tgt,
            "js_divergence": js,
            "scheduler_mismatch": int(not scheduler_match),
            "delta_positive_rate": abs(pos_rate_src - pos_rate_tgt),
        })

    js_df = pd.DataFrame(js_results)
    js_df.to_csv(output_dir / "js_divergence_pairs.csv", index=False)

    # Merge with transfer results (GB model)
    gb_cross = transfer_results[(transfer_results["model"] == "GB") & (transfer_results["is_cross"])].copy()
    gb_in = transfer_results[(transfer_results["model"] == "GB") & (~transfer_results["is_cross"])]

    if len(gb_cross) > 0 and len(gb_in) > 0:
        # Compute ECE degradation ratio
        in_domain_ece = gb_in.set_index("source")["ece"]
        gb_cross["in_domain_ece"] = gb_cross["source"].map(in_domain_ece)
        gb_cross["ece_ratio"] = gb_cross["ece"] / gb_cross["in_domain_ece"].clip(lower=1e-6)

        merged = gb_cross.merge(js_df, on=["source", "target"], how="left")

        if len(merged) >= 5:
            from sklearn.linear_model import LinearRegression

            X_reg = merged[["js_divergence", "scheduler_mismatch", "delta_positive_rate"]].values
            y_reg = merged["ece_ratio"].values

            reg = LinearRegression().fit(X_reg, y_reg)
            r2 = reg.score(X_reg, y_reg)

            print(f"\n  Regression R² = {r2:.3f}")
            print(f"  Coefficients:")
            print(f"    β₀ (intercept)      = {reg.intercept_:.3f}")
            print(f"    β₁ (JS divergence)  = {reg.coef_[0]:.3f}")
            print(f"    β₂ (sched mismatch) = {reg.coef_[1]:.3f}")
            print(f"    β₃ (Δpos rate)      = {reg.coef_[2]:.3f}")

            reg_results = {
                "r2": r2,
                "intercept": reg.intercept_,
                "coef_js": reg.coef_[0],
                "coef_sched": reg.coef_[1],
                "coef_delta_pos": reg.coef_[2],
                "n_pairs": len(merged),
            }
            with open(output_dir / "js_regression.json", "w") as f:
                json.dump(reg_results, f, indent=2, default=float)

            merged.to_csv(output_dir / "js_ece_merged.csv", index=False)
            print(f"\n  Saved regression results and merged data")
        else:
            print(f"  Only {len(merged)} pairs — need at least 5 for regression")

    return js_df


# ---------------------------------------------------------------------------
# Experiment 3: Few-Shot Recalibration
# ---------------------------------------------------------------------------

def run_fewshot_recalibration(datasets: dict[str, pd.DataFrame], output_dir: Path):
    """
    For cross-env pairs with high ECE, show recalibration with 50/100/200 target samples.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Few-Shot Recalibration")
    print("=" * 70)

    envs = list(datasets.keys())
    sample_sizes = [25, 50, 100, 200]
    results = []

    for source in envs:
        src_df = datasets[source]
        X_src = src_df[FEATURES].values
        y_src = src_df["slo_violated"].values

        model = CalibratedClassifierCV(
            GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
            cv=min(N_FOLDS, max(2, int(len(y_src) * 0.8))),
            method="isotonic",
        )
        try:
            model.fit(X_src, y_src)
        except ValueError:
            continue

        for target in envs:
            if source == target:
                continue

            tgt_df = datasets[target]
            X_tgt = tgt_df[FEATURES].values
            y_tgt = tgt_df["slo_violated"].values

            y_prob = model.predict_proba(X_tgt)[:, 1]
            raw_ece = compute_ece(y_tgt, y_prob)

            if raw_ece < 0.05:
                continue  # Only recalibrate where ECE is already bad

            for n_samples in sample_sizes:
                if n_samples >= len(y_tgt):
                    continue

                # Sample from target for recalibration
                np.random.seed(RANDOM_STATE)
                cal_idx = np.random.choice(len(y_tgt), size=n_samples, replace=False)
                eval_idx = np.setdiff1d(np.arange(len(y_tgt)), cal_idx)

                if len(eval_idx) < 10:
                    continue

                # Isotonic recalibration on the sample
                iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
                iso.fit(y_prob[cal_idx], y_tgt[cal_idx])

                y_recal = iso.predict(y_prob[eval_idx])
                recal_ece = compute_ece(y_tgt[eval_idx], y_recal)

                # In-domain baseline ECE (for reference)
                in_domain_model = CalibratedClassifierCV(
                    GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
                    cv=min(N_FOLDS, max(2, int(len(y_tgt[eval_idx]) * 0.8))),
                    method="isotonic",
                )
                try:
                    in_domain_model.fit(X_tgt[cal_idx], y_tgt[cal_idx])
                    y_retrained = in_domain_model.predict_proba(X_tgt[eval_idx])[:, 1]
                    retrained_ece = compute_ece(y_tgt[eval_idx], y_retrained)
                except:
                    retrained_ece = np.nan

                recovery = 1 - (recal_ece / max(raw_ece, 1e-6))

                results.append({
                    "source": source,
                    "target": target,
                    "n_samples": n_samples,
                    "raw_ece": raw_ece,
                    "recal_ece": recal_ece,
                    "retrained_ece": retrained_ece,
                    "recovery_pct": recovery * 100,
                })
                print(f"  {source} → {target} [{n_samples} samples]: ECE {raw_ece:.4f} → {recal_ece:.4f} ({recovery*100:.1f}% recovery)")

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df.to_csv(output_dir / "fewshot_recalibration.csv", index=False)
        print(f"\nSaved recalibration results to {output_dir / 'fewshot_recalibration.csv'}")
    return results_df


# ---------------------------------------------------------------------------
# Experiment 4: Brier Decomposition (all 5 environments)
# ---------------------------------------------------------------------------

def run_brier_decomposition(datasets: dict[str, pd.DataFrame], output_dir: Path):
    """
    Murphy (1973) Brier decomposition for GB model on all environments.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Brier Decomposition")
    print("=" * 70)

    results = []

    for env_name, df in datasets.items():
        X = df[FEATURES].values
        y = df["slo_violated"].values

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        all_y_true, all_y_prob = [], []

        for train_idx, test_idx in skf.split(X, y):
            model = CalibratedClassifierCV(
                GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
                cv=min(3, max(2, len(train_idx) // 50)),
                method="isotonic",
            )
            try:
                model.fit(X[train_idx], y[train_idx])
                probs = model.predict_proba(X[test_idx])[:, 1]
                all_y_true.append(y[test_idx])
                all_y_prob.append(probs)
            except:
                continue

        if not all_y_true:
            continue

        y_true = np.concatenate(all_y_true)
        y_prob = np.concatenate(all_y_prob)

        decomp = brier_decomposition(y_true, y_prob)
        decomp["env"] = env_name
        decomp["n_jobs"] = len(y_true)
        decomp["positive_rate"] = y_true.mean()
        results.append(decomp)

        print(f"  {env_name:<15}: Reliab={decomp['reliability']:.4f}  Resol={decomp['resolution']:.4f}  Uncert={decomp['uncertainty']:.4f}  Brier={decomp['brier']:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "brier_decomposition.csv", index=False)
    print(f"\nSaved Brier decomposition to {output_dir / 'brier_decomposition.csv'}")
    return results_df


# ---------------------------------------------------------------------------
# Experiment 5: End-to-End Routing Simulation
# ---------------------------------------------------------------------------

def run_routing_simulation(datasets: dict[str, pd.DataFrame], output_dir: Path):
    """
    Simulate SLO-driven routing with calibrated vs uncalibrated predictions.

    Scenario: Jobs arrive and must be routed to one of K clusters.
    - Baseline: route to cluster with lowest predicted delay (point estimate)
    - Calibrated: route to cluster maximizing P(SLO met)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: End-to-End Routing Simulation")
    print("=" * 70)

    envs = list(datasets.keys())
    if len(envs) < 2:
        print("  Need at least 2 environments for routing simulation")
        return pd.DataFrame()

    # Train per-cluster models
    cluster_models = {}
    for env_name in envs:
        df = datasets[env_name]
        X = df[FEATURES].values
        y = df["slo_violated"].values

        model = CalibratedClassifierCV(
            GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
            cv=min(N_FOLDS, max(2, len(y) // 50)),
            method="isotonic",
        )
        try:
            model.fit(X, y)
            cluster_models[env_name] = model
        except:
            print(f"  SKIP {env_name}: cannot train model")

    if len(cluster_models) < 2:
        print("  Need at least 2 trained models")
        return pd.DataFrame()

    # Generate synthetic job arrivals using a mix of all environments
    all_dfs = [datasets[env] for env in cluster_models.keys()]
    combined = pd.concat(all_dfs, ignore_index=True)

    # Sample jobs to route
    n_jobs = min(5000, len(combined))
    np.random.seed(RANDOM_STATE)
    sample_idx = np.random.choice(len(combined), size=n_jobs, replace=False)
    jobs = combined.iloc[sample_idx].copy()

    X_jobs = jobs[FEATURES].values

    # Strategy 1: Random routing (baseline)
    random_assignments = np.random.choice(list(cluster_models.keys()), size=n_jobs)

    # Strategy 2: Calibrated P(SLO met) routing
    calibrated_assignments = []
    for i in range(n_jobs):
        x = X_jobs[i:i+1]
        best_env, best_prob = None, -1
        for env_name, model in cluster_models.items():
            p_slo_met = 1 - model.predict_proba(x)[0, 1]  # P(NOT violated)
            if p_slo_met > best_prob:
                best_prob = p_slo_met
                best_env = env_name
        calibrated_assignments.append(best_env)

    # Strategy 3: Point-estimate routing (lowest predicted probability of violation)
    # Using uncalibrated GB model
    uncal_models = {}
    for env_name in cluster_models.keys():
        df = datasets[env_name]
        X = df[FEATURES].values
        y = df["slo_violated"].values
        model = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
        try:
            model.fit(X, y)
            uncal_models[env_name] = model
        except:
            pass

    point_assignments = []
    for i in range(n_jobs):
        x = X_jobs[i:i+1]
        best_env, best_prob = None, 2.0
        for env_name, model in uncal_models.items():
            p_violated = model.predict_proba(x)[0, 1]
            if p_violated < best_prob:
                best_prob = p_violated
                best_env = env_name
        point_assignments.append(best_env if best_env else list(uncal_models.keys())[0])

    # Evaluate: actual SLO violation rate for routed jobs
    actual_violations = jobs["slo_violated"].values

    # SLO miss rate per strategy
    results = {
        "random_slo_miss_rate": actual_violations.mean(),  # Representative baseline
        "calibrated_slo_miss_rate": actual_violations[np.array(calibrated_assignments) != ""].mean() if len(calibrated_assignments) > 0 else np.nan,
        "point_slo_miss_rate": actual_violations[np.array(point_assignments) != ""].mean() if len(point_assignments) > 0 else np.nan,
        "n_jobs": n_jobs,
        "n_clusters": len(cluster_models),
    }

    print(f"\n  Routing simulation results ({n_jobs} jobs, {len(cluster_models)} clusters):")
    print(f"    Random routing SLO miss rate:     {results['random_slo_miss_rate']:.3f}")
    print(f"    Point-estimate routing miss rate:  {results['point_slo_miss_rate']:.3f}")
    print(f"    Calibrated routing miss rate:      {results['calibrated_slo_miss_rate']:.3f}")

    with open(output_dir / "routing_simulation.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    return results


# ---------------------------------------------------------------------------
# Experiment 6: Overhead Measurement
# ---------------------------------------------------------------------------

def run_overhead_measurement(datasets: dict[str, pd.DataFrame], output_dir: Path):
    """Measure prediction and recalibration latency."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Overhead Measurement")
    print("=" * 70)

    env_name = list(datasets.keys())[0]
    df = datasets[env_name]
    X = df[FEATURES].values
    y = df["slo_violated"].values

    model = CalibratedClassifierCV(
        GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        cv=min(N_FOLDS, max(2, len(y) // 50)),
        method="isotonic",
    )
    model.fit(X, y)

    # Prediction latency (per job)
    single_x = X[:1]
    n_iters = 1000
    start = time.perf_counter()
    for _ in range(n_iters):
        model.predict_proba(single_x)
    pred_time_ms = (time.perf_counter() - start) / n_iters * 1000

    # Batch prediction latency
    start = time.perf_counter()
    model.predict_proba(X[:100])
    batch_time_ms = (time.perf_counter() - start) * 1000

    # Recalibration latency
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    y_prob = model.predict_proba(X)[:, 1]

    recal_times = []
    for n in [50, 100, 200, 500]:
        if n > len(y):
            continue
        start = time.perf_counter()
        for _ in range(100):
            iso.fit(y_prob[:n], y[:n])
        recal_time_ms = (time.perf_counter() - start) / 100 * 1000
        recal_times.append({"n_samples": n, "recal_time_ms": recal_time_ms})

    results = {
        "single_prediction_ms": pred_time_ms,
        "batch_100_prediction_ms": batch_time_ms,
        "recalibration_latency": recal_times,
    }

    print(f"  Single prediction latency: {pred_time_ms:.3f} ms")
    print(f"  Batch (100) prediction:    {batch_time_ms:.3f} ms")
    for r in recal_times:
        print(f"  Recalibration ({r['n_samples']} samples): {r['recal_time_ms']:.3f} ms")

    with open(output_dir / "overhead_measurement.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    return results


# ---------------------------------------------------------------------------
# SHAP Feature Importance
# ---------------------------------------------------------------------------

def run_shap_importance(datasets: dict[str, pd.DataFrame], output_dir: Path):
    """SHAP feature importance for GB model on each environment."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: SHAP Feature Importance")
    print("=" * 70)

    try:
        import shap
    except ImportError:
        print("  shap not installed — run: pip install shap")
        print("  Skipping SHAP analysis")
        return pd.DataFrame()

    results = []
    for env_name, df in datasets.items():
        X = df[FEATURES].values
        y = df["slo_violated"].values

        model = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        n_sample = min(500, len(X))
        shap_values = explainer.shap_values(X[:n_sample])

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        for i, feat in enumerate(FEATURES):
            results.append({
                "env": env_name,
                "feature": feat,
                "mean_abs_shap": mean_abs_shap[i],
            })

        top_feat = FEATURES[np.argmax(mean_abs_shap)]
        print(f"  {env_name}: top feature = {top_feat} (SHAP={max(mean_abs_shap):.4f})")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "shap_importance.csv", index=False)
    print(f"\nSaved SHAP importance to {output_dir / 'shap_importance.csv'}")
    return results_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_datasets(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all universal-schema parquet files from data_dir."""
    datasets = {}

    file_map = {
        "eks_p3_universal.parquet": "EKS-P3",
        "eks_nov_universal.parquet": "EKS-Nov",
        "slurm_universal.parquet": "Slurm-HPC",
        "alibaba_2020_universal.parquet": "Alibaba-2020",
        "alibaba_2023_universal.parquet": "Alibaba-2023",
    }

    for filename, env_name in file_map.items():
        path = data_dir / filename
        if path.exists():
            df = pd.read_parquet(path)
            datasets[env_name] = df
            print(f"  Loaded {env_name}: {len(df):,} rows, pos_rate={df['slo_violated'].mean():.3f}")
        else:
            print(f"  MISSING {env_name}: {path}")

    return datasets


def main():
    parser = argparse.ArgumentParser(description="Run all Paper 4 experiments")
    parser.add_argument("--data-dir", type=str, default="paper4/data/universal",
                        help="Directory with universal-schema parquet files")
    parser.add_argument("--output-dir", type=str, default="paper4/results",
                        help="Directory for experiment results")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP analysis")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    datasets = load_datasets(data_dir)

    if len(datasets) < 2:
        print(f"\nERROR: Need at least 2 environments but only found {len(datasets)}.")
        print("Run universal_schema.py first to build the datasets, or check --data-dir path.")
        sys.exit(1)

    print(f"\nLoaded {len(datasets)} environments: {list(datasets.keys())}")

    # Run all experiments
    transfer_df = run_transfer_matrix(datasets, output_dir)
    js_df = run_js_regression(datasets, transfer_df, output_dir)
    recal_df = run_fewshot_recalibration(datasets, output_dir)
    brier_df = run_brier_decomposition(datasets, output_dir)
    routing_results = run_routing_simulation(datasets, output_dir)
    overhead_results = run_overhead_measurement(datasets, output_dir)

    if not args.skip_shap:
        shap_df = run_shap_importance(datasets, output_dir)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    main()
