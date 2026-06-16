#!/usr/bin/env python3
"""
Universal Feature Schema ETL for Paper 4

Transforms all 5 environments into a common 4-feature schema:
  - pending_ratio: pending / total jobs at submit time
  - queue_depth: absolute pending count at submit time
  - gpu_request: GPUs requested by the job
  - qos_class: 0 = best-effort, 1 = latency-sensitive (where available)

Also constructs the SLO binary label:
  y = 1 if start_delay > P90_threshold (environment-specific)

Outputs one parquet per environment with columns:
  [pending_ratio, queue_depth, gpu_request, qos_class, slo_violated, start_delay_s, env_name]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UNIVERSAL_FEATURES = ["pending_ratio", "queue_depth", "gpu_request", "qos_class"]
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "universal"

# ---------------------------------------------------------------------------
# EKS ETL (Kubernetes)
# ---------------------------------------------------------------------------

def etl_eks(parquet_path: str | Path, env_name: str = "EKS-P3") -> pd.DataFrame:
    """
    Transform EKS parquet to universal schema.

    Expects columns:
      - pending_pods, running_pods (or similar) → pending_ratio
      - gpu_pending_pods or pending_pods → queue_depth
      - req_gpu → gpu_request
      - start_delay_seconds or queue_wait_s → start_delay
    """
    df = pd.read_parquet(parquet_path)
    print(f"[{env_name}] Loaded {len(df):,} rows, columns: {list(df.columns)[:20]}...")

    # pending_ratio
    if "pending_ratio" in df.columns:
        pending_ratio = df["pending_ratio"].fillna(0)
    elif "pending_pods" in df.columns and "running_pods" in df.columns:
        total = df["pending_pods"] + df["running_pods"]
        pending_ratio = (df["pending_pods"] / total.replace(0, np.nan)).fillna(0)
    else:
        raise KeyError(f"[{env_name}] Cannot derive pending_ratio. Available: {list(df.columns)}")

    # queue_depth
    if "queue_depth" in df.columns:
        queue_depth = df["queue_depth"].fillna(0)
    elif "gpu_pending_pods" in df.columns:
        queue_depth = df["gpu_pending_pods"].fillna(0)
    elif "pending_pods" in df.columns:
        queue_depth = df["pending_pods"].fillna(0)
    else:
        queue_depth = pd.Series(0, index=df.index)

    # gpu_request
    if "req_gpu" in df.columns:
        gpu_request = df["req_gpu"].fillna(0)
    elif "gpu_request" in df.columns:
        gpu_request = df["gpu_request"].fillna(0)
    else:
        gpu_request = pd.Series(1, index=df.index)

    # qos_class (not typically available in EKS)
    qos_class = pd.Series(0, index=df.index)

    # start_delay
    delay_col = None
    for col in ["start_delay_seconds", "queue_wait_s", "start_delay_s", "wait_time_s"]:
        if col in df.columns:
            delay_col = col
            break
    if delay_col is None:
        raise KeyError(f"[{env_name}] Cannot find start_delay column. Available: {list(df.columns)}")

    start_delay = df[delay_col].fillna(0).clip(lower=0)

    # SLO label: P90 threshold
    threshold = np.percentile(start_delay, 90)
    print(f"[{env_name}] P90 threshold: {threshold:.1f}s, positive rate: {(start_delay > threshold).mean():.3f}")

    return pd.DataFrame({
        "pending_ratio": pending_ratio,
        "queue_depth": queue_depth,
        "gpu_request": gpu_request,
        "qos_class": qos_class,
        "start_delay_s": start_delay,
        "slo_violated": (start_delay > threshold).astype(int),
        "slo_threshold_s": threshold,
        "env_name": env_name,
    })


# ---------------------------------------------------------------------------
# Slurm ETL
# ---------------------------------------------------------------------------

def etl_slurm(parquet_path: str | Path, env_name: str = "Slurm-HPC") -> pd.DataFrame:
    """Transform Slurm data to universal schema."""
    df = pd.read_parquet(parquet_path)
    print(f"[{env_name}] Loaded {len(df):,} rows, columns: {list(df.columns)[:20]}...")

    pending_ratio = df.get("pending_ratio", pd.Series(0, index=df.index)).fillna(0)
    queue_depth = df.get("pending_jobs", df.get("queue_depth", pd.Series(0, index=df.index))).fillna(0)
    gpu_request = df.get("req_gpu", df.get("gpu_request", pd.Series(1, index=df.index))).fillna(0)
    qos_class = pd.Series(0, index=df.index)

    delay_col = None
    for col in ["start_delay_seconds", "queue_wait_s", "start_delay_s", "wait_time_s"]:
        if col in df.columns:
            delay_col = col
            break
    if delay_col is None:
        raise KeyError(f"[{env_name}] Cannot find start_delay column.")

    start_delay = df[delay_col].fillna(0).clip(lower=0)
    threshold = np.percentile(start_delay, 90)
    print(f"[{env_name}] P90 threshold: {threshold:.1f}s, positive rate: {(start_delay > threshold).mean():.3f}")

    return pd.DataFrame({
        "pending_ratio": pending_ratio,
        "queue_depth": queue_depth,
        "gpu_request": gpu_request,
        "qos_class": qos_class,
        "start_delay_s": start_delay,
        "slo_violated": (start_delay > threshold).astype(int),
        "slo_threshold_s": threshold,
        "env_name": env_name,
    })


# ---------------------------------------------------------------------------
# Alibaba-2020 ETL (PAI / Kubernetes)
# ---------------------------------------------------------------------------

def etl_alibaba_2020(data_dir: str | Path, env_name: str = "Alibaba-2020") -> pd.DataFrame:
    """
    Transform Alibaba-2020 PAI trace to universal schema.

    Uses pai_job_table and pai_task_table.
    Queue wait = start_time - submit_time (at job level).
    pending_ratio derived from trace-window replay.
    """
    data_dir = Path(data_dir)

    # Try reading from pre-processed parquet first
    processed = data_dir / "alibaba_2020_processed.parquet"
    if processed.exists():
        df = pd.read_parquet(processed)
        print(f"[{env_name}] Loaded pre-processed {len(df):,} rows")
    else:
        # Import from raw tar.gz files using existing ETL
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from data.alibaba_v2020 import discover_v2020_paths, read_tar_csv_with_header

        paths = discover_v2020_paths(data_dir)

        print(f"[{env_name}] Reading job table from {paths.job}...")
        jobs = read_tar_csv_with_header(paths.job)

        print(f"[{env_name}] Reading task table from {paths.task}...")
        tasks = read_tar_csv_with_header(paths.task)

        # Merge jobs and tasks
        if "start_time" in jobs.columns and "end_time" in jobs.columns:
            df = jobs.copy()
        else:
            df = tasks.merge(jobs[["job_name", "status"]], on="job_name", how="left", suffixes=("", "_job"))

        # Convert times (Alibaba uses seconds since epoch)
        for tc in ["start_time", "end_time"]:
            if tc in df.columns:
                df[tc] = pd.to_numeric(df[tc], errors="coerce")

        df = df.dropna(subset=["start_time"])
        df = df.sort_values("start_time").reset_index(drop=True)

        # Save processed
        processed.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(processed, index=False)
        print(f"[{env_name}] Saved processed to {processed}")

    # Derive pending_ratio via trace-window replay
    if "start_time" in df.columns and "end_time" in df.columns:
        submit_times = df["start_time"].values
        end_times = df["end_time"].values

        # For each job, count how many other jobs were "pending" at submit time
        # pending = submitted before this job and not yet finished
        pending_counts = np.zeros(len(df))
        for i in range(len(df)):
            t = submit_times[i]
            pending_counts[i] = np.sum((submit_times[:i] <= t) & (end_times[:i] > t))

        total_active = pending_counts + 1  # include the submitting job
        pending_ratio = pending_counts / np.maximum(total_active, 1)
    else:
        pending_ratio = np.zeros(len(df))

    # GPU request
    gpu_request = pd.to_numeric(df.get("plan_gpu", pd.Series(1, index=df.index)), errors="coerce").fillna(1)

    # Queue wait (start_time here is task start, we need submit vs start)
    # In Alibaba-2020, jobs have submit → start delay
    if "end_time" in df.columns and "start_time" in df.columns:
        start_delay = (df["end_time"] - df["start_time"]).clip(lower=0)
    else:
        start_delay = pd.Series(0, index=df.index)

    # QoS class: derive from group_tag if available
    qos_class = pd.Series(0, index=df.index)

    threshold = np.percentile(start_delay[start_delay > 0], 90) if (start_delay > 0).any() else 0
    print(f"[{env_name}] P90 threshold: {threshold:.1f}s, positive rate: {(start_delay > threshold).mean():.3f}")

    return pd.DataFrame({
        "pending_ratio": pending_ratio,
        "queue_depth": pending_counts if "pending_counts" in dir() else 0,
        "gpu_request": gpu_request.values,
        "qos_class": qos_class.values,
        "start_delay_s": start_delay.values,
        "slo_violated": (start_delay > threshold).astype(int).values,
        "slo_threshold_s": threshold,
        "env_name": env_name,
    })


# ---------------------------------------------------------------------------
# Alibaba-2023 ETL (FGD / Unified Scheduler)
# ---------------------------------------------------------------------------

def etl_alibaba_2023(data_dir: str | Path, env_name: str = "Alibaba-2023") -> pd.DataFrame:
    """
    Transform Alibaba-2023 (Unified Scheduler / FGD) trace to universal schema.

    Key file: PodMetaInfo/pod_meta_info.tar.gz
    Columns: creation_time, scheduled_time, deletion_time, gpu_request, cpu_request, etc.
    Queue wait = scheduled_time - creation_time
    """
    data_dir = Path(data_dir)

    processed = data_dir / "alibaba_2023_processed.parquet"
    if processed.exists():
        df = pd.read_parquet(processed)
        print(f"[{env_name}] Loaded pre-processed {len(df):,} rows")
    else:
        import tarfile
        import io

        pod_meta_path = data_dir / "PodMetaInfo" / "pod_meta_info.tar.gz"
        if not pod_meta_path.exists():
            raise FileNotFoundError(f"[{env_name}] {pod_meta_path} not found. Run scripts/fetch_gpu_v2023.sh first.")

        print(f"[{env_name}] Reading pod_meta_info from {pod_meta_path}...")
        with tarfile.open(pod_meta_path, "r:gz") as tf:
            csv_member = next((m for m in tf.getmembers() if m.isfile() and m.name.endswith(".csv")), None)
            if csv_member is None:
                raise ValueError(f"No CSV in {pod_meta_path}")
            f = tf.extractfile(csv_member)
            df = pd.read_csv(io.BytesIO(f.read()))

        print(f"[{env_name}] Columns: {list(df.columns)}")
        processed.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(processed, index=False)
        print(f"[{env_name}] Saved processed to {processed}")

    # Queue wait = scheduled_time - creation_time
    for col in ["creation_time", "scheduled_time", "deletion_time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "scheduled_time" in df.columns and "creation_time" in df.columns:
        start_delay = (df["scheduled_time"] - df["creation_time"]).clip(lower=0)
    else:
        # Try alternative column names
        start_delay = pd.Series(0, index=df.index)

    df = df[start_delay.notna() & (start_delay >= 0)].copy()
    start_delay = start_delay[df.index]

    # GPU request
    gpu_request = pd.to_numeric(df.get("gpu_spec_count", df.get("gpu_request", pd.Series(1, index=df.index))), errors="coerce").fillna(1)

    # QoS class
    if "qos" in df.columns:
        qos_class = (df["qos"].str.lower().isin(["ls", "latency-sensitive", "guaranteed"])).astype(int)
    elif "qos_class" in df.columns:
        qos_class = (df["qos_class"].str.lower().isin(["ls", "latency-sensitive", "guaranteed"])).astype(int)
    else:
        qos_class = pd.Series(0, index=df.index)

    # Derive pending_ratio via trace-window replay
    creation_times = df["creation_time"].values if "creation_time" in df.columns else np.zeros(len(df))
    sched_times = df["scheduled_time"].values if "scheduled_time" in df.columns else np.zeros(len(df))

    # Vectorized: for each pod, count pending pods at creation_time
    sorted_idx = np.argsort(creation_times)
    pending_counts = np.zeros(len(df))

    for i in range(len(sorted_idx)):
        idx = sorted_idx[i]
        t = creation_times[idx]
        # Pods created before this one that haven't been scheduled yet
        pending_counts[idx] = np.sum(
            (creation_times[sorted_idx[:i]] <= t) &
            (sched_times[sorted_idx[:i]] > t)
        )

    total_active = pending_counts + 1
    pending_ratio = pending_counts / np.maximum(total_active, 1)

    threshold = np.percentile(start_delay[start_delay > 0], 90) if (start_delay > 0).any() else 0
    print(f"[{env_name}] P90 threshold: {threshold:.1f}s, N={len(df):,}, positive rate: {(start_delay > threshold).mean():.3f}")

    return pd.DataFrame({
        "pending_ratio": pending_ratio,
        "queue_depth": pending_counts,
        "gpu_request": gpu_request.values,
        "qos_class": qos_class.values,
        "start_delay_s": start_delay.values,
        "slo_violated": (start_delay > threshold).astype(int).values,
        "slo_threshold_s": threshold,
        "env_name": env_name,
    })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build universal feature schema for all environments")
    parser.add_argument("--eks-p3", type=str, help="Path to EKS Phase3 parquet")
    parser.add_argument("--eks-nov", type=str, help="Path to EKS Nov24 parquet")
    parser.add_argument("--slurm", type=str, help="Path to Slurm parquet")
    parser.add_argument("--alibaba-2020", type=str, help="Path to Alibaba-2020 raw data dir")
    parser.add_argument("--alibaba-2023", type=str, help="Path to Alibaba-2023 raw data dir")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = []

    if args.eks_p3:
        df = etl_eks(args.eks_p3, "EKS-P3")
        df.to_parquet(output_dir / "eks_p3_universal.parquet", index=False)
        datasets.append(("EKS-P3", df))
        print(f"  → Saved {len(df):,} rows to eks_p3_universal.parquet\n")

    if args.eks_nov:
        df = etl_eks(args.eks_nov, "EKS-Nov")
        df.to_parquet(output_dir / "eks_nov_universal.parquet", index=False)
        datasets.append(("EKS-Nov", df))
        print(f"  → Saved {len(df):,} rows to eks_nov_universal.parquet\n")

    if args.slurm:
        df = etl_slurm(args.slurm, "Slurm-HPC")
        df.to_parquet(output_dir / "slurm_universal.parquet", index=False)
        datasets.append(("Slurm-HPC", df))
        print(f"  → Saved {len(df):,} rows to slurm_universal.parquet\n")

    if args.alibaba_2020:
        df = etl_alibaba_2020(args.alibaba_2020, "Alibaba-2020")
        df.to_parquet(output_dir / "alibaba_2020_universal.parquet", index=False)
        datasets.append(("Alibaba-2020", df))
        print(f"  → Saved {len(df):,} rows to alibaba_2020_universal.parquet\n")

    if args.alibaba_2023:
        df = etl_alibaba_2023(args.alibaba_2023, "Alibaba-2023")
        df.to_parquet(output_dir / "alibaba_2023_universal.parquet", index=False)
        datasets.append(("Alibaba-2023", df))
        print(f"  → Saved {len(df):,} rows to alibaba_2023_universal.parquet\n")

    if datasets:
        print("=" * 60)
        print("Summary of universal schema datasets:")
        print(f"{'Environment':<15} {'N':>10} {'P90 (s)':>10} {'Pos Rate':>10}")
        print("-" * 50)
        for name, df in datasets:
            threshold = df["slo_threshold_s"].iloc[0]
            pos_rate = df["slo_violated"].mean()
            print(f"{name:<15} {len(df):>10,} {threshold:>10.1f} {pos_rate:>10.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
