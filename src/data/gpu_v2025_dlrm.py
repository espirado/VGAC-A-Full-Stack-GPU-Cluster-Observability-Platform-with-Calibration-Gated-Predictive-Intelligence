from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class V2025Paths:
    root: Path
    csv: Path


def discover_v2025_paths(root: str | Path) -> V2025Paths:
    r = Path(root).expanduser().resolve()
    csv = r / "disaggregated_DLRM_trace.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Expected CSV at {csv}; run the preview fetch first.")
    return V2025Paths(root=r, csv=csv)


def _to_seconds(s: pd.Series) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce")
    med = s2.dropna().median()
    if pd.notna(med) and np.isfinite(med) and med > 1e11:
        s2 = s2 / 1000.0
    return s2


def build_modeling_table(
    paths: V2025Paths,
    sample_rows: Optional[int] = None,
    wait_percentile: Optional[float] = 0.10,
    underutil_percentile: Optional[float] = 0.10,
) -> pd.DataFrame:
    # Load CSV
    usecols = None  # read all; it's not huge in preview
    df = pd.read_csv(paths.csv, nrows=sample_rows, usecols=usecols)

    # Normalize timestamps
    for c in ["creation_time", "scheduled_time", "deletion_time"]:
        if c in df.columns:
            df[c] = _to_seconds(df[c])

    # Filter GPU instances
    if "gpu_request" in df.columns:
        df["gpu_request"] = pd.to_numeric(df["gpu_request"], errors="coerce").fillna(0)
        df = df[df["gpu_request"] > 0].copy()

    # Compute wait and sojourn
    df["wait_time"] = np.nan
    has_wait = df["creation_time"].notna() & df["scheduled_time"].notna()
    df.loc[has_wait, "wait_time"] = df.loc[has_wait, "scheduled_time"] - df.loc[has_wait, "creation_time"]

    df["sojourn_time"] = np.nan
    has_soj = df["scheduled_time"].notna() & df["deletion_time"].notna()
    df.loc[has_soj, "sojourn_time"] = df.loc[has_soj, "deletion_time"] - df.loc[has_soj, "scheduled_time"]

    # Choose label source: prefer wait_time if non-degenerate, else sojourn_time
    wt_valid = df["wait_time"].fillna(0)
    use_wait = (wt_valid > 0).sum() >= max(50, int(0.02 * len(df)))
    label_source = "wait_time" if use_wait else "sojourn_time"

    # Label: long_wait via percentile
    if wait_percentile is None:
        wait_percentile = 0.10
    s = df[label_source].dropna()
    thr = float(s.quantile(wait_percentile)) if len(s) > 0 else float("nan")
    df["long_wait"] = (df[label_source] >= thr).astype(int) if np.isfinite(thr) else 0

    # Optional label: GPU underutilization via proxy, if realized usage/util columns exist
    # Supported columns (first match wins): utilization percent or usage amount
    util_col_candidates = [
        "gpu_utilization", "gpu_util", "gpu_util_pct",
        "gpu_usage", "gpu_usage_mean", "avg_gpu_util",
    ]
    util_col = next((c for c in util_col_candidates if c in df.columns), None)
    df["underutilization"] = np.nan
    df.attrs["underutil_label_source"] = None
    df.attrs["underutil_threshold"] = float("nan")
    if util_col is not None and underutil_percentile is not None:
        # If utilization appears to be in [0,100], convert to ratio
        util_series = pd.to_numeric(df[util_col], errors="coerce")
        # Heuristic: values mostly >1 implies percent
        median_abs = util_series.dropna().abs().median()
        if pd.notna(median_abs) and median_abs > 1.0:
            util_ratio = (util_series / 100.0).clip(0.0, 1.0)
        else:
            util_ratio = util_series.clip(0.0, 1.0)
        # Define a gap proxy: requested GPUs multiplied by unused share
        if "gpu_request" in df.columns:
            gap = pd.to_numeric(df["gpu_request"], errors="coerce").fillna(0) * (1.0 - util_ratio)
        else:
            gap = (1.0 - util_ratio)
        s2 = gap.dropna()
        thr_u = float(s2.quantile(float(underutil_percentile))) if len(s2) > 0 else float("nan")
        df.loc[gap.notna(), "underutilization"] = (gap >= thr_u).astype(int) if np.isfinite(thr_u) else np.nan
        df.attrs["underutil_label_source"] = util_col
        df.attrs["underutil_threshold"] = thr_u

    # Minimal feature set
    num_candidates = [
        "cpu_request",
        "memory_request",
        "gpu_request",
        "rdma_request",
        "disk_request",
        "cpu_limit",
        "memory_limit",
        "gpu_limit",
        "rdma_limit",
        "disk_limit",
        "max_instance_per_node",
    ]
    num_cols = [c for c in num_candidates if c in df.columns]

    cat_candidates = ["role", "app_name"]
    cat_cols = [c for c in cat_candidates if c in df.columns]

    # Identifier
    if "instance_sn" in df.columns:
        df["example_id"] = df["instance_sn"].astype(str)
    else:
        df["example_id"] = df.index.astype(str)

    keep = [
        *num_cols,
        *cat_cols,
        "example_id",
        "long_wait",
        "underutilization",
        "wait_time",
        "sojourn_time",
        "creation_time",
        "scheduled_time",
        "deletion_time",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # Metadata
    df.attrs["long_wait_label_source"] = label_source
    df.attrs["long_wait_threshold"] = thr
    df.attrs["wait_percentile"] = float(wait_percentile)
    df.attrs["underutil_percentile"] = float(underutil_percentile) if underutil_percentile is not None else None

    return df




