from __future__ import annotations

"""
Unified feature schema for queue-risk and TTS modeling across clusters.

This schema captures:
- Identifiers and timestamps
- Submit-time resource requests and constraints
- Cluster-state snapshot context (counts/ratios)
- GPU context (when available in the archive)
- Labels (binary long-wait and continuous TTS)

Utilities provided:
- COLUMN_SPECS: canonical column names and logical dtypes
- ensure_unified_columns(df): add any missing columns with sensible defaults
- cast_dtypes(df): best-effort cast to target dtypes without being lossy
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str  # pandas dtype string (logical target)
    default: object


# Core identifiers and time
IDENTIFIER_COLS: List[ColumnSpec] = [
    ColumnSpec("cluster", "string", None),
    ColumnSpec("namespace", "string", None),
    ColumnSpec("pod", "string", None),
    ColumnSpec("submit_ts", "datetime64[ns, UTC]", pd.NaT),
]

# Submit-time resource requests and constraints
SUBMIT_TIME_COLS: List[ColumnSpec] = [
    ColumnSpec("num_containers", "Int64", pd.NA),
    ColumnSpec("req_cpu_m", "float64", np.nan),
    ColumnSpec("req_mem_mb", "float64", np.nan),
    ColumnSpec("req_gpu", "Int64", pd.NA),
    ColumnSpec("has_node_selector", "Int8", 0),
    ColumnSpec("node_selector_keys", "Int64", pd.NA),
    ColumnSpec("constraint_count", "Int64", 0),
    ColumnSpec("has_affinity", "Int8", 0),
    ColumnSpec("na_required_terms", "Int64", pd.NA),
    ColumnSpec("na_preferred_terms", "Int64", pd.NA),
    ColumnSpec("num_tolerations", "Int64", pd.NA),
    ColumnSpec("toleration_keys_count", "Int64", pd.NA),
    ColumnSpec("tolerations_effect_noSchedule", "Int64", pd.NA),
    ColumnSpec("num_images_ecr", "Int64", pd.NA),
    ColumnSpec("num_images_dockerhub", "Int64", pd.NA),
    ColumnSpec("num_images_other", "Int64", pd.NA),
    ColumnSpec("image_pull_always", "Int64", pd.NA),
    ColumnSpec("job_parallelism", "Int64", pd.NA),
    ColumnSpec("job_completions", "Int64", pd.NA),
    ColumnSpec("priority_class", "string", None),
    ColumnSpec("priority_value", "float64", np.nan),
    ColumnSpec("namespace_gpu_pending_total", "Int64", pd.NA),
    ColumnSpec("namespace_avg_wait_seconds", "float64", np.nan),
    ColumnSpec("namespace_avg_req_gpu", "float64", np.nan),
    ColumnSpec("priority_gpu_pending_total", "Int64", pd.NA),
    ColumnSpec("priority_avg_wait_seconds", "float64", np.nan),
    ColumnSpec("priority_avg_req_gpu", "float64", np.nan),
    ColumnSpec("namespace_pods_roll_5m", "float64", np.nan),
    ColumnSpec("namespace_pods_roll_15m", "float64", np.nan),
    ColumnSpec("namespace_req_gpu_roll_mean_5m", "float64", np.nan),
    ColumnSpec("namespace_req_gpu_roll_mean_15m", "float64", np.nan),
    ColumnSpec("priority_pods_roll_5m", "float64", np.nan),
    ColumnSpec("priority_pods_roll_15m", "float64", np.nan),
    ColumnSpec("priority_req_gpu_roll_mean_5m", "float64", np.nan),
    ColumnSpec("priority_req_gpu_roll_mean_15m", "float64", np.nan),
]

# Cluster snapshot context (point-in-time)
CLUSTER_STATE_COLS: List[ColumnSpec] = [
    ColumnSpec("cluster_node_count", "Int64", pd.NA),
    ColumnSpec("cluster_gpu_capacity", "Int64", pd.NA),
    ColumnSpec("cluster_gpu_allocatable", "Int64", pd.NA),
    ColumnSpec("cluster_cpu_capacity_m", "float64", np.nan),
    ColumnSpec("cluster_mem_capacity_mb", "float64", np.nan),
    ColumnSpec("recent_failed_scheduling_ns", "Int64", pd.NA),
    ColumnSpec("recent_image_pull_err_ns", "Int64", pd.NA),
    ColumnSpec("recent_backoff_ns", "Int64", pd.NA),
    # Optional rollups (add when present)
    ColumnSpec("pending_ratio", "float64", np.nan),
    ColumnSpec("node_cpu_rate", "float64", np.nan),
    ColumnSpec("node_mem_util", "float64", np.nan),
    ColumnSpec("node_load5", "float64", np.nan),
]

# Queue + telemetry augmentations (Paper 2 focus)
TELEMETRY_COLS: List[ColumnSpec] = [
    ColumnSpec("gpu_pending_pods", "Int64", pd.NA),
    ColumnSpec("gpu_running_pods", "Int64", pd.NA),
    ColumnSpec("pending_pods", "Int64", pd.NA),
    ColumnSpec("running_pods", "Int64", pd.NA),
    ColumnSpec("total_pods", "Int64", pd.NA),
    ColumnSpec("gpu_pending_per_alloc_gpu", "float64", np.nan),
    ColumnSpec("gpu_running_per_alloc_gpu", "float64", np.nan),
    ColumnSpec("pending_per_node", "float64", np.nan),
    ColumnSpec("total_pods_per_node", "float64", np.nan),
    ColumnSpec("gpu_utilization_cluster", "float64", np.nan),
    ColumnSpec("cpu_utilization_cluster", "float64", np.nan),
    ColumnSpec("mem_utilization_cluster", "float64", np.nan),
    ColumnSpec("avg_wait_seconds", "float64", np.nan),
    ColumnSpec("max_wait_seconds", "float64", np.nan),
    ColumnSpec("p90_wait_seconds", "float64", np.nan),
    ColumnSpec("pending_ratio_roll_mean_5m", "float64", np.nan),
    ColumnSpec("pending_ratio_roll_mean_15m", "float64", np.nan),
    ColumnSpec("pending_ratio_delta_5m", "float64", np.nan),
    ColumnSpec("pending_ratio_delta_15m", "float64", np.nan),
    ColumnSpec("pending_ratio_roll_std_5m", "float64", np.nan),
    ColumnSpec("pending_ratio_roll_std_15m", "float64", np.nan),
    ColumnSpec("gpu_pending_pods_roll_mean_5m", "float64", np.nan),
    ColumnSpec("gpu_pending_pods_roll_mean_15m", "float64", np.nan),
    ColumnSpec("gpu_pending_pods_delta_5m", "float64", np.nan),
    ColumnSpec("gpu_pending_pods_delta_15m", "float64", np.nan),
    ColumnSpec("gpu_pending_pods_roll_std_5m", "float64", np.nan),
    ColumnSpec("gpu_pending_pods_roll_std_15m", "float64", np.nan),
    ColumnSpec("gpu_running_pods_roll_mean_5m", "float64", np.nan),
    ColumnSpec("gpu_running_pods_roll_mean_15m", "float64", np.nan),
    ColumnSpec("gpu_running_pods_delta_5m", "float64", np.nan),
    ColumnSpec("gpu_running_pods_delta_15m", "float64", np.nan),
    ColumnSpec("gpu_running_pods_roll_std_5m", "float64", np.nan),
    ColumnSpec("gpu_running_pods_roll_std_15m", "float64", np.nan),
    ColumnSpec("pending_pods_roll_mean_5m", "float64", np.nan),
    ColumnSpec("pending_pods_roll_mean_15m", "float64", np.nan),
    ColumnSpec("pending_pods_delta_5m", "float64", np.nan),
    ColumnSpec("pending_pods_delta_15m", "float64", np.nan),
    ColumnSpec("pending_pods_roll_std_5m", "float64", np.nan),
    ColumnSpec("pending_pods_roll_std_15m", "float64", np.nan),
    ColumnSpec("running_pods_roll_mean_5m", "float64", np.nan),
    ColumnSpec("running_pods_roll_mean_15m", "float64", np.nan),
    ColumnSpec("running_pods_delta_5m", "float64", np.nan),
    ColumnSpec("running_pods_delta_15m", "float64", np.nan),
    ColumnSpec("running_pods_roll_std_5m", "float64", np.nan),
    ColumnSpec("running_pods_roll_std_15m", "float64", np.nan),
    ColumnSpec("total_pods_roll_mean_5m", "float64", np.nan),
    ColumnSpec("total_pods_roll_mean_15m", "float64", np.nan),
    ColumnSpec("total_pods_delta_5m", "float64", np.nan),
    ColumnSpec("total_pods_delta_15m", "float64", np.nan),
    ColumnSpec("total_pods_roll_std_5m", "float64", np.nan),
    ColumnSpec("total_pods_roll_std_15m", "float64", np.nan),
    ColumnSpec("gpu_utilization_cluster_roll_mean_5m", "float64", np.nan),
    ColumnSpec("gpu_utilization_cluster_roll_mean_15m", "float64", np.nan),
    ColumnSpec("gpu_utilization_cluster_delta_5m", "float64", np.nan),
    ColumnSpec("gpu_utilization_cluster_delta_15m", "float64", np.nan),
    ColumnSpec("gpu_utilization_cluster_roll_std_5m", "float64", np.nan),
    ColumnSpec("gpu_utilization_cluster_roll_std_15m", "float64", np.nan),
    ColumnSpec("cpu_utilization_cluster_roll_mean_5m", "float64", np.nan),
    ColumnSpec("cpu_utilization_cluster_roll_mean_15m", "float64", np.nan),
    ColumnSpec("cpu_utilization_cluster_delta_5m", "float64", np.nan),
    ColumnSpec("cpu_utilization_cluster_delta_15m", "float64", np.nan),
    ColumnSpec("cpu_utilization_cluster_roll_std_5m", "float64", np.nan),
    ColumnSpec("cpu_utilization_cluster_roll_std_15m", "float64", np.nan),
    ColumnSpec("mem_utilization_cluster_roll_mean_5m", "float64", np.nan),
    ColumnSpec("mem_utilization_cluster_roll_mean_15m", "float64", np.nan),
    ColumnSpec("mem_utilization_cluster_delta_5m", "float64", np.nan),
    ColumnSpec("mem_utilization_cluster_delta_15m", "float64", np.nan),
    ColumnSpec("mem_utilization_cluster_roll_std_5m", "float64", np.nan),
    ColumnSpec("mem_utilization_cluster_roll_std_15m", "float64", np.nan),
    ColumnSpec("avg_wait_seconds_roll_mean_5m", "float64", np.nan),
    ColumnSpec("avg_wait_seconds_roll_mean_15m", "float64", np.nan),
    ColumnSpec("avg_wait_seconds_delta_5m", "float64", np.nan),
    ColumnSpec("avg_wait_seconds_delta_15m", "float64", np.nan),
]

# GPU/DCGM context (optional; only if present in archive)
GPU_CONTEXT_COLS: List[ColumnSpec] = [
    ColumnSpec("gpu_util_avg", "float64", np.nan),
    ColumnSpec("gpu_util_avg_max", "float64", np.nan),
    ColumnSpec("gpu_util_avg_std", "float64", np.nan),
    ColumnSpec("gpu_util_avg_p10", "float64", np.nan),
    ColumnSpec("gpu_util_avg_p90", "float64", np.nan),
    ColumnSpec("gpu_util_avg_p95", "float64", np.nan),
    ColumnSpec("gpu_util_avg_cv", "float64", np.nan),
    ColumnSpec("gpu_util_avg_p90_p10", "float64", np.nan),
    ColumnSpec("gpu_util_avg_gt90_frac", "float64", np.nan),
    ColumnSpec("gpu_fb_used_mb", "float64", np.nan),
    ColumnSpec("gpu_fb_used_mb_max", "float64", np.nan),
    ColumnSpec("gpu_fb_used_mb_std", "float64", np.nan),
    ColumnSpec("gpu_fb_used_mb_p10", "float64", np.nan),
    ColumnSpec("gpu_fb_used_mb_p90", "float64", np.nan),
    ColumnSpec("gpu_fb_used_mb_p95", "float64", np.nan),
    ColumnSpec("gpu_fb_used_mb_cv", "float64", np.nan),
    ColumnSpec("gpu_fb_used_mb_p90_p10", "float64", np.nan),
    ColumnSpec("gpu_power_w", "float64", np.nan),
    ColumnSpec("gpu_power_w_max", "float64", np.nan),
    ColumnSpec("gpu_power_w_std", "float64", np.nan),
    ColumnSpec("gpu_power_w_p10", "float64", np.nan),
    ColumnSpec("gpu_power_w_p90", "float64", np.nan),
    ColumnSpec("gpu_power_w_p95", "float64", np.nan),
    ColumnSpec("gpu_power_w_cv", "float64", np.nan),
    ColumnSpec("gpu_power_w_p90_p10", "float64", np.nan),
    ColumnSpec("gpu_temp_c", "float64", np.nan),
    ColumnSpec("gpu_temp_c_max", "float64", np.nan),
    ColumnSpec("gpu_temp_c_std", "float64", np.nan),
    ColumnSpec("gpu_temp_c_p10", "float64", np.nan),
    ColumnSpec("gpu_temp_c_p90", "float64", np.nan),
    ColumnSpec("gpu_temp_c_p95", "float64", np.nan),
    ColumnSpec("gpu_temp_c_cv", "float64", np.nan),
    ColumnSpec("gpu_temp_c_p90_p10", "float64", np.nan),
    ColumnSpec("gpu_temp_c_gt80_frac", "float64", np.nan),
    # Modal-aligned thermal thresholds (88-90°C = degradation zone)
    ColumnSpec("gpu_temp_c_gt88_frac", "float64", np.nan),  # Fraction of GPUs > 88°C (perf degradation threshold)
    ColumnSpec("gpu_temp_c_gt90_frac", "float64", np.nan),  # Fraction of GPUs > 90°C (critical thermal violation)
]

# Labels and outcomes
LABEL_COLS: List[ColumnSpec] = [
    ColumnSpec("phase", "string", None),
    ColumnSpec("tts_seconds", "float64", np.nan),
    ColumnSpec("age_seconds", "float64", np.nan),
    ColumnSpec("label_long_wait", "Int8", 0),
]

ALL_SPECS: List[ColumnSpec] = (
    IDENTIFIER_COLS
    + SUBMIT_TIME_COLS
    + CLUSTER_STATE_COLS
    + TELEMETRY_COLS
    + GPU_CONTEXT_COLS
    + LABEL_COLS
)

COLUMN_SPECS: Dict[str, ColumnSpec] = {c.name: c for c in ALL_SPECS}
UNIFIED_COLUMNS: List[str] = [c.name for c in ALL_SPECS]


def ensure_unified_columns(df: pd.DataFrame, cluster_name: str | None = None) -> pd.DataFrame:
    """
    Ensure df contains all columns in the unified schema, adding missing ones with defaults.
    Optionally set the 'cluster' identifier if provided and missing.
    """
    out = df.copy()
    for name, spec in COLUMN_SPECS.items():
        if name not in out.columns:
            out[name] = spec.default
    if cluster_name is not None:
        if "cluster" in out.columns and out["cluster"].isna().all():
            out["cluster"] = cluster_name
    # Order columns for consistency
    out = out[UNIFIED_COLUMNS]
    return out


def cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Best-effort cast to the logical target dtypes. This avoids raising on incompatible
    values by falling back to object where necessary.
    """
    out = df.copy()
    for name, spec in COLUMN_SPECS.items():
        if name not in out.columns:
            continue
        try:
            if spec.dtype.startswith("datetime64"):
                # Let pandas infer timezone-aware; coerce errors to NaT
                out[name] = pd.to_datetime(out[name], utc=True, errors="coerce")
            elif spec.dtype in ("Int8", "Int16", "Int32", "Int64"):
                out[name] = pd.to_numeric(out[name], errors="coerce").astype(spec.dtype)
            elif spec.dtype in ("float32", "float64"):
                out[name] = pd.to_numeric(out[name], errors="coerce").astype(spec.dtype)
            elif spec.dtype == "string":
                out[name] = out[name].astype("string")
            else:
                # Fallback attempt
                out[name] = out[name].astype(spec.dtype)
        except Exception:
            # Leave as-is if casting would be lossy
            continue
    return out


def align_to_unified_schema(df: pd.DataFrame, cluster_name: str | None = None) -> pd.DataFrame:
    """
    Convenience helper: fill missing columns and cast to target dtypes,
    returning a frame that matches UNIFIED_COLUMNS in order.
    """
    return cast_dtypes(ensure_unified_columns(df, cluster_name=cluster_name))













