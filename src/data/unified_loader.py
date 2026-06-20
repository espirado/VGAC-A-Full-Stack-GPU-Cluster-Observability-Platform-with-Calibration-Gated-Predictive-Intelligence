from __future__ import annotations

"""
Dataset unification loaders.

Goal:
- Provide a single entry point to load Azure, EKS (archive), Alibaba, and Google datasets
  into the unified feature schema used by our modeling stack.

Notes:
- For EKS, we reuse features produced by `src/feature/extract_k8s_submit_features.py`
  (parquet/csv). If needed, we can call the extractor upstream in notebooks/scripts.
- For Azure/Alibaba/Google, we expose minimal adapters that map available columns into
  the unified schema. These can be extended incrementally as we enrich sources.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.features.schema import align_to_unified_schema


def load_eks_features(features_path: Path, cluster_name: Optional[str] = "eks") -> pd.DataFrame:
    """
    Load features previously produced by `extract_k8s_submit_features.py` and align
    to the unified schema.
    """
    p = Path(features_path)
    if not p.exists():
        raise FileNotFoundError(f"Features path not found: {p}")
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    return align_to_unified_schema(df, cluster_name=cluster_name)


def load_azure_submit_table(df: pd.DataFrame, cluster_name: Optional[str] = "azure") -> pd.DataFrame:
    """
    Map an Azure submit-time table into the unified schema.
    Expected columns (example; update as needed):
      - namespace, pod/job identifiers (or equivalents)
      - cpu_m, mem_mb, gpu (requested)
      - timestamps for submission/start (to derive tts_seconds)
    """
    out = pd.DataFrame()
    # Identifiers
    out["namespace"] = df.get("namespace", pd.Series([None] * len(df)))
    out["pod"] = df.get("pod", df.get("job_name", pd.Series([None] * len(df))))
    out["submit_ts"] = pd.to_datetime(df.get("submit_ts"), utc=True, errors="coerce")
    # Requests
    out["num_containers"] = df.get("num_containers", 1)
    out["req_cpu_m"] = df.get("cpu_m", df.get("req_cpu_m"))
    out["req_mem_mb"] = df.get("mem_mb", df.get("req_mem_mb"))
    out["req_gpu"] = df.get("gpu", df.get("req_gpu", 0))
    # Labels
    start_ts = pd.to_datetime(df.get("start_ts"), utc=True, errors="coerce")
    tts = (start_ts - out["submit_ts"]).dt.total_seconds()
    out["tts_seconds"] = tts
    out["phase"] = df.get("phase", None)
    out["age_seconds"] = df.get("age_seconds", None)
    # Align and return
    return align_to_unified_schema(out, cluster_name=cluster_name)


def load_alibaba_sample(df: pd.DataFrame, cluster_name: Optional[str] = "alibaba") -> pd.DataFrame:
    """
    Minimal adapter for Alibaba-style job tables into the unified schema.
    Assumes presence of job submit time and resource requests where available.
    """
    out = pd.DataFrame()
    out["namespace"] = df.get("queue", None)
    out["pod"] = df.get("job_id", None)
    out["submit_ts"] = pd.to_datetime(df.get("submission_time"), utc=True, errors="coerce")
    out["num_containers"] = df.get("num_containers", 1)
    out["req_cpu_m"] = df.get("cpu_request_m", None)
    out["req_mem_mb"] = df.get("memory_gb", None) * 1024 if "memory_gb" in df else df.get("req_mem_mb", None)
    out["req_gpu"] = df.get("gpu_request", 0)
    out["tts_seconds"] = df.get("queue_time_seconds", None)
    return align_to_unified_schema(out, cluster_name=cluster_name)


def load_google_sample(df: pd.DataFrame, cluster_name: Optional[str] = "google2019") -> pd.DataFrame:
    """
    Minimal adapter for Google/Borg-like traces into the unified schema.
    """
    out = pd.DataFrame()
    out["namespace"] = df.get("pool", None)
    out["pod"] = df.get("task_id", df.get("job_id", None))
    out["submit_ts"] = pd.to_datetime(df.get("submit_time"), utc=True, errors="coerce")
    out["num_containers"] = df.get("num_instances", 1)
    out["req_cpu_m"] = df.get("requested_cpu_m", None)
    out["req_mem_mb"] = df.get("requested_mem_mb", None)
    out["req_gpu"] = df.get("requested_gpu", 0)
    # If start/end timestamps exist, derive tts_seconds
    start_ts = pd.to_datetime(df.get("start_time"), utc=True, errors="coerce")
    tts = (start_ts - out["submit_ts"]).dt.total_seconds()
    out["tts_seconds"] = tts
    return align_to_unified_schema(out, cluster_name=cluster_name)













