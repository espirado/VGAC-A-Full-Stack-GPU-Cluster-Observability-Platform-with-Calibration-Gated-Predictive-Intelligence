from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import io
import tarfile

import numpy as np
import pandas as pd


@dataclass
class V2020Paths:
    root: Path
    job: Path
    task: Path
    instance: Path
    sensor: Path
    group_tag: Optional[Path]
    machine_spec: Path
    machine_metric: Optional[Path]


def discover_v2020_paths(root: Path | str = "data/raw/alibaba_2020") -> V2020Paths:
    r = Path(root)
    return V2020Paths(
        root=r,
        job=r / "pai_job_table.tar.gz",
        task=r / "pai_task_table.tar.gz",
        instance=r / "pai_instance_table.tar.gz",
        sensor=r / "pai_sensor_table.tar.gz",
        group_tag=(r / "pai_group_tag_table.tar.gz" if (r / "pai_group_tag_table.tar.gz").exists() else None),
        machine_spec=r / "pai_machine_spec.tar.gz",
        machine_metric=(r / "pai_machine_metric.tar.gz" if (r / "pai_machine_metric.tar.gz").exists() else None),
    )


def _read_first_member_bytes(tar_path: Path) -> bytes:
    with tarfile.open(tar_path, mode="r:gz") as tf:
        member = next((m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".csv")), None)
        if member is None:
            raise ValueError(f"No CSV member in {tar_path}")
        f = tf.extractfile(member)
        assert f is not None
        return f.read()


def _maybe_read_header_bytes(tar_dir: Path, tar_name: str) -> Optional[bytes]:
    # Strip .tar.gz if present
    if tar_name.endswith('.tar.gz'):
        base = tar_name[:-7]
    else:
        base = Path(tar_name).stem
    header_path = tar_dir / f"{base}.header"
    if header_path.exists():
        return header_path.read_bytes()
    return None


def read_tar_csv_with_header(tar_path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read CSV inside .tar.gz; if a sibling .header exists, apply it explicitly.

    Assumes header file sits alongside the .tar.gz with name <stem>.header.
    """
    raw = _read_first_member_bytes(tar_path)
    header_bytes = _maybe_read_header_bytes(tar_path.parent, tar_path.name)

    # Fallback expected schemas from v2020 README
    expected: Dict[str, list[str]] = {
        "pai_job_table": [
            "job_name", "inst_id", "user", "status", "start_time", "end_time",
        ],
        "pai_task_table": [
            "job_name", "task_name", "inst_num", "status", "start_time", "end_time",
            "plan_cpu", "plan_mem", "plan_gpu", "gpu_type",
        ],
        "pai_instance_table": [
            "job_name", "task_name", "inst_name", "worker_name", "inst_id", "status",
            "start_time", "end_time", "machine",
        ],
        "pai_sensor_table": [
            "job_name", "task_name", "worker_name", "inst_id", "machine", "gpu_name",
            "cpu_usage", "gpu_wrk_util", "avg_mem", "max_mem", "avg_gpu_wrk_mem", "max_gpu_wrk_mem",
            "read", "write", "read_count", "write_count",
        ],
        "pai_group_tag_table": [
            "inst_id", "user", "gpu_type_spec", "group", "workload",
        ],
        "pai_machine_spec": [
            "machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu",
        ],
        "pai_machine_metric": [
            "worker_name", "machine", "start_time", "end_time", "machine_cpu_iowait",
            "machine_cpu_kernel", "machine_cpu_usr", "machine_gpu", "machine_load_1",
            "machine_net_receive", "machine_num_worker", "machine_cpu",
        ],
    }

    stem = (tar_path.name[:-7] if tar_path.name.endswith('.tar.gz') else tar_path.stem)
    names_from_expected = expected.get(stem)

    if header_bytes is not None:
        header_line = header_bytes.decode("utf-8", errors="ignore").strip()
        names = [c.strip() for c in header_line.split(",")]
        if names_from_expected and len(names) != len(names_from_expected):
            # Header length mismatch; trust expected schema
            names = names_from_expected
        return pd.read_csv(io.BytesIO(raw), nrows=nrows, header=None, names=names, low_memory=False)
    else:
        if names_from_expected:
            return pd.read_csv(io.BytesIO(raw), nrows=nrows, header=None, names=names_from_expected, low_memory=False)
        return pd.read_csv(io.BytesIO(raw), nrows=nrows, header="infer", low_memory=False)


def build_modeling_table(
    paths: V2020Paths,
    sample_rows: int = 200_000,
    underutil_threshold: float = 0.5,
    wait_threshold: float = 3600.0,
    wait_percentile: Optional[float] = None,
) -> pd.DataFrame:
    """Create a compact modeling table for ML from v2020 traces.

    - Features: plan_cpu, plan_gpu, plan_mem, gpu_type, machine capacities, simple time gaps
    - Labels:
      - underutilized: (gpu_wrk_util / plan_gpu) < threshold where plan_gpu>0
      - long_wait: (min task.start - job.start) > wait_threshold
    Note: sampling is applied during CSV read to keep memory bounded.
    """
    # Load tables (sampled)
    task = read_tar_csv_with_header(paths.task, nrows=sample_rows)
    job = read_tar_csv_with_header(paths.job, nrows=sample_rows)
    instance = read_tar_csv_with_header(paths.instance, nrows=sample_rows)
    sensor = read_tar_csv_with_header(paths.sensor, nrows=sample_rows)
    machine_spec = read_tar_csv_with_header(paths.machine_spec, nrows=100_000)

    # Ensure expected columns exist (rename if case/spacing differs)
    def norm(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        return df

    task, job, instance, sensor, machine_spec = map(norm, (task, job, instance, sensor, machine_spec))

    # Compute per-job wait time from job and task tables
    # Resolve time columns (Alibaba v2020 may use different names, e.g., gmt_create/gmt_start)
    def pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        for name in candidates:
            if name in df.columns:
                return name
        return None

    job_start_col = pick_col(job, ["start_time", "gmt_create", "gmt_submit", "submit_time", "gmt_start", "gmt_start_time"]) or "start_time"
    task_start_col = pick_col(task, ["start_time", "gmt_start", "gmt_start_time"]) or "start_time"

    # Normalize time columns to numeric (seconds). If values look like ms, scale down.
    def coerce_time(s: pd.Series) -> pd.Series:
        s2 = pd.to_numeric(s, errors="coerce")
        # Heuristic: if median > 1e11, assume milliseconds, convert to seconds
        med = s2.dropna().median()
        if med is not None and np.isfinite(med) and med > 1e11:
            s2 = s2 / 1000.0
        return s2

    if job_start_col in job.columns:
        job[job_start_col] = coerce_time(job[job_start_col])
    job_end_col = pick_col(job, ["end_time", "gmt_finish", "finish_time", "gmt_end", "gmt_end_time"]) or "end_time"
    if job_end_col in job.columns:
        job[job_end_col] = coerce_time(job[job_end_col])
    if task_start_col in task.columns:
        task[task_start_col] = coerce_time(task[task_start_col])

    # Earliest task.start per job
    task_agg = task.groupby("job_name", as_index=False).agg(task_start_min=(task_start_col, "min"))
    job2 = job.merge(task_agg, on="job_name", how="left")
    job2["wait_time"] = job2["task_start_min"] - job2[job_start_col]
    # Sojourn time (fallback when submission is unavailable): job end - job start
    if job_end_col in job2.columns:
        job2["sojourn_time"] = job2[job_end_col] - job2[job_start_col]
    else:
        job2["sojourn_time"] = np.nan

    # Select relevant columns
    task_sel = task[[
        "job_name", "task_name", "inst_num", "status", "start_time", "end_time", "plan_cpu", "plan_mem", "plan_gpu", "gpu_type"
    ]].copy()

    instance_sel = instance[[
        "job_name", "task_name", "inst_name", "worker_name", "inst_id", "status", "start_time", "end_time", "machine"
    ]].copy()

    sensor_sel = sensor[[
        "job_name", "task_name", "worker_name", "inst_id", "machine", "gpu_name", "cpu_usage", "gpu_wrk_util", "avg_mem", "max_mem", "avg_gpu_wrk_mem", "max_gpu_wrk_mem"
    ]].copy()

    mspec_sel = machine_spec[["machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu"]].copy()

    # Join instance -> sensor (instance lifetime metrics)
    inst_sensor = instance_sel.merge(sensor_sel, on=["job_name", "task_name", "worker_name", "inst_id", "machine"], how="left")

    # Join task -> instance/sensor (task allocations with realized usage)
    task_inst = inst_sensor.merge(task_sel, on=["job_name", "task_name"], how="left", suffixes=("", "_task"))

    # Join machine specs for normalization
    task_inst = task_inst.merge(mspec_sel, on="machine", how="left", suffixes=("", "_mspec"))

    # Join job wait time (prefer the earliest wait per job replicated to its tasks)
    task_inst = task_inst.merge(job2[["job_name", "wait_time", "sojourn_time"]], on="job_name", how="left")
    # Sanity: ensure non-negative wait_time; clip negatives to zero
    if "wait_time" in task_inst.columns:
        task_inst["wait_time"] = pd.to_numeric(task_inst["wait_time"], errors="coerce")
        task_inst.loc[task_inst["wait_time"] < 0, "wait_time"] = 0.0

    # Compute labels
    df = task_inst.copy()
    for col in ["plan_gpu", "gpu_wrk_util", "plan_cpu", "plan_mem", "cap_cpu", "cap_mem", "cap_gpu", "wait_time", "sojourn_time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Avoid division by zero / NaNs
    df["gpu_efficiency"] = np.where(df["plan_gpu"].fillna(0) > 0, df["gpu_wrk_util"] / df["plan_gpu"], np.nan)
    df["underutilized"] = (df["gpu_efficiency"] < underutil_threshold).astype("Int8")
    if wait_percentile is not None:
        # Compute threshold from available wait_time values; if degenerate, fall back to sojourn_time percentiles
        wt = df["wait_time"].dropna()
        used_fallback = False
        if len(wt) > 0 and wt.abs().sum() > 0:
            thr = float(wt.quantile(wait_percentile))
            mask = df["wait_time"]
        else:
            st = df["sojourn_time"].dropna()
            if len(st) > 0 and st.abs().sum() > 0:
                thr = float(st.quantile(wait_percentile))
                mask = df["sojourn_time"]
                used_fallback = True
            else:
                thr = wait_threshold
                mask = df["wait_time"]
        df["long_wait"] = (mask >= thr).astype("Int8")
        df.attrs["long_wait_threshold"] = thr
        df.attrs["long_wait_percentile"] = wait_percentile
        df.attrs["long_wait_used_fallback_sojourn"] = used_fallback
    else:
        df["long_wait"] = (df["wait_time"] > wait_threshold).astype("Int8")
        df.attrs["long_wait_threshold"] = wait_threshold
        df.attrs["long_wait_percentile"] = None

    # Minimal feature set (no realized usage to avoid leakage for underutilized label)
    features = [
        "plan_cpu", "plan_mem", "plan_gpu", "gpu_type_task", "cap_cpu", "cap_mem", "cap_gpu"
    ]
    # If joined gpu_type duplicated, prefer task's gpu_type
    if "gpu_type_task" not in df.columns and "gpu_type" in df.columns:
        df.rename(columns={"gpu_type": "gpu_type_task"}, inplace=True)

    out_cols = [
        "job_name", "task_name", "worker_name", "machine",
        *features,
        "underutilized", "long_wait", "gpu_efficiency", "wait_time", "sojourn_time",
    ]
    df_out = df[out_cols].dropna(subset=["plan_cpu", "plan_mem", "plan_gpu"]).reset_index(drop=True)
    return df_out


