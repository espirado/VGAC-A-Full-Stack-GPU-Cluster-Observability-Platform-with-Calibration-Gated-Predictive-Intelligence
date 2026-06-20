import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        # Handle Z suffix
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def parse_cpu_millicores(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value)
    try:
        if s.endswith("m"):
            return float(s[:-1])
        # cores -> millicores
        return float(s) * 1000.0
    except Exception:
        return None


def parse_memory_bytes(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value)
    units = [
        ("Ki", 1024),
        ("Mi", 1024 ** 2),
        ("Gi", 1024 ** 3),
        ("Ti", 1024 ** 4),
        ("K", 1000),
        ("M", 1000 ** 2),
        ("G", 1000 ** 3),
    ]
    for suf, mult in units:
        if s.endswith(suf):
            try:
                return float(s[: -len(suf)]) * mult
            except Exception:
                return None
    try:
        return float(s)
    except Exception:
        return None


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return {}


def build_job_index(jobs: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for j in jobs.get("items", []):
        ns = (j.get("metadata") or {}).get("namespace") or "default"
        name = (j.get("metadata") or {}).get("name") or ""
        idx[(ns, name)] = j
    return idx


def recent_event_counts(events: Dict[str, Any], window_minutes: int = 15) -> Dict[str, int]:
    by_ns: Counter[str] = Counter()
    now = datetime.now(timezone.utc)
    for e in events.get("items", []):
        ns = (e.get("metadata") or {}).get("namespace") or "default"
        t = parse_iso((e.get("lastTimestamp") or e.get("eventTime") or ""))
        if t is None:
            continue
        # treat timezone-naive as UTC
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        if (now - t).total_seconds() <= window_minutes * 60:
            by_ns[ns] += 1
    return dict(by_ns)


def main() -> None:
    p = argparse.ArgumentParser(description="Extract submit-time features from k8s snapshots")
    p.add_argument("--pods", type=Path, required=True)
    p.add_argument("--jobs", type=Path, required=True)
    p.add_argument("--nodes", type=Path, required=True)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Output Parquet path (on mounted disk)")
    p.add_argument("--threshold_sec", type=int, default=300, help="Long-wait label threshold in seconds")
    args = p.parse_args()

    pods = load_json(args.pods)
    jobs = load_json(args.jobs)
    nodes = load_json(args.nodes)
    events = load_json(args.events)

    job_idx = build_job_index(jobs)
    ev_recent = recent_event_counts(events, window_minutes=15)
    # Event-derived signals
    def recent_reason_counts(reasons: List[str], window_minutes: int = 15) -> Dict[str, int]:
        by_ns: Counter[str] = Counter()
        now = datetime.now(timezone.utc)
        for e in events.get("items", []):
            if (e.get("reason") or "") not in reasons:
                continue
            ns = (e.get("metadata") or {}).get("namespace") or "default"
            t = parse_iso((e.get("lastTimestamp") or e.get("eventTime") or ""))
            if t is None:
                continue
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            if (now - t).total_seconds() <= window_minutes * 60:
                by_ns[ns] += 1
        return dict(by_ns)

    ev_imgpull = recent_reason_counts(["ErrImagePull", "ImagePullBackOff"], window_minutes=15)
    ev_backoff = recent_reason_counts(["BackOff"], window_minutes=15)

    node_count = len(nodes.get("items", []))
    gpu_capacity = 0
    for n in nodes.get("items", []):
        cap = (n.get("status", {}) or {}).get("capacity", {}) or {}
        try:
            gpu_capacity += int(cap.get("nvidia.com/gpu", "0"))
        except Exception:
            pass

    rows: List[Dict[str, Any]] = []
    for pod in pods.get("items", []):
        md = pod.get("metadata", {}) or {}
        st = pod.get("status", {}) or {}
        sp = pod.get("spec", {}) or {}
        ns = md.get("namespace") or "default"
        name = md.get("name") or ""
        creation = parse_iso(md.get("creationTimestamp"))
        start = parse_iso(st.get("startTime"))
        tts = None
        if creation and start and start >= creation:
            tts = (start - creation).total_seconds()

        # resources
        cpu_m = 0.0
        mem_b = 0.0
        gpu_req = 0
        num_cont = 0
        for c in (sp.get("containers") or []):
            num_cont += 1
            req = (c.get("resources") or {}).get("requests") or {}
            cpu_v = parse_cpu_millicores(req.get("cpu"))
            mem_v = parse_memory_bytes(req.get("memory"))
            gpu_v = req.get("nvidia.com/gpu") or req.get("gpu")
            if cpu_v is not None:
                cpu_m += cpu_v
            if mem_v is not None:
                mem_b += mem_v
            try:
                if gpu_v is not None:
                    gpu_req += int(str(gpu_v))
            except Exception:
                pass

        # constraints
        node_selector = sp.get("nodeSelector") or {}
        has_node_selector = 1 if node_selector else 0
        node_selector_keys = len(node_selector)

        affinity = sp.get("affinity") or {}
        has_affinity = 1 if affinity else 0
        # node affinity specifics
        na_req_terms = 0
        na_pref_terms = 0
        try:
            na = (affinity.get("nodeAffinity") or {})
            req = (na.get("requiredDuringSchedulingIgnoredDuringExecution") or {}).get("nodeSelectorTerms") or []
            na_req_terms = len(req)
            pref = na.get("preferredDuringSchedulingIgnoredDuringExecution") or []
            na_pref_terms = len(pref)
        except Exception:
            pass

        tolerations = sp.get("tolerations") or []
        num_tolerations = len(tolerations)
        tol_keys = set()
        tol_effect_nosched = 0
        for t in tolerations:
            k = t.get("key")
            if k is not None:
                tol_keys.add(str(k))
            if str(t.get("effect")) == "NoSchedule":
                tol_effect_nosched += 1

        # container images & policies
        num_images_ecr = 0
        num_images_dockerhub = 0
        num_images_other = 0
        image_pull_always = 0
        for c in (sp.get("containers") or []):
            img = str(c.get("image") or "")
            if ".amazonaws.com/" in img:
                num_images_ecr += 1
            elif "/" not in img or ("/" in img and "://" not in img):
                # Likely Docker Hub library image when no registry domain present
                num_images_dockerhub += 1
            else:
                num_images_other += 1
            if str(c.get("imagePullPolicy") or "") == "Always":
                image_pull_always += 1

        # job linkage
        job_parallelism = None
        job_completions = None
        for ref in (md.get("ownerReferences") or []):
            if str(ref.get("kind")) == "Job":
                jname = ref.get("name")
                j = job_idx.get((ns, jname))
                if j:
                    spj = j.get("spec", {}) or {}
                    job_parallelism = spj.get("parallelism")
                    job_completions = spj.get("completions")
                break

        # compute age now for pending labeling
        now = datetime.now(timezone.utc)
        age_seconds = None
        if creation is not None:
            c = creation
            if c.tzinfo is None:
                c = c.replace(tzinfo=timezone.utc)
            age_seconds = (now - c).total_seconds()

        rows.append(
            {
                "namespace": ns,
                "pod": name,
                "num_containers": num_cont,
                "req_cpu_m": cpu_m,
                "req_mem_mb": mem_b / (1024 ** 2) if mem_b else 0.0,
                "req_gpu": gpu_req,
                "has_node_selector": has_node_selector,
                "has_affinity": has_affinity,
                "num_tolerations": num_tolerations,
                "toleration_keys_count": len(tol_keys),
                "tolerations_effect_noSchedule": tol_effect_nosched,
                "node_selector_keys": node_selector_keys,
                "na_required_terms": na_req_terms,
                "na_preferred_terms": na_pref_terms,
                "num_images_ecr": num_images_ecr,
                "num_images_dockerhub": num_images_dockerhub,
                "num_images_other": num_images_other,
                "image_pull_always": image_pull_always,
                "job_parallelism": job_parallelism,
                "job_completions": job_completions,
                "cluster_node_count": node_count,
                "cluster_gpu_capacity": gpu_capacity,
                "recent_failed_scheduling_ns": ev_recent.get(ns, 0),
                "recent_image_pull_err_ns": ev_imgpull.get(ns, 0),
                "recent_backoff_ns": ev_backoff.get(ns, 0),
                "phase": st.get("phase"),
                "tts_seconds": tts,
                "age_seconds": age_seconds,
            }
        )

    if not rows:
        print("No rows extracted; aborting.")
        return

    df = pd.DataFrame(rows)
    # Label: started pods by tts; pending pods by wall time since creation
    thr = float(args.threshold_sec)
    long_started = df["tts_seconds"].notna() & (df["tts_seconds"] > thr)
    long_pending = (df["tts_seconds"].isna()) & (df["phase"].isin(["Pending", "Unknown"])) & (df["age_seconds"].notna()) & (df["age_seconds"] > thr)
    df["label_long_wait"] = np.where(long_started | long_pending, 1, 0)

    # Write features to mounted disk; prefer Parquet, fallback to CSV if engine missing
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if str(args.out).endswith(".parquet"):
            df.to_parquet(args.out, index=False)
            print(f"Wrote features: {args.out} rows={len(df)}")
        else:
            df.to_csv(args.out, index=False)
            print(f"Wrote features: {args.out} rows={len(df)}")
    except Exception as e:
        # Fallback to CSV next to requested path
        csv_path = args.out.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"Parquet unavailable ({e}); wrote CSV instead: {csv_path} rows={len(df)}")


if __name__ == "__main__":
    main()


