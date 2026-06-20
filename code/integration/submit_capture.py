"""Submit-time queue-state capture (L1).

This is the observability primitive whose absence produces the paper's
inverted correlation (Pearson r = -0.27 under naive 30 s polling) and
whose presence restores the expected sign (r = +0.44). The contribution
is methodological rather than algorithmic: capture cluster state at the
exact moment the user submits, not at the next poll.

Two reference adapters are provided so reviewers can see how the same
``QueueSnapshot`` schema bridges into Kubernetes and Slurm without
duplicating logic:

    - ``K8sAdmissionAdapter``  : called from a ValidatingAdmissionWebhook
      that watches ``CREATE`` on Pods/Jobs.
    - ``SlurmJobSubmitAdapter``: called from a Slurm ``job_submit/lua``
      hook on submit.

Both adapters call the shared ``SubmitTimeCapturer`` which queries a
state provider for the *current* queue depth, fragmentation score, and
GPU availability, then emits a ``QueueSnapshot``. State providers are
pluggable - tests use ``InMemoryStateProvider``; production deployments
plug a Kubernetes informer or a Slurm ``squeue`` reader.

NOTE: The actual webhook server / job_submit shim are out of scope for
this short paper (4 pages); they are referenced in
``docs/ARCHITECTURE.md`` Section 7. This module is the
language-agnostic core.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Protocol


@dataclass(frozen=True)
class QueueSnapshot:
    """Submit-time observation of cluster queue state.

    The fields here are exactly the L1 columns described in
    ``docs/DATA_DICTIONARY.md`` (section "submit-time features"). They
    feed directly into the universal feature schema.
    """

    submit_ts: float
    job_id: str
    pending_jobs: int
    pending_gpus: int
    running_jobs: int
    running_gpus: int
    gpu_nodes_alloc: int
    gpu_nodes_total: int
    job_type: str
    job_gpu_request: int
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def fragmentation_score(self) -> float:
        """Fraction of GPU-capable nodes already partially in use.

        Defined as ``alloc_nodes / total_nodes`` and clamped to [0, 1].
        Higher = more fragmented = harder to find a contiguous block.
        """
        if self.gpu_nodes_total <= 0:
            return 0.0
        return float(min(1.0, max(0.0, self.gpu_nodes_alloc / self.gpu_nodes_total)))

    @property
    def pending_ratio(self) -> float:
        """``pending_gpus / max(running_gpus, 1)`` - the headline
        feature whose correlation flipped sign in Section 4."""
        return float(self.pending_gpus) / float(max(self.running_gpus, 1))

    def to_feature_row(self) -> Dict[str, Any]:
        """Flat dict suitable for downstream feature stores."""
        row = asdict(self)
        row["fragmentation_score"] = self.fragmentation_score
        row["pending_ratio"] = self.pending_ratio
        return row


# ---------------------------------------------------------------------------
# State providers
# ---------------------------------------------------------------------------
class StateProvider(Protocol):
    """Anything that can answer 'what is the queue state right now?'."""

    def snapshot(self) -> Dict[str, Any]:
        ...


@dataclass
class InMemoryStateProvider:
    """Test/development state provider backed by a mutable dict.

    Production code uses a Kubernetes informer or ``squeue`` reader;
    those live in deployment manifests outside this artifact.
    """

    state: Dict[str, Any] = field(default_factory=lambda: {
        "pending_jobs": 0,
        "pending_gpus": 0,
        "running_jobs": 0,
        "running_gpus": 0,
        "gpu_nodes_alloc": 0,
        "gpu_nodes_total": 0,
    })

    def snapshot(self) -> Dict[str, Any]:
        return dict(self.state)


# ---------------------------------------------------------------------------
# Capturer
# ---------------------------------------------------------------------------
@dataclass
class SubmitTimeCapturer:
    """Synthesizes a ``QueueSnapshot`` from a state provider on submit.

    Polling cadence is *not* the responsibility of this class - the
    state provider is expected to be event-driven (Kubernetes informer
    is push-based; Slurm ``job_submit`` runs synchronously inside the
    submit RPC). We poll every 5 s only as a fallback when neither is
    available, matching the paper's footnote on baseline cadence.
    """

    state_provider: StateProvider

    def capture(
        self,
        job_id: str,
        job_type: str,
        gpu_request: int,
        submit_ts: Optional[float] = None,
        **extra: Any,
    ) -> QueueSnapshot:
        if submit_ts is None:
            submit_ts = time.time()
        s = self.state_provider.snapshot()
        return QueueSnapshot(
            submit_ts=float(submit_ts),
            job_id=str(job_id),
            pending_jobs=int(s.get("pending_jobs", 0)),
            pending_gpus=int(s.get("pending_gpus", 0)),
            running_jobs=int(s.get("running_jobs", 0)),
            running_gpus=int(s.get("running_gpus", 0)),
            gpu_nodes_alloc=int(s.get("gpu_nodes_alloc", 0)),
            gpu_nodes_total=int(s.get("gpu_nodes_total", 0)),
            job_type=str(job_type),
            job_gpu_request=int(gpu_request),
            extra=dict(extra),
        )


# ---------------------------------------------------------------------------
# Reference adapters
# ---------------------------------------------------------------------------
@dataclass
class K8sAdmissionAdapter:
    """Adapter hook for a Kubernetes ValidatingAdmissionWebhook.

    The webhook receives an ``AdmissionRequest`` for a Pod or Job;
    ``capture_from_admission`` extracts the relevant fields and
    delegates to the shared capturer. Returns the snapshot so the
    webhook can persist it before deciding to admit.
    """

    capturer: SubmitTimeCapturer

    def capture_from_admission(self, req: Dict[str, Any]) -> QueueSnapshot:
        obj = req.get("object", {})
        meta = obj.get("metadata", {}) or {}
        spec = obj.get("spec", {}) or {}
        containers = spec.get("containers", []) or []
        gpu_request = 0
        for c in containers:
            res = (c.get("resources", {}) or {}).get("requests", {}) or {}
            gpu_request += int(res.get("nvidia.com/gpu", 0) or 0)
        return self.capturer.capture(
            job_id=str(meta.get("uid") or meta.get("name") or "unknown"),
            job_type=meta.get("labels", {}).get("job-type", "unknown"),
            gpu_request=gpu_request,
            namespace=meta.get("namespace", "default"),
        )


@dataclass
class SlurmJobSubmitAdapter:
    """Adapter hook for a Slurm ``job_submit/lua`` plugin.

    Slurm passes a job-record table; this method shapes it into a
    ``QueueSnapshot`` via the shared capturer. Returning ``None`` from
    the underlying Lua hook is fine - this adapter is non-blocking.
    """

    capturer: SubmitTimeCapturer

    def capture_from_job_record(self, job_record: Dict[str, Any]) -> QueueSnapshot:
        return self.capturer.capture(
            job_id=str(job_record.get("job_id", job_record.get("name", "unknown"))),
            job_type=str(job_record.get("partition", "default")),
            gpu_request=int(job_record.get("tres_per_node", {}).get("gpu", 0) or 0),
            account=job_record.get("account"),
            qos=job_record.get("qos"),
        )


if __name__ == "__main__":
    sp = InMemoryStateProvider(state={
        "pending_jobs": 12,
        "pending_gpus": 32,
        "running_jobs": 7,
        "running_gpus": 24,
        "gpu_nodes_alloc": 6,
        "gpu_nodes_total": 16,
    })
    cap = SubmitTimeCapturer(sp)
    snap = cap.capture(job_id="job-001", job_type="train", gpu_request=4)
    print("snapshot      :", snap)
    print("frag_score    :", snap.fragmentation_score)
    print("pending_ratio :", snap.pending_ratio)
    print("feature_row   :", snap.to_feature_row())

    k8s = K8sAdmissionAdapter(cap)
    pod_req = {
        "object": {
            "metadata": {"uid": "abc", "name": "trainer", "namespace": "ml", "labels": {"job-type": "train"}},
            "spec": {"containers": [{"resources": {"requests": {"nvidia.com/gpu": "2"}}}]},
        }
    }
    print("k8s snapshot  :", k8s.capture_from_admission(pod_req))
