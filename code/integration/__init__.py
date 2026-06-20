"""Submit-time capture and admission integration for VGAC.

Modules:
    submit_capture - submit-time queue-state capture (the L1 layer
                     described in ``docs/ARCHITECTURE.md``); contains
                     reference adapters for Kubernetes admission
                     webhooks and Slurm ``job_submit`` hooks.
    policy_translate - probability -> policy-band translator used by
                       ``code/policy``.
"""

from .submit_capture import (
    QueueSnapshot,
    SubmitTimeCapturer,
    SlurmJobSubmitAdapter,
    K8sAdmissionAdapter,
)

__all__ = [
    "QueueSnapshot",
    "SubmitTimeCapturer",
    "SlurmJobSubmitAdapter",
    "K8sAdmissionAdapter",
]
