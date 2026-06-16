"""
GPU-Ext Bridge

Integration layer between VGAC predictions and gpu_ext eBPF policies.
Translates scheduling policies into gpu_ext-compatible configurations.

Reference: https://arxiv.org/abs/2512.12615
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List
import json

from .generator import SchedulingPolicy, PriorityClass, MemoryPolicy, PreemptionPolicy

logger = logging.getLogger(__name__)


class PolicyAction(Enum):
    """Actions that gpu_ext can execute."""
    # Scheduling actions
    SET_PRIORITY = "set_priority"
    ENABLE_PREEMPTION = "enable_preemption"
    SET_TIME_SLICE = "set_time_slice"
    
    # Memory actions
    SET_MEMORY_POLICY = "set_memory_policy"
    ENABLE_PREFETCH = "enable_prefetch"
    SET_EVICTION_POLICY = "set_eviction_policy"
    
    # Colocation actions
    ENABLE_PACKING = "enable_packing"
    SET_INTERFERENCE_THRESHOLD = "set_interference_threshold"
    
    # Observability actions
    ENABLE_PROFILING = "enable_profiling"
    SET_TRACE_LEVEL = "set_trace_level"


@dataclass
class EBPFPolicyConfig:
    """Configuration for gpu_ext eBPF programs."""
    
    # Scheduling
    priority_class: int = 0          # 0-255, higher = more priority
    preempt_eligible: bool = True
    time_slice_us: int = 100000      # microseconds
    
    # Memory
    prefetch_enabled: bool = True
    prefetch_distance: int = 4       # pages ahead
    eviction_policy: str = "lru"     # lru, lfu, fifo
    memory_pressure_threshold: float = 0.8
    
    # Colocation
    packing_enabled: bool = False
    interference_threshold: float = 0.1
    min_gpu_util_for_pack: float = 0.3
    
    # Observability
    profiling_enabled: bool = False
    trace_level: int = 0             # 0=off, 1=summary, 2=detailed
    
    def to_ebpf_map(self) -> Dict[str, Any]:
        """Convert to format suitable for eBPF map."""
        return {
            "sched": {
                "priority": self.priority_class,
                "preempt": 1 if self.preempt_eligible else 0,
                "time_slice_us": self.time_slice_us,
            },
            "mem": {
                "prefetch": 1 if self.prefetch_enabled else 0,
                "prefetch_dist": self.prefetch_distance,
                "eviction": {"lru": 0, "lfu": 1, "fifo": 2}.get(self.eviction_policy, 0),
                "pressure_thresh": int(self.memory_pressure_threshold * 100),
            },
            "pack": {
                "enabled": 1 if self.packing_enabled else 0,
                "interference_thresh": int(self.interference_threshold * 100),
                "min_util": int(self.min_gpu_util_for_pack * 100),
            },
            "obs": {
                "profile": 1 if self.profiling_enabled else 0,
                "trace": self.trace_level,
            },
        }


class GPUExtBridge:
    """
    Bridge between VGAC scheduling policies and gpu_ext eBPF runtime.
    
    This class:
    1. Translates VGAC policies to gpu_ext eBPF configurations
    2. Manages eBPF program loading (placeholder for actual gpu_ext integration)
    3. Tracks policy applications for validation
    
    In production, this would interface with gpu_ext's control plane
    to dynamically update eBPF maps with policy configurations.
    """
    
    # Priority class mapping (VGAC → gpu_ext)
    PRIORITY_MAP = {
        PriorityClass.CRITICAL: 255,
        PriorityClass.HIGH: 200,
        PriorityClass.NORMAL: 128,
        PriorityClass.LOW: 64,
        PriorityClass.BEST_EFFORT: 0,
    }
    
    # Memory policy mapping
    MEMORY_PREFETCH_MAP = {
        MemoryPolicy.CONSERVATIVE: (False, 0),
        MemoryPolicy.BALANCED: (True, 2),
        MemoryPolicy.AGGRESSIVE: (True, 8),
        MemoryPolicy.STREAMING: (False, 0),
    }
    
    def __init__(
        self,
        gpu_ext_socket: Optional[str] = None,
        dry_run: bool = True,
    ):
        """
        Initialize gpu_ext bridge.
        
        Args:
            gpu_ext_socket: Path to gpu_ext control socket (if available)
            dry_run: If True, don't actually apply policies (for testing)
        """
        self.gpu_ext_socket = gpu_ext_socket
        self.dry_run = dry_run
        self._connected = False
        self._policy_cache: Dict[str, EBPFPolicyConfig] = {}
        self._application_log: List[Dict[str, Any]] = []
    
    def connect(self) -> bool:
        """
        Connect to gpu_ext control plane.
        
        Returns:
            True if connected successfully
        """
        if self.dry_run:
            logger.info("gpu_ext bridge in dry-run mode")
            self._connected = True
            return True
        
        if not self.gpu_ext_socket:
            logger.warning("No gpu_ext socket configured")
            return False
        
        try:
            # In production: connect to gpu_ext control socket
            # For now, this is a placeholder
            logger.info(f"Would connect to gpu_ext at {self.gpu_ext_socket}")
            self._connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect to gpu_ext: {e}")
            return False
    
    def translate_policy(self, policy: SchedulingPolicy) -> EBPFPolicyConfig:
        """
        Translate VGAC policy to gpu_ext eBPF configuration.
        
        Args:
            policy: VGAC scheduling policy
            
        Returns:
            gpu_ext eBPF configuration
        """
        # Priority
        base_priority = self.PRIORITY_MAP.get(policy.priority, 128)
        final_priority = max(0, min(255, base_priority + policy.priority_boost))
        
        # Preemption
        preempt_eligible = policy.preemption not in (
            PreemptionPolicy.NEVER,
            PreemptionPolicy.RELUCTANT,
        )
        
        # Memory
        prefetch_enabled, prefetch_distance = self.MEMORY_PREFETCH_MAP.get(
            policy.memory, (True, 2)
        )
        
        # Eviction policy
        if policy.memory == MemoryPolicy.STREAMING:
            eviction_policy = "fifo"
        elif policy.memory == MemoryPolicy.CONSERVATIVE:
            eviction_policy = "lru"
        else:
            eviction_policy = "lfu"
        
        # Colocation/packing
        from .generator import ColocationPolicy
        packing_enabled = policy.colocation == ColocationPolicy.PACKING
        
        return EBPFPolicyConfig(
            priority_class=final_priority,
            preempt_eligible=preempt_eligible,
            time_slice_us=policy.time_slice_ms * 1000,
            prefetch_enabled=prefetch_enabled,
            prefetch_distance=prefetch_distance,
            eviction_policy=eviction_policy,
            packing_enabled=packing_enabled,
            profiling_enabled=True,  # Always profile for validation
            trace_level=1,
        )
    
    def apply_policy(
        self,
        job_id: str,
        policy: SchedulingPolicy,
    ) -> Dict[str, Any]:
        """
        Apply scheduling policy for a job via gpu_ext.
        
        Args:
            job_id: Job identifier
            policy: VGAC scheduling policy
            
        Returns:
            Application result with status and details
        """
        # Translate to eBPF config
        ebpf_config = self.translate_policy(policy)
        
        # Cache for later retrieval
        self._policy_cache[job_id] = ebpf_config
        
        # Log application
        application = {
            "job_id": job_id,
            "policy": policy.to_dict(),
            "ebpf_config": ebpf_config.to_ebpf_map(),
            "dry_run": self.dry_run,
            "applied": True,
        }
        self._application_log.append(application)
        
        if self.dry_run:
            logger.info(
                f"[DRY-RUN] Would apply policy for {job_id}: "
                f"priority={ebpf_config.priority_class}, "
                f"prefetch={ebpf_config.prefetch_enabled}"
            )
            return {
                "status": "dry_run",
                "job_id": job_id,
                "config": ebpf_config.to_ebpf_map(),
            }
        
        try:
            # In production: send to gpu_ext control plane
            # self._send_to_gpu_ext(job_id, ebpf_config)
            
            logger.info(f"Applied gpu_ext policy for {job_id}")
            return {
                "status": "applied",
                "job_id": job_id,
                "config": ebpf_config.to_ebpf_map(),
            }
        except Exception as e:
            logger.error(f"Failed to apply policy for {job_id}: {e}")
            return {
                "status": "error",
                "job_id": job_id,
                "error": str(e),
            }
    
    def get_policy(self, job_id: str) -> Optional[EBPFPolicyConfig]:
        """Get cached policy for a job."""
        return self._policy_cache.get(job_id)
    
    def remove_policy(self, job_id: str) -> bool:
        """Remove policy when job completes."""
        if job_id in self._policy_cache:
            del self._policy_cache[job_id]
            logger.debug(f"Removed policy for {job_id}")
            return True
        return False
    
    def get_application_log(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get recent policy applications."""
        return self._application_log[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "connected": self._connected,
            "dry_run": self.dry_run,
            "active_policies": len(self._policy_cache),
            "total_applications": len(self._application_log),
        }


# Singleton instance
_bridge: Optional[GPUExtBridge] = None


def get_bridge(dry_run: bool = True) -> GPUExtBridge:
    """Get or create gpu_ext bridge singleton."""
    global _bridge
    if _bridge is None:
        _bridge = GPUExtBridge(dry_run=dry_run)
        _bridge.connect()
    return _bridge

