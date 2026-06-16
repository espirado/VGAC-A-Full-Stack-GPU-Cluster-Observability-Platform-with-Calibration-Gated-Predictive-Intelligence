"""
Policy Generator

Converts VGAC predictions into scheduling policies.
Maps calibrated probabilities to concrete policy actions.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class PriorityClass(Enum):
    """Job priority classes."""
    CRITICAL = "critical"      # System-critical, never preempt
    HIGH = "high"              # User-facing, minimize wait
    NORMAL = "normal"          # Default
    LOW = "low"                # Batch, can wait
    BEST_EFFORT = "best_effort"  # Fill capacity, preempt freely


class PreemptionPolicy(Enum):
    """Preemption behavior."""
    NEVER = "never"            # Never preempt this job
    RELUCTANT = "reluctant"    # Only preempt if critical
    NORMAL = "normal"          # Standard preemption rules
    EAGER = "eager"            # Preempt for higher priority
    IMMEDIATE = "immediate"    # Can be preempted immediately


class MemoryPolicy(Enum):
    """GPU memory management policy."""
    CONSERVATIVE = "conservative"  # Minimize memory, swap to host
    BALANCED = "balanced"          # Default memory management
    AGGRESSIVE = "aggressive"      # Prefetch aggressively
    STREAMING = "streaming"        # Stream data, don't cache


class ColocationPolicy(Enum):
    """Job colocation behavior."""
    ISOLATED = "isolated"      # Run alone on GPU
    TOLERANT = "tolerant"      # Can share with compatible jobs
    PACKING = "packing"        # Actively pack with others


@dataclass
class SchedulingPolicy:
    """Complete scheduling policy for a job."""
    priority: PriorityClass = PriorityClass.NORMAL
    preemption: PreemptionPolicy = PreemptionPolicy.NORMAL
    memory: MemoryPolicy = MemoryPolicy.BALANCED
    colocation: ColocationPolicy = ColocationPolicy.TOLERANT
    
    # Numeric parameters
    priority_boost: int = 0          # -100 to 100
    time_slice_ms: int = 100         # Time-slice duration
    memory_limit_mb: Optional[int] = None
    
    # Metadata
    reason: str = ""
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "priority": self.priority.value,
            "preemption": self.preemption.value,
            "memory": self.memory.value,
            "colocation": self.colocation.value,
            "priority_boost": self.priority_boost,
            "time_slice_ms": self.time_slice_ms,
            "memory_limit_mb": self.memory_limit_mb,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass
class PolicyResult:
    """Result of policy generation."""
    policy: SchedulingPolicy
    use_default: bool = False
    annotations: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PredictionInput:
    """Input predictions for policy generation."""
    probability_long_wait: float
    probability_fragmentation: float = 0.0
    probability_under_utilization: float = 0.0
    probability_phase_imbalance: float = 0.0
    confidence: float = 1.0
    estimated_wait_seconds: float = 0.0
    estimated_runtime_seconds: float = 0.0
    
    # Job characteristics
    gpu_request: int = 0
    cpu_request_m: int = 1000
    memory_request_mb: int = 4096


class PolicyGenerator:
    """
    Generate scheduling policies from VGAC predictions.
    
    Maps calibrated probabilities to concrete scheduling actions,
    respecting confidence intervals and operational constraints.
    """
    
    # Thresholds for policy decisions
    LONG_WAIT_HIGH_THRESHOLD = 0.7
    LONG_WAIT_MEDIUM_THRESHOLD = 0.4
    FRAGMENTATION_HIGH_THRESHOLD = 0.5
    FRAGMENTATION_MEDIUM_THRESHOLD = 0.3
    UNDER_UTIL_THRESHOLD = 0.4
    PHASE_IMBALANCE_THRESHOLD = 0.6
    CONFIDENCE_THRESHOLD = 0.6
    
    def __init__(
        self,
        enable_aggressive_policies: bool = False,
        respect_confidence: bool = True,
    ):
        self.enable_aggressive_policies = enable_aggressive_policies
        self.respect_confidence = respect_confidence
    
    def generate(self, prediction: PredictionInput) -> PolicyResult:
        """
        Generate scheduling policy from prediction.
        
        Decision tree:
        1. Check confidence - if too low, use defaults
        2. Check long wait probability - adjust priority
        3. Check fragmentation risk - adjust memory policy
        4. Check under-utilization - adjust colocation
        5. Combine into final policy
        """
        warnings = []
        
        # Low confidence → use defaults
        if self.respect_confidence and prediction.confidence < self.CONFIDENCE_THRESHOLD:
            logger.debug(f"Low confidence ({prediction.confidence:.2f}), using defaults")
            return PolicyResult(
                policy=SchedulingPolicy(reason="Low confidence, using defaults"),
                use_default=True,
                annotations={"vgac.io/policy-reason": "low-confidence"},
                warnings=["Prediction confidence below threshold"],
            )
        
        # Start with base policy
        policy = SchedulingPolicy()
        reasons = []
        
        # =====================================================================
        # PRIORITY DECISION
        # =====================================================================
        if prediction.probability_long_wait > self.LONG_WAIT_HIGH_THRESHOLD:
            # High wait probability → boost priority
            policy.priority = PriorityClass.HIGH
            policy.priority_boost = 50
            policy.preemption = PreemptionPolicy.RELUCTANT
            reasons.append(f"P(long_wait)={prediction.probability_long_wait:.2f}>0.7")
            
        elif prediction.probability_long_wait > self.LONG_WAIT_MEDIUM_THRESHOLD:
            # Medium wait probability → slight boost
            policy.priority = PriorityClass.NORMAL
            policy.priority_boost = 20
            reasons.append(f"P(long_wait)={prediction.probability_long_wait:.2f}>0.4")
            
        else:
            # Low wait probability → can be deprioritized if needed
            policy.priority = PriorityClass.NORMAL
            policy.preemption = PreemptionPolicy.NORMAL
        
        # =====================================================================
        # MEMORY POLICY DECISION
        # =====================================================================
        if prediction.probability_fragmentation > self.FRAGMENTATION_HIGH_THRESHOLD:
            # High fragmentation risk → conservative memory
            policy.memory = MemoryPolicy.CONSERVATIVE
            if prediction.memory_request_mb > 0:
                policy.memory_limit_mb = int(prediction.memory_request_mb * 1.1)
            reasons.append(f"P(frag)={prediction.probability_fragmentation:.2f}>0.5")
            
        elif prediction.probability_fragmentation > self.FRAGMENTATION_MEDIUM_THRESHOLD:
            # Medium fragmentation risk → balanced with monitoring
            policy.memory = MemoryPolicy.BALANCED
            reasons.append(f"P(frag)={prediction.probability_fragmentation:.2f}>0.3")
            
        else:
            # Low fragmentation risk → can prefetch aggressively
            policy.memory = MemoryPolicy.AGGRESSIVE
        
        # =====================================================================
        # COLOCATION DECISION
        # =====================================================================
        if prediction.probability_under_utilization > self.UNDER_UTIL_THRESHOLD:
            # Likely to under-utilize → pack with other jobs
            policy.colocation = ColocationPolicy.PACKING
            reasons.append(f"P(under_util)={prediction.probability_under_utilization:.2f}>0.4")
            
        elif prediction.gpu_request > 4:
            # Large GPU job → isolate
            policy.colocation = ColocationPolicy.ISOLATED
            reasons.append(f"Large GPU job ({prediction.gpu_request} GPUs)")
            
        else:
            policy.colocation = ColocationPolicy.TOLERANT
        
        # =====================================================================
        # TIME SLICE DECISION
        # =====================================================================
        if prediction.gpu_request == 0:
            # CPU-only job → longer time slices
            policy.time_slice_ms = 200
        elif prediction.estimated_runtime_seconds > 3600:
            # Long job → standard slices
            policy.time_slice_ms = 100
        else:
            # Short job → shorter slices for responsiveness
            policy.time_slice_ms = 50
        
        # =====================================================================
        # FINALIZE
        # =====================================================================
        policy.confidence = prediction.confidence
        policy.reason = "; ".join(reasons) if reasons else "Default policy"
        
        # Build annotations for Kubernetes
        annotations = {
            "vgac.io/policy-version": "v1",
            "vgac.io/prediction-confidence": f"{prediction.confidence:.3f}",
            "vgac.io/probability-long-wait": f"{prediction.probability_long_wait:.3f}",
            "vgac.io/policy-reason": policy.reason[:200],
        }
        
        # Build labels
        labels = {
            "vgac.io/priority": policy.priority.value,
            "vgac.io/colocation": policy.colocation.value,
        }
        
        return PolicyResult(
            policy=policy,
            use_default=False,
            annotations=annotations,
            labels=labels,
            warnings=warnings,
        )
    
    def generate_from_dict(self, prediction_dict: Dict[str, Any]) -> PolicyResult:
        """Generate policy from prediction dictionary."""
        prediction = PredictionInput(
            probability_long_wait=prediction_dict.get("probability_long_wait", 0.5),
            probability_fragmentation=prediction_dict.get("probability_fragmentation", 0.0),
            probability_under_utilization=prediction_dict.get("probability_under_utilization", 0.0),
            probability_phase_imbalance=prediction_dict.get("probability_phase_imbalance", 0.0),
            confidence=prediction_dict.get("confidence", 1.0),
            estimated_wait_seconds=prediction_dict.get("estimated_wait_seconds", 0.0),
            estimated_runtime_seconds=prediction_dict.get("estimated_runtime_seconds", 0.0),
            gpu_request=prediction_dict.get("gpu_request", 0),
            cpu_request_m=prediction_dict.get("cpu_request_m", 1000),
            memory_request_mb=prediction_dict.get("memory_request_mb", 4096),
        )
        return self.generate(prediction)

