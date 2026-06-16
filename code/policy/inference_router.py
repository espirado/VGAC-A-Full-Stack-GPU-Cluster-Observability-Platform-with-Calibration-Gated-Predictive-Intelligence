"""
Inference Router

Routes LLM inference requests to optimal workers based on
predicted phase characteristics (prefill vs decode).

Integrates with llm-d/vLLM disaggregated inference architecture.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class WorkerType(Enum):
    """Types of inference workers in disaggregated architecture."""
    PREFILL_OPTIMIZED = "prefill"      # Compute-optimized GPUs
    DECODE_OPTIMIZED = "decode"        # Memory-bandwidth-optimized GPUs
    COLOCATED = "colocated"            # Traditional single-worker
    HYBRID = "hybrid"                  # Can handle both phases


class CacheStrategy(Enum):
    """KV cache management strategy."""
    FULL_CACHE = "full_cache"          # Cache everything
    PREFIX_CACHE = "prefix_cache"      # Cache common prefixes
    NO_CACHE = "no_cache"              # Don't cache (streaming)
    PARTIAL = "partial"                # Cache based on prediction


@dataclass
class InferencePrediction:
    """Predictions for an inference request."""
    # Phase characteristics
    prefill_intensity: float           # 0-1, compute intensity
    decode_memory_bw_tb_s: float       # Required memory bandwidth
    
    # Timing predictions
    estimated_ttft_ms: float           # Time to first token
    estimated_tpot_ms: float           # Time per output token
    estimated_total_ms: float          # Total request time
    
    # Cache predictions
    kv_cache_hit_probability: float    # 0-1, prefix cache hit
    kv_cache_size_mb: float            # Estimated cache size
    
    # Risk scores
    fragmentation_risk: float          # 0-1
    phase_imbalance: float             # 0-1, benefit from disaggregation
    
    # Confidence
    confidence: float = 1.0


@dataclass
class RoutingDecision:
    """Routing decision for an inference request."""
    worker_type: WorkerType
    cache_strategy: CacheStrategy
    priority: int                      # 0-255
    
    # Expected outcomes
    expected_ttft_ms: float
    expected_throughput_tokens_s: float
    
    # Routing metadata
    reason: str
    confidence: float
    
    # Fallback
    fallback_worker_type: Optional[WorkerType] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_type": self.worker_type.value,
            "cache_strategy": self.cache_strategy.value,
            "priority": self.priority,
            "expected_ttft_ms": self.expected_ttft_ms,
            "expected_throughput_tokens_s": self.expected_throughput_tokens_s,
            "reason": self.reason,
            "confidence": self.confidence,
            "fallback_worker_type": (
                self.fallback_worker_type.value if self.fallback_worker_type else None
            ),
        }


@dataclass
class InferenceRequest:
    """Input for inference routing."""
    request_id: str
    model_name: str
    prompt_tokens: int
    max_new_tokens: int
    
    # Optional hints
    batch_size: int = 1
    priority: str = "normal"           # low, normal, high
    stream: bool = False
    prefix_id: Optional[str] = None    # For prefix caching
    
    # Model info (if known)
    model_size_gb: Optional[float] = None
    context_length: Optional[int] = None


class InferenceRouter:
    """
    Route LLM inference requests based on predicted phase characteristics.
    
    Decision logic:
    1. Predict phase intensity (prefill vs decode)
    2. Predict cache behavior (hit probability, size)
    3. Route to optimal worker type
    4. Set cache strategy
    """
    
    # Thresholds
    PREFILL_INTENSITY_THRESHOLD = 0.7
    DECODE_MEMORY_THRESHOLD = 1.0  # TB/s
    PHASE_IMBALANCE_THRESHOLD = 0.5
    CACHE_HIT_THRESHOLD = 0.3
    CONFIDENCE_THRESHOLD = 0.6
    
    # Constants for prediction
    BYTES_PER_TOKEN_KV = 2 * 128 * 2  # 2 (K+V) * hidden_dim * 2 bytes
    
    def __init__(
        self,
        enable_disaggregation: bool = True,
        enable_prefix_cache: bool = True,
        default_worker: WorkerType = WorkerType.COLOCATED,
    ):
        self.enable_disaggregation = enable_disaggregation
        self.enable_prefix_cache = enable_prefix_cache
        self.default_worker = default_worker
        
        self._routing_log: List[Dict[str, Any]] = []
    
    def predict(self, request: InferenceRequest) -> InferencePrediction:
        """
        Predict inference characteristics for a request.
        
        In production, this would use trained models. For now,
        we use heuristics based on token counts and model size.
        """
        # Heuristic: prefill intensity scales with prompt length
        # Short prompts = more decode-heavy, long prompts = more prefill-heavy
        prefill_ratio = request.prompt_tokens / (
            request.prompt_tokens + request.max_new_tokens
        )
        prefill_intensity = min(1.0, prefill_ratio * 1.5)
        
        # Memory bandwidth: scales with model size and output length
        model_size_gb = request.model_size_gb or 14  # Default ~7B model
        decode_tokens = request.max_new_tokens
        decode_memory_bw = (model_size_gb * 2) / 1000  # Simplified estimate
        
        # Timing estimates (very rough heuristics)
        # In production: use historical data + model-specific profiles
        estimated_ttft_ms = request.prompt_tokens * 0.05 + 20  # ~50ms per 1K tokens
        estimated_tpot_ms = 20 + model_size_gb * 0.5  # Depends on model size
        estimated_total_ms = estimated_ttft_ms + decode_tokens * estimated_tpot_ms
        
        # Cache predictions
        # If prefix_id provided, assume some cache hit potential
        kv_cache_hit_probability = 0.4 if request.prefix_id else 0.1
        kv_cache_size_mb = (
            request.prompt_tokens * self.BYTES_PER_TOKEN_KV / 1e6
        )
        
        # Phase imbalance: difference between prefill and decode requirements
        phase_imbalance = abs(prefill_intensity - 0.5) * 2
        
        # Fragmentation: higher for larger cache sizes
        fragmentation_risk = min(1.0, kv_cache_size_mb / 1000)
        
        return InferencePrediction(
            prefill_intensity=prefill_intensity,
            decode_memory_bw_tb_s=decode_memory_bw,
            estimated_ttft_ms=estimated_ttft_ms,
            estimated_tpot_ms=estimated_tpot_ms,
            estimated_total_ms=estimated_total_ms,
            kv_cache_hit_probability=kv_cache_hit_probability,
            kv_cache_size_mb=kv_cache_size_mb,
            fragmentation_risk=fragmentation_risk,
            phase_imbalance=phase_imbalance,
            confidence=0.7,  # Heuristic confidence
        )
    
    def route(self, request: InferenceRequest) -> RoutingDecision:
        """
        Route inference request to optimal worker.
        
        Decision tree:
        1. Predict phase characteristics
        2. If disaggregation enabled and beneficial → route to specialized worker
        3. Set cache strategy based on predictions
        4. Return routing decision with confidence
        """
        prediction = self.predict(request)
        
        # Log for validation
        self._routing_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request.request_id,
            "prediction": {
                "prefill_intensity": prediction.prefill_intensity,
                "decode_memory_bw": prediction.decode_memory_bw_tb_s,
                "phase_imbalance": prediction.phase_imbalance,
            },
        })
        
        # Low confidence → use default
        if prediction.confidence < self.CONFIDENCE_THRESHOLD:
            return RoutingDecision(
                worker_type=self.default_worker,
                cache_strategy=CacheStrategy.FULL_CACHE,
                priority=128,
                expected_ttft_ms=prediction.estimated_ttft_ms,
                expected_throughput_tokens_s=1000 / prediction.estimated_tpot_ms,
                reason="Low confidence, using default worker",
                confidence=prediction.confidence,
            )
        
        # Determine worker type
        reasons = []
        worker_type = self.default_worker
        
        if self.enable_disaggregation:
            if prediction.phase_imbalance > self.PHASE_IMBALANCE_THRESHOLD:
                # Significant phase imbalance → use disaggregation
                if prediction.prefill_intensity > self.PREFILL_INTENSITY_THRESHOLD:
                    worker_type = WorkerType.PREFILL_OPTIMIZED
                    reasons.append(f"Prefill-heavy (intensity={prediction.prefill_intensity:.2f})")
                elif prediction.decode_memory_bw_tb_s > self.DECODE_MEMORY_THRESHOLD:
                    worker_type = WorkerType.DECODE_OPTIMIZED
                    reasons.append(f"Decode-heavy (bw={prediction.decode_memory_bw_tb_s:.2f}TB/s)")
                else:
                    worker_type = WorkerType.HYBRID
                    reasons.append("Moderate imbalance, using hybrid")
            else:
                worker_type = WorkerType.COLOCATED
                reasons.append(f"Balanced phases (imbalance={prediction.phase_imbalance:.2f})")
        
        # Determine cache strategy
        if prediction.kv_cache_hit_probability > self.CACHE_HIT_THRESHOLD:
            cache_strategy = CacheStrategy.PREFIX_CACHE
            reasons.append(f"Prefix cache likely (p={prediction.kv_cache_hit_probability:.2f})")
        elif request.stream:
            cache_strategy = CacheStrategy.NO_CACHE
            reasons.append("Streaming mode, no cache")
        elif prediction.fragmentation_risk > 0.5:
            cache_strategy = CacheStrategy.PARTIAL
            reasons.append(f"High fragmentation risk ({prediction.fragmentation_risk:.2f})")
        else:
            cache_strategy = CacheStrategy.FULL_CACHE
        
        # Set priority
        priority_map = {"low": 64, "normal": 128, "high": 200}
        priority = priority_map.get(request.priority, 128)
        
        return RoutingDecision(
            worker_type=worker_type,
            cache_strategy=cache_strategy,
            priority=priority,
            expected_ttft_ms=prediction.estimated_ttft_ms,
            expected_throughput_tokens_s=1000 / prediction.estimated_tpot_ms,
            reason="; ".join(reasons),
            confidence=prediction.confidence,
            fallback_worker_type=WorkerType.COLOCATED,
        )
    
    def get_routing_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent routing decisions for validation."""
        return self._routing_log[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        if not self._routing_log:
            return {"total_routes": 0}
        
        worker_counts = {}
        for entry in self._routing_log:
            # Count would require storing decision, simplified here
            pass
        
        return {
            "total_routes": len(self._routing_log),
            "disaggregation_enabled": self.enable_disaggregation,
            "prefix_cache_enabled": self.enable_prefix_cache,
        }


# Singleton instance
_router: Optional[InferenceRouter] = None


def get_router() -> InferenceRouter:
    """Get or create inference router singleton."""
    global _router
    if _router is None:
        _router = InferenceRouter()
    return _router

