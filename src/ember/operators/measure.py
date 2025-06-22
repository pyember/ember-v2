"""Built-in measurement and metrics for operators.

Following Larry Page's principle: measure everything, then iterate based on data.
"""

import functools
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar

__all__ = [
    'measure',
    'OperatorMetrics',
    'MetricsReport',
]

T = TypeVar('T')


@dataclass
class CallMetrics:
    """Metrics for a single function call."""
    
    function_name: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    error: Optional[str] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    cache_hit: bool = False
    
    @property
    def timestamp(self) -> datetime:
        """Get timestamp of call."""
        return datetime.fromtimestamp(self.start_time)


@dataclass
class FunctionMetrics:
    """Aggregated metrics for a function."""
    
    name: str
    call_count: int = 0
    error_count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    cache_hits: int = 0
    last_called: Optional[float] = None
    
    @property
    def avg_time_ms(self) -> float:
        """Average execution time."""
        return self.total_time_ms / self.call_count if self.call_count > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        """Error rate as percentage."""
        return (self.error_count / self.call_count * 100) if self.call_count > 0 else 0.0
    
    @property 
    def cache_hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        return (self.cache_hits / self.call_count * 100) if self.call_count > 0 else 0.0
    
    def update(self, call: CallMetrics) -> None:
        """Update metrics with a new call."""
        self.call_count += 1
        self.total_time_ms += call.duration_ms
        self.min_time_ms = min(self.min_time_ms, call.duration_ms)
        self.max_time_ms = max(self.max_time_ms, call.duration_ms)
        self.last_called = call.end_time
        
        if not call.success:
            self.error_count += 1
        if call.cache_hit:
            self.cache_hits += 1


@dataclass
class MetricsReport:
    """Comprehensive metrics report."""
    
    total_calls: int
    total_functions: int
    avg_latency_ms: float
    error_rate: float
    cache_hit_rate: float
    tier_distribution: Dict[str, float]
    top_functions: List[Dict[str, Any]]
    performance_insights: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class OperatorMetrics:
    """Global metrics collection for operators."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern for global metrics."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize metrics storage."""
        if self._initialized:
            return
            
        self.function_metrics: Dict[str, FunctionMetrics] = {}
        self.recent_calls: List[CallMetrics] = []
        self.tier_counts = defaultdict(int)
        self._initialized = True
        self._max_recent = 10000  # Keep last 10k calls
    
    def record_call(self, call: CallMetrics, tier: str = "simple") -> None:
        """Record a function call.
        
        Args:
            call: Call metrics to record
            tier: Which tier (simple, advanced, experimental)
        """
        # Update function metrics
        if call.function_name not in self.function_metrics:
            self.function_metrics[call.function_name] = FunctionMetrics(name=call.function_name)
        
        self.function_metrics[call.function_name].update(call)
        
        # Track recent calls
        self.recent_calls.append(call)
        if len(self.recent_calls) > self._max_recent:
            self.recent_calls = self.recent_calls[-self._max_recent:]
        
        # Track tier distribution
        self.tier_counts[tier] += 1
    
    @classmethod
    def report(cls) -> MetricsReport:
        """Generate comprehensive metrics report.
        
        Returns:
            MetricsReport with all collected metrics
        """
        metrics = cls()
        
        if not metrics.function_metrics:
            return MetricsReport(
                total_calls=0,
                total_functions=0,
                avg_latency_ms=0.0,
                error_rate=0.0,
                cache_hit_rate=0.0,
                tier_distribution={},
                top_functions=[],
                performance_insights=[]
            )
        
        # Calculate aggregate metrics
        total_calls = sum(fm.call_count for fm in metrics.function_metrics.values())
        total_errors = sum(fm.error_count for fm in metrics.function_metrics.values())
        total_cache_hits = sum(fm.cache_hits for fm in metrics.function_metrics.values())
        total_time = sum(fm.total_time_ms for fm in metrics.function_metrics.values())
        
        # Tier distribution
        tier_total = sum(metrics.tier_counts.values())
        tier_distribution = {
            tier: (count / tier_total * 100) if tier_total > 0 else 0.0
            for tier, count in metrics.tier_counts.items()
        }
        
        # Top functions by call count
        top_functions = sorted(
            metrics.function_metrics.values(),
            key=lambda fm: fm.call_count,
            reverse=True
        )[:10]
        
        top_functions_data = [
            {
                'name': fm.name,
                'calls': fm.call_count,
                'avg_ms': round(fm.avg_time_ms, 2),
                'error_rate': round(fm.error_rate, 2),
                'cache_hit_rate': round(fm.cache_hit_rate, 2),
            }
            for fm in top_functions
        ]
        
        # Performance insights
        insights = []
        
        # Find slow functions
        slow_functions = [
            fm for fm in metrics.function_metrics.values()
            if fm.avg_time_ms > 100 and fm.call_count > 10
        ]
        if slow_functions:
            slowest = max(slow_functions, key=lambda fm: fm.avg_time_ms)
            insights.append(
                f"Function '{slowest.name}' averages {slowest.avg_time_ms:.1f}ms - consider optimization"
            )
        
        # Find high error rate functions
        error_functions = [
            fm for fm in metrics.function_metrics.values()
            if fm.error_rate > 5 and fm.call_count > 10
        ]
        if error_functions:
            worst = max(error_functions, key=lambda fm: fm.error_rate)
            insights.append(
                f"Function '{worst.name}' has {worst.error_rate:.1f}% error rate - needs investigation"
            )
        
        # Check cache effectiveness
        avg_cache_hit_rate = (total_cache_hits / total_calls * 100) if total_calls > 0 else 0
        if avg_cache_hit_rate < 50 and total_calls > 1000:
            insights.append(
                f"Cache hit rate is only {avg_cache_hit_rate:.1f}% - consider cache tuning"
            )
        
        return MetricsReport(
            total_calls=total_calls,
            total_functions=len(metrics.function_metrics),
            avg_latency_ms=total_time / total_calls if total_calls > 0 else 0.0,
            error_rate=(total_errors / total_calls * 100) if total_calls > 0 else 0.0,
            cache_hit_rate=avg_cache_hit_rate,
            tier_distribution=tier_distribution,
            top_functions=top_functions_data,
            performance_insights=insights
        )
    
    @classmethod
    def reset(cls) -> None:
        """Reset all metrics (useful for testing)."""
        metrics = cls()
        metrics.function_metrics.clear()
        metrics.recent_calls.clear()
        metrics.tier_counts.clear()


def measure(func: Optional[Callable[..., T]] = None, *,
            tier: str = "simple",
            track_size: bool = False) -> Callable[..., T]:
    """Decorator to measure operator execution.
    
    Args:
        func: Function to measure
        tier: Tier classification (simple, advanced, experimental)
        track_size: Whether to track input/output sizes
        
    Returns:
        Measured version of function
    """
    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(f)
        def measured(*args, **kwargs) -> T:
            start_time = time.time()
            success = True
            error_msg = None
            input_size = None
            output_size = None
            cache_hit = False
            
            try:
                # Track input size if requested
                if track_size and args:
                    input_size = _estimate_size(args[0])
                
                # Check if result might be cached
                if hasattr(f, '_cache'):
                    cache_hit = True  # Simplified cache detection
                
                # Execute function
                result = f(*args, **kwargs)
                
                # Track output size if requested
                if track_size:
                    output_size = _estimate_size(result)
                
                return result
                
            except Exception as e:
                success = False
                error_msg = str(e)
                raise
                
            finally:
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                
                # Record metrics
                call_metrics = CallMetrics(
                    function_name=f.__name__,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=duration_ms,
                    success=success,
                    error=error_msg,
                    input_size=input_size,
                    output_size=output_size,
                    cache_hit=cache_hit
                )
                
                OperatorMetrics().record_call(call_metrics, tier=tier)
        
        # Add metrics access to function
        measured.metrics = lambda: OperatorMetrics().function_metrics.get(f.__name__)
        
        return measured
    
    if func is None:
        return decorator
    return decorator(func)


def _estimate_size(obj: Any) -> int:
    """Estimate size of object for metrics.
    
    Args:
        obj: Object to estimate
        
    Returns:
        Estimated size in bytes
    """
    if hasattr(obj, '__len__'):
        return len(obj)
    elif hasattr(obj, 'shape'):
        # NumPy-like arrays
        import numpy as np
        return np.prod(obj.shape)
    elif isinstance(obj, dict):
        return sum(_estimate_size(v) for v in obj.values())
    else:
        return 1  # Default size