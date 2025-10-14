"""
Analytics Service - Query Performance Monitoring
Tracks database query performance and timeout violations for Analytics service
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class QueryMetrics:
    """Query performance metrics"""
    query_type: str
    duration_ms: float
    success: bool
    timeout_violation: bool
    error_message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class AnalyticsQueryMonitor:
    """Monitors database query performance and timeouts for Analytics service"""
    
    def __init__(self):
        self.service_name = "analytics"
        self.metrics: Dict[str, QueryMetrics] = {}
        self.timeout_threshold_ms = 5000  # 5 seconds
        self.slow_query_threshold_ms = 1000  # 1 second
        
    def record_query(self, query_type: str, duration_ms: float, success: bool, 
                    error_message: Optional[str] = None) -> QueryMetrics:
        """Record query performance metrics"""
        timeout_violation = duration_ms > self.timeout_threshold_ms
        
        metric = QueryMetrics(
            query_type=query_type,
            duration_ms=duration_ms,
            success=success,
            timeout_violation=timeout_violation,
            error_message=error_message
        )
        
        # Store metric with timestamp as key
        key = f"{query_type}_{int(time.time() * 1000)}"
        self.metrics[key] = metric
        
        # Log performance issues
        if timeout_violation:
            logger.warning(f"⚠️ [ANALYTICS] Query timeout violation: {query_type} took {duration_ms:.2f}ms")
        elif duration_ms > self.slow_query_threshold_ms:
            logger.info(f"🐌 [ANALYTICS] Slow query: {query_type} took {duration_ms:.2f}ms")
        
        return metric
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get query performance summary"""
        if not self.metrics:
            return {"total_queries": 0}
        
        recent_metrics = list(self.metrics.values())[-100:]  # Last 100 queries
        
        total_queries = len(recent_metrics)
        successful_queries = sum(1 for m in recent_metrics if m.success)
        timeout_violations = sum(1 for m in recent_metrics if m.timeout_violation)
        slow_queries = sum(1 for m in recent_metrics if m.duration_ms > self.slow_query_threshold_ms)
        
        avg_duration = sum(m.duration_ms for m in recent_metrics) / total_queries
        max_duration = max(m.duration_ms for m in recent_metrics)
        
        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "failed_queries": total_queries - successful_queries,
            "timeout_violations": timeout_violations,
            "slow_queries": slow_queries,
            "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
            "avg_duration_ms": round(avg_duration, 2),
            "max_duration_ms": round(max_duration, 2),
            "timeout_threshold_ms": self.timeout_threshold_ms,
            "slow_query_threshold_ms": self.slow_query_threshold_ms
        }
    
    def clear_metrics(self):
        """Clear stored metrics"""
        self.metrics.clear()
        logger.info(f"🧹 [ANALYTICS] Query metrics cleared")

# Global query monitor instance for analytics service
_analytics_query_monitor: Optional[AnalyticsQueryMonitor] = None

def get_analytics_query_monitor() -> AnalyticsQueryMonitor:
    """Get or create query monitor for analytics service"""
    global _analytics_query_monitor
    if _analytics_query_monitor is None:
        _analytics_query_monitor = AnalyticsQueryMonitor()
    return _analytics_query_monitor

@asynccontextmanager
async def monitor_query(query_type: str, operation: Callable, *args, **kwargs):
    """Context manager for monitoring database queries in analytics service"""
    monitor = get_analytics_query_monitor()
    start_time = time.time()
    success = False
    error_message = None
    
    try:
        result = await operation(*args, **kwargs)
        success = True
        yield result
    except Exception as e:
        error_message = str(e)
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        monitor.record_query(query_type, duration_ms, success, error_message)

def log_analytics_query_performance():
    """Log current query performance for analytics service"""
    monitor = get_analytics_query_monitor()
    summary = monitor.get_performance_summary()
    
    if summary["total_queries"] > 0:
        logger.info(f"📊 [ANALYTICS] Query Performance Summary:")
        logger.info(f"   Total queries: {summary['total_queries']}")
        logger.info(f"   Success rate: {summary['success_rate']:.2%}")
        logger.info(f"   Avg duration: {summary['avg_duration_ms']}ms")
        logger.info(f"   Max duration: {summary['max_duration_ms']}ms")
        logger.info(f"   Timeout violations: {summary['timeout_violations']}")
        logger.info(f"   Slow queries: {summary['slow_queries']}")
    else:
        logger.info(f"📊 [ANALYTICS] No query metrics available")
