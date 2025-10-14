"""
Training Service - Query Performance Monitoring
Tracks database query performance, timeouts, and slow queries
"""

import asyncio
import time
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict, Any, Callable, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class QueryMonitor:
    """Query performance monitor for training service"""
    
    def __init__(self, service_name: str, slow_query_threshold_ms: int = 1000, timeout_threshold_ms: int = 5000):
        self.service_name = service_name
        self.metrics = defaultdict(lambda: {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_duration_ms": 0.0,
            "max_duration_ms": 0.0,
            "min_duration_ms": float('inf'),
            "timeout_violations": 0,
            "slow_queries": 0,
            "last_call_timestamp": None,
            "last_error": None
        })
        self.slow_query_threshold_ms = slow_query_threshold_ms
        self.timeout_threshold_ms = timeout_threshold_ms
        self.lock = asyncio.Lock()
    
    async def record_query(self, query_name: str, duration_ms: float, success: bool, error: Optional[str] = None):
        """Record query performance metrics"""
        async with self.lock:
            metrics = self.metrics[query_name]
            metrics["total_calls"] += 1
            metrics["total_duration_ms"] += duration_ms
            metrics["max_duration_ms"] = max(metrics["max_duration_ms"], duration_ms)
            metrics["min_duration_ms"] = min(metrics["min_duration_ms"], duration_ms)
            metrics["last_call_timestamp"] = time.time()
            
            if success:
                metrics["successful_calls"] += 1
            else:
                metrics["failed_calls"] += 1
                metrics["last_error"] = error
            
            if duration_ms > self.timeout_threshold_ms:
                metrics["timeout_violations"] += 1
                logger.warning(f"🚨 [{self.service_name.upper()}] Query '{query_name}' timed out! Duration: {duration_ms:.2f}ms")
            elif duration_ms > self.slow_query_threshold_ms:
                metrics["slow_queries"] += 1
                logger.warning(f"🐢 [{self.service_name.upper()}] Slow query detected: '{query_name}'. Duration: {duration_ms:.2f}ms")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all queries"""
        summary = {}
        for query_name, metrics in self.metrics.items():
            avg_duration = metrics["total_duration_ms"] / metrics["total_calls"] if metrics["total_calls"] > 0 else 0
            success_rate = metrics["successful_calls"] / metrics["total_calls"] if metrics["total_calls"] > 0 else 0
            
            summary[query_name] = {
                "total_calls": metrics["total_calls"],
                "successful_calls": metrics["successful_calls"],
                "failed_calls": metrics["failed_calls"],
                "avg_duration_ms": f"{avg_duration:.2f}",
                "max_duration_ms": f"{metrics['max_duration_ms']:.2f}",
                "min_duration_ms": f"{metrics['min_duration_ms']:.2f}",
                "success_rate": f"{success_rate:.2%}",
                "timeout_violations": metrics["timeout_violations"],
                "slow_queries": metrics["slow_queries"],
                "last_call_timestamp": datetime.fromtimestamp(metrics["last_call_timestamp"]).isoformat() if metrics["last_call_timestamp"] else None,
                "last_error": metrics["last_error"]
            }
        
        return summary
    
    def clear_metrics(self):
        """Clear all metrics"""
        asyncio.create_task(self._clear_metrics_async())
    
    async def _clear_metrics_async(self):
        """Clear metrics asynchronously"""
        async with self.lock:
            self.metrics.clear()
            logger.info(f"📊 [{self.service_name.upper()}] Query performance metrics cleared.")

# Global query monitors
_query_monitors: Dict[str, QueryMonitor] = {}
_monitor_lock = asyncio.Lock()

async def get_training_query_monitor() -> QueryMonitor:
    """Get or create training service query monitor"""
    async with _monitor_lock:
        if "training" not in _query_monitors:
            _query_monitors["training"] = QueryMonitor("training")
        return _query_monitors["training"]

@asynccontextmanager
async def monitor_query(query_name: str, func: Callable, *args, **kwargs):
    """Monitor a database query operation"""
    monitor = await get_training_query_monitor()
    start_time = time.perf_counter()
    success = False
    result = None
    error_message = None
    
    try:
        result = await func(*args, **kwargs)
        success = True
        yield result
    except Exception as e:
        error_message = str(e)
        logger.error(f"❌ [{monitor.service_name.upper()}] Query '{query_name}' failed: {error_message}")
        raise
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        await monitor.record_query(query_name, duration_ms, success, error_message)

async def log_training_query_performance():
    """Log current query performance summary"""
    monitor = await get_training_query_monitor()
    summary = monitor.get_performance_summary()
    
    if summary:
        logger.info(f"📊 [{monitor.service_name.upper()}] Current Query Performance Summary:")
        for query_name, metrics in summary.items():
            logger.info(f"  Query: {query_name}")
            for key, value in metrics.items():
                logger.info(f"    {key}: {value}")
    else:
        logger.info(f"📊 [{monitor.service_name.upper()}] No query performance data available.")
