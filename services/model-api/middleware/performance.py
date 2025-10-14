"""
Performance Monitoring Middleware for Model API Service
Enforces performance budgets and SLA monitoring
"""

import time
import logging
from prometheus_client import Histogram, Counter
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Performance budget metrics
REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint', 'status'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

# SLA thresholds (in milliseconds)
PERFORMANCE_BUDGETS = {
    "/predict": {"p95_ms": 100, "p99_ms": 500},
    "/predict/batch": {"p95_ms": 200, "p99_ms": 1000},
    "/predict/trained": {"p95_ms": 150, "p99_ms": 750},
    "/models/load": {"p95_ms": 5000, "p99_ms": 10000},
    "/models/unload": {"p95_ms": 1000, "p99_ms": 2000},
    "/health": {"p95_ms": 50, "p99_ms": 100},
    "/health/deep": {"p95_ms": 200, "p99_ms": 500},
    "/metrics": {"p95_ms": 100, "p99_ms": 200}
}

class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Enforces performance budgets and SLA monitoring"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        logger.info("✅ PerformanceMonitoringMiddleware initialized")

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_seconds = time.time() - start_time
        duration_ms = duration_seconds * 1000
        
        # Extract endpoint path (remove query params)
        endpoint = request.url.path
        
        # Record metrics
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code
        ).observe(duration_seconds)
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code
        ).inc()
        
        # Check performance budget
        budget = PERFORMANCE_BUDGETS.get(endpoint)
        if budget:
            self._check_performance_budget(
                endpoint=endpoint,
                duration_ms=duration_ms,
                budget=budget,
                method=request.method
            )
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        response.headers["X-Request-ID"] = getattr(request.state, 'request_id', 'unknown')
        
        # Log performance metrics
        logger.info(f"Request completed: {request.method} {endpoint} - Status: {response.status_code} - Duration: {duration_ms:.2f}ms")
        
        return response
    
    def _check_performance_budget(
        self, 
        endpoint: str, 
        duration_ms: float, 
        budget: dict, 
        method: str
    ):
        """Check if request violates performance budget"""
        
        # Check P95 threshold
        if duration_ms > budget["p95_ms"]:
            logger.warning(f"SLA violation P95: {endpoint} {method} - {duration_ms:.2f}ms > {budget['p95_ms']}ms (ratio: {duration_ms / budget['p95_ms']:.2f})")
        
        # Check P99 threshold
        if duration_ms > budget["p99_ms"]:
            logger.error(f"SLA violation P99: {endpoint} {method} - {duration_ms:.2f}ms > {budget['p99_ms']}ms (ratio: {duration_ms / budget['p99_ms']:.2f})")

def get_performance_metrics() -> dict:
    """Get current performance metrics summary"""
    return {
        "budgets": PERFORMANCE_BUDGETS,
        "total_requests": REQUEST_COUNT._value.sum(),
        "avg_duration": REQUEST_DURATION._sum._value / max(REQUEST_DURATION._count._value, 1)
    }