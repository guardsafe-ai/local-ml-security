"""
Distributed Tracing Middleware for Model API Service
Properly propagates trace context across services
"""

from opentelemetry.propagate import inject, extract
from opentelemetry import trace
from opentelemetry.context import attach, detach
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp
import logging

logger = logging.getLogger(__name__)

class TracingMiddleware(BaseHTTPMiddleware):
    """Properly propagate trace context across services"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.tracer = trace.get_tracer(__name__)
        logger.info("✅ TracingMiddleware initialized")

    async def dispatch(self, request: Request, call_next):
        # Extract trace context from incoming request
        ctx = extract(request.headers)
        
        # Set as current context
        token = attach(ctx)
        
        try:
            # Create span for this request
            with self.tracer.start_as_current_span(
                f"{request.method} {request.url.path}",
                attributes={
                    "http.method": request.method,
                    "http.url": str(request.url),
                    "http.route": request.url.path,
                    "service.name": "model-api"
                }
            ) as span:
                # Process request with trace context
                response = await call_next(request)
                
                # Add response attributes to span
                span.set_attributes({
                    "http.status_code": response.status_code,
                    "http.response_size": len(response.body) if hasattr(response, 'body') else 0
                })
                
                return response
        finally:
            detach(token)

def inject_trace_context(headers: dict) -> dict:
    """
    Inject current trace context into outgoing request headers
    
    Args:
        headers: Dictionary of headers to add trace context to
        
    Returns:
        Updated headers with trace context
    """
    inject(headers)
    return headers

def create_span(name: str, attributes: dict = None):
    """
    Create a new span for tracing
    
    Args:
        name: Name of the span
        attributes: Dictionary of span attributes
        
    Returns:
        Span context manager
    """
    tracer = trace.get_tracer(__name__)
    return tracer.start_as_current_span(name, attributes=attributes or {})
