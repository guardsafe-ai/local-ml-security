"""
Distributed Tracing Setup for ML Security Service

This module provides OpenTelemetry tracing configuration for all services.
It automatically instruments FastAPI, requests, Redis, and PostgreSQL.
"""

import os
import logging
from opentelemetry import trace
try:
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
except ImportError:
    from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
try:
    from opentelemetry.instrumentation.redis import RedisInstrumentor
except ImportError:
    # Fallback for different package structure
    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor
    except ImportError:
        RedisInstrumentor = None
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_tracing(service_name: str, app=None):
    """
    Set up distributed tracing for a service.
    
    Args:
        service_name: Name of the service (e.g., 'model-api', 'red-team')
        app: FastAPI app instance (optional)
    """
    
    # Get Jaeger configuration from environment
    jaeger_host = os.getenv('JAEGER_AGENT_HOST', 'localhost')
    jaeger_port = int(os.getenv('JAEGER_AGENT_PORT', '14268'))
    
    logger.info(f"Setting up tracing for {service_name} with Jaeger at {jaeger_host}:{jaeger_port}")
    
    # Create resource with service name
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
        "deployment.environment": "development"
    })
    
    # Set up tracer provider
    trace.set_tracer_provider(TracerProvider(resource=resource))
    tracer = trace.get_tracer(__name__)
    
    # Set up Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name=jaeger_host,
        agent_port=jaeger_port,
    )
    
    # Add span processor
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Instrument libraries
    try:
        # Instrument FastAPI if app is provided
        if app:
            FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.get_tracer_provider())
            logger.info(f"FastAPI instrumented for {service_name}")
        
        # Instrument requests library
        RequestsInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())
        logger.info(f"Requests library instrumented for {service_name}")
        
        # Instrument Redis
        if RedisInstrumentor:
            RedisInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())
            logger.info(f"Redis instrumented for {service_name}")
        else:
            logger.warning(f"Redis instrumentation not available for {service_name}")
        
        # Instrument PostgreSQL
        Psycopg2Instrumentor().instrument(tracer_provider=trace.get_tracer_provider())
        logger.info(f"PostgreSQL instrumented for {service_name}")
        
    except Exception as e:
        logger.warning(f"Failed to instrument some libraries: {e}")
    
    logger.info(f"Tracing setup complete for {service_name}")
    return tracer

def get_tracer(service_name: str = None):
    """Get a tracer instance for the current service."""
    return trace.get_tracer(service_name or "ml-security")

def create_span(tracer, operation_name: str, **kwargs):
    """Create a new span for an operation."""
    return tracer.start_span(operation_name, **kwargs)

def add_span_attributes(span, **attributes):
    """Add attributes to a span."""
    for key, value in attributes.items():
        span.set_attribute(key, value)

def add_span_event(span, event_name: str, **attributes):
    """Add an event to a span."""
    span.add_event(event_name, attributes)

def set_span_status(span, status_code, description=None):
    """Set the status of a span."""
    from opentelemetry.trace import Status, StatusCode
    if status_code == "ok":
        span.set_status(Status(StatusCode.OK, description))
    elif status_code == "error":
        span.set_status(Status(StatusCode.ERROR, description))
    else:
        span.set_status(Status(StatusCode.UNSET, description))

# Example usage for request tracing
def trace_request(tracer, operation_name: str, request_data: dict = None):
    """Context manager for tracing requests."""
    from contextlib import contextmanager
    
    @contextmanager
    def _trace_request():
        with tracer.start_as_current_span(operation_name) as span:
            if request_data:
                add_span_attributes(span, **request_data)
            try:
                yield span
            except Exception as e:
                set_span_status(span, "error", str(e))
                raise
            else:
                set_span_status(span, "ok")
    
    return _trace_request()

# Example usage for database operations
def trace_db_operation(tracer, operation_name: str, query: str = None, **attributes):
    """Context manager for tracing database operations."""
    from contextlib import contextmanager
    
    @contextmanager
    def _trace_db_operation():
        with tracer.start_as_current_span(operation_name) as span:
            if query:
                add_span_attributes(span, **{"db.statement": query})
            add_span_attributes(span, **attributes)
            try:
                yield span
            except Exception as e:
                set_span_status(span, "error", str(e))
                raise
            else:
                set_span_status(span, "ok")
    
    return _trace_db_operation()

# Example usage for external API calls
def trace_external_call(tracer, operation_name: str, url: str = None, method: str = None, **attributes):
    """Context manager for tracing external API calls."""
    from contextlib import contextmanager
    
    @contextmanager
    def _trace_external_call():
        with tracer.start_as_current_span(operation_name) as span:
            if url:
                add_span_attributes(span, **{"http.url": url})
            if method:
                add_span_attributes(span, **{"http.method": method})
            add_span_attributes(span, **attributes)
            try:
                yield span
            except Exception as e:
                set_span_status(span, "error", str(e))
                raise
            else:
                set_span_status(span, "ok")
    
    return _trace_external_call()
