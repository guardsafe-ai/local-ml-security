import os
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
# from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from contextlib import contextmanager

def setup_tracing(service_name: str, app=None):
    """Sets up OpenTelemetry tracing for the service."""
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    
    jaeger_host = os.getenv("JAEGER_AGENT_HOST", "localhost")
    jaeger_port = int(os.getenv("JAEGER_AGENT_PORT", 6831)) # Default UDP port for Jaeger agent

    jaeger_exporter = JaegerExporter(
        agent_host_name=jaeger_host,
        agent_port=jaeger_port,
    )
    span_processor = BatchSpanProcessor(jaeger_exporter)
    provider.add_span_processor(span_processor)
    trace.set_tracer_provider(provider)

    if app:
        FastAPIInstrumentor.instrument_app(app)
    RequestsInstrumentor().instrument()
    RedisInstrumentor().instrument()
    # Psycopg2Instrumentor().instrument()

    return trace.get_tracer(service_name)

def get_tracer(service_name: str):
    """Returns a tracer for the given service name."""
    return trace.get_tracer(service_name)

@contextmanager
def trace_request(tracer, span_name, **kwargs):
    """Context manager to create a span for a request."""
    with tracer.start_as_current_span(span_name) as span:
        for key, value in kwargs.items():
            span.set_attribute(key, value)
        yield span

def add_span_attributes(span, **kwargs):
    """Adds attributes to an existing span."""
    for key, value in kwargs.items():
        span.set_attribute(key, value)
