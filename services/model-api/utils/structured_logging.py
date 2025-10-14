"""
Structured Logging Utility for Model API Service
Provides consistent, structured logging across the service
"""

import structlog
import logging
import sys
import json
from typing import Any, Dict, Optional
from datetime import datetime

def setup_structured_logging(service_name: str, log_level: str = "INFO") -> structlog.BoundLogger:
    """
    Configure structured logging for the service
    
    Args:
        service_name: Name of the service (e.g., "model-api")
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured structured logger
    """
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )
    
    # Create service-specific logger
    logger = structlog.get_logger(service_name)
    
    # Add service context
    logger = logger.bind(service=service_name, version="1.0.0")
    
    return logger

def log_ml_operation(
    logger: structlog.BoundLogger,
    operation: str,
    model_name: str,
    status: str = "started",
    **kwargs
) -> None:
    """
    Log ML-specific operations with structured data
    
    Args:
        logger: Structured logger instance
        operation: Operation being performed (predict, train, load, etc.)
        model_name: Name of the model
        status: Operation status (started, completed, failed)
        **kwargs: Additional context data
    """
    logger.info(
        "ml_operation",
        operation=operation,
        model_name=model_name,
        status=status,
        timestamp=datetime.now().isoformat(),
        **kwargs
    )

def log_prediction_request(
    logger: structlog.BoundLogger,
    request_id: str,
    text_length: int,
    models: list,
    ensemble: bool = False,
    **kwargs
) -> None:
    """
    Log prediction request with structured data
    
    Args:
        logger: Structured logger instance
        request_id: Unique request identifier
        text_length: Length of input text
        models: List of models used
        ensemble: Whether ensemble prediction was used
        **kwargs: Additional context data
    """
    logger.info(
        "prediction_request",
        request_id=request_id,
        text_length=text_length,
        models=models,
        ensemble=ensemble,
        timestamp=datetime.now().isoformat(),
        **kwargs
    )

def log_prediction_result(
    logger: structlog.BoundLogger,
    request_id: str,
    prediction: str,
    confidence: float,
    processing_time_ms: float,
    model_name: str,
    **kwargs
) -> None:
    """
    Log prediction result with structured data
    
    Args:
        logger: Structured logger instance
        request_id: Unique request identifier
        prediction: Model prediction
        confidence: Prediction confidence
        processing_time_ms: Processing time in milliseconds
        model_name: Name of the model used
        **kwargs: Additional context data
    """
    logger.info(
        "prediction_result",
        request_id=request_id,
        prediction=prediction,
        confidence=confidence,
        processing_time_ms=processing_time_ms,
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
        **kwargs
    )

def log_model_lifecycle(
    logger: structlog.BoundLogger,
    event: str,
    model_name: str,
    version: str = None,
    **kwargs
) -> None:
    """
    Log model lifecycle events
    
    Args:
        logger: Structured logger instance
        event: Lifecycle event (loaded, unloaded, trained, deployed, etc.)
        model_name: Name of the model
        version: Model version
        **kwargs: Additional context data
    """
    logger.info(
        "model_lifecycle",
        event=event,
        model_name=model_name,
        version=version,
        timestamp=datetime.now().isoformat(),
        **kwargs
    )

def log_error_with_context(
    logger: structlog.BoundLogger,
    error: Exception,
    operation: str,
    **kwargs
) -> None:
    """
    Log errors with structured context
    
    Args:
        logger: Structured logger instance
        error: Exception that occurred
        operation: Operation that failed
        **kwargs: Additional context data
    """
    logger.error(
        "operation_error",
        operation=operation,
        error_type=type(error).__name__,
        error_message=str(error),
        timestamp=datetime.now().isoformat(),
        **kwargs
    )

def log_performance_metric(
    logger: structlog.BoundLogger,
    metric_name: str,
    value: float,
    unit: str = "ms",
    **kwargs
) -> None:
    """
    Log performance metrics
    
    Args:
        logger: Structured logger instance
        metric_name: Name of the metric
        value: Metric value
        unit: Unit of measurement
        **kwargs: Additional context data
    """
    logger.info(
        "performance_metric",
        metric_name=metric_name,
        value=value,
        unit=unit,
        timestamp=datetime.now().isoformat(),
        **kwargs
    )

def log_cache_operation(
    logger: structlog.BoundLogger,
    operation: str,
    key: str,
    hit: bool = None,
    ttl: int = None,
    **kwargs
) -> None:
    """
    Log cache operations
    
    Args:
        logger: Structured logger instance
        operation: Cache operation (get, set, delete, invalidate)
        key: Cache key
        hit: Whether it was a cache hit (for get operations)
        ttl: Time to live in seconds
        **kwargs: Additional context data
    """
    logger.info(
        "cache_operation",
        operation=operation,
        key=key,
        hit=hit,
        ttl=ttl,
        timestamp=datetime.now().isoformat(),
        **kwargs
    )

def log_health_check(
    logger: structlog.BoundLogger,
    check_name: str,
    status: str,
    latency_ms: float = None,
    error: str = None,
    **kwargs
) -> None:
    """
    Log health check results
    
    Args:
        logger: Structured logger instance
        check_name: Name of the health check
        status: Check status (healthy, unhealthy, degraded)
        latency_ms: Check latency in milliseconds
        error: Error message if check failed
        **kwargs: Additional context data
    """
    logger.info(
        "health_check",
        check_name=check_name,
        status=status,
        latency_ms=latency_ms,
        error=error,
        timestamp=datetime.now().isoformat(),
        **kwargs
    )

# Convenience function to get logger for Model API
def get_model_api_logger() -> structlog.BoundLogger:
    """Get structured logger for Model API service"""
    return setup_structured_logging("model-api", "INFO")
