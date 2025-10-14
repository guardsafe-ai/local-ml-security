"""
Structured Logging Utility for Training Service
Provides consistent, structured logging across the service
"""

import structlog
import logging
import sys
import json
from typing import Any, Dict, Optional
from datetime import datetime

def setup_structured_logging(service_name: str, log_level: str = "INFO") -> structlog.BoundLogger:
    """Configure structured logging for the service"""
    
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
    
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )
    
    logger = structlog.get_logger(service_name)
    logger = logger.bind(service=service_name, version="1.0.0")
    
    return logger

def log_training_job(
    logger: structlog.BoundLogger,
    job_id: str,
    model_name: str,
    status: str,
    **kwargs
) -> None:
    """Log training job events"""
    logger.info(
        "training_job",
        job_id=job_id,
        model_name=model_name,
        status=status,
        timestamp=datetime.now().isoformat(),
        **kwargs
    )

def log_model_metrics(
    logger: structlog.BoundLogger,
    model_name: str,
    version: str,
    metrics: Dict[str, float],
    **kwargs
) -> None:
    """Log model performance metrics"""
    logger.info(
        "model_metrics",
        model_name=model_name,
        version=version,
        metrics=metrics,
        timestamp=datetime.now().isoformat(),
        **kwargs
    )

def get_training_logger() -> structlog.BoundLogger:
    """Get structured logger for Training service"""
    return setup_structured_logging("training", "INFO")
