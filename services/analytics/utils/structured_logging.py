"""
Structured Logging Utility for Analytics Service
Provides consistent, structured logging across the service
"""

import structlog
import logging
import sys
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

def log_drift_detection(
    logger: structlog.BoundLogger,
    model_name: str,
    drift_score: float,
    drift_detected: bool,
    **kwargs
) -> None:
    """Log drift detection events"""
    logger.info(
        "drift_detection",
        model_name=model_name,
        drift_score=drift_score,
        drift_detected=drift_detected,
        timestamp=datetime.now().isoformat(),
        **kwargs
    )

def log_retrain_trigger(
    logger: structlog.BoundLogger,
    model_name: str,
    trigger_reason: str,
    job_id: str,
    **kwargs
) -> None:
    """Log retraining trigger events"""
    logger.info(
        "retrain_trigger",
        model_name=model_name,
        trigger_reason=trigger_reason,
        job_id=job_id,
        timestamp=datetime.now().isoformat(),
        **kwargs
    )

def get_analytics_logger() -> structlog.BoundLogger:
    """Get structured logger for Analytics service"""
    return setup_structured_logging("analytics", "INFO")
