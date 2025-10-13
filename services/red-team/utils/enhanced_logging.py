"""
Enhanced Logging Utilities
Provides structured logging with full context and stack traces
"""

import logging
import traceback
import uuid
from typing import Any, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def log_error_with_context(
    error: Exception,
    operation: str,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    model_name: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None,
    level: int = logging.ERROR
) -> None:
    """
    Log error with full context and stack trace
    
    Args:
        error: The exception that occurred
        operation: Description of the operation that failed
        user_id: ID of the user making the request
        request_id: Unique request identifier
        model_name: Name of the model being used
        additional_context: Additional context data
        level: Logging level (default: ERROR)
    """
    context = {
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.utcnow().isoformat(),
        "stack_trace": traceback.format_exc()
    }
    
    if user_id:
        context["user_id"] = user_id
    if request_id:
        context["request_id"] = request_id
    if model_name:
        context["model_name"] = model_name
    if additional_context:
        context.update(additional_context)
    
    # Log with structured context
    logger.log(
        level,
        f"❌ [{operation}] {type(error).__name__}: {str(error)}",
        extra={"context": context},
        exc_info=True
    )


def log_warning_with_context(
    message: str,
    operation: str,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    model_name: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log warning with context
    
    Args:
        message: Warning message
        operation: Description of the operation
        user_id: ID of the user making the request
        request_id: Unique request identifier
        model_name: Name of the model being used
        additional_context: Additional context data
    """
    context = {
        "operation": operation,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if user_id:
        context["user_id"] = user_id
    if request_id:
        context["request_id"] = request_id
    if model_name:
        context["model_name"] = model_name
    if additional_context:
        context.update(additional_context)
    
    logger.warning(
        f"⚠️ [{operation}] {message}",
        extra={"context": context}
    )


def log_info_with_context(
    message: str,
    operation: str,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    model_name: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log info with context
    
    Args:
        message: Info message
        operation: Description of the operation
        user_id: ID of the user making the request
        request_id: Unique request identifier
        model_name: Name of the model being used
        additional_context: Additional context data
    """
    context = {
        "operation": operation,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if user_id:
        context["user_id"] = user_id
    if request_id:
        context["request_id"] = request_id
    if model_name:
        context["model_name"] = model_name
    if additional_context:
        context.update(additional_context)
    
    logger.info(
        f"ℹ️ [{operation}] {message}",
        extra={"context": context}
    )


def generate_request_id() -> str:
    """Generate a unique request ID"""
    return str(uuid.uuid4())


def extract_context_from_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract context information from request data
    
    Args:
        request_data: Request data dictionary
        
    Returns:
        Dictionary with extracted context
    """
    context = {}
    
    if "model_name" in request_data:
        context["model_name"] = request_data["model_name"]
    if "user_id" in request_data:
        context["user_id"] = request_data["user_id"]
    if "request_id" in request_data:
        context["request_id"] = request_data["request_id"]
    if "text" in request_data:
        context["text_length"] = len(str(request_data["text"]))
    if "confidence_threshold" in request_data:
        context["confidence_threshold"] = request_data["confidence_threshold"]
    
    return context
