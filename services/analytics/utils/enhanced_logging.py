"""
Analytics Service - Enhanced Error Logging
Service-specific enhanced logging with context information
"""

import logging
import traceback
import uuid
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
import sys
import os

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Log levels for enhanced logging"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ErrorCategory(Enum):
    """Error categories for better classification"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NETWORK = "network"
    DATABASE = "database"
    EXTERNAL_API = "external_api"
    ML_MODEL = "ml_model"
    DATA_PROCESSING = "data_processing"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    ANALYTICS = "analytics"
    DRIFT_DETECTION = "drift_detection"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """Comprehensive error context information"""
    # Request information
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Service information
    service_name: Optional[str] = None
    service_version: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    
    # Error information
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_category: Optional[ErrorCategory] = None
    stack_trace: Optional[str] = None
    
    # Analytics-specific information
    model_name: Optional[str] = None
    analysis_type: Optional[str] = None  # drift, performance, baseline
    dataset_size: Optional[int] = None
    drift_score: Optional[float] = None
    confidence_score: Optional[float] = None
    
    # System information
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    disk_usage_percent: Optional[float] = None
    
    # Additional context
    additional_data: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None

class AnalyticsEnhancedLogger:
    """Enhanced logger for Analytics service with context-aware error logging"""
    
    def __init__(self, service_version: str = "1.0.0"):
        self.service_name = "analytics"
        self.service_version = service_version
        self.logger = logging.getLogger(f"analytics_enhanced")
        
        # Set up structured logging
        self._setup_structured_logging()
    
    def _setup_structured_logging(self):
        """Set up structured logging with JSON formatting"""
        # Create a custom formatter for structured logging
        class StructuredFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "service": "analytics",
                    "version": self.service_version,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                    "thread": record.thread,
                    "process": record.process
                }
                
                # Add context if available
                if hasattr(record, 'context') and record.context:
                    log_entry["context"] = record.context
                
                # Add exception info if available
                if record.exc_info:
                    log_entry["exception"] = {
                        "type": record.exc_info[0].__name__,
                        "message": str(record.exc_info[1]),
                        "traceback": traceback.format_exception(*record.exc_info)
                    }
                
                return json.dumps(log_entry, default=str)
        
        # Set up handler with structured formatter
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""
        try:
            # Try to get system info, fallback to basic info if psutil not available
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                
                return {
                    "memory_usage_mb": memory_info.rss / 1024 / 1024,
                    "cpu_usage_percent": process.cpu_percent(),
                    "disk_usage_percent": psutil.disk_usage('/').percent,
                    "pid": process.pid,
                    "thread_count": process.num_threads()
                }
            except ImportError:
                # Fallback without psutil
                return {
                    "pid": os.getpid(),
                    "memory_usage_mb": None,
                    "cpu_usage_percent": None,
                    "disk_usage_percent": None
                }
        except Exception:
            return {}
    
    def _categorize_error(self, exception: Exception) -> ErrorCategory:
        """Categorize error based on exception type"""
        error_type = type(exception).__name__
        
        if "ValidationError" in error_type or "ValueError" in error_type:
            return ErrorCategory.VALIDATION
        elif "AuthenticationError" in error_type or "Unauthorized" in error_type:
            return ErrorCategory.AUTHENTICATION
        elif "PermissionError" in error_type or "Forbidden" in error_type:
            return ErrorCategory.AUTHORIZATION
        elif "ConnectionError" in error_type or "TimeoutError" in error_type:
            return ErrorCategory.NETWORK
        elif "DatabaseError" in error_type or "SQLAlchemyError" in error_type:
            return ErrorCategory.DATABASE
        elif "HTTPError" in error_type or "RequestException" in error_type:
            return ErrorCategory.EXTERNAL_API
        elif "ModelError" in error_type or "PredictionError" in error_type:
            return ErrorCategory.ML_MODEL
        elif "DataError" in error_type or "ProcessingError" in error_type:
            return ErrorCategory.DATA_PROCESSING
        elif "ConfigurationError" in error_type or "ConfigError" in error_type:
            return ErrorCategory.CONFIGURATION
        elif "MemoryError" in error_type or "ResourceError" in error_type:
            return ErrorCategory.RESOURCE
        elif "DriftError" in error_type or "AnalyticsError" in error_type:
            return ErrorCategory.ANALYTICS
        else:
            return ErrorCategory.UNKNOWN
    
    def log_error_with_context(
        self,
        error: Exception,
        operation: str,
        context: Optional[ErrorContext] = None,
        level: LogLevel = LogLevel.ERROR,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """Log error with comprehensive context"""
        try:
            # Create context if not provided
            if context is None:
                context = ErrorContext()
            
            # Fill in basic information
            context.service_name = self.service_name
            context.service_version = self.service_version
            context.error_type = type(error).__name__
            context.error_message = str(error)
            context.error_category = self._categorize_error(error)
            context.stack_trace = traceback.format_exc()
            context.timestamp = datetime.now().isoformat()
            
            # Add system information
            system_info = self._get_system_info()
            context.memory_usage_mb = system_info.get("memory_usage_mb")
            context.cpu_usage_percent = system_info.get("cpu_usage_percent")
            context.disk_usage_percent = system_info.get("disk_usage_percent")
            
            # Add additional data
            if additional_data:
                context.additional_data = additional_data
            
            # Create log record with context
            log_record = self.logger.makeRecord(
                name=self.logger.name,
                level=getattr(logging, level.value),
                fn="",
                lno=0,
                msg=f"Analytics Error in {operation}: {str(error)}",
                args=(),
                exc_info=sys.exc_info()
            )
            
            # Add context to the record
            log_record.context = asdict(context)
            
            # Log the record
            self.logger.handle(log_record)
            
        except Exception as e:
            # Fallback to basic logging if enhanced logging fails
            self.logger.error(f"Failed to log error with context: {e}")
            self.logger.error(f"Original error: {error}")
    
    def log_analytics_event(
        self,
        event_type: str,
        model_name: Optional[str] = None,
        analysis_type: Optional[str] = None,
        dataset_size: Optional[int] = None,
        drift_score: Optional[float] = None,
        confidence_score: Optional[float] = None,
        request_id: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """Log analytics-specific events with context"""
        context = ErrorContext(
            request_id=request_id,
            model_name=model_name,
            analysis_type=analysis_type,
            dataset_size=dataset_size,
            drift_score=drift_score,
            confidence_score=confidence_score,
            additional_data={
                **(additional_data or {}),
                "event_type": event_type
            }
        )
        
        log_record = self.logger.makeRecord(
            name=self.logger.name,
            level=logging.INFO,
            fn="",
            lno=0,
            msg=f"Analytics event: {event_type} for model {model_name or 'unknown'}",
            args=(),
            exc_info=None
        )
        
        log_record.context = asdict(context)
        self.logger.handle(log_record)

# Global enhanced logger instance for Analytics service
_analytics_logger = None

def get_analytics_logger(service_version: str = "1.0.0") -> AnalyticsEnhancedLogger:
    """Get or create the Analytics enhanced logger"""
    global _analytics_logger
    if _analytics_logger is None:
        _analytics_logger = AnalyticsEnhancedLogger(service_version)
    return _analytics_logger

# Convenience functions for common logging patterns
def log_analytics_error(
    error: Exception,
    operation: str,
    request_id: Optional[str] = None,
    model_name: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None
):
    """Convenience function for logging Analytics errors with context"""
    logger = get_analytics_logger()
    
    context = ErrorContext(
        request_id=request_id,
        model_name=model_name,
        additional_data=additional_context
    )
    
    logger.log_error_with_context(error, operation, context)

def log_analytics_event(
    event_type: str,
    model_name: Optional[str] = None,
    analysis_type: Optional[str] = None,
    dataset_size: Optional[int] = None,
    drift_score: Optional[float] = None,
    confidence_score: Optional[float] = None,
    request_id: Optional[str] = None,
    additional_data: Optional[Dict[str, Any]] = None
):
    """Convenience function for logging Analytics events"""
    logger = get_analytics_logger()
    logger.log_analytics_event(
        event_type=event_type,
        model_name=model_name,
        analysis_type=analysis_type,
        dataset_size=dataset_size,
        drift_score=drift_score,
        confidence_score=confidence_score,
        request_id=request_id,
        additional_data=additional_data
    )