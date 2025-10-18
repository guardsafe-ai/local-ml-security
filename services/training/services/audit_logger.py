"""
Audit Logger Service
Provides comprehensive audit logging for compliance and security
"""

import logging
import time
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Types of audit events"""
    MODEL_PREDICT = "model_predict"
    MODEL_LOAD = "model_load"
    MODEL_UNLOAD = "model_unload"
    DATA_ACCESS = "data_access"
    TRAINING_START = "training_start"
    TRAINING_COMPLETE = "training_complete"
    MODEL_TRAIN = "model_train"
    DRIFT_DETECTED = "drift_detected"
    MODEL_PROMOTION = "model_promotion"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    ADMIN_ACTION = "admin_action"

class AuditSeverity(Enum):
    """Audit event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuditLogger:
    """Comprehensive audit logging service"""
    
    def __init__(self):
        self.logger = logging.getLogger("audit")
        # Set up audit-specific logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    async def log_event(self, 
                       event_type: AuditEventType,
                       user_id: str,
                       session_id: Optional[str] = None,
                       ip_address: Optional[str] = None,
                       resource: Optional[str] = None,
                       action: Optional[str] = None,
                       details: Optional[Dict[str, Any]] = None,
                       severity: AuditSeverity = AuditSeverity.LOW,
                       success: bool = True) -> None:
        """
        Log an audit event
        
        Args:
            event_type: Type of event
            user_id: User who triggered the event
            session_id: Session identifier
            ip_address: IP address of the user
            resource: Resource being accessed
            action: Action being performed
            details: Additional event details
            severity: Event severity level
            success: Whether the action was successful
        """
        try:
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type.value,
                "user_id": user_id,
                "session_id": session_id,
                "ip_address": ip_address,
                "resource": resource,
                "action": action,
                "details": details or {},
                "severity": severity.value,
                "success": success
            }
            
            # Log to audit logger
            log_message = f"Event: {event_type.value} | User: {user_id} | Resource: {resource} | Action: {action} | Success: {success}"
            if details:
                log_message += f" | Details: {details}"
            
            if severity == AuditSeverity.CRITICAL:
                self.logger.critical(log_message)
            elif severity == AuditSeverity.HIGH:
                self.logger.error(log_message)
            elif severity == AuditSeverity.MEDIUM:
                self.logger.warning(log_message)
            else:
                self.logger.info(log_message)
            
            # In production, this would also write to a secure audit database
            # For now, we just log to the audit logger
            
        except Exception as e:
            logger.error(f"❌ Failed to log audit event: {e}")
    
    def log_security_event(self, 
                          event_type: str,
                          user_id: str,
                          details: Dict[str, Any],
                          severity: AuditSeverity = AuditSeverity.HIGH) -> None:
        """Log a security-related event"""
        self.log_event(
            event_type=AuditEventType(event_type),
            user_id=user_id,
            details=details,
            severity=severity
        )
    
    def log_data_access(self, 
                       user_id: str,
                       resource: str,
                       action: str,
                       details: Dict[str, Any]) -> None:
        """Log data access events"""
        self.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id=user_id,
            resource=resource,
            action=action,
            details=details
        )
