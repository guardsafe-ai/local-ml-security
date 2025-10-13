"""
Comprehensive Audit Logging Service
Implements GDPR-compliant audit trails for all system activities
"""

import logging
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hashlib

logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Types of audit events"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    DATA_UPLOAD = "data_upload"
    DATA_DOWNLOAD = "data_download"
    DATA_DELETE = "data_delete"
    MODEL_TRAIN = "model_train"
    MODEL_DEPLOY = "model_deploy"
    MODEL_PREDICT = "model_predict"
    DRIFT_DETECT = "drift_detect"
    MODEL_PROMOTE = "model_promote"
    SYSTEM_CONFIG = "system_config"
    ACCESS_DENIED = "access_denied"
    PII_DETECTED = "pii_detected"
    SECURITY_ALERT = "security_alert"

class AuditSeverity(Enum):
    """Severity levels for audit events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AuditEvent:
    """Represents an audit event"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource: str
    action: str
    details: Dict[str, Any]
    severity: AuditSeverity
    success: bool
    error_message: Optional[str] = None
    data_classification: Optional[str] = None
    retention_period: Optional[int] = None

class AuditLogger:
    """Handles comprehensive audit logging"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.events_buffer = []
        self.buffer_size = 100
        self.flush_interval = 30  # seconds
        self._start_background_flush()
        
    def _start_background_flush(self):
        """Start background task to flush audit events"""
        asyncio.create_task(self._background_flush())
    
    async def _background_flush(self):
        """Background task to flush audit events periodically"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush_events()
            except Exception as e:
                logger.error(f"❌ [AUDIT] Background flush error: {e}")
    
    async def log_event(self, event: AuditEvent):
        """Log an audit event"""
        try:
            # Add to buffer
            self.events_buffer.append(event)
            
            # Flush if buffer is full
            if len(self.events_buffer) >= self.buffer_size:
                await self.flush_events()
            
            logger.info(f"📝 [AUDIT] Logged {event.event_type.value} event: {event.event_id}")
            
        except Exception as e:
            logger.error(f"❌ [AUDIT] Failed to log event: {e}")
    
    async def flush_events(self):
        """Flush all buffered events to database"""
        if not self.events_buffer:
            return
        
        try:
            events_to_flush = self.events_buffer.copy()
            self.events_buffer.clear()
            
            if self.db_manager:
                await self._store_events_in_db(events_to_flush)
            else:
                # Fallback to file logging
                await self._store_events_in_file(events_to_flush)
            
            logger.info(f"✅ [AUDIT] Flushed {len(events_to_flush)} events")
            
        except Exception as e:
            logger.error(f"❌ [AUDIT] Failed to flush events: {e}")
            # Re-add events to buffer on failure
            self.events_buffer.extend(events_to_flush)
    
    async def _store_events_in_db(self, events: List[AuditEvent]):
        """Store events in database with transaction support"""
        try:
            async with self.db_manager.transaction() as conn:
                for event in events:
                    await conn.execute("""
                        INSERT INTO audit_log (
                            event_id, event_type, timestamp, user_id, session_id,
                            ip_address, user_agent, resource, action, details,
                            severity, success, error_message, data_classification,
                            retention_period
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    """, 
                    event.event_id,
                    event.event_type.value,
                    event.timestamp,
                    event.user_id,
                    event.session_id,
                    event.ip_address,
                    event.user_agent,
                    event.resource,
                    event.action,
                    json.dumps(event.details),
                    event.severity.value,
                    event.success,
                    event.error_message,
                    event.data_classification,
                    event.retention_period
                    )
        except Exception as e:
            logger.error(f"❌ [AUDIT] Failed to store events in DB: {e}")
            raise
    
    async def _store_events_in_file(self, events: List[AuditEvent]):
        """Store events in file as fallback"""
        try:
            audit_file = "/tmp/audit_log.jsonl"
            with open(audit_file, "a") as f:
                for event in events:
                    event_dict = asdict(event)
                    event_dict["timestamp"] = event.timestamp.isoformat()
                    f.write(json.dumps(event_dict) + "\n")
        except Exception as e:
            logger.error(f"❌ [AUDIT] Failed to store events in file: {e}")
    
    async def query_events(self, 
                          user_id: Optional[str] = None,
                          event_type: Optional[AuditEventType] = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          severity: Optional[AuditSeverity] = None,
                          limit: int = 1000) -> List[Dict[str, Any]]:
        """Query audit events with filters"""
        try:
            # Input validation
            if not isinstance(limit, int) or limit < 1:
                limit = 1000
            limit = min(limit, 10000)  # Cap at reasonable maximum
            
            query = "SELECT * FROM audit_log WHERE 1=1"
            params = []
            param_count = 0
            
            if user_id:
                param_count += 1
                query += f" AND user_id = ${param_count}"
                params.append(user_id)
            
            if event_type:
                param_count += 1
                query += f" AND event_type = ${param_count}"
                params.append(event_type.value)
            
            if start_time:
                param_count += 1
                query += f" AND timestamp >= ${param_count}"
                params.append(start_time)
            
            if end_time:
                param_count += 1
                query += f" AND timestamp <= ${param_count}"
                params.append(end_time)
            
            if severity:
                param_count += 1
                query += f" AND severity = ${param_count}"
                params.append(severity.value)
            
            # Parameterize LIMIT clause to prevent SQL injection
            param_count += 1
            query += f" ORDER BY timestamp DESC LIMIT ${param_count}"
            # Validate and cap limit to prevent abuse
            params.append(min(limit, 10000))  # Cap at reasonable maximum
            
            if self.db_manager:
                results = await self.db_manager.execute_query(query, *params)
                return results
            else:
                return []
                
        except Exception as e:
            logger.error(f"❌ [AUDIT] Failed to query events: {e}")
            return []
    
    async def get_user_activity_summary(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get activity summary for a user"""
        try:
            end_time = datetime.now(timezone.utc)
            start_time = datetime.fromtimestamp(
                end_time.timestamp() - (days * 24 * 60 * 60),
                tz=timezone.utc
            )
            
            events = await self.query_events(
                user_id=user_id,
                start_time=start_time,
                end_time=end_time
            )
            
            # Analyze events
            event_counts = {}
            resource_access = set()
            failed_attempts = 0
            
            for event in events:
                event_type = event.get("event_type", "unknown")
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                
                if event.get("resource"):
                    resource_access.add(event["resource"])
                
                if not event.get("success", True):
                    failed_attempts += 1
            
            return {
                "user_id": user_id,
                "period_days": days,
                "total_events": len(events),
                "event_counts": event_counts,
                "unique_resources": len(resource_access),
                "failed_attempts": failed_attempts,
                "success_rate": (len(events) - failed_attempts) / len(events) if events else 0
            }
            
        except Exception as e:
            logger.error(f"❌ [AUDIT] Failed to get user activity summary: {e}")
            return {}
    
    async def detect_suspicious_activity(self, user_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Detect suspicious activity patterns"""
        try:
            end_time = datetime.now(timezone.utc)
            start_time = datetime.fromtimestamp(
                end_time.timestamp() - (hours * 60 * 60),
                tz=timezone.utc
            )
            
            events = await self.query_events(
                user_id=user_id,
                start_time=start_time,
                end_time=end_time
            )
            
            suspicious_events = []
            
            # Check for patterns
            failed_logins = [e for e in events if e.get("event_type") == "user_login" and not e.get("success")]
            if len(failed_logins) > 5:
                suspicious_events.append({
                    "type": "multiple_failed_logins",
                    "count": len(failed_logins),
                    "events": failed_logins[:5]  # First 5 events
                })
            
            # Check for unusual access patterns
            access_denied = [e for e in events if e.get("event_type") == "access_denied"]
            if len(access_denied) > 3:
                suspicious_events.append({
                    "type": "multiple_access_denials",
                    "count": len(access_denied),
                    "events": access_denied[:3]
                })
            
            # Check for PII access
            pii_events = [e for e in events if e.get("event_type") == "pii_detected"]
            if pii_events:
                suspicious_events.append({
                    "type": "pii_access",
                    "count": len(pii_events),
                    "events": pii_events
                })
            
            return suspicious_events
            
        except Exception as e:
            logger.error(f"❌ [AUDIT] Failed to detect suspicious activity: {e}")
            return []

# Convenience functions for common audit events
async def log_user_login(user_id: str, ip_address: str, user_agent: str, success: bool, session_id: str = None):
    """Log user login event"""
    event = AuditEvent(
        event_id=str(uuid.uuid4()),
        event_type=AuditEventType.USER_LOGIN,
        timestamp=datetime.now(timezone.utc),
        user_id=user_id,
        session_id=session_id,
        ip_address=ip_address,
        user_agent=user_agent,
        resource="authentication",
        action="login",
        details={"login_method": "password"},
        severity=AuditSeverity.MEDIUM if success else AuditSeverity.HIGH,
        success=success
    )
    await audit_logger.log_event(event)

async def log_data_access(user_id: str, resource: str, action: str, data_classification: str, ip_address: str = None):
    """Log data access event"""
    event = AuditEvent(
        event_id=str(uuid.uuid4()),
        event_type=AuditEventType.DATA_DOWNLOAD if action == "download" else AuditEventType.DATA_UPLOAD,
        timestamp=datetime.now(timezone.utc),
        user_id=user_id,
        ip_address=ip_address,
        resource=resource,
        action=action,
        details={"data_classification": data_classification},
        severity=AuditSeverity.HIGH if data_classification == "confidential" else AuditSeverity.MEDIUM,
        success=True,
        data_classification=data_classification,
        retention_period=2555  # 7 years for GDPR compliance
    )
    await audit_logger.log_event(event)

async def log_model_prediction(user_id: str, model_name: str, input_hash: str, prediction: str, confidence: float, ip_address: str = None):
    """Log model prediction event"""
    event = AuditEvent(
        event_id=str(uuid.uuid4()),
        event_type=AuditEventType.MODEL_PREDICT,
        timestamp=datetime.now(timezone.utc),
        user_id=user_id,
        ip_address=ip_address,
        resource=f"model:{model_name}",
        action="predict",
        details={
            "model_name": model_name,
            "input_hash": input_hash,
            "prediction": prediction,
            "confidence": confidence
        },
        severity=AuditSeverity.LOW,
        success=True,
        retention_period=365  # 1 year
    )
    await audit_logger.log_event(event)

async def log_security_alert(alert_type: str, details: Dict[str, Any], severity: AuditSeverity, user_id: str = None):
    """Log security alert"""
    event = AuditEvent(
        event_id=str(uuid.uuid4()),
        event_type=AuditEventType.SECURITY_ALERT,
        timestamp=datetime.now(timezone.utc),
        user_id=user_id,
        resource="security",
        action="alert",
        details=details,
        severity=severity,
        success=False
    )
    await audit_logger.log_event(event)

# Global audit logger instance
audit_logger = AuditLogger()
