"""
WebSocket Streaming Module
Real-time streaming of red team testing events and metrics via WebSocket.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import uuid

try:
    from fastapi import WebSocket, WebSocketDisconnect
    from fastapi.websockets import WebSocketState
except ImportError:
    # Fallback for environments without FastAPI
    class WebSocket:
        def __init__(self, *args, **kwargs): pass
        async def send_text(self, *args, **kwargs): pass
        async def send_json(self, *args, **kwargs): pass
        async def close(self, *args, **kwargs): pass
        @property
        def client_state(self): return None
    class WebSocketDisconnect(Exception): pass
    class WebSocketState:
        CONNECTED = "connected"
        DISCONNECTED = "disconnected"

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of streaming events"""
    ATTACK_START = "attack_start"
    ATTACK_PROGRESS = "attack_progress"
    ATTACK_COMPLETE = "attack_complete"
    VULNERABILITY_DETECTED = "vulnerability_detected"
    COMPLIANCE_VIOLATION = "compliance_violation"
    PRIVACY_BREACH = "privacy_breach"
    SYSTEM_ALERT = "system_alert"
    METRICS_UPDATE = "metrics_update"
    WORKER_STATUS = "worker_status"
    QUEUE_UPDATE = "queue_update"

class EventSeverity(Enum):
    """Event severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class StreamingEvent:
    """Base streaming event"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    severity: EventSeverity
    data: Dict[str, Any]
    source: str = "redteam"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "data": self.data,
            "source": self.source
        }

@dataclass
class AttackEvent(StreamingEvent):
    """Attack-specific event"""
    attack_type: str
    model_type: str
    session_id: str
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None

@dataclass
class VulnerabilityEvent(StreamingEvent):
    """Vulnerability detection event"""
    vulnerability_id: str
    attack_type: str
    model_type: str
    severity_score: float
    description: str
    remediation: Optional[str] = None

@dataclass
class ComplianceEvent(StreamingEvent):
    """Compliance violation event"""
    framework: str
    control: str
    violation_type: str
    description: str
    remediation: Optional[str] = None

@dataclass
class SystemEvent(StreamingEvent):
    """System-level event"""
    component: str
    status: str
    message: str
    metrics: Optional[Dict[str, Any]] = None

class WebSocketStreaming:
    """WebSocket streaming manager for real-time events"""
    
    def __init__(self):
        self.connections: Set[WebSocket] = set()
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        self.event_history: List[StreamingEvent] = []
        self.max_history_size = 1000
        self.subscriptions: Dict[WebSocket, Set[EventType]] = {}
        
    async def connect(self, websocket: WebSocket, 
                     subscribed_events: Optional[List[EventType]] = None):
        """Accept a new WebSocket connection"""
        try:
            await websocket.accept()
            self.connections.add(websocket)
            
            if subscribed_events:
                self.subscriptions[websocket] = set(subscribed_events)
            else:
                # Subscribe to all events by default
                self.subscriptions[websocket] = set(EventType)
            
            # Send connection confirmation
            await self._send_event(websocket, StreamingEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.SYSTEM_ALERT,
                timestamp=datetime.now(),
                severity=EventSeverity.INFO,
                data={"message": "Connected to red team streaming"},
                source="system"
            ))
            
            logger.info(f"WebSocket connected. Total connections: {len(self.connections)}")
            
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            await self.disconnect(websocket)
    
    async def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket"""
        try:
            if websocket in self.connections:
                self.connections.remove(websocket)
            
            if websocket in self.subscriptions:
                del self.subscriptions[websocket]
            
            await websocket.close()
            logger.info(f"WebSocket disconnected. Total connections: {len(self.connections)}")
            
        except Exception as e:
            logger.error(f"Failed to disconnect WebSocket: {e}")
    
    async def broadcast_event(self, event: StreamingEvent):
        """Broadcast event to all connected clients"""
        try:
            # Add to history
            self.event_history.append(event)
            if len(self.event_history) > self.max_history_size:
                self.event_history.pop(0)
            
            # Call registered handlers
            if event.event_type in self.event_handlers:
                for handler in self.event_handlers[event.event_type]:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"Event handler failed: {e}")
            
            # Broadcast to subscribed connections
            disconnected = set()
            for websocket in self.connections.copy():
                try:
                    if (websocket in self.subscriptions and 
                        event.event_type in self.subscriptions[websocket]):
                        await self._send_event(websocket, event)
                except WebSocketDisconnect:
                    disconnected.add(websocket)
                except Exception as e:
                    logger.error(f"Failed to send event to WebSocket: {e}")
                    disconnected.add(websocket)
            
            # Clean up disconnected connections
            for websocket in disconnected:
                await self.disconnect(websocket)
                
        except Exception as e:
            logger.error(f"Failed to broadcast event: {e}")
    
    async def _send_event(self, websocket: WebSocket, event: StreamingEvent):
        """Send event to specific WebSocket"""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(event.to_dict())
        except Exception as e:
            logger.error(f"Failed to send event to WebSocket: {e}")
            raise
    
    def register_event_handler(self, event_type: EventType, handler: Callable):
        """Register event handler for specific event type"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def unregister_event_handler(self, event_type: EventType, handler: Callable):
        """Unregister event handler"""
        if event_type in self.event_handlers and handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
    
    async def stream_attack_start(self, attack_type: str, model_type: str, 
                                session_id: str, data: Dict[str, Any] = None):
        """Stream attack start event"""
        event = AttackEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.ATTACK_START,
            timestamp=datetime.now(),
            severity=EventSeverity.INFO,
            data=data or {},
            attack_type=attack_type,
            model_type=model_type,
            session_id=session_id
        )
        await self.broadcast_event(event)
    
    async def stream_attack_progress(self, attack_type: str, model_type: str, 
                                   session_id: str, progress: float, 
                                   data: Dict[str, Any] = None):
        """Stream attack progress event"""
        event = AttackEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.ATTACK_PROGRESS,
            timestamp=datetime.now(),
            severity=EventSeverity.INFO,
            data=data or {},
            attack_type=attack_type,
            model_type=model_type,
            session_id=session_id,
            progress=progress
        )
        await self.broadcast_event(event)
    
    async def stream_attack_complete(self, attack_type: str, model_type: str, 
                                   session_id: str, result: Dict[str, Any], 
                                   success: bool):
        """Stream attack completion event"""
        severity = EventSeverity.INFO if success else EventSeverity.ERROR
        event = AttackEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.ATTACK_COMPLETE,
            timestamp=datetime.now(),
            severity=severity,
            data={"success": success},
            attack_type=attack_type,
            model_type=model_type,
            session_id=session_id,
            result=result
        )
        await self.broadcast_event(event)
    
    async def stream_vulnerability(self, vulnerability_id: str, attack_type: str, 
                                 model_type: str, severity_score: float, 
                                 description: str, remediation: str = None):
        """Stream vulnerability detection event"""
        severity = self._score_to_severity(severity_score)
        event = VulnerabilityEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.VULNERABILITY_DETECTED,
            timestamp=datetime.now(),
            severity=severity,
            data={},
            vulnerability_id=vulnerability_id,
            attack_type=attack_type,
            model_type=model_type,
            severity_score=severity_score,
            description=description,
            remediation=remediation
        )
        await self.broadcast_event(event)
    
    async def stream_compliance_violation(self, framework: str, control: str, 
                                        violation_type: str, description: str, 
                                        remediation: str = None):
        """Stream compliance violation event"""
        event = ComplianceEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.COMPLIANCE_VIOLATION,
            timestamp=datetime.now(),
            severity=EventSeverity.WARNING,
            data={},
            framework=framework,
            control=control,
            violation_type=violation_type,
            description=description,
            remediation=remediation
        )
        await self.broadcast_event(event)
    
    async def stream_privacy_breach(self, breach_type: str, data_type: str, 
                                  description: str, severity_score: float):
        """Stream privacy breach event"""
        severity = self._score_to_severity(severity_score)
        event = StreamingEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.PRIVACY_BREACH,
            timestamp=datetime.now(),
            severity=severity,
            data={
                "breach_type": breach_type,
                "data_type": data_type,
                "description": description,
                "severity_score": severity_score
            }
        )
        await self.broadcast_event(event)
    
    async def stream_system_alert(self, component: str, status: str, 
                                message: str, metrics: Dict[str, Any] = None):
        """Stream system alert event"""
        severity = self._status_to_severity(status)
        event = SystemEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.SYSTEM_ALERT,
            timestamp=datetime.now(),
            severity=severity,
            data={},
            component=component,
            status=status,
            message=message,
            metrics=metrics
        )
        await self.broadcast_event(event)
    
    async def stream_metrics_update(self, metrics: Dict[str, Any]):
        """Stream metrics update event"""
        event = StreamingEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.METRICS_UPDATE,
            timestamp=datetime.now(),
            severity=EventSeverity.INFO,
            data=metrics
        )
        await self.broadcast_event(event)
    
    async def stream_worker_status(self, worker_id: str, status: str, 
                                 metrics: Dict[str, Any] = None):
        """Stream worker status update"""
        event = SystemEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.WORKER_STATUS,
            timestamp=datetime.now(),
            severity=EventSeverity.INFO,
            data={},
            component=f"worker_{worker_id}",
            status=status,
            message=f"Worker {worker_id} status: {status}",
            metrics=metrics
        )
        await self.broadcast_event(event)
    
    async def stream_queue_update(self, queue_type: str, size: int, 
                                metrics: Dict[str, Any] = None):
        """Stream queue update event"""
        event = StreamingEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.QUEUE_UPDATE,
            timestamp=datetime.now(),
            severity=EventSeverity.INFO,
            data={
                "queue_type": queue_type,
                "size": size,
                "metrics": metrics or {}
            }
        )
        await self.broadcast_event(event)
    
    def _score_to_severity(self, score: float) -> EventSeverity:
        """Convert numeric score to severity level"""
        if score >= 0.8:
            return EventSeverity.CRITICAL
        elif score >= 0.6:
            return EventSeverity.ERROR
        elif score >= 0.4:
            return EventSeverity.WARNING
        else:
            return EventSeverity.INFO
    
    def _status_to_severity(self, status: str) -> EventSeverity:
        """Convert status string to severity level"""
        status_lower = status.lower()
        if status_lower in ["error", "failed", "critical"]:
            return EventSeverity.CRITICAL
        elif status_lower in ["warning", "degraded"]:
            return EventSeverity.WARNING
        else:
            return EventSeverity.INFO
    
    def get_event_history(self, event_type: Optional[EventType] = None, 
                         limit: int = 100) -> List[StreamingEvent]:
        """Get event history"""
        events = self.event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:] if limit else events
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.connections)
    
    def get_subscription_stats(self) -> Dict[EventType, int]:
        """Get subscription statistics"""
        stats = {}
        for event_type in EventType:
            count = sum(1 for subs in self.subscriptions.values() 
                       if event_type in subs)
            stats[event_type] = count
        return stats
    
    async def cleanup_disconnected(self):
        """Clean up disconnected WebSocket connections"""
        try:
            disconnected = set()
            for websocket in self.connections.copy():
                try:
                    # Try to send a ping to check if connection is alive
                    await websocket.send_text("ping")
                except:
                    disconnected.add(websocket)
            
            for websocket in disconnected:
                await self.disconnect(websocket)
                
        except Exception as e:
            logger.error(f"Failed to cleanup disconnected connections: {e}")
    
    async def start_heartbeat(self, interval: int = 30):
        """Start heartbeat to keep connections alive"""
        try:
            while True:
                await self.cleanup_disconnected()
                await asyncio.sleep(interval)
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
    
    def export_event_data(self, format_type: str = "json") -> str:
        """Export event data in specified format"""
        try:
            if format_type == "json":
                events_data = [event.to_dict() for event in self.event_history]
                return json.dumps(events_data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            logger.error(f"Failed to export event data: {e}")
            return "[]"
