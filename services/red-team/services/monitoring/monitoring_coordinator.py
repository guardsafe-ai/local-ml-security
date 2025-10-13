"""
Monitoring Coordinator
Coordinates all monitoring components including metrics, streaming, and alerts.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import json

from .prometheus_metrics import PrometheusMetrics
from .websocket_streaming import WebSocketStreaming, StreamingEvent, EventType
from .alert_manager import AlertManager, AlertSeverity

logger = logging.getLogger(__name__)

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enable_metrics: bool = True
    enable_streaming: bool = True
    enable_alerts: bool = True
    metrics_interval: int = 30
    cleanup_interval: int = 3600
    alert_cooldown: int = 300
    max_event_history: int = 1000

class MonitoringCoordinator:
    """Coordinates all monitoring components"""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        
        # Initialize components
        self.metrics = PrometheusMetrics() if self.config.enable_metrics else None
        self.streaming = WebSocketStreaming() if self.config.enable_streaming else None
        self.alerts = AlertManager() if self.config.enable_alerts else None
        
        # State
        self.is_running = False
        self.tasks = []
        
        # Setup integration
        self._setup_integration()
    
    def _setup_integration(self):
        """Setup integration between components"""
        try:
            if self.streaming and self.alerts:
                # Register alert handler for streaming events
                self.streaming.register_event_handler(
                    EventType.VULNERABILITY_DETECTED,
                    self._handle_vulnerability_event
                )
                self.streaming.register_event_handler(
                    EventType.COMPLIANCE_VIOLATION,
                    self._handle_compliance_event
                )
                self.streaming.register_event_handler(
                    EventType.PRIVACY_BREACH,
                    self._handle_privacy_event
                )
                self.streaming.register_event_handler(
                    EventType.SYSTEM_ALERT,
                    self._handle_system_event
                )
                self.streaming.register_event_handler(
                    EventType.ATTACK_COMPLETE,
                    self._handle_attack_event
                )
                self.streaming.register_event_handler(
                    EventType.WORKER_STATUS,
                    self._handle_worker_event
                )
                self.streaming.register_event_handler(
                    EventType.QUEUE_UPDATE,
                    self._handle_queue_event
                )
                self.streaming.register_event_handler(
                    EventType.METRICS_UPDATE,
                    self._handle_metrics_event
                )
            
            logger.info("Monitoring integration setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup monitoring integration: {e}")
    
    async def _handle_vulnerability_event(self, event: StreamingEvent):
        """Handle vulnerability detection event"""
        try:
            if self.alerts:
                event_data = {
                    "event_type": "vulnerability_detected",
                    "severity": event.data.get("severity", "medium"),
                    "attack_type": event.data.get("attack_type", "unknown"),
                    "model_type": event.data.get("model_type", "unknown"),
                    "vulnerability_id": event.data.get("vulnerability_id", ""),
                    "description": event.data.get("description", ""),
                    "severity_score": event.data.get("severity_score", 0.5)
                }
                await self.alerts.evaluate_event(event_data)
                
        except Exception as e:
            logger.error(f"Failed to handle vulnerability event: {e}")
    
    async def _handle_compliance_event(self, event: StreamingEvent):
        """Handle compliance violation event"""
        try:
            if self.alerts:
                event_data = {
                    "event_type": "compliance_violation",
                    "framework": event.data.get("framework", "unknown"),
                    "control": event.data.get("control", "unknown"),
                    "violation_type": event.data.get("violation_type", "unknown"),
                    "description": event.data.get("description", ""),
                    "severity": "medium"
                }
                await self.alerts.evaluate_event(event_data)
                
        except Exception as e:
            logger.error(f"Failed to handle compliance event: {e}")
    
    async def _handle_privacy_event(self, event: StreamingEvent):
        """Handle privacy breach event"""
        try:
            if self.alerts:
                event_data = {
                    "event_type": "privacy_breach",
                    "breach_type": event.data.get("breach_type", "unknown"),
                    "data_type": event.data.get("data_type", "unknown"),
                    "description": event.data.get("description", ""),
                    "severity_score": event.data.get("severity_score", 0.5)
                }
                await self.alerts.evaluate_event(event_data)
                
        except Exception as e:
            logger.error(f"Failed to handle privacy event: {e}")
    
    async def _handle_system_event(self, event: StreamingEvent):
        """Handle system alert event"""
        try:
            if self.alerts:
                event_data = {
                    "event_type": "system_alert",
                    "component": event.data.get("component", "unknown"),
                    "status": event.data.get("status", "unknown"),
                    "message": event.data.get("message", ""),
                    "severity": event.severity.value
                }
                await self.alerts.evaluate_event(event_data)
                
        except Exception as e:
            logger.error(f"Failed to handle system event: {e}")
    
    async def _handle_attack_event(self, event: StreamingEvent):
        """Handle attack completion event"""
        try:
            if self.alerts:
                success = event.data.get("success", True)
                event_data = {
                    "event_type": "attack_complete",
                    "success": success,
                    "attack_type": event.data.get("attack_type", "unknown"),
                    "model_type": event.data.get("model_type", "unknown"),
                    "failure_rate": 0.0 if success else 1.0
                }
                await self.alerts.evaluate_event(event_data)
                
        except Exception as e:
            logger.error(f"Failed to handle attack event: {e}")
    
    async def _handle_worker_event(self, event: StreamingEvent):
        """Handle worker status event"""
        try:
            if self.alerts:
                event_data = {
                    "event_type": "worker_status",
                    "status": event.data.get("status", "unknown"),
                    "worker_id": event.data.get("worker_id", "unknown"),
                    "component": event.data.get("component", "worker")
                }
                await self.alerts.evaluate_event(event_data)
                
        except Exception as e:
            logger.error(f"Failed to handle worker event: {e}")
    
    async def _handle_queue_event(self, event: StreamingEvent):
        """Handle queue update event"""
        try:
            if self.alerts:
                event_data = {
                    "event_type": "queue_update",
                    "size": event.data.get("size", 0),
                    "queue_type": event.data.get("queue_type", "unknown")
                }
                await self.alerts.evaluate_event(event_data)
                
        except Exception as e:
            logger.error(f"Failed to handle queue event: {e}")
    
    async def _handle_metrics_event(self, event: StreamingEvent):
        """Handle metrics update event"""
        try:
            if self.alerts:
                metrics = event.data
                event_data = {
                    "event_type": "metrics_update",
                    "cpu_usage": metrics.get("cpu_usage", 0),
                    "memory_usage": metrics.get("memory_usage", 0),
                    "disk_usage": metrics.get("disk_usage", {}),
                    "gpu_utilization": metrics.get("gpu_utilization", 0)
                }
                await self.alerts.evaluate_event(event_data)
                
        except Exception as e:
            logger.error(f"Failed to handle metrics event: {e}")
    
    async def start(self):
        """Start monitoring coordinator"""
        try:
            if self.is_running:
                logger.warning("Monitoring coordinator already running")
                return
            
            self.is_running = True
            
            # Start metrics collection
            if self.metrics:
                task = asyncio.create_task(
                    self.metrics.start_metrics_collection(self.config.metrics_interval)
                )
                self.tasks.append(task)
            
            # Start streaming heartbeat
            if self.streaming:
                task = asyncio.create_task(
                    self.streaming.start_heartbeat()
                )
                self.tasks.append(task)
            
            # Start alert cleanup
            if self.alerts:
                task = asyncio.create_task(
                    self.alerts.start_cleanup_task(self.config.cleanup_interval)
                )
                self.tasks.append(task)
            
            logger.info("Monitoring coordinator started")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring coordinator: {e}")
            await self.stop()
    
    async def stop(self):
        """Stop monitoring coordinator"""
        try:
            self.is_running = False
            
            # Cancel all tasks
            for task in self.tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            
            self.tasks.clear()
            logger.info("Monitoring coordinator stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring coordinator: {e}")
    
    def record_attack_execution(self, attack_type: str, model_type: str, 
                              duration: float, success: bool):
        """Record attack execution"""
        try:
            if self.metrics:
                self.metrics.record_attack_execution(attack_type, model_type, duration, success)
            
            if self.streaming:
                asyncio.create_task(
                    self.streaming.stream_attack_complete(
                        attack_type, model_type, "session_id", 
                        {"success": success, "duration": duration}, success
                    )
                )
                
        except Exception as e:
            logger.error(f"Failed to record attack execution: {e}")
    
    def record_vulnerability(self, vulnerability_id: str, attack_type: str, 
                           model_type: str, severity_score: float, 
                           description: str, remediation: str = None):
        """Record vulnerability detection"""
        try:
            if self.metrics:
                severity = self._score_to_severity(severity_score)
                self.metrics.record_vulnerability(severity, attack_type, model_type)
            
            if self.streaming:
                asyncio.create_task(
                    self.streaming.stream_vulnerability(
                        vulnerability_id, attack_type, model_type, 
                        severity_score, description, remediation
                    )
                )
                
        except Exception as e:
            logger.error(f"Failed to record vulnerability: {e}")
    
    def record_compliance_violation(self, framework: str, control: str, 
                                  violation_type: str, description: str, 
                                  remediation: str = None):
        """Record compliance violation"""
        try:
            if self.metrics:
                self.metrics.record_compliance_violation(framework, control, "medium")
            
            if self.streaming:
                asyncio.create_task(
                    self.streaming.stream_compliance_violation(
                        framework, control, violation_type, description, remediation
                    )
                )
                
        except Exception as e:
            logger.error(f"Failed to record compliance violation: {e}")
    
    def record_privacy_breach(self, breach_type: str, data_type: str, 
                            description: str, severity_score: float):
        """Record privacy breach"""
        try:
            if self.metrics:
                self.metrics.record_privacy_breach(breach_type, data_type)
            
            if self.streaming:
                asyncio.create_task(
                    self.streaming.stream_privacy_breach(
                        breach_type, data_type, description, severity_score
                    )
                )
                
        except Exception as e:
            logger.error(f"Failed to record privacy breach: {e}")
    
    def update_system_metrics(self, cpu_percent: float, memory_bytes: int, 
                            disk_usage: Dict[str, int], gpu_utilization: Dict[str, float] = None):
        """Update system metrics"""
        try:
            if self.metrics:
                self.metrics.update_resource_usage(cpu_percent, memory_bytes, disk_usage)
                
                if gpu_utilization:
                    for gpu_id, utilization in gpu_utilization.items():
                        self.metrics.update_gpu_utilization(gpu_id, "unknown", utilization)
            
            if self.streaming:
                metrics_data = {
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory_bytes,
                    "disk_usage": disk_usage,
                    "gpu_utilization": gpu_utilization or {}
                }
                asyncio.create_task(
                    self.streaming.stream_metrics_update(metrics_data)
                )
                
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def update_worker_status(self, worker_id: str, status: str, 
                           metrics: Dict[str, Any] = None):
        """Update worker status"""
        try:
            if self.streaming:
                asyncio.create_task(
                    self.streaming.stream_worker_status(worker_id, status, metrics)
                )
                
        except Exception as e:
            logger.error(f"Failed to update worker status: {e}")
    
    def update_queue_size(self, queue_type: str, size: int, 
                         metrics: Dict[str, Any] = None):
        """Update queue size"""
        try:
            if self.metrics:
                self.metrics.update_queue_size(queue_type, size)
            
            if self.streaming:
                asyncio.create_task(
                    self.streaming.stream_queue_update(queue_type, size, metrics)
                )
                
        except Exception as e:
            logger.error(f"Failed to update queue size: {e}")
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics"""
        if self.metrics:
            return self.metrics.get_metrics()
        return ""
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        if self.alerts:
            return self.alerts.get_alert_stats()
        return {}
    
    def get_connection_count(self) -> int:
        """Get WebSocket connection count"""
        if self.streaming:
            return self.streaming.get_connection_count()
        return 0
    
    def get_subscription_stats(self) -> Dict[str, int]:
        """Get subscription statistics"""
        if self.streaming:
            return self.streaming.get_subscription_stats()
        return {}
    
    def _score_to_severity(self, score: float) -> str:
        """Convert numeric score to severity string"""
        if score >= 0.8:
            return "critical"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def export_monitoring_data(self, format_type: str = "json") -> str:
        """Export monitoring data"""
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "metrics_available": self.metrics is not None,
                "streaming_available": self.streaming is not None,
                "alerts_available": self.alerts is not None,
                "connection_count": self.get_connection_count(),
                "alert_stats": self.get_alert_stats(),
                "subscription_stats": self.get_subscription_stats()
            }
            
            if format_type == "json":
                return json.dumps(data, indent=2)
            else:
                return str(data)
                
        except Exception as e:
            logger.error(f"Failed to export monitoring data: {e}")
            return "{}"
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of monitoring components"""
        try:
            status = {
                "overall": "healthy",
                "components": {
                    "metrics": "healthy" if self.metrics else "disabled",
                    "streaming": "healthy" if self.streaming else "disabled",
                    "alerts": "healthy" if self.alerts else "disabled"
                },
                "is_running": self.is_running,
                "active_tasks": len(self.tasks),
                "timestamp": datetime.now().isoformat()
            }
            
            # Check for any unhealthy components
            unhealthy_components = [
                name for name, health in status["components"].items() 
                if health == "unhealthy"
            ]
            
            if unhealthy_components:
                status["overall"] = "degraded"
                status["unhealthy_components"] = unhealthy_components
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {
                "overall": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
