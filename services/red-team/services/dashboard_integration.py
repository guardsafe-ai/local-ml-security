"""
Dashboard Integration Service
Provides real-time dashboard integration for security metrics
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import json
import httpx
from dataclasses import dataclass
from enum import Enum
import websockets
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class DashboardType(Enum):
    """Dashboard types"""
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    COMPLIANCE = "compliance"
    REAL_TIME = "real_time"
    SECURITY_OPERATIONS = "security_operations"


class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class DashboardMetric:
    """Dashboard metric data structure"""
    name: str
    value: float
    unit: str
    metric_type: MetricType
    labels: Dict[str, str]
    timestamp: datetime
    description: str
    trend: Optional[str] = None
    threshold: Optional[float] = None
    status: Optional[str] = None


@dataclass
class DashboardAlert:
    """Dashboard alert data structure"""
    alert_id: str
    title: str
    message: str
    severity: str
    status: str
    timestamp: datetime
    source: str
    metric_name: str
    threshold_value: float
    current_value: float
    resolved: bool = False


class DashboardIntegrationService:
    """
    Service for real-time dashboard integration
    """
    
    def __init__(self):
        """Initialize dashboard integration service"""
        self.metrics: Dict[str, DashboardMetric] = {}
        self.alerts: List[DashboardAlert] = []
        self.websocket_connections: List[websockets.WebSocketServerProtocol] = []
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.dashboard_configs: Dict[DashboardType, Dict[str, Any]] = {}
        self.http_client: Optional[httpx.AsyncClient] = None
        self._setup_dashboard_configs()
        logger.info("✅ Dashboard Integration Service initialized")
    
    def _setup_dashboard_configs(self):
        """Setup dashboard configurations"""
        try:
            # Executive dashboard config
            self.dashboard_configs[DashboardType.EXECUTIVE] = {
                "title": "Executive Security Dashboard",
                "description": "High-level security metrics for executives",
                "refresh_interval": 30,  # seconds
                "metrics": [
                    "overall_security_score",
                    "critical_vulnerabilities",
                    "compliance_score",
                    "incident_count",
                    "threat_level"
                ],
                "charts": [
                    "security_trend",
                    "vulnerability_distribution",
                    "compliance_status"
                ]
            }
            
            # Technical dashboard config
            self.dashboard_configs[DashboardType.TECHNICAL] = {
                "title": "Technical Security Dashboard",
                "description": "Detailed technical metrics for security teams",
                "refresh_interval": 10,  # seconds
                "metrics": [
                    "attack_detection_rate",
                    "false_positive_rate",
                    "response_time",
                    "model_accuracy",
                    "threat_intelligence_updates"
                ],
                "charts": [
                    "attack_timeline",
                    "model_performance",
                    "threat_landscape"
                ]
            }
            
            # Compliance dashboard config
            self.dashboard_configs[DashboardType.COMPLIANCE] = {
                "title": "Compliance Dashboard",
                "description": "Compliance metrics and audit status",
                "refresh_interval": 60,  # seconds
                "metrics": [
                    "soc2_compliance",
                    "iso27001_compliance",
                    "owasp_compliance",
                    "audit_status",
                    "control_coverage"
                ],
                "charts": [
                    "compliance_trends",
                    "audit_timeline",
                    "control_status"
                ]
            }
            
            # Real-time dashboard config
            self.dashboard_configs[DashboardType.REAL_TIME] = {
                "title": "Real-time Security Dashboard",
                "description": "Live security monitoring and alerts",
                "refresh_interval": 5,  # seconds
                "metrics": [
                    "active_attacks",
                    "blocked_requests",
                    "system_health",
                    "alert_count",
                    "response_time"
                ],
                "charts": [
                    "live_attacks",
                    "system_metrics",
                    "alert_timeline"
                ]
            }
            
            # Security operations dashboard config
            self.dashboard_configs[DashboardType.SECURITY_OPERATIONS] = {
                "title": "Security Operations Dashboard",
                "description": "SOC metrics and operational status",
                "refresh_interval": 15,  # seconds
                "metrics": [
                    "incident_response_time",
                    "threat_hunting_results",
                    "vulnerability_scan_results",
                    "security_tool_status",
                    "team_performance"
                ],
                "charts": [
                    "incident_timeline",
                    "threat_hunting_metrics",
                    "tool_status"
                ]
            }
            
        except Exception as e:
            logger.error(f"Dashboard configs setup failed: {e}")
    
    async def initialize(self):
        """Initialize dashboard integration service"""
        try:
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=50)
            )
            
            # Initialize default metrics
            await self._initialize_default_metrics()
            
            logger.info("✅ Dashboard Integration Service initialized")
            
        except Exception as e:
            logger.error(f"Dashboard integration initialization failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.http_client:
                await self.http_client.aclose()
            
            # Close all WebSocket connections
            for ws in self.websocket_connections:
                await ws.close()
            
            logger.info("✅ Dashboard Integration Service cleaned up")
        except Exception as e:
            logger.error(f"Dashboard integration cleanup failed: {e}")
    
    async def _initialize_default_metrics(self):
        """Initialize default metrics"""
        try:
            default_metrics = [
                {
                    "name": "overall_security_score",
                    "value": 85.0,
                    "unit": "%",
                    "metric_type": MetricType.GAUGE,
                    "labels": {"category": "security"},
                    "description": "Overall security score",
                    "threshold": 80.0
                },
                {
                    "name": "critical_vulnerabilities",
                    "value": 0,
                    "unit": "count",
                    "metric_type": MetricType.COUNTER,
                    "labels": {"severity": "critical"},
                    "description": "Number of critical vulnerabilities",
                    "threshold": 5.0
                },
                {
                    "name": "attack_detection_rate",
                    "value": 95.0,
                    "unit": "%",
                    "metric_type": MetricType.GAUGE,
                    "labels": {"category": "detection"},
                    "description": "Attack detection rate",
                    "threshold": 90.0
                },
                {
                    "name": "false_positive_rate",
                    "value": 2.0,
                    "unit": "%",
                    "metric_type": MetricType.GAUGE,
                    "labels": {"category": "detection"},
                    "description": "False positive rate",
                    "threshold": 5.0
                },
                {
                    "name": "response_time",
                    "value": 150.0,
                    "unit": "ms",
                    "metric_type": MetricType.HISTOGRAM,
                    "labels": {"category": "performance"},
                    "description": "Average response time",
                    "threshold": 200.0
                }
            ]
            
            for metric_data in default_metrics:
                metric = DashboardMetric(
                    name=metric_data["name"],
                    value=metric_data["value"],
                    unit=metric_data["unit"],
                    metric_type=metric_data["metric_type"],
                    labels=metric_data["labels"],
                    timestamp=datetime.now(),
                    description=metric_data["description"],
                    threshold=metric_data.get("threshold")
                )
                
                self.metrics[metric.name] = metric
                self.metric_history[metric.name].append(metric.value)
            
            logger.info("✅ Default metrics initialized")
            
        except Exception as e:
            logger.error(f"Default metrics initialization failed: {e}")
    
    async def update_metric(self, 
                           name: str, 
                           value: float, 
                           labels: Dict[str, str] = None,
                           description: str = None) -> bool:
        """
        Update a metric value
        
        Args:
            name: Metric name
            value: New value
            labels: Metric labels
            description: Metric description
            
        Returns:
            True if successful
        """
        try:
            if name in self.metrics:
                # Update existing metric
                metric = self.metrics[name]
                metric.value = value
                metric.timestamp = datetime.now()
                
                # Update labels if provided
                if labels:
                    metric.labels.update(labels)
                
                # Update description if provided
                if description:
                    metric.description = description
                
                # Calculate trend
                if len(self.metric_history[name]) > 1:
                    previous_value = self.metric_history[name][-2]
                    if value > previous_value:
                        metric.trend = "up"
                    elif value < previous_value:
                        metric.trend = "down"
                    else:
                        metric.trend = "stable"
                
                # Check threshold
                if metric.threshold:
                    if value > metric.threshold:
                        metric.status = "warning"
                        await self._create_alert(name, value, metric.threshold, "high")
                    elif value < metric.threshold * 0.8:  # 20% below threshold
                        metric.status = "good"
                    else:
                        metric.status = "normal"
                
            else:
                # Create new metric
                metric = DashboardMetric(
                    name=name,
                    value=value,
                    unit="",
                    metric_type=MetricType.GAUGE,
                    labels=labels or {},
                    timestamp=datetime.now(),
                    description=description or ""
                )
                self.metrics[name] = metric
            
            # Update history
            self.metric_history[name].append(value)
            
            # Broadcast to WebSocket connections
            await self._broadcast_metric_update(metric)
            
            logger.debug(f"Updated metric {name}: {value}")
            return True
            
        except Exception as e:
            logger.error(f"Metric update failed for {name}: {e}")
            return False
    
    async def _create_alert(self, metric_name: str, current_value: float, threshold: float, severity: str):
        """Create an alert for metric threshold breach"""
        try:
            alert_id = f"alert_{metric_name}_{datetime.now().timestamp()}"
            
            alert = DashboardAlert(
                alert_id=alert_id,
                title=f"Metric Threshold Breached: {metric_name}",
                message=f"{metric_name} is {current_value} (threshold: {threshold})",
                severity=severity,
                status="active",
                timestamp=datetime.now(),
                source="dashboard",
                metric_name=metric_name,
                threshold_value=threshold,
                current_value=current_value
            )
            
            self.alerts.append(alert)
            
            # Broadcast alert to WebSocket connections
            await self._broadcast_alert(alert)
            
            logger.warning(f"Alert created: {alert_id}")
            
        except Exception as e:
            logger.error(f"Alert creation failed: {e}")
    
    async def _broadcast_metric_update(self, metric: DashboardMetric):
        """Broadcast metric update to WebSocket connections"""
        try:
            if not self.websocket_connections:
                return
            
            message = {
                "type": "metric_update",
                "data": {
                    "name": metric.name,
                    "value": metric.value,
                    "unit": metric.unit,
                    "labels": metric.labels,
                    "timestamp": metric.timestamp.isoformat(),
                    "trend": metric.trend,
                    "status": metric.status
                }
            }
            
            # Send to all connected clients
            disconnected = []
            for ws in self.websocket_connections:
                try:
                    await ws.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected.append(ws)
            
            # Remove disconnected clients
            for ws in disconnected:
                self.websocket_connections.remove(ws)
            
        except Exception as e:
            logger.error(f"Metric broadcast failed: {e}")
    
    async def _broadcast_alert(self, alert: DashboardAlert):
        """Broadcast alert to WebSocket connections"""
        try:
            if not self.websocket_connections:
                return
            
            message = {
                "type": "alert",
                "data": {
                    "alert_id": alert.alert_id,
                    "title": alert.title,
                    "message": alert.message,
                    "severity": alert.severity,
                    "status": alert.status,
                    "timestamp": alert.timestamp.isoformat(),
                    "source": alert.source,
                    "metric_name": alert.metric_name,
                    "threshold_value": alert.threshold_value,
                    "current_value": alert.current_value
                }
            }
            
            # Send to all connected clients
            disconnected = []
            for ws in self.websocket_connections:
                try:
                    await ws.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected.append(ws)
            
            # Remove disconnected clients
            for ws in disconnected:
                self.websocket_connections.remove(ws)
            
        except Exception as e:
            logger.error(f"Alert broadcast failed: {e}")
    
    async def get_dashboard_data(self, dashboard_type: DashboardType) -> Dict[str, Any]:
        """
        Get dashboard data for specified type
        
        Args:
            dashboard_type: Type of dashboard
            
        Returns:
            Dashboard data
        """
        try:
            config = self.dashboard_configs.get(dashboard_type, {})
            
            # Get metrics for this dashboard
            dashboard_metrics = {}
            for metric_name in config.get("metrics", []):
                if metric_name in self.metrics:
                    metric = self.metrics[metric_name]
                    dashboard_metrics[metric_name] = {
                        "name": metric.name,
                        "value": metric.value,
                        "unit": metric.unit,
                        "labels": metric.labels,
                        "timestamp": metric.timestamp.isoformat(),
                        "trend": metric.trend,
                        "status": metric.status,
                        "description": metric.description
                    }
            
            # Get recent alerts
            recent_alerts = [
                {
                    "alert_id": alert.alert_id,
                    "title": alert.title,
                    "message": alert.message,
                    "severity": alert.severity,
                    "status": alert.status,
                    "timestamp": alert.timestamp.isoformat(),
                    "metric_name": alert.metric_name
                }
                for alert in self.alerts[-10:]  # Last 10 alerts
            ]
            
            # Get metric history for charts
            chart_data = {}
            for chart_name in config.get("charts", []):
                chart_data[chart_name] = await self._get_chart_data(chart_name)
            
            return {
                "dashboard_type": dashboard_type.value,
                "title": config.get("title", ""),
                "description": config.get("description", ""),
                "refresh_interval": config.get("refresh_interval", 30),
                "metrics": dashboard_metrics,
                "alerts": recent_alerts,
                "charts": chart_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Dashboard data retrieval failed: {e}")
            return {"error": str(e)}
    
    async def _get_chart_data(self, chart_name: str) -> Dict[str, Any]:
        """Get chart data for specified chart"""
        try:
            if chart_name == "security_trend":
                return {
                    "type": "line",
                    "data": {
                        "labels": [f"T-{i}" for i in range(24, 0, -1)],
                        "datasets": [{
                            "label": "Security Score",
                            "data": list(self.metric_history.get("overall_security_score", deque([85.0] * 24))),
                            "borderColor": "rgb(75, 192, 192)",
                            "tension": 0.1
                        }]
                    }
                }
            
            elif chart_name == "vulnerability_distribution":
                return {
                    "type": "doughnut",
                    "data": {
                        "labels": ["Critical", "High", "Medium", "Low"],
                        "datasets": [{
                            "data": [0, 2, 5, 12],
                            "backgroundColor": ["#ff6384", "#ff9f40", "#ffcd56", "#4bc0c0"]
                        }]
                    }
                }
            
            elif chart_name == "attack_timeline":
                return {
                    "type": "bar",
                    "data": {
                        "labels": [f"Hour {i}" for i in range(24)],
                        "datasets": [{
                            "label": "Attacks",
                            "data": [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 0, 1, 2, 1, 0, 1, 0, 2, 1, 0, 1, 0, 0, 1],
                            "backgroundColor": "rgba(255, 99, 132, 0.6)"
                        }]
                    }
                }
            
            else:
                return {"type": "empty", "data": {}}
                
        except Exception as e:
            logger.error(f"Chart data retrieval failed for {chart_name}: {e}")
            return {"type": "error", "data": {}}
    
    async def add_websocket_connection(self, websocket: websockets.WebSocketServerProtocol):
        """Add WebSocket connection for real-time updates"""
        try:
            self.websocket_connections.append(websocket)
            logger.info(f"WebSocket connection added. Total connections: {len(self.websocket_connections)}")
        except Exception as e:
            logger.error(f"WebSocket connection addition failed: {e}")
    
    async def remove_websocket_connection(self, websocket: websockets.WebSocketServerProtocol):
        """Remove WebSocket connection"""
        try:
            if websocket in self.websocket_connections:
                self.websocket_connections.remove(websocket)
            logger.info(f"WebSocket connection removed. Total connections: {len(self.websocket_connections)}")
        except Exception as e:
            logger.error(f"WebSocket connection removal failed: {e}")
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        try:
            summary = {
                "total_metrics": len(self.metrics),
                "total_alerts": len(self.alerts),
                "active_alerts": len([a for a in self.alerts if not a.resolved]),
                "websocket_connections": len(self.websocket_connections),
                "metrics_by_status": {
                    "good": len([m for m in self.metrics.values() if m.status == "good"]),
                    "normal": len([m for m in self.metrics.values() if m.status == "normal"]),
                    "warning": len([m for m in self.metrics.values() if m.status == "warning"])
                },
                "alerts_by_severity": {
                    "critical": len([a for a in self.alerts if a.severity == "critical"]),
                    "high": len([a for a in self.alerts if a.severity == "high"]),
                    "medium": len([a for a in self.alerts if a.severity == "medium"]),
                    "low": len([a for a in self.alerts if a.severity == "low"])
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Metrics summary generation failed: {e}")
            return {"error": str(e)}
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        try:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.status = "resolved"
                    
                    # Broadcast resolution
                    await self._broadcast_alert(alert)
                    
                    logger.info(f"Alert resolved: {alert_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Alert resolution failed: {e}")
            return False
    
    async def get_dashboard_configs(self) -> Dict[str, Any]:
        """Get all dashboard configurations"""
        try:
            configs = {}
            for dashboard_type, config in self.dashboard_configs.items():
                configs[dashboard_type.value] = config
            
            return {
                "status": "success",
                "configs": configs,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Dashboard configs retrieval failed: {e}")
            return {"error": str(e)}
    
    async def export_dashboard_data(self, dashboard_type: DashboardType, format: str = "json") -> Dict[str, Any]:
        """Export dashboard data in specified format"""
        try:
            data = await self.get_dashboard_data(dashboard_type)
            
            if format.lower() == "json":
                return {
                    "status": "success",
                    "format": "json",
                    "data": data,
                    "exported_at": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported format: {format}"
                }
                
        except Exception as e:
            logger.error(f"Dashboard data export failed: {e}")
            return {"error": str(e)}
