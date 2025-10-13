"""
Dashboard Integration Routes
Provides endpoints for real-time dashboard integration
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import json

from services.dashboard_integration import DashboardIntegrationService, DashboardType, MetricType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# Initialize dashboard integration service
dashboard_service = DashboardIntegrationService()


@router.get("/data/{dashboard_type}")
async def get_dashboard_data(dashboard_type: str) -> Dict[str, Any]:
    """
    Get dashboard data for specified type
    
    Args:
        dashboard_type: Type of dashboard
        
    Returns:
        Dashboard data
    """
    try:
        try:
            dashboard_enum = DashboardType(dashboard_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid dashboard type: {dashboard_type}")
        
        data = await dashboard_service.get_dashboard_data(dashboard_enum)
        
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/update")
async def update_metric(
    name: str,
    value: float,
    labels: Optional[Dict[str, str]] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update a metric value
    
    Args:
        name: Metric name
        value: New value
        labels: Metric labels
        description: Metric description
        
    Returns:
        Update status
    """
    try:
        success = await dashboard_service.update_metric(name, value, labels, description)
        
        if success:
            return {
                "status": "success",
                "message": f"Metric {name} updated successfully",
                "value": value,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to update metric {name}"
            }
            
    except Exception as e:
        logger.error(f"Metric update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_all_metrics() -> Dict[str, Any]:
    """
    Get all metrics
    
    Returns:
        All metrics data
    """
    try:
        metrics = {}
        for name, metric in dashboard_service.metrics.items():
            metrics[name] = {
                "name": metric.name,
                "value": metric.value,
                "unit": metric.unit,
                "metric_type": metric.metric_type.value,
                "labels": metric.labels,
                "timestamp": metric.timestamp.isoformat(),
                "trend": metric.trend,
                "status": metric.status,
                "description": metric.description,
                "threshold": metric.threshold
            }
        
        return {
            "status": "success",
            "metrics": metrics,
            "count": len(metrics),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{metric_name}")
async def get_metric(metric_name: str) -> Dict[str, Any]:
    """
    Get specific metric
    
    Args:
        metric_name: Name of the metric
        
    Returns:
        Metric data
    """
    try:
        if metric_name not in dashboard_service.metrics:
            raise HTTPException(status_code=404, detail="Metric not found")
        
        metric = dashboard_service.metrics[metric_name]
        
        return {
            "status": "success",
            "metric": {
                "name": metric.name,
                "value": metric.value,
                "unit": metric.unit,
                "metric_type": metric.metric_type.value,
                "labels": metric.labels,
                "timestamp": metric.timestamp.isoformat(),
                "trend": metric.trend,
                "status": metric.status,
                "description": metric.description,
                "threshold": metric.threshold
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metric retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    status: Optional[str] = Query(None, description="Filter by status")
) -> Dict[str, Any]:
    """
    Get alerts with optional filtering
    
    Args:
        severity: Filter by severity
        status: Filter by status
        
    Returns:
        Alerts data
    """
    try:
        alerts = []
        for alert in dashboard_service.alerts:
            if severity and alert.severity != severity:
                continue
            if status and alert.status != status:
                continue
            
            alerts.append({
                "alert_id": alert.alert_id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity,
                "status": alert.status,
                "timestamp": alert.timestamp.isoformat(),
                "source": alert.source,
                "metric_name": alert.metric_name,
                "threshold_value": alert.threshold_value,
                "current_value": alert.current_value,
                "resolved": alert.resolved
            })
        
        return {
            "status": "success",
            "alerts": alerts,
            "count": len(alerts),
            "filters": {
                "severity": severity,
                "status": status
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Alerts retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str) -> Dict[str, Any]:
    """
    Resolve an alert
    
    Args:
        alert_id: ID of the alert to resolve
        
    Returns:
        Resolution status
    """
    try:
        success = await dashboard_service.resolve_alert(alert_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Alert {alert_id} resolved successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "message": f"Alert {alert_id} not found"
            }
            
    except Exception as e:
        logger.error(f"Alert resolution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_metrics_summary() -> Dict[str, Any]:
    """
    Get metrics summary
    
    Returns:
        Metrics summary
    """
    try:
        summary = await dashboard_service.get_metrics_summary()
        
        return {
            "status": "success",
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Metrics summary retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/configs")
async def get_dashboard_configs() -> Dict[str, Any]:
    """
    Get all dashboard configurations
    
    Returns:
        Dashboard configurations
    """
    try:
        configs = await dashboard_service.get_dashboard_configs()
        
        return {
            "status": "success",
            "configs": configs,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Dashboard configs retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/{dashboard_type}")
async def export_dashboard_data(
    dashboard_type: str,
    format: str = Query("json", description="Export format")
) -> Dict[str, Any]:
    """
    Export dashboard data
    
    Args:
        dashboard_type: Type of dashboard
        format: Export format
        
    Returns:
        Exported data
    """
    try:
        try:
            dashboard_enum = DashboardType(dashboard_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid dashboard type: {dashboard_type}")
        
        export_data = await dashboard_service.export_dashboard_data(dashboard_enum, format)
        
        return export_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard data export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates
    
    Args:
        websocket: WebSocket connection
    """
    await websocket.accept()
    await dashboard_service.add_websocket_connection(websocket)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}))
            elif message.get("type") == "subscribe":
                # Handle subscription to specific metrics or dashboards
                await websocket.send_text(json.dumps({
                    "type": "subscribed",
                    "subscription": message.get("subscription"),
                    "timestamp": datetime.now().isoformat()
                }))
            
    except WebSocketDisconnect:
        await dashboard_service.remove_websocket_connection(websocket)
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await dashboard_service.remove_websocket_connection(websocket)


@router.get("/health")
async def get_dashboard_health() -> Dict[str, Any]:
    """
    Get dashboard service health
    
    Returns:
        Health status
    """
    try:
        health_status = {
            "status": "healthy",
            "service": "dashboard_integration",
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "total_metrics": len(dashboard_service.metrics),
                "total_alerts": len(dashboard_service.alerts),
                "active_alerts": len([a for a in dashboard_service.alerts if not a.resolved]),
                "websocket_connections": len(dashboard_service.websocket_connections)
            },
            "capabilities": [
                "Real-time metric updates",
                "WebSocket streaming",
                "Alert management",
                "Multiple dashboard types",
                "Data export",
                "Threshold monitoring"
            ]
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Dashboard health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "dashboard_integration",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/capabilities")
async def get_dashboard_capabilities() -> Dict[str, Any]:
    """
    Get dashboard capabilities
    
    Returns:
        Dashboard capabilities
    """
    try:
        capabilities = {
            "dashboard_types": [
                {
                    "type": "executive",
                    "title": "Executive Security Dashboard",
                    "description": "High-level security metrics for executives",
                    "refresh_interval": 30,
                    "target_audience": "C-Level Executives, Board Members"
                },
                {
                    "type": "technical",
                    "title": "Technical Security Dashboard",
                    "description": "Detailed technical metrics for security teams",
                    "refresh_interval": 10,
                    "target_audience": "Security Engineers, DevOps Teams"
                },
                {
                    "type": "compliance",
                    "title": "Compliance Dashboard",
                    "description": "Compliance metrics and audit status",
                    "refresh_interval": 60,
                    "target_audience": "Compliance Officers, Auditors"
                },
                {
                    "type": "real_time",
                    "title": "Real-time Security Dashboard",
                    "description": "Live security monitoring and alerts",
                    "refresh_interval": 5,
                    "target_audience": "SOC Analysts, Incident Response Teams"
                },
                {
                    "type": "security_operations",
                    "title": "Security Operations Dashboard",
                    "description": "SOC metrics and operational status",
                    "refresh_interval": 15,
                    "target_audience": "SOC Managers, Security Operations Teams"
                }
            ],
            "features": [
                "Real-time metric updates",
                "WebSocket streaming",
                "Alert management and resolution",
                "Multiple dashboard types",
                "Data export (JSON)",
                "Threshold monitoring",
                "Trend analysis",
                "Interactive charts",
                "Customizable metrics",
                "Historical data tracking"
            ],
            "chart_types": [
                "Line charts (trends)",
                "Bar charts (comparisons)",
                "Doughnut charts (distributions)",
                "Gauge charts (metrics)",
                "Timeline charts (events)"
            ],
            "metric_types": [
                "Counter (cumulative values)",
                "Gauge (current values)",
                "Histogram (distributions)",
                "Summary (statistics)"
            ]
        }
        
        return {
            "status": "success",
            "capabilities": capabilities,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Capabilities retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
