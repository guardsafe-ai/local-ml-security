"""
Monitoring Service - API Routes
REST API endpoints for monitoring data
"""

import logging
from datetime import datetime
from typing import Dict, List, Any
from fastapi import APIRouter, HTTPException
from services.data_collector import DataCollector
from services.visualization import VisualizationService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
data_collector = DataCollector()
visualization_service = VisualizationService()


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "monitoring",
        "version": "1.0.0",
        "status": "running",
        "description": "Real-time monitoring and visualization service"
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "service": "monitoring",
            "timestamp": datetime.now(),
            "uptime_seconds": 0.0,  # Would calculate actual uptime
            "dependencies": {
                "redis": True,
                "postgres": True,
                "data_collector": True
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard-data")
async def get_dashboard_data():
    """Get complete dashboard data"""
    try:
        data = data_collector.get_dashboard_data()
        return data
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-loading-status")
async def get_model_loading_status():
    """Get model loading status"""
    try:
        data = data_collector.get_model_loading_status()
        return {"data": data, "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Error getting model loading status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-status")
async def get_training_status():
    """Get training status"""
    try:
        data = data_collector.get_training_status()
        return {"data": data, "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system-metrics")
async def get_system_metrics():
    """Get system metrics"""
    try:
        data = data_collector.get_system_metrics()
        return {"data": data, "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/service-health")
async def get_service_health():
    """Get service health status"""
    try:
        data = data_collector.get_service_health()
        return {"data": data, "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Error getting service health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts():
    """Get alerts"""
    try:
        data = data_collector.get_alerts()
        return {"data": data, "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/charts/model-loading")
async def get_model_loading_chart():
    """Get model loading chart data"""
    try:
        data = data_collector.get_model_loading_status()
        chart = visualization_service.create_model_loading_chart(data)
        return {"chart": chart.to_json(), "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Error creating model loading chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/charts/training-progress")
async def get_training_progress_chart():
    """Get training progress chart data"""
    try:
        data = data_collector.get_training_status()
        chart = visualization_service.create_training_progress_chart(data)
        return {"chart": chart.to_json(), "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Error creating training progress chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/charts/system-metrics")
async def get_system_metrics_chart():
    """Get system metrics chart data"""
    try:
        data = data_collector.get_system_metrics()
        chart = visualization_service.create_system_metrics_chart(data)
        return {"chart": chart.to_json(), "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Error creating system metrics chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/charts/service-health")
async def get_service_health_chart():
    """Get service health chart data"""
    try:
        data = data_collector.get_service_health()
        chart = visualization_service.create_service_health_chart(data)
        return {"chart": chart.to_json(), "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Error creating service health chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/charts/alerts-timeline")
async def get_alerts_timeline_chart():
    """Get alerts timeline chart data"""
    try:
        data = data_collector.get_alerts()
        chart = visualization_service.create_alerts_timeline(data)
        return {"chart": chart.to_json(), "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Error creating alerts timeline chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))
