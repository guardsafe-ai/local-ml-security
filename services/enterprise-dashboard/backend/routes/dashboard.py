"""
Enterprise Dashboard Backend - Dashboard Routes
Main dashboard and metrics endpoints
"""

import logging
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from models.responses import DashboardMetrics, ModelInfo, TrainingJob, AttackResult
from services.dashboard_service import DashboardService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize dashboard service
dashboard_service = DashboardService()


@router.get("/metrics", response_model=DashboardMetrics)
async def get_dashboard_metrics():
    """Get comprehensive dashboard metrics"""
    try:
        return await dashboard_service.get_dashboard_metrics()
    except Exception as e:
        logger.error(f"Failed to get dashboard metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/overview")
async def get_models_overview():
    """Get comprehensive models overview"""
    try:
        return await dashboard_service.get_models_overview()
    except Exception as e:
        logger.error(f"Failed to get models overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/overview")
async def get_training_overview():
    """Get training jobs overview"""
    try:
        return await dashboard_service.get_training_overview()
    except Exception as e:
        logger.error(f"Failed to get training overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/red-team/overview")
async def get_red_team_overview():
    """Get red team testing overview"""
    try:
        return await dashboard_service.get_red_team_overview()
    except Exception as e:
        logger.error(f"Failed to get red team overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/health")
async def get_system_health_overview():
    """Get system health overview"""
    try:
        return await dashboard_service.get_system_health_overview()
    except Exception as e:
        logger.error(f"Failed to get system health overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/activity/recent")
async def get_recent_activity(limit: int = 10):
    """Get recent system activity"""
    try:
        return await dashboard_service.get_recent_activity(limit)
    except Exception as e:
        logger.error(f"Failed to get recent activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/trends")
async def get_performance_trends(hours: int = 24):
    """Get performance trends over time"""
    try:
        return await dashboard_service.get_performance_trends(hours)
    except Exception as e:
        logger.error(f"Failed to get performance trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/overview")
async def get_analytics_overview():
    """Get analytics overview"""
    try:
        return await dashboard_service.get_analytics_overview()
    except Exception as e:
        logger.error(f"Failed to get analytics overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/business-metrics/overview")
async def get_business_metrics_overview():
    """Get business metrics overview"""
    try:
        return await dashboard_service.get_business_metrics_overview()
    except Exception as e:
        logger.error(f"Failed to get business metrics overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-privacy/overview")
async def get_data_privacy_overview():
    """Get data privacy overview"""
    try:
        return await dashboard_service.get_data_privacy_overview()
    except Exception as e:
        logger.error(f"Failed to get data privacy overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/realtime")
async def get_realtime_metrics():
    """Get real-time metrics for WebSocket broadcasting"""
    try:
        # Get fresh metrics without cache
        metrics = await dashboard_service.get_dashboard_metrics()
        health_status = await dashboard_service.api_client.get_all_services_health()
        
        return {
            "metrics": {
                "total_models": metrics.total_models,
                "active_jobs": metrics.active_jobs,
                "total_attacks": metrics.total_attacks,
                "detection_rate": metrics.detection_rate,
                "system_health": metrics.system_health,
                "last_updated": metrics.last_updated.isoformat()
            },
            "services": health_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get real-time metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/score")
async def get_health_score():
    """Get calculated health score with algorithm"""
    try:
        health_status = await dashboard_service.api_client.get_all_services_health()
        
        if not health_status:
            return {"score": 0, "status": "unhealthy", "details": "No services available"}
        
        # Calculate health score with weighted algorithm
        total_services = len(health_status)
        healthy_services = sum(1 for service in health_status if service.get("status") == "healthy")
        
        # Weight by response time (lower is better)
        response_times = [service.get("response_time", 0) for service in health_status if service.get("response_time", 0) > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Base score from healthy services percentage
        base_score = (healthy_services / total_services) * 100
        
        # Adjust for response time (penalty for slow responses)
        response_penalty = min(avg_response_time * 2, 20)  # Max 20 point penalty
        
        # Final score
        final_score = max(base_score - response_penalty, 0)
        
        # Determine status
        if final_score >= 90:
            status = "excellent"
        elif final_score >= 75:
            status = "good"
        elif final_score >= 50:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "score": round(final_score, 2),
            "status": status,
            "healthy_services": healthy_services,
            "total_services": total_services,
            "average_response_time": round(avg_response_time, 3),
            "response_penalty": round(response_penalty, 2),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to calculate health score: {e}")
        raise HTTPException(status_code=500, detail=str(e))
