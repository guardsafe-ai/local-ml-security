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
