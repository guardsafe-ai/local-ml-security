"""
Enterprise Dashboard Backend - Health Routes
Health check and system status endpoints
"""

import logging
from datetime import datetime
from typing import List
from fastapi import APIRouter
from models.responses import HealthResponse, ServiceHealth
from services.api_client import APIClient

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize API client
api_client = APIClient()


@router.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="healthy",
        service="enterprise-dashboard-backend",
        timestamp=datetime.now(),
        version="1.0.0"
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    try:
        return HealthResponse(
            status="healthy",
            service="enterprise-dashboard-backend",
            timestamp=datetime.now(),
            version="1.0.0",
            uptime_seconds=3600.0  # This would be calculated in practice
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            service="enterprise-dashboard-backend",
            timestamp=datetime.now(),
            version="1.0.0"
        )


@router.get("/services/health", response_model=List[ServiceHealth])
async def get_all_services_health():
    """Get health status of all ML Security services"""
    try:
        health_status = await api_client.get_all_services_health()
        return [ServiceHealth(**service) for service in health_status]
    except Exception as e:
        logger.error(f"Failed to get services health: {e}")
        return []


@router.get("/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        # Get all services health
        health_status = await api_client.get_all_services_health()
        
        # Calculate overall health
        healthy_services = sum(1 for service in health_status if service.get("status") == "healthy")
        total_services = len(health_status)
        overall_health = (healthy_services / total_services * 100) if total_services > 0 else 0.0
        
        return {
            "status": "healthy" if overall_health >= 80 else "degraded" if overall_health >= 50 else "unhealthy",
            "overall_health_percentage": round(overall_health, 2),
            "healthy_services": healthy_services,
            "total_services": total_services,
            "services": health_status,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "uptime_seconds": 3600.0  # This would be calculated in practice
        }
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
