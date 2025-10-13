"""
Red Team Service - Health Routes
Health check and system status endpoints
"""

import logging
from datetime import datetime
from fastapi import APIRouter
from models.responses import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="healthy",
        service="red-team",
        timestamp=datetime.now()
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    try:
        # Check dependencies
        dependencies = {
            "model_api": True,  # This would be checked in practice
            "redis": True,      # This would be checked in practice
            "database": True,   # This would be checked in practice
            "mlflow": True      # This would be checked in practice
        }
        
        return HealthResponse(
            status="healthy",
            service="red-team",
            timestamp=datetime.now(),
            dependencies=dependencies,
            running=False,  # This would be dynamic
            total_tests=0,  # This would be dynamic
            uptime_seconds=3600.0  # This would be calculated
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            service="red-team",
            timestamp=datetime.now()
        )
