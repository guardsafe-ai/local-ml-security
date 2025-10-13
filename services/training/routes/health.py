"""
Training Service - Health Routes
Health check and system status endpoints
"""

import logging
from datetime import datetime
from fastapi import APIRouter, Response
from models.responses import HealthResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="healthy",
        service="training",
        timestamp=datetime.now()
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    try:
        # Check dependencies
        dependencies = {
            "mlflow": True,  # This would be checked in practice
            "redis": True,   # This would be checked in practice
            "minio": True,   # This would be checked in practice
            "database": True # This would be checked in practice
        }
        
        return HealthResponse(
            status="healthy",
            service="training",
            timestamp=datetime.now(),
            dependencies=dependencies,
            models_loaded=1,  # This would be dynamic
            active_jobs=0,    # This would be dynamic
            uptime_seconds=3600.0  # This would be calculated
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            service="training",
            timestamp=datetime.now()
        )


@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
