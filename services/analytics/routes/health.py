"""
Health Check Routes
"""

from fastapi import APIRouter, Response
from datetime import datetime
from models.responses import HealthResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="analytics",
        timestamp=datetime.now()
    )


@router.get("/status", response_model=HealthResponse)
async def status():
    """Status endpoint"""
    return HealthResponse(
        status="healthy",
        service="analytics", 
        timestamp=datetime.now()
    )


@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
