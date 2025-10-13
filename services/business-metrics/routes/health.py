"""
Business Metrics Service - Health Routes
Health check and status endpoints
"""

import logging
import time
from datetime import datetime
from fastapi import APIRouter, HTTPException
from models.responses import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Service start time for uptime calculation
start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        uptime = time.time() - start_time
        
        # Check dependencies (simplified)
        dependencies = {
            "redis": True,  # Would check actual Redis connection
            "postgres": True,  # Would check actual PostgreSQL connection
            "analytics": True,  # Would check analytics service
            "model_api": True  # Would check model API service
        }
        
        return HealthResponse(
            status="healthy",
            service="business-metrics",
            timestamp=datetime.now(),
            uptime_seconds=uptime,
            dependencies=dependencies
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
