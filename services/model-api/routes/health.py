"""
Health Check Routes
"""

import asyncio
import logging
import httpx
from fastapi import APIRouter, HTTPException
from datetime import datetime
from models.responses import HealthResponse
from typing import Dict, Any

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["health"])


async def check_database_connection() -> bool:
    """Check database connectivity"""
    try:
        # This would be injected from the main app
        # For now, return True as a placeholder
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


async def check_mlflow_connection() -> bool:
    """Check MLflow connectivity"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://mlflow:5000/health", timeout=5.0)
            return response.status_code == 200
    except Exception as e:
        logger.error(f"MLflow health check failed: {e}")
        return False


async def check_redis_connection() -> bool:
    """Check Redis connectivity"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://redis:6379/ping", timeout=5.0)
            return response.status_code == 200
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False


async def check_models_status() -> Dict[str, Any]:
    """Check model loading status"""
    try:
        # This would be injected from the main app
        # For now, return placeholder data
        return {
            "loaded_models": 0,
            "total_models": 0,
            "memory_usage_mb": 0
        }
    except Exception as e:
        logger.error(f"Models status check failed: {e}")
        return {"error": str(e)}


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint"""
    return HealthResponse(
        status="healthy",
        available_models=[],
        total_models=0,
        timestamp=datetime.now()
    )


@router.get("/deep")
async def deep_health_check():
    """Deep health check with dependency verification"""
    try:
        # Run all health checks in parallel
        checks = await asyncio.gather(
            check_database_connection(),
            check_mlflow_connection(),
            check_redis_connection(),
            return_exceptions=True
        )
        
        models_status = await check_models_status()
        
        # Process results
        db_healthy = checks[0] if not isinstance(checks[0], Exception) else False
        mlflow_healthy = checks[1] if not isinstance(checks[1], Exception) else False
        redis_healthy = checks[2] if not isinstance(checks[2], Exception) else False
        
        checks_result = {
            "service": "healthy",
            "database": db_healthy,
            "mlflow": mlflow_healthy,
            "redis": redis_healthy,
            "models": models_status
        }
        
        # Determine overall status
        critical_services = [db_healthy, mlflow_healthy]
        overall_status = "healthy" if all(critical_services) else "degraded"
        
        if not any(critical_services):
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "checks": checks_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Deep health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    try:
        # Check if service is ready to accept traffic
        # This would check if models are loaded, DB is connected, etc.
        return {"status": "ready", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    try:
        # Check if service is alive (basic process check)
        return {"status": "alive", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        raise HTTPException(status_code=500, detail="Service not alive")
