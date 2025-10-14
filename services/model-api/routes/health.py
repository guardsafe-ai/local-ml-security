"""
Enhanced Health Check Routes with Liveness, Readiness, and Startup Probes
"""

import asyncio
import logging
import httpx
import time
# import psutil  # Will be installed later
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timedelta
from models.responses import HealthResponse
from typing import Dict, Any, Optional
import redis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["health"])

# Global state for health checks
_service_start_time = None
_startup_complete = False
_ready_state = False


def set_service_start_time():
    """Set service start time for uptime calculation"""
    global _service_start_time
    _service_start_time = time.time()

def set_startup_complete():
    """Mark startup as complete"""
    global _startup_complete
    _startup_complete = True

def set_ready_state(ready: bool):
    """Set service ready state"""
    global _ready_state
    _ready_state = ready

async def check_database_connection() -> Dict[str, Any]:
    """Check database connectivity with detailed metrics"""
    try:
        start_time = time.time()
        # This would be injected from the main app
        # For now, return True as a placeholder
        latency_ms = (time.time() - start_time) * 1000
        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 2),
            "connection_pool": "active"
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "latency_ms": None
        }

async def check_mlflow_connection() -> Dict[str, Any]:
    """Check MLflow connectivity with detailed metrics"""
    try:
        start_time = time.time()
        async with httpx.AsyncClient() as client:
            response = await client.get("http://mlflow:5000/health", timeout=5.0)
            latency_ms = (time.time() - start_time) * 1000
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "latency_ms": round(latency_ms, 2),
                "status_code": response.status_code
            }
    except Exception as e:
        logger.error(f"MLflow health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "latency_ms": None
        }

async def check_redis_connection() -> Dict[str, Any]:
    """Check Redis connectivity with detailed metrics"""
    try:
        start_time = time.time()
        # Use the existing Redis client from the main app
        from main import model_manager
        if hasattr(model_manager, 'redis_client') and model_manager.redis_client:
            model_manager.redis_client.ping()
            latency_ms = (time.time() - start_time) * 1000
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "memory_usage": "available"
            }
        else:
            # Fallback to direct connection
            redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
            redis_client.ping()
            latency_ms = (time.time() - start_time) * 1000
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "memory_usage": "available"
            }
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "latency_ms": None
        }

async def check_models_status() -> Dict[str, Any]:
    """Check model loading status with detailed metrics"""
    try:
        # This would be injected from the main app
        # For now, return placeholder data
        return {
            "loaded_models": 0,
            "total_models": 0,
            "memory_usage_mb": 0,
            "status": "healthy"
        }
    except Exception as e:
        logger.error(f"Models status check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "loaded_models": 0,
            "total_models": 0,
            "memory_usage_mb": 0
        }

async def check_system_resources() -> Dict[str, Any]:
    """Check system resource usage"""
    try:
        # For now, return basic system info without psutil
        return {
            "status": "healthy",
            "cpu_percent": "N/A (psutil not available)",
            "memory": {
                "total_gb": "N/A (psutil not available)",
                "available_gb": "N/A (psutil not available)",
                "percent_used": "N/A (psutil not available)"
            },
            "disk": {
                "total_gb": "N/A (psutil not available)",
                "free_gb": "N/A (psutil not available)",
                "percent_used": "N/A (psutil not available)"
            }
        }
    except Exception as e:
        logger.error(f"System resources check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


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
    """Comprehensive health check with dependency verification and metrics"""
    try:
        # Run all health checks in parallel
        checks = await asyncio.gather(
            check_database_connection(),
            check_mlflow_connection(),
            check_redis_connection(),
            check_models_status(),
            check_system_resources(),
            return_exceptions=True
        )
        
        # Process results
        db_check = checks[0] if not isinstance(checks[0], Exception) else {"status": "unhealthy", "error": str(checks[0])}
        mlflow_check = checks[1] if not isinstance(checks[1], Exception) else {"status": "unhealthy", "error": str(checks[1])}
        redis_check = checks[2] if not isinstance(checks[2], Exception) else {"status": "unhealthy", "error": str(checks[2])}
        models_check = checks[3] if not isinstance(checks[3], Exception) else {"status": "unhealthy", "error": str(checks[3])}
        system_check = checks[4] if not isinstance(checks[4], Exception) else {"status": "unhealthy", "error": str(checks[4])}
        
        checks_result = {
            "service": "healthy",
            "database": db_check,
            "mlflow": mlflow_check,
            "redis": redis_check,
            "models": models_check,
            "system": system_check
        }
        
        # Determine overall status
        critical_services = [
            db_check.get("status") == "healthy",
            mlflow_check.get("status") == "healthy"
        ]
        overall_status = "healthy" if all(critical_services) else "degraded"
        
        if not any(critical_services):
            overall_status = "unhealthy"
        
        # Calculate uptime
        uptime_seconds = 0
        if _service_start_time:
            uptime_seconds = time.time() - _service_start_time
        
        return {
            "status": overall_status,
            "checks": checks_result,
            "uptime_seconds": round(uptime_seconds, 2),
            "startup_complete": _startup_complete,
            "ready_state": _ready_state,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Deep health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe - checks if service is ready to accept traffic"""
    try:
        # Check critical dependencies for readiness
        checks = await asyncio.gather(
            check_database_connection(),
            check_mlflow_connection(),
            return_exceptions=True
        )
        
        db_ready = checks[0].get("status") == "healthy" if not isinstance(checks[0], Exception) else False
        mlflow_ready = checks[1].get("status") == "healthy" if not isinstance(checks[1], Exception) else False
        
        # Service is ready if critical dependencies are healthy and startup is complete
        is_ready = db_ready and mlflow_ready and _startup_complete
        
        if is_ready:
            set_ready_state(True)
            return {
                "status": "ready",
                "dependencies": {
                    "database": db_ready,
                    "mlflow": mlflow_ready
                },
                "startup_complete": _startup_complete,
                "timestamp": datetime.now().isoformat()
            }
        else:
            set_ready_state(False)
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "dependencies": {
                        "database": db_ready,
                        "mlflow": mlflow_ready
                    },
                    "startup_complete": _startup_complete,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe - checks if service is alive and responsive"""
    try:
        # Basic liveness check - service is alive if it can respond
        # Check system resources to ensure service isn't in a bad state
        system_check = await check_system_resources()
        
        # Service is alive if system resources are reasonable
        # For now, just check if system check is healthy (psutil not available)
        is_alive = system_check.get("status") == "healthy"
        
        if is_alive:
            return {
                "status": "alive",
                "system_resources": system_check,
                "uptime_seconds": round(time.time() - _service_start_time, 2) if _service_start_time else 0,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "not_alive",
                    "reason": "system_resources_exhausted",
                    "system_resources": system_check,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        raise HTTPException(status_code=500, detail="Service not alive")


@router.get("/startup")
async def startup_check():
    """Kubernetes startup probe - checks if service has completed startup"""
    try:
        # Check if startup is complete
        if _startup_complete:
            return {
                "status": "started",
                "startup_complete": True,
                "uptime_seconds": round(time.time() - _service_start_time, 2) if _service_start_time else 0,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Service is still starting up
            startup_time = time.time() - _service_start_time if _service_start_time else 0
            return {
                "status": "starting",
                "startup_complete": False,
                "startup_time_seconds": round(startup_time, 2),
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Startup check failed: {e}")
        raise HTTPException(status_code=500, detail="Startup check failed")
