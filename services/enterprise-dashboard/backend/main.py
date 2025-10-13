"""
Enterprise ML Security Dashboard - FastAPI Backend (Modular)
API Gateway that aggregates all ML Security services with modular architecture
"""

import asyncio
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


# Import modular components
from models.responses import HealthResponse
from routes import (
    health_router,
    dashboard_router,
    models_router,
    training_router,
    red_team_router,
    analytics_router,
    business_metrics_router,
    data_privacy_router,
    mlflow_router,
    monitoring_router,
    websocket_router
)
from routes.data_management import router as data_management_router
from routes.performance_cache import router as performance_cache_router
from services.websocket_manager import manager
from utils.logging import setup_logging
from utils.config import get_config

# Setup logging
config = get_config()
setup_logging(config["log_level"])
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Enterprise Dashboard Backend starting up...")
    
    # Startup tasks
    try:
        # Initialize any required services
        logger.info("Initializing services...")
        
        
        # Start WebSocket cleanup task
        if config["enable_websocket"]:
            import asyncio
            asyncio.create_task(periodic_websocket_cleanup())
        
        logger.info("Enterprise Dashboard Backend startup completed")
        
    except Exception as e:
        logger.error(f"Failed to initialize Enterprise Dashboard Backend: {e}")
        raise
    
    yield
    
    # Shutdown tasks
    logger.info("Enterprise Dashboard Backend shutting down...")
    try:
        # Cleanup WebSocket connections
        if config["enable_websocket"]:
            await manager.broadcast({
                "type": "shutdown",
                "data": {"message": "Server is shutting down"},
                "timestamp": datetime.now().isoformat()
            })
        
        
        logger.info("Enterprise Dashboard Backend shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


async def periodic_websocket_cleanup():
    """Periodic cleanup of inactive WebSocket connections"""
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            await manager.cleanup_inactive_connections()
        except Exception as e:
            logger.error(f"WebSocket cleanup error: {e}")


app = FastAPI(
    title="Enterprise ML Security Dashboard Backend (Modular)",
    version=config["version"],
    description="Modular API Gateway for ML Security services",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom CORS headers middleware
@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    return response

# Include routers
app.include_router(health_router, tags=["Health"])
app.include_router(dashboard_router, prefix="/dashboard", tags=["Dashboard"])
app.include_router(models_router, prefix="/models", tags=["Models"])
app.include_router(training_router, prefix="/training", tags=["Training"])
app.include_router(data_management_router, prefix="/data", tags=["Data Management"])
app.include_router(performance_cache_router, prefix="/training", tags=["Performance Cache"])
app.include_router(red_team_router, prefix="/red-team", tags=["Red Team"])
app.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])
app.include_router(business_metrics_router, prefix="/business-metrics", tags=["Business Metrics"])
app.include_router(data_privacy_router, prefix="/data-privacy", tags=["Data Privacy"])
app.include_router(mlflow_router, prefix="/mlflow", tags=["MLflow"])
app.include_router(monitoring_router, prefix="/monitoring", tags=["Monitoring"])

# Include WebSocket router if enabled
if config["enable_websocket"]:
    app.include_router(websocket_router, tags=["WebSocket"])

# Global exception handlers
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with proper JSON responses"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "message": exc.detail,
            "path": str(request.url),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with proper JSON responses"""
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "status_code": 422,
            "message": "Validation error",
            "details": exc.errors(),
            "path": str(request.url),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors with proper JSON responses"""
    return JSONResponse(
        status_code=404,
        content={
            "error": True,
            "status_code": 404,
            "message": f"Endpoint not found: {request.url.path}",
            "available_endpoints": [
                "/health",
                "/docs",
                "/openapi.json",
                "/dashboard/metrics",
                "/dashboard/red-team/overview",
                "/dashboard/training/overview",
                "/dashboard/analytics/overview",
                "/dashboard/business-metrics/overview",
                "/dashboard/data-privacy/overview",
                "/models",
                "/models/registry",
                "/training/jobs",
                "/red-team/results",
                "/analytics/summary",
                "/business-metrics/summary",
                "/data-privacy/summary"
            ],
            "path": str(request.url),
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)
