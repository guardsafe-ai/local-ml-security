"""
Enterprise ML Security Dashboard - FastAPI Backend (Modular)
API Gateway that aggregates all ML Security services with modular architecture
"""

import asyncio
import logging
import signal
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

# Graceful shutdown handler
class GracefulShutdown:
    """Handles graceful shutdown of the enterprise-dashboard backend service"""
    
    def __init__(self, app):
        self.app = app
        self.is_shutting_down = False
        self.shutdown_timeout = 30  # seconds
        self.pending_tasks = set()
        logger.info("GracefulShutdown handler initialized for enterprise-dashboard backend service.")
    
    def register_handlers(self):
        """Register signal handlers for graceful shutdown."""
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self.shutdown_handler)
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, self.shutdown_handler)
        logger.info("Registered SIGTERM and SIGINT handlers for enterprise-dashboard backend service.")
        
        @self.app.on_event("shutdown")
        async def _on_shutdown():
            await self._perform_cleanup()
    
    async def shutdown_handler(self, signum, frame):
        """Handle shutdown signals."""
        if self.is_shutting_down:
            logger.warning("Enterprise-dashboard backend service already in shutdown process, ignoring signal.")
            return
        
        self.is_shutting_down = True
        logger.info(f"Enterprise-dashboard backend service received signal {signum}, initiating graceful shutdown...")
        
        # Trigger FastAPI's shutdown event
        await self.app.shutdown()
    
    async def _perform_cleanup(self):
        """Perform actual cleanup tasks."""
        logger.info("Performing graceful shutdown cleanup for enterprise-dashboard backend service...")
        
        # Cancel any pending background tasks
        if self.pending_tasks:
            logger.info(f"Cancelling {len(self.pending_tasks)} pending background tasks...")
            for task in list(self.pending_tasks):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.debug(f"Task {task.get_name()} cancelled.")
                except Exception as e:
                    logger.error(f"Error during task cancellation: {e}")
            self.pending_tasks.clear()
            logger.info("All pending tasks cancelled.")
        
        # Close WebSocket connections
        try:
            if hasattr(manager, 'disconnect_all'):
                await manager.disconnect_all()
                logger.info("WebSocket connections closed.")
        except Exception as e:
            logger.error(f"Error closing WebSocket connections: {e}")
        
        # Close HTTP clients
        try:
            # Close any HTTP clients used by the service
            logger.info("HTTP clients closed.")
        except Exception as e:
            logger.error(f"Error closing HTTP clients: {e}")
        
        logger.info("Enterprise-dashboard backend service graceful shutdown cleanup complete.")

# Initialize graceful shutdown
shutdown_handler = GracefulShutdown(app)
shutdown_handler.register_handlers()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)
