"""
ML Security Analytics Service - Modular Version
Stores and analyzes red team test results and model performance metrics
"""

import logging
import asyncio
import signal
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import modular components
from models import HealthResponse
from routes import (
    health_router,
    red_team_router, 
    model_performance_router,
    analytics_router
)
from routes.drift_detection import router as drift_detection_router
from routes.auto_retrain import router as auto_retrain_router
from services.auto_retrain import auto_retrain_service
from utils import setup_logging, get_config
from database.connection import db_manager
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from utils.enhanced_logging import get_analytics_logger, log_analytics_error, log_analytics_event
from utils.http_client import get_http_client, close_http_client
from background_tasks import start_drift_monitoring, stop_drift_monitoring

# Import tracing setup
import sys
import os
sys.path.append('/app')
from tracing_setup import setup_tracing, get_tracer, trace_request, add_span_attributes

# Setup logging
setup_logging(level="INFO", service_name="analytics")
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Create FastAPI app
app = FastAPI(
    title="ML Security Analytics",
    version=config["service"]["version"],
    description="Analytics service for ML Security system"
)

# Setup distributed tracing
setup_tracing("analytics", app)

# Prometheus metrics
ANALYTICS_REQUEST_COUNT = Counter('analytics_requests_total', 'Total number of analytics requests', ['method', 'endpoint', 'status'])
ANALYTICS_REQUEST_DURATION = Histogram('analytics_request_duration_seconds', 'Analytics request duration in seconds', ['method', 'endpoint'])
DRIFT_DETECTION_COUNT = Counter('drift_detections_total', 'Total number of drift detections', ['model_name', 'drift_type'])
DRIFT_SCORE = Gauge('drift_score', 'Current drift score', ['model_name', 'drift_type'])
AUTO_RETRAIN_COUNT = Counter('auto_retrains_total', 'Total number of auto-retrain triggers', ['model_name', 'reason'])

# ML Operation Metrics
MODEL_PROMOTION_DECISIONS = Counter('model_promotion_decisions_total', 'Model promotion decisions', ['model_name', 'decision', 'reason'])
TRAINING_DATA_PIPELINE_ERRORS = Counter('training_data_pipeline_errors_total', 'Data pipeline errors', ['pipeline_stage', 'error_type'])

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(red_team_router)
app.include_router(model_performance_router)
app.include_router(analytics_router)
app.include_router(drift_detection_router, prefix="/drift", tags=["Drift Detection"])
app.include_router(auto_retrain_router, prefix="/auto-retrain", tags=["Auto Retrain"])

# Root endpoint
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="healthy",
        service="analytics",
        timestamp=datetime.now()
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database schema on startup"""
    logger.info("Starting Analytics Service...")
    
    # Initialize HTTP client
    try:
        await get_http_client()  # Initialize shared HTTP client
        logger.info("✅ HTTP client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize HTTP client: {e}")
        raise
    
    # Initialize database connection pool
    try:
        await db_manager.connect()
        logger.info("✅ Analytics database connected successfully")
        
        # Initialize database schema
        try:
            await db_manager.initialize_schema()
            logger.info("✅ Database schema initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing schema: {e}")
            
    except Exception as e:
        logger.error(f"❌ Failed to connect to database: {e}")
        raise
    
    # Start auto-retrain monitoring in background
    try:
        asyncio.create_task(auto_retrain_service.start_monitoring())
        logger.info("🔄 Auto-retrain monitoring started")
    except Exception as e:
        logger.error(f"Failed to start auto-retrain monitoring: {e}")
    
    # Start drift monitoring
    try:
        from services.drift_detection import drift_detector
        from services.email_notifications import email_service
        await start_drift_monitoring(drift_detector, email_service, db_manager)
        logger.info("✅ Drift monitoring started")
    except Exception as e:
        logger.error(f"❌ Failed to start drift monitoring: {e}")
    
    logger.info("Analytics Service ready")

# Graceful shutdown handler
class GracefulShutdown:
    """Handles graceful shutdown of the analytics service"""
    
    def __init__(self, app):
        self.app = app
        self.is_shutting_down = False
        self.shutdown_timeout = 30  # seconds
        self.pending_tasks = set()
        logger.info("GracefulShutdown handler initialized for analytics service.")
    
    def register_handlers(self):
        """Register signal handlers for graceful shutdown."""
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self.shutdown_handler)
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, self.shutdown_handler)
        logger.info("Registered SIGTERM and SIGINT handlers for analytics service.")
        
        @self.app.on_event("shutdown")
        async def _on_shutdown():
            await self._perform_cleanup()
    
    async def shutdown_handler(self, signum, frame):
        """Handle shutdown signals."""
        if self.is_shutting_down:
            logger.warning("Analytics service already in shutdown process, ignoring signal.")
            return
        
        self.is_shutting_down = True
        logger.info(f"Analytics service received signal {signum}, initiating graceful shutdown...")
        
        # Trigger FastAPI's shutdown event
        await self.app.shutdown()
    
    async def _perform_cleanup(self):
        """Perform actual cleanup tasks."""
        logger.info("Performing graceful shutdown cleanup for analytics service...")
        
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
        
        # Close HTTP clients
        try:
            await close_http_client()
            logger.info("HTTP client closed.")
        except Exception as e:
            logger.error(f"Error closing HTTP clients: {e}")
        
        # Stop drift monitoring
        try:
            await stop_drift_monitoring()
            logger.info("Drift monitoring stopped.")
        except Exception as e:
            logger.error(f"Error stopping drift monitoring: {e}")
        
        # Close database connections
        try:
            await db_manager.disconnect()
            logger.info("Database connection pool closed.")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
        
        logger.info("Analytics service graceful shutdown cleanup complete.")

# Initialize graceful shutdown
shutdown_handler = GracefulShutdown(app)
shutdown_handler.register_handlers()

# Note: Startup event is already defined above

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    await shutdown_handler._perform_cleanup()

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=config["service"]["host"], 
        port=config["service"]["port"]
    )
