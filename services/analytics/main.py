"""
ML Security Analytics Service - Modular Version
Stores and analyzes red team test results and model performance metrics
"""

import logging
import asyncio
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
from utils.http_client import get_http_client, close_http_client

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
DRIFT_DETECTION_FAILURES = Counter('drift_detection_failures_total', 'Drift detection errors', ['detection_type', 'error_type'])
MODEL_PROMOTION_DECISIONS = Counter('model_promotion_decisions_total', 'Model promotion decisions', ['model_name', 'decision', 'reason'])
TRAINING_DATA_PIPELINE_ERRORS = Counter('training_data_pipeline_errors_total', 'Data pipeline errors', ['pipeline_stage', 'error_type'])
ML_OPERATION_DURATION = Histogram('ml_operation_duration_seconds', 'ML operation duration', ['operation_type', 'model_name'], buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0])

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
            with open("/app/schema.sql", "r") as f:
                schema_sql = f.read()
            await db_manager.execute_command(schema_sql)
            logger.info("✅ Database schema initialized successfully")
        except FileNotFoundError:
            logger.warning("⚠️ Schema file not found, skipping schema initialization")
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

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down Analytics Service...")
    
    try:
        await db_manager.disconnect()
        logger.info("✅ Analytics database disconnected")
    except Exception as e:
        logger.error(f"Error during database disconnect: {e}")
    
    try:
        await close_http_client()
        logger.info("✅ HTTP client closed")
    except Exception as e:
        logger.error(f"Error during HTTP client cleanup: {e}")

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=config["service"]["host"], 
        port=config["service"]["port"]
    )
