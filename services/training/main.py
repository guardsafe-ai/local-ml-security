"""
Training Service - Modular Version
Model training service with modular architecture
"""

import logging
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import mlflow

# Import modular components
from models.responses import HealthResponse
from routes import health_router, models_router, training_router, data_router
from routes.efficient_data import router as efficient_data_router
from routes.data_augmentation import router as data_augmentation_router
from routes.training_queue import router as training_queue_router
from services.model_trainer import ModelTrainer
from services.mlflow_service import MLflowService
# Removed separate DatabaseService - using main db_manager for all database operations
from services.training_queue import training_queue
from utils.logging import setup_logging
from utils.config import get_config
from database.async_connection import db_manager
from efficient_data_manager import EfficientDataManager
import httpx
import asyncio
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Import tracing setup
import sys
import os
sys.path.append('/app')
from tracing_setup import setup_tracing, get_tracer, trace_request, add_span_attributes

# Import audit logging
# Audit logging is handled by individual services

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize core services
config = get_config()
model_trainer = ModelTrainer()
mlflow_service = MLflowService()
data_manager = EfficientDataManager()

# Audit logging is handled by individual services

# External service URLs
BUSINESS_METRICS_URL = "http://business-metrics:8004"
DATA_PRIVACY_URL = "http://data-privacy:8005"

async def initialize_external_services():
    """Initialize connections to external services"""
    try:
        # Test business metrics service
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BUSINESS_METRICS_URL}/health", timeout=5.0)
            if response.status_code == 200:
                logger.info("✅ Business Metrics service connected")
            else:
                logger.warning("⚠️ Business Metrics service not responding")
    except Exception as e:
        logger.warning(f"⚠️ Could not connect to Business Metrics service: {e}")
    
    try:
        # Test data privacy service
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{DATA_PRIVACY_URL}/health", timeout=5.0)
            if response.status_code == 200:
                logger.info("✅ Data Privacy service connected")
            else:
                logger.warning("⚠️ Data Privacy service not responding")
    except Exception as e:
        logger.warning(f"⚠️ Could not connect to Data Privacy service: {e}")

async def send_business_metric(metric_name: str, value: float, tags: dict = None, metadata: dict = None):
    """Send metric to business metrics service"""
    try:
        metric_data = {
            "metric_name": metric_name,
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "tags": tags or {},
            "metadata": metadata or {}
        }
        
        async with httpx.AsyncClient() as client:
            await client.post(f"{BUSINESS_METRICS_URL}/metrics", json=metric_data, timeout=5.0)
    except Exception as e:
        logger.warning(f"⚠️ Failed to send business metric {metric_name}: {e}")

async def classify_training_data(data: dict, data_id: str):
    """Classify training data for privacy compliance"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DATA_PRIVACY_URL}/classify",
                params={"data_id": data_id},
                json=data,
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"⚠️ Data classification failed: {response.status_code}")
                return None
    except Exception as e:
        logger.warning(f"⚠️ Failed to classify data {data_id}: {e}")
        return None

app = FastAPI(
    title="ML Security Training Service (Modular)",
    version="1.0.0",
    description="Modular training service for ML security models"
)

# Setup distributed tracing
setup_tracing("training", app)

# Prometheus metrics
TRAINING_REQUEST_COUNT = Counter('training_requests_total', 'Total number of training requests', ['method', 'endpoint', 'status'])
TRAINING_REQUEST_DURATION = Histogram('training_request_duration_seconds', 'Training request duration in seconds', ['method', 'endpoint'])
TRAINING_JOB_COUNT = Counter('training_jobs_total', 'Total number of training jobs', ['model_name', 'status'])
TRAINING_DURATION = Histogram('training_duration_seconds', 'Training duration in seconds', ['model_name'])
ACTIVE_TRAINING_JOBS = Gauge('training_active_jobs', 'Number of active training jobs')
QUEUE_SIZE = Gauge('training_queue_size', 'Number of jobs in training queue')

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Type", "Content-Length"],
)

# Include routers
app.include_router(health_router, tags=["Health"])
app.include_router(models_router, prefix="/models", tags=["Models"])
app.include_router(training_router, prefix="/training", tags=["Training"])
app.include_router(training_queue_router, prefix="/training", tags=["Training Queue"])
app.include_router(data_router, prefix="/data", tags=["Data"])
app.include_router(efficient_data_router, prefix="/data/efficient", tags=["Efficient Data"])
app.include_router(data_augmentation_router, prefix="/data/augmentation", tags=["Data Augmentation"])

# Prometheus metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    from fastapi import Response
    
    # Basic metrics for training service
    metrics_data = f"""# HELP training_service_up Training service status
# TYPE training_service_up gauge
training_service_up 1

# HELP training_jobs_total Total number of training jobs
# TYPE training_jobs_total counter
training_jobs_total 0

# HELP training_jobs_active Active training jobs
# TYPE training_jobs_active gauge
training_jobs_active 0
"""
    
    return Response(content=metrics_data, media_type="text/plain")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Training service starting up...")
    try:
        # Initialize database
        await db_manager.connect()
        await db_manager.initialize_schema()
        logger.info("Database initialized successfully")
        
        # Training configuration now handled by main database manager
        logger.info("Training configuration database service initialized successfully")
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
        logger.info("MLflow initialized successfully")
        
        # Initialize training queue
        await training_queue.initialize()
        logger.info("Training queue initialized")
        
        # Initialize external service integrations
        await initialize_external_services()
        logger.info("External service integrations initialized")
        
        logger.info("Training service startup completed")
    except Exception as e:
        logger.error(f"Failed to initialize training service: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Training service shutting down...")
    try:
        # Close training queue
        await training_queue.close()
        logger.info("✅ Training queue closed")
        
        await db_manager.disconnect()
        logger.info("✅ Training service shutdown completed")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
