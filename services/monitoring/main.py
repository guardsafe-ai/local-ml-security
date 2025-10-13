"""
Monitoring Service - Modular Main
Modularized monitoring service with clean architecture
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import modular components
from routes import api_router
from utils.logging import setup_logging
from utils.config import get_config

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Monitoring Service starting up...")
    
    # Initialize any required services
    try:
        logger.info("Monitoring service initialized successfully")
    except Exception as e:
        logger.error(f"Monitoring service initialization failed: {e}")
        raise
    
    yield
    
    logger.info("Monitoring Service shutting down...")
    
    # Cleanup
    try:
        logger.info("Monitoring service cleanup completed")
    except Exception as e:
        logger.error(f"Monitoring service cleanup failed: {e}")


# Create FastAPI application
app = FastAPI(
    title="ML Security Monitoring Service (Modular)",
    version="1.0.0",
    description="Real-time monitoring and visualization service for ML Security platform",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router, prefix="", tags=["monitoring"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "monitoring",
        "version": "1.0.0",
        "status": "running",
        "description": "Real-time monitoring and visualization service"
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config["host"],
        port=config["port"],
        log_level=config["log_level"].lower()
    )
