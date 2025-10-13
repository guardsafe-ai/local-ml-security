"""
Tracing Service - Modular Main
Distributed tracing service for ML Security platform
"""

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="ML Security Tracing Service (Modular)",
    version="1.0.0",
    description="Distributed tracing service for ML Security platform"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "tracing",
        "version": "1.0.0",
        "status": "running",
        "description": "Distributed tracing service for ML Security platform"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "tracing",
        "message": "Tracing service is running"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8009)
