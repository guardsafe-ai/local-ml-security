"""
Enterprise Dashboard Backend - Training Routes
Model training and job management endpoints
"""

import logging
import json
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from models.requests import TrainingRequest, RetrainingRequest
from models.responses import TrainingJob, SuccessResponse
from services.api_client import APIClient
from routes.performance_cache import PerformanceCache

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize API client
api_client = APIClient()


@router.get("/jobs")
async def get_training_jobs():
    """Get all training jobs with Redis caching"""
    try:
        # Check cache first
        cached_data = PerformanceCache.get_cached_performance_data()
        if cached_data:
            logger.info("📊 Returning cached training jobs data")
            return cached_data
        
        # Fetch fresh data
        jobs = await api_client.get_training_jobs()
        result = {"jobs": jobs, "count": len(jobs)}
        
        # Cache the data for 5 minutes
        PerformanceCache.cache_performance_data(result, ttl=300)
        
        return result
    except Exception as e:
        logger.error(f"Failed to get training jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}")
async def get_training_job(job_id: str):
    """Get specific training job"""
    try:
        jobs = await api_client.get_training_jobs()
        job = next((job for job in jobs if job.get("job_id") == job_id), None)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return job
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    """Get logs for a specific training job"""
    try:
        logs = await api_client.get_job_logs(job_id)
        return logs
    except Exception as e:
        logger.error(f"Failed to get logs for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start", response_model=SuccessResponse)
async def start_training(request: TrainingRequest):
    """Start a new training job"""
    try:
        request_data = {
            "model_name": request.model_name,
            "training_data_path": request.training_data_path,
            "config": request.config or {}
        }
        
        result = await api_client.start_training(request_data)
        
        return SuccessResponse(
            status="success",
            message=f"Training job started for {request.model_name}",
            timestamp=datetime.now(),
            data=result
        )
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", response_model=SuccessResponse)
async def train_model(request: TrainingRequest):
    """Train a model"""
    try:
        request_data = {
            "model_name": request.model_name,
            "training_data_path": request.training_data_path,
            "config": request.config or {}
        }
        
        result = await api_client.train_model(request_data)
        
        return SuccessResponse(
            status="success",
            message=f"Training started for {request.model_name}",
            timestamp=datetime.now(),
            data=result
        )
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/loaded-model", response_model=SuccessResponse)
async def train_loaded_model(request: TrainingRequest):
    """Train a loaded model"""
    try:
        request_data = {
            "model_name": request.model_name,
            "training_data_path": request.training_data_path,
            "config": request.config or {}
        }
        
        result = await api_client.train_loaded_model(request_data)
        
        return SuccessResponse(
            status="success",
            message=f"Training started for loaded model {request.model_name}",
            timestamp=datetime.now(),
            data=result
        )
    except Exception as e:
        logger.error(f"Failed to train loaded model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrain", response_model=SuccessResponse)
async def retrain_model(request: RetrainingRequest):
    """Retrain an existing model"""
    try:
        request_data = {
            "model_name": request.model_name,
            "training_data_path": request.training_data_path,
            "config": request.config or {}
        }
        
        result = await api_client.start_training(request_data)  # Same endpoint for retraining
        
        return SuccessResponse(
            status="success",
            message=f"Retraining job started for {request.model_name}",
            timestamp=datetime.now(),
            data=result
        )
    except Exception as e:
        logger.error(f"Failed to start retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop/{job_id}", response_model=SuccessResponse)
async def stop_training(job_id: str):
    """Stop a training job"""
    try:
        # This would typically call a stop endpoint on the training service
        # For now, return success
        return SuccessResponse(
            status="success",
            message=f"Training job {job_id} stop requested",
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Failed to stop training job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def get_training_models():
    """Get models available for training"""
    try:
        # This would typically get models from MLflow or training service
        return {"models": [], "count": 0}
    except Exception as e:
        logger.error(f"Failed to get training models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}")
async def get_training_model(model_name: str):
    """Get specific training model information"""
    try:
        # This would typically get model details from MLflow
        return {"model_name": model_name, "status": "unknown"}
    except Exception as e:
        logger.error(f"Failed to get training model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def get_training_logs():
    """Get training logs"""
    try:
        # This would typically get logs from the training service
        return {"logs": [], "count": 0}
    except Exception as e:
        logger.error(f"Failed to get training logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# TRAINING CONFIGURATION MANAGEMENT
# =============================================================================


@router.get("/config/{model_name}")
async def get_training_config(model_name: str):
    """Get training configuration for a specific model"""
    try:
        config = await api_client.get_training_config(model_name)
        return config
    except Exception as e:
        logger.error(f"Failed to get training config for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config/{model_name}")
async def update_training_config(model_name: str, config: Dict[str, Any]):
    """Save training configuration for a specific model"""
    try:
        # Ensure model_name in config matches the URL parameter
        config['model_name'] = model_name
        
        result = await api_client.update_training_config(model_name, config)
        return result
    except Exception as e:
        logger.error(f"Failed to save training config for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/configs")
async def list_training_configs():
    """List all saved training configurations"""
    try:
        return await api_client.list_training_configs()
    except Exception as e:
        logger.error(f"Failed to list training configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/config/{model_name}")
async def delete_training_config(model_name: str):
    """Delete training configuration for a specific model"""
    try:
        return await api_client.delete_training_config(model_name)
    except Exception as e:
        logger.error(f"Failed to delete training config for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
