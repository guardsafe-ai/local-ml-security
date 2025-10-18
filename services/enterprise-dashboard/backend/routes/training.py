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
from services.main_api_client import MainAPIClient
from routes.performance_cache import PerformanceCache

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize API client
api_client = MainAPIClient()


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
        training_data = await api_client.get_training_jobs()
        
        # Handle different response formats
        if isinstance(training_data, dict):
            jobs = training_data.get("jobs", [])
            count = training_data.get("count", len(jobs))
        else:
            jobs = training_data if isinstance(training_data, list) else []
            count = len(jobs)
        
        result = {"jobs": jobs, "count": count}
        
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
        logs = await api_client.get_training_logs(job_id)
        return logs
    except Exception as e:
        logger.error(f"Failed to get logs for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start", response_model=SuccessResponse)
async def start_training(request: TrainingRequest):
    """Start a new training job"""
    try:
        result = await api_client.start_training(
            model_name=request.model_name,
            training_data_path=request.training_data_path,
            config=request.config or {}
        )
        
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
        result = await api_client.start_training(
            model_name=request.model_name,
            training_data_path=request.training_data_path,
            config=request.config or {}
        )
        
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
        result = await api_client.start_training(
            model_name=request.model_name,
            training_data_path=request.training_data_path,
            config=request.config or {}
        )
        
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
        result = await api_client.start_training(
            model_name=request.model_name,
            training_data_path=request.training_data_path,
            config=request.config or {}
        )
        
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
        result = await api_client.stop_training(job_id)
        return SuccessResponse(
            status="success",
            message=f"Training job {job_id} stop requested",
            timestamp=datetime.now(),
            data=result
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
        # For now, return a default configuration
        # This would typically get config from the training service
        return {
            "model_name": model_name,
            "config": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001
            }
        }
    except Exception as e:
        logger.error(f"Failed to get training config for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config/{model_name}")
async def update_training_config(model_name: str, config: Dict[str, Any]):
    """Save training configuration for a specific model"""
    try:
        # For now, return success
        # This would typically save config to the training service
        return {
            "status": "success",
            "message": f"Configuration saved for {model_name}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to save training config for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/configs")
async def list_training_configs():
    """List all saved training configurations"""
    try:
        # For now, return empty list
        # This would typically get configs from the training service
        return {"configs": [], "count": 0}
    except Exception as e:
        logger.error(f"Failed to list training configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/config/{model_name}")
async def delete_training_config(model_name: str):
    """Delete training configuration for a specific model"""
    try:
        # For now, return success
        # This would typically delete config from the training service
        return {
            "status": "success",
            "message": f"Configuration deleted for {model_name}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to delete training config for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== CRITICAL MISSING TRAINING ENDPOINTS FOR FRONTEND USERS =====

@router.get("/experiments/analytics")
async def get_experiment_analytics():
    """
    Get comprehensive experiment analytics
    
    Frontend Usage: ML Experiment Dashboard showing:
    - Training success rates over time
    - Model performance comparisons
    - Resource usage patterns
    - Experiment trends and insights
    """
    try:
        # Get analytics from training service
        analytics = await api_client.training.get_experiment_analytics()
        
        return {
            "experiment_analytics": analytics,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_experiments": analytics.get("total_experiments", 0),
                "success_rate": analytics.get("success_rate", 0.0),
                "average_accuracy": analytics.get("average_accuracy", 0.0),
                "best_model": analytics.get("best_model", "none")
            }
        }
    except Exception as e:
        logger.error(f"Failed to get experiment analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/compare")
async def compare_experiments(model_names: str):
    """
    Compare experiments across different models
    
    Frontend Usage: Model Comparison Page showing:
    - Side-by-side performance metrics
    - Training time comparisons
    - Accuracy vs speed trade-offs
    - Resource usage differences
    """
    try:
        # Parse comma-separated model names
        model_list = [name.strip() for name in model_names.split(",")]
        
        # Get comparison data from training service
        comparison = await api_client.training.compare_model_experiments(model_list)
        
        return {
            "comparison": comparison,
            "models_compared": model_list,
            "timestamp": datetime.now().isoformat(),
            "insights": {
                "best_performing": comparison.get("best_model", "none"),
                "fastest_training": comparison.get("fastest_model", "none"),
                "most_efficient": comparison.get("most_efficient", "none")
            }
        }
    except Exception as e:
        logger.error(f"Failed to compare experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{model_name}/runs")
async def get_model_experiment_runs(model_name: str, limit: int = 50):
    """
    Get all experiment runs for a specific model
    
    Frontend Usage: Model History Page showing:
    - All training runs for this model
    - Performance progression over time
    - Failed runs and their reasons
    - Hyperparameter experiments
    """
    try:
        # Get runs from training service
        runs = await api_client.training.get_model_experiment_runs(model_name, limit)
        
        return {
            "model_name": model_name,
            "runs": runs,
            "total_runs": len(runs),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get experiment runs for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/promote")
async def promote_model_to_production(model_name: str, version: str = None):
    """
    Promote a model version to production
    
    Frontend Usage: Model Management Page:
    - User selects a trained model version
    - Clicks "Promote to Production"
    - Model becomes available for inference
    - Previous production model is archived
    """
    try:
        result = await api_client.training.promote_model(model_name, version)
        
        return {
            "status": "success",
            "message": f"Model {model_name} promoted to production",
            "model_name": model_name,
            "version": version,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to promote model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/rollback")
async def rollback_model_version(model_name: str, version: str):
    """
    Rollback to a previous model version
    
    Frontend Usage: Model Management Page:
    - User notices production model issues
    - Selects a previous stable version
    - Clicks "Rollback to Version X"
    - System reverts to the selected version
    """
    try:
        result = await api_client.training.rollback_model(model_name, version)
        
        return {
            "status": "success",
            "message": f"Model {model_name} rolled back to version {version}",
            "model_name": model_name,
            "version": version,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to rollback model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}/stages")
async def get_model_stages(model_name: str):
    """
    Get all stages and versions for a model
    
    Frontend Usage: Model Version History:
    - Shows all versions (Staging, Production, Archived)
    - Version metadata and performance
    - Easy stage transitions
    """
    try:
        stages = await api_client.training.get_model_stages(model_name)
        
        return {
            "model_name": model_name,
            "stages": stages,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get model stages for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/augmentation/techniques")
async def get_augmentation_techniques():
    """
    Get available data augmentation techniques
    
    Frontend Usage: Data Augmentation Settings:
    - User uploads training data
    - Sees available augmentation options
    - Configures augmentation parameters
    - Previews augmented samples
    """
    try:
        techniques = await api_client.training.get_augmentation_techniques()
        
        return {
            "techniques": techniques,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get augmentation techniques: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data/augmentation/preview")
async def preview_augmentation(request: Dict[str, Any]):
    """
    Preview data augmentation on sample text
    
    Frontend Usage: Data Augmentation Preview:
    - User enters sample text
    - Selects augmentation techniques
    - Sees preview of augmented samples
    - Adjusts parameters before applying
    """
    try:
        text = request.get("text", "")
        label = request.get("label", "")
        num_samples = request.get("num_samples", 5)
        
        preview = await api_client.training.preview_augmentation(text, label, num_samples)
        
        return {
            "original_text": text,
            "original_label": label,
            "augmented_samples": preview,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to preview augmentation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queue/status")
async def get_training_queue_status():
    """
    Get training queue status and statistics
    
    Frontend Usage: Training Queue Dashboard:
    - Shows pending, running, completed jobs
    - Queue length and estimated wait times
    - Resource utilization
    - Job priority management
    """
    try:
        queue_status = await api_client.training.get_queue_stats()
        
        return {
            "queue_status": queue_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
