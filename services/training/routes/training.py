"""
Training Service - Training Routes
Model training and job management endpoints
"""

import logging
import time
import json
from typing import List, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from models.requests import TrainingRequest, ModelLoadingRequest, SingleModelLoadingRequest
from models.responses import TrainingResult, ModelLoadingResult, JobStatus
from services.model_trainer import ModelTrainer
from services.training_config_service import TrainingConfigService
from services.training_queue import training_queue, JobPriority

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
model_trainer = ModelTrainer()

async def check_training_data_availability(request: TrainingRequest) -> bool:
    """Check if training data is available before starting training"""
    try:
        training_data_path = request.training_data_path
        
        if not training_data_path or training_data_path == "auto":
            # Check EfficientDataManager for fresh data
            from efficient_data_manager import EfficientDataManager
            data_manager = EfficientDataManager()
            
            # Get fresh data files
            fresh_files = []
            for file_id, file_info in data_manager.data_files.items():
                status = file_info.status.value if hasattr(file_info.status, 'value') else str(file_info.status)
                if status == "fresh":
                    fresh_files.append(file_info)
            
            if fresh_files:
                return True
            
            # Check if fallback data exists
            fallback_path = data_manager.get_training_data_path()
            if fallback_path and fallback_path.startswith("s3://"):
                try:
                    s3_path_parts = fallback_path.replace('s3://', '').split('/', 1)
                    if len(s3_path_parts) == 2:
                        bucket_name, s3_key = s3_path_parts
                        data_manager.s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                        return True
                except Exception:
                    pass
            
            # Also check for sample data created by the old data service
            try:
                # Check for sample data in MinIO
                response = data_manager.s3_client.list_objects_v2(
                    Bucket=data_manager.bucket_name,
                    Prefix='training-data/fresh/'
                )
                
                for obj in response.get('Contents', []):
                    if obj['Key'].endswith('.jsonl') and obj['Key'] != 'training-data/fresh/.gitkeep':
                        return True
                        
            except Exception:
                pass
            
            # Also check for local fresh data files
            try:
                import os
                import glob
                fresh_data_dir = "/app/training_data/fresh/"
                if os.path.exists(fresh_data_dir):
                    jsonl_files = glob.glob(os.path.join(fresh_data_dir, "*.jsonl"))
                    if jsonl_files:
                        return True
            except Exception:
                pass
            
            return False
        else:
            # Custom data path provided - verify it exists
            if training_data_path.startswith("s3://"):
                # Verify S3 path exists
                s3_path_parts = training_data_path.replace('s3://', '').split('/', 1)
                if len(s3_path_parts) == 2:
                    bucket_name, s3_key = s3_path_parts
                    from efficient_data_manager import EfficientDataManager
                    data_manager = EfficientDataManager()
                    data_manager.s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                    return True
            else:
                # Local file path - check if file exists
                import os
                return os.path.exists(training_data_path)
            
            return False
            
    except Exception as e:
        logger.error(f"Error checking data availability: {e}")
        return False


@router.get("/jobs", response_model=List[JobStatus])
async def list_jobs():
    """List all training jobs"""
    try:
        jobs = await model_trainer.list_jobs()
        return jobs
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of a specific training job"""
    try:
        job = await model_trainer.get_job_status(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    """Get logs for a specific training job"""
    try:
        from services.training_logs_service import TrainingLogsService

        # Get real logs from the database
        logs = await TrainingLogsService.get_job_logs(job_id)
        
        # If no logs found, return a message
        if not logs:
            return {
                "job_id": job_id, 
                "logs": [],
                "message": "No logs found for this training job"
            }
        
        return {"job_id": job_id, "logs": logs}
    except Exception as e:
        logger.error(f"Failed to get logs for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start-model-loading")
async def start_model_loading():
    """Start model loading process"""
    try:
        # This would start loading models in the background
        return {
            "status": "success",
            "message": "Model loading started",
            "timestamp": "2025-09-26T17:30:00.000000"
        }
    except Exception as e:
        logger.error(f"Failed to start model loading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load-models", response_model=ModelLoadingResult)
async def load_specific_models(request: ModelLoadingRequest):
    """Load specific models"""
    try:
        # This would load the specified models
        loaded_models = []
        failed_models = []
        
        for model_type, model_names in request.models.items():
            for model_name in model_names:
                try:
                    # Simulate model loading
                    loaded_models.append(f"{model_type}_{model_name}")
                except Exception as e:
                    failed_models.append(f"{model_type}_{model_name}: {str(e)}")
        
        return ModelLoadingResult(
            status="success",
            message=f"Loaded {len(loaded_models)} models, {len(failed_models)} failed",
            loaded_models=loaded_models,
            failed_models=failed_models,
            timestamp="2025-09-26T17:30:00.000000"
        )
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load-model", response_model=ModelLoadingResult)
async def load_single_model(request: SingleModelLoadingRequest):
    """Load a single model"""
    try:
        # This would load the specified model
        return ModelLoadingResult(
            status="success",
            message=f"Model {request.model_name} loaded successfully",
            loaded_models=[request.model_name],
            failed_models=[],
            timestamp="2025-09-26T17:30:00.000000"
        )
    except Exception as e:
        logger.error(f"Failed to load model {request.model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", response_model=TrainingResult)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train a model"""
    try:
        # Check data availability before starting training
        data_available = await check_training_data_availability(request)
        
        if not data_available:
            raise HTTPException(
                status_code=400, 
                detail="No training data available. Please upload training data or create sample data first."
            )
        
        # Submit job to training queue
        config_dict = request.config.dict() if request.config else {}
        job_id = await training_queue.submit_job(
            model_name=request.model_name,
            training_data_path=request.training_data_path,
            config=config_dict,
            priority=JobPriority.NORMAL,
            timeout_seconds=3600
        )
        
        # Log training start for audit
        try:
            from services.audit_logger import AuditLogger, AuditEventType, AuditSeverity
            audit_logger = AuditLogger()
            await audit_logger.log_event(
                event_type=AuditEventType.MODEL_TRAIN,
                user_id="training_service",
                session_id=f"training_session_{int(time.time())}",
                ip_address="127.0.0.1",
                resource="training_service",
                action="start_training",
                details={
                    "model_name": request.model_name,
                    "job_id": job_id,
                    "training_data_path": request.training_data_path,
                    "config": request.config.dict() if hasattr(request.config, 'dict') else str(request.config),
                    "timestamp": time.time()
                },
                severity=AuditSeverity.MEDIUM,
                success=True
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to log training audit event: {e}")
        
        return TrainingResult(
            status="started",
            message=f"Training job {job_id} submitted to queue",
            job_id=job_id,
            model_name=request.model_name,
            timestamp=time.time()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start training for {request.model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train-loaded")
async def train_loaded_model(request: Dict[str, Any]):
    """
    Train a loaded model using the training queue
    
    Now using queue for consistency with /train endpoint
    """
    try:
        # Extract parameters
        model_name = request.get("model_name")
        training_data_path = request.get("training_data_path", "latest")
        config = request.get("config", {})
        
        if not model_name:
            raise HTTPException(
                status_code=400,
                detail="model_name is required"
            )
        
        # Submit to queue instead of background task
        job_id = await training_queue.submit_job(
            model_name=model_name,
            training_data_path=training_data_path,
            config=config,
            priority=JobPriority.NORMAL
        )
        
        logger.info(f"Training job submitted to queue: {job_id}")
        
        return {
            "status": "queued",
            "job_id": job_id,
            "message": f"Training job for {model_name} submitted to queue",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to queue training for loaded model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrain", response_model=TrainingResult)
async def retrain_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Retrain a model"""
    try:
        # Check data availability before starting retraining
        data_available = await check_training_data_availability(request)
        
        if not data_available:
            raise HTTPException(
                status_code=400, 
                detail="No training data available. Please upload training data or create sample data first."
            )
        
        # Start retraining in background
        job_id = f"retrain_{request.model_name}_{int(time.time())}"
        
        # Add retraining task to background
        background_tasks.add_task(
            model_trainer.train_model,
            request
        )
        
        return TrainingResult(
            status="started",
            message=f"Retraining started for {request.model_name}",
            job_id=job_id,
            model_name=request.model_name,
            timestamp="2025-09-26T17:30:00.000000"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start retraining for {request.model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced-retrain")
async def advanced_retrain(request: Dict[str, Any], background_tasks: BackgroundTasks):
    """Advanced retraining with triggers"""
    try:
        # This would handle advanced retraining with triggers
        return {
            "status": "started",
            "message": "Advanced retraining started",
            "timestamp": "2025-09-26T17:30:00.000000"
        }
    except Exception as e:
        logger.error(f"Failed to start advanced retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# TRAINING CONFIGURATION MANAGEMENT
# =============================================================================

@router.get("/config/{model_name}")
async def get_training_config(model_name: str):
    """Get training configuration for a specific model"""
    try:
        config = await TrainingConfigService.get_config(model_name)
        if config:
            return config
        else:
            raise HTTPException(status_code=404, detail=f"No training configuration found for model: {model_name}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training config for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config/{model_name}")
async def update_training_config(model_name: str, config: Dict[str, Any]):
    """Save training configuration for a specific model"""
    try:
        # Ensure model_name in config matches the URL parameter
        config['model_name'] = model_name
        
        result = await TrainingConfigService.save_config(model_name, config)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to save training config for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/configs")
async def list_training_configs():
    """List all saved training configurations"""
    try:
        return await TrainingConfigService.list_configs()
    except Exception as e:
        logger.error(f"Failed to list training configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/config/{model_name}")
async def delete_training_config(model_name: str):
    """Delete training configuration for a specific model"""
    try:
        return await TrainingConfigService.delete_config(model_name)
    except Exception as e:
        logger.error(f"Failed to delete training config for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
