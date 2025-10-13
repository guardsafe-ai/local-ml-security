"""
Training Service - Model Routes
Model management and information endpoints
"""

import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from models.responses import ModelListResponse, ModelRegistryResponse, ModelVersionsResponse, ModelInfo
from services.model_trainer import ModelTrainer
from services.mlflow_service import MLflowService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
model_trainer = ModelTrainer()
mlflow_service = MLflowService()


@router.get("/models", response_model=ModelListResponse)
async def get_available_models():
    """Get available models for training"""
    try:
        models = await model_trainer.get_available_models()
        available_models = list(models.keys())
        
        # Get MLflow models
        mlflow_models = await mlflow_service.get_model_registry()
        mlflow_model_names = [model["name"] for model in mlflow_models]
        
        return ModelListResponse(
            models=models,
            available_models=available_models,
            mlflow_models=mlflow_model_names
        )
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-registry", response_model=ModelRegistryResponse)
async def get_model_registry():
    """Get MLflow model registry"""
    try:
        models = await mlflow_service.get_model_registry()
        return ModelRegistryResponse(
            models=models,
            total_models=len(models)
        )
    except Exception as e:
        logger.error(f"Failed to get model registry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest-models")
async def get_latest_models():
    """Get latest versions of all models"""
    try:
        models = await mlflow_service.get_latest_models()
        return {"models": models, "count": len(models)}
    except Exception as e:
        logger.error(f"Failed to get latest models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/best-models")
async def get_best_models():
    """Get best performing models"""
    try:
        models = await mlflow_service.get_best_models()
        return {"models": models, "count": len(models)}
    except Exception as e:
        logger.error(f"Failed to get best models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}/versions", response_model=ModelVersionsResponse)
async def get_model_versions(model_name: str):
    """Get versions for a specific model"""
    try:
        versions = await mlflow_service.get_model_versions(model_name)
        latest_version = versions[0]["version"] if versions else "none"
        
        return ModelVersionsResponse(
            model_name=model_name,
            versions=versions,
            latest_version=latest_version
        )
    except Exception as e:
        logger.error(f"Failed to get versions for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}/info")
async def get_model_info(model_name: str):
    """Get detailed information about a model"""
    try:
        info = await mlflow_service.get_model_info(model_name)
        if not info:
            raise HTTPException(status_code=404, detail="Model not found")
        return info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}/status")
async def get_training_status(model_name: str):
    """Get training status for a model"""
    try:
        # This would check if model is currently being trained
        return {
            "model_name": model_name,
            "status": "available",
            "message": "Model is available for training"
        }
    except Exception as e:
        logger.error(f"Failed to get training status for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-loading-status")
async def get_model_loading_status():
    """Get model loading status"""
    try:
        return {
            "status": "ready",
            "message": "Model loading service is ready"
        }
    except Exception as e:
        logger.error(f"Failed to get model loading status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/promote")
async def promote_model_to_production(model_name: str, version: str = None):
    """Promote model from Staging to Production"""
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # If no version specified, get the latest Staging version
        if not version:
            versions = client.search_model_versions(f"name='{model_name}'")
            staging_versions = [v for v in versions if v.current_stage == 'Staging']
            if not staging_versions:
                raise HTTPException(status_code=404, detail=f"No Staging version found for model {model_name}")
            version = staging_versions[0].version
        
        # Archive existing production versions
        all_versions = client.search_model_versions(f"name='{model_name}'")
        prod_versions = [v for v in all_versions if v.current_stage == 'Production']
        for prod_version in prod_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=prod_version.version,
                stage="Archived"
            )
        
        # Promote to Production
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        
        return {
            "status": "success",
            "message": f"Model {model_name} v{version} promoted to Production",
            "model_name": model_name,
            "version": version,
            "stage": "Production"
        }
    except Exception as e:
        logger.error(f"Failed to promote model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/rollback")
async def rollback_model(model_name: str, version: str):
    """Rollback to a previous Production version"""
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Archive current production
        all_versions = client.search_model_versions(f"name='{model_name}'")
        prod_versions = [v for v in all_versions if v.current_stage == 'Production']
        for prod_version in prod_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=prod_version.version,
                stage="Archived"
            )
        
        # Promote specified version to Production
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
        return {
            "status": "success",
            "message": f"Model {model_name} rolled back to v{version}",
            "model_name": model_name,
            "version": version,
            "stage": "Production"
        }
    except Exception as e:
        logger.error(f"Failed to rollback model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}/stages")
async def get_model_stages(model_name: str):
    """Get all stages and versions for a model"""
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        versions = client.search_model_versions(f"name='{model_name}'")
        
        stages = {}
        for version in versions:
            stage = version.current_stage
            if stage not in stages:
                stages[stage] = []
            stages[stage].append({
                "version": version.version,
                "creation_timestamp": version.creation_timestamp,
                "description": version.description
            })
        
        return {
            "model_name": model_name,
            "stages": stages,
            "total_versions": len(versions)
        }
    except Exception as e:
        logger.error(f"Failed to get model stages for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
