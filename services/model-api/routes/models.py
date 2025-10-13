"""
Model Management Routes
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from models.responses import ModelsResponse, ModelInfo
from models.requests import LoadModelRequest, UnloadModelRequest

router = APIRouter(prefix="/models", tags=["models"])

# These will be injected with services
model_manager = None


@router.get("/", response_model=ModelsResponse)
async def list_models():
    """List all available models"""
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    try:
        models_info = model_manager.get_models_info()
        available_models = model_manager.get_available_models()
        mlflow_models = model_manager.get_mlflow_models()
        
        return ModelsResponse(
            models=models_info,
            available_models=available_models,
            mlflow_models=mlflow_models
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get information about a specific model"""
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    try:
        info = model_manager.get_model_info(model_name)
        if not info:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        return ModelInfo(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_model(request: LoadModelRequest):
    """Load a model into memory"""
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    try:
        success = model_manager.load_model(request.model_name, request.version)
        if success:
            return {"message": f"Model {request.model_name} loaded successfully"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to load model {request.model_name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unload")
async def unload_model(request: UnloadModelRequest):
    """Unload a model from memory"""
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    try:
        success = model_manager.unload_model(request.model_name)
        if success:
            return {"message": f"Model {request.model_name} unloaded successfully"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to unload model {request.model_name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-load")
async def batch_load_models(model_names: List[str]):
    """Batch load multiple models concurrently"""
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    try:
        results = await model_manager.batch_load_models(model_names)
        success_count = sum(results.values())
        
        return {
            "message": f"Batch load completed: {success_count}/{len(model_names)} models loaded",
            "results": results,
            "success_count": success_count,
            "total_count": len(model_names)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/warm-cache/{model_name}")
async def warm_model_cache(model_name: str):
    """Warm up the cache for a specific model"""
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    try:
        await model_manager._warm_cache(model_name)
        return {"message": f"Cache warmed for model {model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preload-status")
async def get_preload_status():
    """Get status of model preloading tasks"""
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    try:
        preload_tasks = {}
        for model_name, task in model_manager._preload_tasks.items():
            preload_tasks[model_name] = {
                "status": "running" if not task.done() else "completed",
                "done": task.done(),
                "cancelled": task.cancelled()
            }
        
        return {
            "preload_tasks": preload_tasks,
            "priority_models": model_manager._preload_priority_models,
            "cache_warming_enabled": model_manager._cache_warming_enabled
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
