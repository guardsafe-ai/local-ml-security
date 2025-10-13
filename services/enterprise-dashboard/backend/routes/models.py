"""
Enterprise Dashboard Backend - Model Management Routes
Model loading, unloading, and prediction endpoints
"""

import logging
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from models.requests import PredictionRequest, BatchPredictionRequest, ModelLoadRequest, ModelUnloadRequest, ModelReloadRequest
from models.responses import PredictionResponse, BatchPredictionResponse, ModelInfo, SuccessResponse
from services.api_client import APIClient

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize API client
api_client = APIClient()


@router.get("/")
async def get_models():
    """Get all models (alias for /available)"""
    try:
        return await api_client.get_available_models()
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available")
async def get_available_models():
    """Get all available models"""
    try:
        return await api_client.get_available_models()
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registry")
async def get_model_registry():
    """Get model registry information"""
    try:
        return await api_client.get_model_registry()
    except Exception as e:
        logger.error(f"Failed to get model registry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest")
async def get_latest_models():
    """Get latest versions of all models"""
    try:
        return await api_client.get_latest_models()
    except Exception as e:
        logger.error(f"Failed to get latest models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/best")
async def get_best_models():
    """Get best performing models"""
    try:
        return await api_client.get_best_models()
    except Exception as e:
        logger.error(f"Failed to get best models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load", response_model=SuccessResponse)
async def load_model(request: ModelLoadRequest):
    """Load a model"""
    try:
        result = await api_client.load_model(request.model_name)
        return SuccessResponse(
            status="success",
            message=f"Model {request.model_name} loaded successfully",
            timestamp=datetime.now(),
            data=result
        )
    except Exception as e:
        logger.error(f"Failed to load model {request.model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unload", response_model=SuccessResponse)
async def unload_model(request: ModelUnloadRequest):
    """Unload a model"""
    try:
        result = await api_client.unload_model(request.model_name)
        return SuccessResponse(
            status="success",
            message=f"Model {request.model_name} unloaded successfully",
            timestamp=datetime.now(),
            data=result
        )
    except Exception as e:
        logger.error(f"Failed to unload model {request.model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload", response_model=SuccessResponse)
async def reload_model(request: ModelReloadRequest):
    """Reload a model"""
    try:
        # First unload, then load
        unload_result = await api_client.unload_model(request.model_name)
        load_result = await api_client.load_model(request.model_name)
        
        return SuccessResponse(
            status="success",
            message=f"Model {request.model_name} reloaded successfully",
            timestamp=datetime.now(),
            data={"unload": unload_result, "load": load_result}
        )
    except Exception as e:
        logger.error(f"Failed to reload model {request.model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=PredictionResponse)
async def predict_model(request: PredictionRequest):
    """Make a model prediction"""
    try:
        result = await api_client.predict_model({
            "text": request.text,
            "model_name": request.model_name,
            "ensemble": request.ensemble
        })
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return PredictionResponse(
            text=request.text,
            prediction=result.get("prediction", "unknown"),
            confidence=result.get("confidence", 0.0),
            probabilities=result.get("probabilities", {}),
            model_name=result.get("model_name", request.model_name or "default"),
            processing_time_ms=result.get("processing_time_ms", 0.0),
            timestamp=datetime.now()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to make prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions"""
    try:
        predictions = []
        total_processing_time = 0.0
        
        for text in request.texts:
            result = await api_client.predict_model({
                "text": text,
                "model_name": request.model_name,
                "ensemble": request.ensemble
            })
            
            if "error" not in result:
                prediction = PredictionResponse(
                    text=text,
                    prediction=result.get("prediction", "unknown"),
                    confidence=result.get("confidence", 0.0),
                    probabilities=result.get("probabilities", {}),
                    model_name=result.get("model_name", request.model_name or "default"),
                    processing_time_ms=result.get("processing_time_ms", 0.0),
                    timestamp=datetime.now()
                )
                predictions.append(prediction)
                total_processing_time += result.get("processing_time_ms", 0.0)
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processing_time_ms=total_processing_time,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Failed to make batch predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info/{model_name}")
async def get_model_info(model_name: str):
    """Get detailed information about a specific model"""
    try:
        models_data = await api_client.get_available_models()
        models = models_data.get("models", {})
        
        if model_name in models:
            return models[model_name]
        else:
            raise HTTPException(status_code=404, detail="Model not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
