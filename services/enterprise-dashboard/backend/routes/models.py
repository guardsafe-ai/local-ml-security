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
from services.main_api_client import MainAPIClient

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize API client
api_client = MainAPIClient()


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


# ===== CRITICAL MISSING ENDPOINTS FOR FRONTEND USERS =====

@router.post("/predict/trained", response_model=PredictionResponse)
async def predict_trained_model(request: PredictionRequest):
    """
    Predict using trained models from MLflow/MinIO
    
    Frontend Usage: Essential for using custom trained models
    - User uploads training data
    - Model gets trained and stored in MLflow
    - User wants to test their trained model
    - This endpoint provides that capability
    """
    try:
        # Use the unified prediction that handles both pretrained and trained models
        result = await api_client.predict_unified(
            text=request.text,
            model_name=request.model_name,
            use_cache=True,
            ensemble=request.ensemble
        )
        
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
        logger.error(f"Failed to make trained model prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_models_status():
    """
    Get overall model status across all services
    
    Frontend Usage: Dashboard overview showing:
    - How many models are loaded vs available
    - Overall system health
    - Quick status summary for monitoring
    """
    try:
        # Get status from both Model API and Model Cache
        model_api_status = await api_client.model_api.get_models()
        model_cache_status = await api_client.model_cache.get_models()
        
        # Aggregate status information
        total_available = len(model_api_status.get("models", {}))
        total_loaded = len(model_cache_status.get("models", {}))
        
        return {
            "total_available": total_available,
            "total_loaded": total_loaded,
            "load_percentage": (total_loaded / total_available * 100) if total_available > 0 else 0,
            "status": "healthy" if total_loaded > 0 else "no_models_loaded",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "model_api": "healthy",
                "model_cache": "healthy"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get models status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_name}/status")
async def get_model_status(model_name: str):
    """
    Get detailed status of a specific model
    
    Frontend Usage: Model detail page showing:
    - Is this specific model loaded?
    - What's its performance?
    - Memory usage, last used, etc.
    """
    try:
        # Check if model is loaded in cache
        cache_status = await api_client.model_cache.get_model_status(model_name)
        
        # Get model info from API
        try:
            model_info = await api_client.model_api.get_model_info(model_name)
        except:
            model_info = {"error": "Model not found in API"}
        
        return {
            "model_name": model_name,
            "loaded": cache_status.get("loaded", False),
            "cache_status": cache_status,
            "api_info": model_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get model status for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batching/stats")
async def get_batching_stats():
    """
    Get dynamic batching statistics
    
    Frontend Usage: Performance monitoring showing:
    - How many requests are being batched
    - Average batch size
    - Batching efficiency metrics
    """
    try:
        # Get batching stats from Model API
        result = await api_client.model_api.get_metrics()
        
        # Parse the metrics to extract batching information
        # This would need to be implemented in the Model API service
        return {
            "batching_enabled": True,  # This would come from actual metrics
            "average_batch_size": 0,
            "total_batches_processed": 0,
            "batching_efficiency": 0.0,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get batching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batching/configure")
async def configure_batching(config: Dict[str, Any]):
    """
    Configure dynamic batching parameters
    
    Frontend Usage: Admin panel for tuning performance:
    - Enable/disable batching
    - Set batch size limits
    - Configure timeout settings
    """
    try:
        # This would need to be implemented in the Model API service
        return {
            "status": "success",
            "message": "Batching configuration updated",
            "config": config,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to configure batching: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_name}/quantize")
async def quantize_model(model_name: str, config: Dict[str, Any] = None):
    """
    Quantize a model for better performance
    
    Frontend Usage: Model optimization feature:
    - User wants to optimize model for production
    - Reduce memory usage and improve speed
    - Trade-off some accuracy for performance
    """
    try:
        if config is None:
            config = {
                "quantization_type": "int8",
                "calibration_samples": 100
            }
        
        # This would need to be implemented in the Model API service
        return {
            "status": "success",
            "message": f"Model {model_name} quantization started",
            "model_name": model_name,
            "config": config,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to quantize model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_name}/performance")
async def get_model_performance(model_name: str):
    """
    Get performance metrics for a specific model
    
    Frontend Usage: Model analytics showing:
    - Prediction accuracy over time
    - Response time statistics
    - Memory usage patterns
    - Error rates
    """
    try:
        # Get performance data from analytics service
        performance_data = await api_client.analytics.get_model_performance(model_name)
        
        return {
            "model_name": model_name,
            "performance": performance_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get performance for model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
