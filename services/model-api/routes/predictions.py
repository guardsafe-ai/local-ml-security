"""
Prediction Routes
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from models.requests import PredictionRequest
from models.responses import PredictionResponse

router = APIRouter(prefix="/predict", tags=["predictions"])

# These will be injected with services
prediction_service = None
cache_service = None


@router.post("/", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction on text"""
    if not prediction_service:
        raise HTTPException(status_code=500, detail="Prediction service not initialized")
    
    try:
        # Check cache first if available
        if cache_service and request.models and len(request.models) == 1:
            cached = cache_service.get_cached_prediction(request.text, request.models[0])
            if cached:
                return PredictionResponse(**cached)
        
        # Make prediction
        result = prediction_service.predict(request)
        
        # Cache result if single model
        if cache_service and request.models and len(request.models) == 1:
            cache_service.cache_prediction(request.text, request.models[0], result.dict())
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def predict_batch(texts: List[str], models: Optional[List[str]] = None, ensemble: bool = True):
    """Make batch predictions"""
    if not prediction_service:
        raise HTTPException(status_code=500, detail="Prediction service not initialized")
    
    try:
        results = []
        for text in texts:
            request = PredictionRequest(
                text=text,
                models=models,
                ensemble=ensemble
            )
            result = prediction_service.predict(request)
            results.append(result.dict())
        
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
