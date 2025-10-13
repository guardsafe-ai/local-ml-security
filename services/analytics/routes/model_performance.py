"""
Model Performance Routes
"""

from fastapi import APIRouter, HTTPException
from models.requests import ModelPerformance
from models.responses import SuccessResponse
from services.model_performance_service import ModelPerformanceService

router = APIRouter(prefix="/model", tags=["model-performance"])

# Initialize service
model_performance_service = ModelPerformanceService()


@router.post("/performance", response_model=SuccessResponse)
async def store_model_performance(performance: ModelPerformance):
    """Store model performance metrics"""
    try:
        return model_performance_service.store_performance(performance)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
