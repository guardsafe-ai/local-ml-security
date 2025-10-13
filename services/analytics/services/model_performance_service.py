"""
Model Performance Service - Business Logic
"""

import logging
from typing import Dict
from database.repositories import ModelPerformanceRepository
from models.requests import ModelPerformance
from models.responses import SuccessResponse

logger = logging.getLogger(__name__)


class ModelPerformanceService:
    """Business logic for model performance operations"""
    
    def __init__(self):
        self.repository = ModelPerformanceRepository()
    
    def store_performance(self, performance: ModelPerformance) -> SuccessResponse:
        """Store model performance metrics"""
        try:
            logger.info(f"Storing performance metrics for model: {performance.model_name}")
            result = self.repository.store_performance(performance)
            return SuccessResponse(**result)
        except Exception as e:
            logger.error(f"Error storing model performance: {e}")
            raise
