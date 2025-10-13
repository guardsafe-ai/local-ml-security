"""
Red Team Service - Business Logic
"""

import logging
from typing import Dict, List, Any
from database.repositories import RedTeamRepository
from models.requests import RedTeamTestResult
from models.responses import RedTeamSummary, ModelComparison

logger = logging.getLogger(__name__)


class RedTeamService:
    """Business logic for red team operations"""
    
    def __init__(self):
        self.repository = RedTeamRepository()
    
    def store_test_result(self, result: RedTeamTestResult) -> Dict[str, str]:
        """Store red team test result"""
        try:
            logger.info(f"Storing red team test result for model: {result.model_name}")
            return self.repository.store_test_result(result)
        except Exception as e:
            logger.error(f"Error storing red team test result: {e}")
            raise
    
    def get_summary(self, days: int = 7) -> RedTeamSummary:
        """Get red team test summary"""
        try:
            logger.info(f"Fetching red team summary for last {days} days")
            summary_data = self.repository.get_summary(days)
            return RedTeamSummary(summary=summary_data)
        except Exception as e:
            logger.error(f"Error fetching red team summary: {e}")
            raise
    
    def get_model_comparison(self, model_name: str, days: int = 30) -> ModelComparison:
        """Get model comparison data"""
        try:
            logger.info(f"Fetching model comparison for {model_name}")
            comparison_data = self.repository.get_model_comparison(model_name, days)
            return ModelComparison(**comparison_data)
        except Exception as e:
            logger.error(f"Error fetching model comparison: {e}")
            raise
