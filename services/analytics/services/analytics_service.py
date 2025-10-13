"""
Analytics Service - Business Logic
"""

import logging
from typing import List, Dict, Any
from database.repositories import AnalyticsRepository
from models.responses import PerformanceTrends

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Business logic for analytics operations"""
    
    def __init__(self):
        self.repository = AnalyticsRepository()
    
    def get_performance_trends(self, days: int = 30) -> PerformanceTrends:
        """Get performance trends over time"""
        try:
            logger.info(f"Fetching performance trends for last {days} days")
            trends_data = self.repository.get_performance_trends(days)
            return PerformanceTrends(trends=trends_data)
        except Exception as e:
            logger.error(f"Error fetching performance trends: {e}")
            raise
