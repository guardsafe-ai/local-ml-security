"""
Analytics Routes
"""

from fastapi import APIRouter, HTTPException
from models.responses import PerformanceTrends
from services.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics", tags=["analytics"])

# Initialize service
analytics_service = AnalyticsService()


@router.get("/trends", response_model=PerformanceTrends)
async def get_performance_trends(days: int = 30):
    """Get performance trends over time"""
    try:
        return analytics_service.get_performance_trends(days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
