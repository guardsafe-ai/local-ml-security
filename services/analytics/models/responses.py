"""
Response Models for Analytics Service
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    timestamp: datetime


class RedTeamSummary(BaseModel):
    """Red team test summary"""
    summary: List[Dict[str, Any]]


class ModelComparison(BaseModel):
    """Model comparison data"""
    model_name: str
    pretrained: Optional[Dict[str, Any]] = None
    trained: Optional[Dict[str, Any]] = None
    improvement: Optional[Dict[str, float]] = None


class PerformanceTrends(BaseModel):
    """Performance trends over time"""
    trends: List[Dict[str, Any]]


class SuccessResponse(BaseModel):
    """Generic success response"""
    message: str
    test_id: Optional[str] = None
