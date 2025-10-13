"""
Business Metrics Service - Request Models
Pydantic models for incoming requests
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pydantic import BaseModel


class AttackSuccessRateRequest(BaseModel):
    """Request model for attack success rate calculation"""
    time_range_days: int = 30
    model_name: Optional[str] = None
    category: Optional[str] = None


class ModelDriftRequest(BaseModel):
    """Request model for model drift detection"""
    model_name: str
    current_data: Dict[str, Any]
    reference_data_id: Optional[str] = None


class CostAnalysisRequest(BaseModel):
    """Request model for cost analysis"""
    time_range_days: int = 30
    include_breakdown: bool = True


class SystemEffectivenessRequest(BaseModel):
    """Request model for system effectiveness calculation"""
    time_range_days: int = 30
    include_metrics: List[str] = ["accuracy", "response_time", "availability"]


class KPICalculationRequest(BaseModel):
    """Request model for KPI calculation"""
    time_range_days: int = 30
    include_recommendations: bool = True
    models: Optional[List[str]] = None
