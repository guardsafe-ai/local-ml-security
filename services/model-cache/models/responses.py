"""
Model Cache Service - Response Models
Pydantic models for outgoing responses
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_predictions: Dict[str, Dict[str, Any]]
    ensemble_used: bool
    processing_time_ms: float
    from_cache: bool = False
    timestamp: float


class ModelStatus(BaseModel):
    model_name: str
    loaded: bool
    loaded_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0


class CacheStats(BaseModel):
    total_predictions: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    model_loads: int
    uptime_seconds: float
    models_loaded: List[str]
    timestamp: datetime


class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: datetime
    uptime_seconds: float
    dependencies: Dict[str, bool]


class SuccessResponse(BaseModel):
    status: str
    message: str
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None
