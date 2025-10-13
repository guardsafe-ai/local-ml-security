"""
Response Models for Model API Service
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from datetime import datetime


class PredictionResponse(BaseModel):
    """Prediction response model"""
    text: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_predictions: Dict[str, Dict[str, Any]]
    ensemble_used: bool
    processing_time_ms: float
    timestamp: datetime


class ModelInfo(BaseModel):
    """Model information model"""
    name: str
    type: str
    loaded: bool
    path: Optional[str]
    labels: List[str]
    performance: Optional[Dict[str, float]]
    model_source: Optional[str] = "Unknown"
    model_version: Optional[str] = "Unknown"


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    available_models: List[str]
    total_models: int
    timestamp: datetime


class ModelsResponse(BaseModel):
    """Models list response model"""
    models: Dict[str, ModelInfo]
    available_models: List[str]
    mlflow_models: List[str]


class ModelVersionsResponse(BaseModel):
    """Model versions response model"""
    model_name: str
    versions: List[Dict[str, Any]]


class CacheStatsResponse(BaseModel):
    """Cache statistics response model"""
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    models_cached: int


class MetricsResponse(BaseModel):
    """Metrics response model"""
    total_requests: int
    total_errors: int
    average_response_time: float
    models_loaded: int
    memory_usage_mb: float
