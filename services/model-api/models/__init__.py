"""
Model API Service Models
Pydantic models for request/response validation
"""

from .requests import PredictionRequest, EnsembleConfig, LoadModelRequest, UnloadModelRequest
from .responses import (
    PredictionResponse, 
    ModelInfo, 
    HealthResponse,
    ModelsResponse,
    ModelVersionsResponse,
    CacheStatsResponse,
    MetricsResponse
)

__all__ = [
    "PredictionRequest",
    "EnsembleConfig", 
    "LoadModelRequest",
    "UnloadModelRequest",
    "PredictionResponse",
    "ModelInfo",
    "HealthResponse",
    "ModelsResponse",
    "ModelVersionsResponse", 
    "CacheStatsResponse",
    "MetricsResponse"
]
