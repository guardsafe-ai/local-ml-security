"""
Model Cache Service - Request Models
Pydantic models for incoming requests
"""

from typing import List, Optional
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    text: str
    models: List[str]
    ensemble: bool = False


class ModelLoadRequest(BaseModel):
    model_name: str
    force_reload: bool = False


class CacheStatsRequest(BaseModel):
    include_details: bool = True
