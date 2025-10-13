"""
Request Models for Model API Service
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """Prediction request model"""
    text: str
    models: Optional[List[str]] = None
    ensemble: bool = True
    return_probabilities: bool = True
    return_embeddings: bool = False


class EnsembleConfig(BaseModel):
    """Ensemble configuration model"""
    models: List[str]
    weights: Optional[Dict[str, float]] = None
    voting_method: str = "soft"  # soft, hard


class LoadModelRequest(BaseModel):
    """Load model request model"""
    model_name: str
    version: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class UnloadModelRequest(BaseModel):
    """Unload model request model"""
    model_name: str
