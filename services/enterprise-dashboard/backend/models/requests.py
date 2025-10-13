"""
Enterprise Dashboard Backend - Request Models
Pydantic models for incoming API requests
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request for model prediction"""
    text: str = Field(..., description="Text to analyze")
    model_name: Optional[str] = Field(None, description="Model name to use")
    ensemble: Optional[bool] = Field(False, description="Use ensemble prediction")


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""
    texts: List[str] = Field(..., description="List of texts to analyze")
    model_name: Optional[str] = Field(None, description="Model name to use")
    ensemble: Optional[bool] = Field(False, description="Use ensemble prediction")


class TrainingRequest(BaseModel):
    """Request for starting training"""
    model_name: str = Field(..., description="Name of the model to train")
    training_data_path: str = Field(..., description="Path to training data")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Training configuration")


class RetrainingRequest(BaseModel):
    """Request for retraining a model"""
    model_name: str = Field(..., description="Name of the model to retrain")
    training_data_path: str = Field(..., description="Path to training data")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Retraining configuration")


class RedTeamTestRequest(BaseModel):
    """Request for red team testing"""
    model_name: Optional[str] = Field(None, description="Model to test")
    attack_categories: Optional[List[str]] = Field(default=None, description="Attack categories")
    num_attacks: Optional[int] = Field(default=50, description="Number of attacks")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Test configuration")


class ModelLoadRequest(BaseModel):
    """Request for loading a model"""
    model_name: str = Field(..., description="Name of the model to load")
    model_type: Optional[str] = Field(default="pytorch", description="Type of model")


class ModelUnloadRequest(BaseModel):
    """Request for unloading a model"""
    model_name: str = Field(..., description="Name of the model to unload")


class ModelReloadRequest(BaseModel):
    """Request for reloading a model"""
    model_name: str = Field(..., description="Name of the model to reload")


class RedTeamResultsRequest(BaseModel):
    """Request for storing red team results"""
    test_id: str = Field(..., description="Test session ID")
    model_name: str = Field(..., description="Model that was tested")
    results: Dict[str, Any] = Field(..., description="Test results")
    timestamp: datetime = Field(default_factory=datetime.now, description="Test timestamp")


class ModelPerformanceRequest(BaseModel):
    """Request for storing model performance"""
    model_name: str = Field(..., description="Name of the model")
    version: str = Field(..., description="Model version")
    metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    test_data_path: Optional[str] = Field(default=None, description="Path to test data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Performance timestamp")


class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")


class ServiceHealthCheckRequest(BaseModel):
    """Request for health check"""
    service_name: str = Field(..., description="Name of the service to check")
    include_details: bool = Field(default=False, description="Include detailed health information")


class MetricsRequest(BaseModel):
    """Request for metrics"""
    time_range_hours: int = Field(default=24, description="Time range in hours")
    service_name: Optional[str] = Field(default=None, description="Specific service")
    include_trends: bool = Field(default=True, description="Include trend analysis")
