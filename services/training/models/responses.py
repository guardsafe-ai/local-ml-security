"""
Training Service - Response Models
Pydantic models for outgoing API responses
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: datetime
    dependencies: Optional[Dict[str, bool]] = None
    models_loaded: Optional[int] = None
    active_jobs: Optional[int] = None
    uptime_seconds: Optional[float] = None


class JobStatus(BaseModel):
    job_id: str
    model_name: str
    status: str  # pending, running, completed, failed
    progress: float
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    training_data_path: Optional[str] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    num_epochs: Optional[int] = None
    max_length: Optional[int] = None
    config: Optional[Dict[str, Any]] = None


class TrainingStatus(BaseModel):
    model_name: str
    status: str
    progress: float
    current_epoch: int
    total_epochs: int
    current_loss: float
    best_metric: float
    start_time: datetime
    estimated_completion: Optional[datetime] = None


class ModelInfo(BaseModel):
    name: str
    type: str
    status: str
    loaded: bool
    path: str
    labels: List[str]
    performance: Optional[Dict[str, float]] = None
    model_source: str
    model_version: str
    description: str


class ModelListResponse(BaseModel):
    models: Dict[str, ModelInfo]
    available_models: List[str]
    mlflow_models: List[str]


class ModelRegistryResponse(BaseModel):
    models: List[Dict[str, Any]]
    total_models: int


class ModelVersionsResponse(BaseModel):
    model_name: str
    versions: List[Dict[str, Any]]
    latest_version: str


class TrainingResult(BaseModel):
    status: str
    message: str
    job_id: str
    model_name: str
    timestamp: datetime


class ModelLoadingResult(BaseModel):
    status: str
    message: str
    loaded_models: List[str]
    failed_models: List[str]
    timestamp: datetime


class DataUploadResult(BaseModel):
    status: str
    message: str
    file_path: str
    data_type: str
    timestamp: datetime


class DataStatistics(BaseModel):
    total_files: int
    fresh_files: int
    used_files: int
    data_types: Dict[str, int]
    total_size_mb: float


class ExperimentSummary(BaseModel):
    total_experiments: int
    active_experiments: int
    completed_experiments: int
    failed_experiments: int
    total_models: int
    best_model: Optional[str] = None


class ModelPerformanceHistory(BaseModel):
    model_name: str
    performance_history: List[Dict[str, Any]]
    best_performance: Optional[Dict[str, float]] = None
    performance_trend: str  # improving, declining, stable


class ModelComparison(BaseModel):
    models: List[str]
    comparison_metrics: Dict[str, Dict[str, float]]
    best_model: str
    comparison_summary: str


class RetrainingHistory(BaseModel):
    model_name: str
    retraining_history: List[Dict[str, Any]]
    total_retrainings: int
    last_retraining: Optional[datetime] = None


class ModelLineage(BaseModel):
    model_name: str
    lineage: List[Dict[str, Any]]
    original_model: str
    current_version: str


class RetrainingRecommendations(BaseModel):
    model_name: str
    recommendations: List[Dict[str, Any]]
    should_retrain: bool
    confidence: float
    reasons: List[str]


class SuccessResponse(BaseModel):
    status: str
    message: str
    timestamp: datetime
