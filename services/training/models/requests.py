"""
Training Service - Request Models
Pydantic models for incoming API requests
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel


class TrainingConfig(BaseModel):
    model_name: str
    max_length: int = 256
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 2
    warmup_steps: int = 100
    weight_decay: float = 0.01
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 500
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True


class TrainingRequest(BaseModel):
    model_name: str
    training_data_path: str
    config: Optional[TrainingConfig] = None
    retrain: bool = False


class ModelLoadingRequest(BaseModel):
    models: Dict[str, List[str]]


class SingleModelLoadingRequest(BaseModel):
    model_name: str
    model_type: str


class RetrainingRequest(BaseModel):
    model_name: str
    training_data_path: str
    config: Optional[TrainingConfig] = None


class AdvancedRetrainingRequest(BaseModel):
    model_name: str
    training_data_path: str
    config: Optional[TrainingConfig] = None
    retraining_trigger: Optional[str] = None
    performance_threshold: Optional[float] = None


class DataUploadRequest(BaseModel):
    data_type: str
    file_path: str
    description: Optional[str] = None


class MultipleDataUploadRequest(BaseModel):
    data_files: List[Dict[str, str]]
    data_type: str


class RedTeamResultsRequest(BaseModel):
    test_id: str
    model_name: str
    results: Dict[str, Any]
    timestamp: datetime


class RetrainingTriggerRequest(BaseModel):
    model_name: str
    performance_metrics: Dict[str, float]
    data_drift_detected: bool = False
    time_since_last_training: Optional[int] = None
