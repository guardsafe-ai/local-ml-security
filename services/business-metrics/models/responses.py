"""
Business Metrics Service - Response Models
Pydantic models for outgoing responses
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pydantic import BaseModel


class AttackSuccessRate(BaseModel):
    total_attacks: int
    successful_attacks: int
    success_rate: float
    by_category: Dict[str, float]
    by_model: Dict[str, float]
    trend_7d: float
    trend_30d: float


class ModelDriftMetrics(BaseModel):
    model_name: str
    drift_detected: bool
    drift_score: float
    confidence_interval: Tuple[float, float]
    last_drift_check: datetime
    features_drifted: List[str]
    severity: str  # low, medium, high, critical


class CostMetrics(BaseModel):
    total_cost_usd: float
    compute_cost: float
    storage_cost: float
    api_calls_cost: float
    model_training_cost: float
    cost_per_prediction: float
    cost_trend_7d: float
    cost_trend_30d: float


class SystemEffectiveness(BaseModel):
    overall_effectiveness: float
    detection_accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    response_time_p95: float
    availability_percent: float
    user_satisfaction_score: float


class BusinessKPI(BaseModel):
    timestamp: datetime
    attack_success_rate: AttackSuccessRate
    model_drift: List[ModelDriftMetrics]
    cost_metrics: CostMetrics
    system_effectiveness: SystemEffectiveness
    recommendations: List[str]


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
