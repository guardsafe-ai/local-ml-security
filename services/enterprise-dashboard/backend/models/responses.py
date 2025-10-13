"""
Enterprise Dashboard Backend - Response Models
Pydantic models for outgoing API responses
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    timestamp: datetime = Field(..., description="Response timestamp")
    version: Optional[str] = Field(default="1.0.0", description="Service version")
    uptime_seconds: Optional[float] = Field(default=None, description="Service uptime")


class ServiceHealth(BaseModel):
    """Individual service health status"""
    name: str = Field(..., description="Service name")
    status: str = Field(..., description="Service status")
    response_time: float = Field(..., description="Response time in milliseconds")
    last_check: datetime = Field(..., description="Last health check time")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")


class DashboardMetrics(BaseModel):
    """Dashboard metrics summary"""
    total_models: int = Field(..., description="Total number of models")
    active_jobs: int = Field(..., description="Number of active training jobs")
    total_attacks: int = Field(..., description="Total number of red team attacks")
    detection_rate: float = Field(..., description="Overall detection rate")
    system_health: float = Field(..., description="Overall system health score")
    last_updated: datetime = Field(..., description="Last metrics update time")


class TrainingJob(BaseModel):
    """Training job information"""
    job_id: str = Field(..., description="Unique job identifier")
    model_name: str = Field(..., description="Name of the model being trained")
    status: str = Field(..., description="Job status")
    progress: float = Field(..., description="Progress percentage")
    start_time: datetime = Field(..., description="Job start time")
    end_time: Optional[datetime] = Field(default=None, description="Job end time")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class AttackResult(BaseModel):
    """Red team attack result"""
    attack_id: str = Field(..., description="Unique attack identifier")
    category: str = Field(..., description="Attack category")
    success: bool = Field(..., description="Whether attack was successful")
    detection_rate: float = Field(..., description="Detection rate for this attack")
    timestamp: datetime = Field(..., description="Attack timestamp")
    severity: Optional[float] = Field(default=None, description="Attack severity")
    security_risk: Optional[str] = Field(default=None, description="Security risk level")


class ModelInfo(BaseModel):
    """Model information"""
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    status: str = Field(..., description="Model status")
    loaded: bool = Field(..., description="Whether model is loaded")
    path: str = Field(..., description="Model path")
    labels: List[str] = Field(..., description="Model labels")
    performance: Optional[Dict[str, Any]] = Field(default=None, description="Performance metrics")
    model_source: str = Field(..., description="Model source")
    model_version: str = Field(..., description="Model version")
    description: str = Field(..., description="Model description")


class PredictionResponse(BaseModel):
    """Model prediction response"""
    text: str = Field(..., description="Input text")
    prediction: str = Field(..., description="Predicted class")
    confidence: float = Field(..., description="Prediction confidence")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    model_name: str = Field(..., description="Model used for prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(..., description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_processing_time_ms: float = Field(..., description="Total processing time")
    timestamp: datetime = Field(..., description="Batch prediction timestamp")


class MLflowExperiment(BaseModel):
    """MLflow experiment information"""
    experiment_id: str = Field(..., description="Experiment ID")
    name: str = Field(..., description="Experiment name")
    lifecycle_stage: str = Field(..., description="Experiment lifecycle stage")
    creation_time: datetime = Field(..., description="Creation time")
    last_update_time: datetime = Field(..., description="Last update time")
    run_count: int = Field(..., description="Number of runs")


class MLflowRun(BaseModel):
    """MLflow run information"""
    run_id: str = Field(..., description="Run ID")
    experiment_id: str = Field(..., description="Experiment ID")
    status: str = Field(..., description="Run status")
    start_time: datetime = Field(..., description="Run start time")
    end_time: Optional[datetime] = Field(default=None, description="Run end time")
    metrics: Dict[str, float] = Field(..., description="Run metrics")
    params: Dict[str, str] = Field(..., description="Run parameters")
    tags: Dict[str, str] = Field(..., description="Run tags")


class AnalyticsSummary(BaseModel):
    """Analytics summary"""
    total_tests: int = Field(..., description="Total number of tests")
    detection_rate: float = Field(..., description="Overall detection rate")
    vulnerability_rate: float = Field(..., description="Vulnerability detection rate")
    model_performance: Dict[str, Dict[str, Any]] = Field(..., description="Model performance data")
    trend_data: List[Dict[str, Any]] = Field(..., description="Trend analysis data")
    last_updated: datetime = Field(..., description="Last update time")


class MonitoringAlert(BaseModel):
    """Monitoring alert"""
    alert_id: str = Field(..., description="Alert identifier")
    service: str = Field(..., description="Service name")
    level: str = Field(..., description="Alert level")
    message: str = Field(..., description="Alert message")
    timestamp: datetime = Field(..., description="Alert timestamp")
    resolved: bool = Field(default=False, description="Whether alert is resolved")


class MonitoringMetrics(BaseModel):
    """Monitoring metrics"""
    service: str = Field(..., description="Service name")
    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    disk_usage: float = Field(..., description="Disk usage percentage")
    network_io: Dict[str, float] = Field(..., description="Network I/O metrics")
    timestamp: datetime = Field(..., description="Metrics timestamp")


class SuccessResponse(BaseModel):
    """Generic success response"""
    status: str = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(..., description="Response timestamp")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Additional data")


class ErrorResponse(BaseModel):
    """Error response"""
    status: str = Field(..., description="Error status")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(..., description="Error timestamp")
    error_code: Optional[str] = Field(default=None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")


class WebSocketResponse(BaseModel):
    """WebSocket response"""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: datetime = Field(..., description="Message timestamp")
    success: bool = Field(default=True, description="Whether operation was successful")
