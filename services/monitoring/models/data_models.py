"""
Monitoring Service - Data Models
Pydantic models for monitoring data structures
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel


class ModelLoadingStatus(BaseModel):
    """Model loading status data"""
    model_name: str
    status: str
    progress: float
    loaded_at: Optional[datetime] = None
    error_message: Optional[str] = None


class TrainingStatus(BaseModel):
    """Training status data"""
    job_id: str
    model_name: str
    status: str
    progress: float
    current_epoch: int
    total_epochs: int
    loss: float
    accuracy: float
    started_at: datetime
    estimated_completion: Optional[datetime] = None


class SystemMetrics(BaseModel):
    """System metrics data"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: float
    active_connections: int


class ServiceHealth(BaseModel):
    """Service health data"""
    service_name: str
    status: str
    response_time: float
    last_check: datetime
    uptime: float
    error_count: int


class Alert(BaseModel):
    """Alert data"""
    alert_id: str
    severity: str
    message: str
    timestamp: datetime
    service: str
    resolved: bool = False


class DashboardData(BaseModel):
    """Complete dashboard data"""
    model_loading_status: List[ModelLoadingStatus]
    training_status: List[TrainingStatus]
    system_metrics: SystemMetrics
    service_health: List[ServiceHealth]
    alerts: List[Alert]
    last_updated: datetime
