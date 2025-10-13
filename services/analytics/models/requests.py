"""
Request Models for Analytics Service
"""

from typing import Dict, List, Optional
from pydantic import BaseModel


class RedTeamTestResult(BaseModel):
    """Red team test result data"""
    test_id: str
    model_name: str
    model_type: str  # 'pre-trained' or 'trained'
    model_version: Optional[str] = None
    model_source: Optional[str] = None
    total_attacks: int
    vulnerabilities_found: int
    detection_rate: float
    test_duration_seconds: Optional[int] = None
    batch_size: Optional[int] = None
    attack_categories: Optional[List[str]] = None
    attack_results: Optional[List[Dict]] = None


class ModelPerformance(BaseModel):
    """Model performance metrics"""
    model_name: str
    model_type: str
    model_version: Optional[str] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    training_duration_seconds: Optional[int] = None
    dataset_size: Optional[int] = None
