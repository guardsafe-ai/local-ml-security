"""
Data Privacy Service - Response Models
Pydantic models for outgoing responses
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel


class DataSubject(BaseModel):
    subject_id: str
    email: Optional[str] = None
    created_at: datetime
    last_accessed: datetime
    data_categories: List[str]
    retention_until: datetime
    consent_given: bool
    consent_withdrawn: bool = False


class DataAnonymization(BaseModel):
    original_text: str
    anonymized_text: str
    anonymization_method: str
    pii_detected: List[str]
    confidence_scores: Dict[str, float]
    created_at: datetime


class AuditLog(BaseModel):
    log_id: str
    timestamp: datetime
    action: str
    subject_id: Optional[str] = None
    user_id: Optional[str] = None
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class RetentionPolicy(BaseModel):
    policy_id: str
    data_category: str
    retention_days: int
    auto_delete: bool
    created_at: datetime
    updated_at: datetime


class ComplianceStatus(BaseModel):
    gdpr_compliant: bool
    ccpa_compliant: bool
    overall_score: float
    issues: List[str]
    recommendations: List[str]
    last_checked: datetime


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
