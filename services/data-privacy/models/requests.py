"""
Data Privacy Service - Request Models
Pydantic models for incoming requests
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel


class DataSubjectRequest(BaseModel):
    """Request model for data subject registration"""
    email: Optional[str] = None
    data_categories: List[str]
    retention_days: int = 365
    consent_given: bool = True


class AnonymizationRequest(BaseModel):
    """Request model for data anonymization"""
    text: str
    anonymization_level: str = "medium"  # low, medium, high
    preserve_format: bool = True


class ConsentWithdrawalRequest(BaseModel):
    """Request model for consent withdrawal"""
    subject_id: str
    reason: Optional[str] = None


class DataDeletionRequest(BaseModel):
    """Request model for data deletion"""
    subject_id: str
    reason: Optional[str] = None
    immediate: bool = False


class AuditLogRequest(BaseModel):
    """Request model for audit log queries"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    action_type: Optional[str] = None
    subject_id: Optional[str] = None
    limit: int = 100


class ComplianceCheckRequest(BaseModel):
    """Request model for compliance checks"""
    check_type: str = "full"  # full, gdpr, ccpa, basic
    include_recommendations: bool = True
