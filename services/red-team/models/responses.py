"""
Red Team Service - Response Models
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
    dependencies: Optional[Dict[str, bool]] = Field(default=None, description="Dependency status")
    running: Optional[bool] = Field(default=None, description="Whether testing is running")
    total_tests: Optional[int] = Field(default=None, description="Total tests performed")
    uptime_seconds: Optional[float] = Field(default=None, description="Service uptime")


class AttackPattern(BaseModel):
    """Individual attack pattern"""
    category: str = Field(..., description="Category of the attack")
    pattern: str = Field(..., description="The attack pattern text")
    severity: float = Field(..., description="Severity score (0-1)")
    description: str = Field(..., description="Description of the attack")
    timestamp: datetime = Field(..., description="When the pattern was created")


class CVSSScore(BaseModel):
    """CVSS v3.1 score for a vulnerability"""
    version: str = Field(default="3.1", description="CVSS version")
    base_score: float = Field(..., description="Base CVSS score")
    temporal_score: float = Field(..., description="Temporal CVSS score")
    environmental_score: float = Field(..., description="Environmental CVSS score")
    overall_score: float = Field(..., description="Overall CVSS score")
    severity: str = Field(..., description="Severity rating (NONE/LOW/MEDIUM/HIGH/CRITICAL)")
    vector_string: str = Field(..., description="CVSS vector string")
    base_metrics: Dict[str, str] = Field(..., description="Base metrics breakdown")


class AttackResult(BaseModel):
    """Result of testing an attack pattern against a model"""
    attack: AttackPattern = Field(..., description="The attack pattern that was tested")
    model_results: Dict[str, Any] = Field(..., description="Results from different model endpoints")
    detected: bool = Field(..., description="Whether the attack was detected")
    confidence: float = Field(..., description="Confidence score of the detection")
    timestamp: datetime = Field(..., description="When the test was performed")
    test_status: str = Field(..., description="Status of the test (PASS/FAIL)")
    pass_fail: bool = Field(..., description="Whether the test passed or failed")
    detection_success: bool = Field(..., description="Whether detection was successful")
    vulnerability_found: bool = Field(..., description="Whether a vulnerability was found")
    security_risk: str = Field(..., description="Security risk level (LOW/MEDIUM/HIGH/CRITICAL)")
    cvss_score: Optional[CVSSScore] = Field(default=None, description="CVSS v3.1 score")
    cve_references: Optional[List[str]] = Field(default=None, description="Related CVE references")
    remediation_steps: Optional[List[str]] = Field(default=None, description="Remediation steps")
    compliance_mapping: Optional[Dict[str, str]] = Field(default=None, description="Compliance framework mapping")


class TestSession(BaseModel):
    """Red team test session"""
    test_id: str = Field(..., description="Unique test session ID")
    timestamp: datetime = Field(..., description="When the test started")
    total_attacks: int = Field(..., description="Total number of attacks")
    vulnerabilities_found: int = Field(..., description="Number of vulnerabilities found")
    detection_rate: float = Field(..., description="Overall detection rate")
    overall_status: str = Field(..., description="Overall test status")
    pass_count: int = Field(..., description="Number of passed tests")
    fail_count: int = Field(..., description="Number of failed tests")
    pass_rate: float = Field(..., description="Pass rate percentage")
    test_summary: Dict[str, Any] = Field(..., description="Test summary statistics")
    security_risk_distribution: Dict[str, int] = Field(..., description="Distribution of security risks")
    risk_summary: Dict[str, int] = Field(..., description="Risk summary statistics")
    attacks: List[AttackPattern] = Field(..., description="List of attack patterns used")
    results: List[AttackResult] = Field(..., description="List of test results")
    vulnerabilities: List[Dict[str, Any]] = Field(..., description="List of vulnerabilities found")


class TestStatus(BaseModel):
    """Current test status"""
    test_id: str = Field(..., description="Test session ID")
    status: str = Field(..., description="Current status")
    progress: float = Field(..., description="Progress percentage")
    current_attack: int = Field(..., description="Current attack number")
    total_attacks: int = Field(..., description="Total attacks planned")
    start_time: datetime = Field(..., description="When the test started")
    estimated_completion: Optional[datetime] = Field(default=None, description="Estimated completion time")
    current_category: Optional[str] = Field(default=None, description="Current attack category")


class ModelInfo(BaseModel):
    """Model information"""
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    status: str = Field(..., description="Model status")
    available: bool = Field(..., description="Whether model is available")
    last_tested: Optional[datetime] = Field(default=None, description="When model was last tested")
    test_count: int = Field(default=0, description="Number of tests performed")
    detection_rate: float = Field(default=0.0, description="Average detection rate")


class MetricsResponse(BaseModel):
    """Metrics response"""
    total_tests: int = Field(..., description="Total number of tests")
    total_attacks: int = Field(..., description="Total number of attacks")
    detection_rate: float = Field(..., description="Overall detection rate")
    vulnerability_rate: float = Field(..., description="Vulnerability detection rate")
    average_confidence: float = Field(..., description="Average confidence score")
    test_duration_avg: float = Field(..., description="Average test duration")
    category_breakdown: Dict[str, Dict[str, Any]] = Field(..., description="Breakdown by category")
    model_performance: Dict[str, Dict[str, Any]] = Field(..., description="Performance by model")
    risk_distribution: Dict[str, int] = Field(..., description="Security risk distribution")
    time_series_data: List[Dict[str, Any]] = Field(..., description="Time series data")
    recent_tests: List[Dict[str, Any]] = Field(..., description="Recent test results")


class LearningStatus(BaseModel):
    """Continuous learning status"""
    enabled: bool = Field(..., description="Whether learning is enabled")
    last_update: Optional[datetime] = Field(default=None, description="Last learning update")
    patterns_learned: int = Field(default=0, description="Number of patterns learned")
    categories_updated: List[str] = Field(default=[], description="Categories that were updated")
    learning_rate: float = Field(default=0.1, description="Current learning rate")
    confidence_threshold: float = Field(default=0.8, description="Confidence threshold")
    next_update: Optional[datetime] = Field(default=None, description="Next scheduled update")


class TestResult(BaseModel):
    """Individual test result"""
    test_id: str = Field(..., description="Test session ID")
    status: str = Field(..., description="Test status")
    message: str = Field(..., description="Status message")
    timestamp: datetime = Field(..., description="When the test completed")
    results: Optional[TestSession] = Field(default=None, description="Test results if available")


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
