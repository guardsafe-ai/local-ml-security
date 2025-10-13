"""
Red Team Service - Request Models
Pydantic models for incoming API requests
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class RedTeamConfig(BaseModel):
    """Configuration for red team testing"""
    model_name: str = Field(..., description="Name of the model to test")
    model_type: str = Field(default="pretrained", description="Type of model (pretrained/trained)")
    attack_categories: List[str] = Field(default=["prompt_injection", "jailbreak", "system_extraction", "code_injection"], description="Categories of attacks to generate")
    num_attacks: int = Field(default=50, description="Number of attacks to generate")
    severity_threshold: float = Field(default=0.7, description="Minimum severity threshold for attacks")
    test_duration_minutes: int = Field(default=10, description="Duration of the test in minutes")
    enable_learning: bool = Field(default=True, description="Enable continuous learning")
    custom_patterns: Optional[List[str]] = Field(default=None, description="Custom attack patterns to include")


class AttackPattern(BaseModel):
    """Individual attack pattern"""
    category: str = Field(..., description="Category of the attack")
    pattern: str = Field(..., description="The attack pattern text")
    severity: float = Field(..., description="Severity score (0-1)")
    description: str = Field(..., description="Description of the attack")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the pattern was created")


class AttackResult(BaseModel):
    """Result of testing an attack pattern against a model"""
    attack: AttackPattern = Field(..., description="The attack pattern that was tested")
    model_results: Dict[str, Any] = Field(..., description="Results from different model endpoints")
    detected: bool = Field(..., description="Whether the attack was detected")
    confidence: float = Field(..., description="Confidence score of the detection")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the test was performed")
    test_status: str = Field(..., description="Status of the test (PASS/FAIL)")
    pass_fail: bool = Field(..., description="Whether the test passed or failed")
    detection_success: bool = Field(..., description="Whether detection was successful")
    vulnerability_found: bool = Field(..., description="Whether a vulnerability was found")
    security_risk: str = Field(..., description="Security risk level (LOW/MEDIUM/HIGH/CRITICAL)")


class TestRequest(BaseModel):
    """Request to run a red team test"""
    config: Optional[RedTeamConfig] = Field(default=None, description="Test configuration")
    test_id: Optional[str] = Field(default=None, description="Custom test ID")
    models: Optional[List[str]] = Field(default=None, description="Specific models to test")
    attack_categories: Optional[List[str]] = Field(default=None, description="Specific attack categories")
    num_attacks: Optional[int] = Field(default=None, description="Number of attacks to generate")


class AdvancedTestRequest(BaseModel):
    """Advanced test request with more options"""
    config: RedTeamConfig = Field(..., description="Test configuration")
    test_id: Optional[str] = Field(default=None, description="Custom test ID")
    models: Optional[List[str]] = Field(default=None, description="Specific models to test")
    custom_attacks: Optional[List[AttackPattern]] = Field(default=None, description="Custom attack patterns")
    learning_config: Optional[Dict[str, Any]] = Field(default=None, description="Learning configuration")


class LearningConfigRequest(BaseModel):
    """Configuration for continuous learning"""
    enabled: bool = Field(default=True, description="Enable continuous learning")
    update_frequency_hours: int = Field(default=24, description="How often to update patterns")
    learning_rate: float = Field(default=0.1, description="Learning rate for pattern updates")
    min_confidence_threshold: float = Field(default=0.8, description="Minimum confidence for learning")
    max_patterns_per_category: int = Field(default=100, description="Maximum patterns per category")


class ModelTestRequest(BaseModel):
    """Request to test specific models"""
    model_names: List[str] = Field(..., description="Names of models to test")
    attack_categories: Optional[List[str]] = Field(default=None, description="Specific attack categories")
    num_attacks: int = Field(default=25, description="Number of attacks per model")
    severity_threshold: float = Field(default=0.7, description="Minimum severity threshold")


class CustomAttackRequest(BaseModel):
    """Request to add custom attack patterns"""
    patterns: List[AttackPattern] = Field(..., description="Custom attack patterns to add")
    category: str = Field(..., description="Category for the patterns")
    persist: bool = Field(default=True, description="Whether to persist the patterns")


class MetricsRequest(BaseModel):
    """Request for specific metrics"""
    time_range_hours: int = Field(default=24, description="Time range for metrics")
    model_name: Optional[str] = Field(default=None, description="Specific model to analyze")
    category: Optional[str] = Field(default=None, description="Specific attack category")
    include_detailed: bool = Field(default=False, description="Include detailed breakdown")


class AdvancedSecurityTestRequest(BaseModel):
    """Request for advanced ML security testing"""
    text: str = Field(..., description="Input text to test")
    attack_categories: Optional[List[str]] = Field(
        default=["gradient", "word_level", "universal_triggers"],
        description="Attack categories to test"
    )
    explainability_methods: Optional[List[str]] = Field(
        default=["shap", "lime", "integrated_gradients", "attention"],
        description="Explainability methods to use"
    )
    evaluation_metrics: Optional[List[str]] = Field(
        default=["robustness", "semantic_preservation", "transferability"],
        description="Evaluation metrics to calculate"
    )
    advanced_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Advanced configuration parameters"
    )
