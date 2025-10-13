"""
Response Models for Attack Data Generator Service
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime


class AttackPattern(BaseModel):
    """Attack pattern model"""
    id: str = Field(..., description="Unique identifier for the pattern")
    pattern_type: str = Field(..., description="Type of attack pattern")
    attack_category: Optional[str] = Field(None, description="OWASP LLM Top 10 category")
    content: str = Field(..., description="Pattern content")
    description: Optional[str] = Field(None, description="Pattern description")
    quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Quality score of the pattern")
    success_rate: float = Field(0.0, ge=0.0, le=1.0, description="Success rate of the pattern")
    complexity: str = Field("medium", description="Complexity level of the pattern")
    tags: List[str] = Field(default_factory=list, description="Tags associated with the pattern")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class PatternGenerationResponse(BaseModel):
    """Response for pattern generation"""
    success: bool = Field(..., description="Whether the generation was successful")
    patterns: List[AttackPattern] = Field(..., description="Generated attack patterns")
    generation_method: str = Field(..., description="Method used for generation")
    quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Average quality score")
    generation_time: float = Field(0.0, ge=0.0, description="Time taken for generation in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if generation failed")


class EvolutionaryGenerationResponse(BaseModel):
    """Response for evolutionary pattern generation"""
    success: bool = Field(..., description="Whether the evolution was successful")
    evolved_patterns: List[AttackPattern] = Field(..., description="Evolved attack patterns")
    fitness_scores: List[float] = Field(..., description="Fitness scores of evolved patterns")
    generation_history: List[Dict[str, Any]] = Field(..., description="Evolution history")
    best_pattern: Optional[AttackPattern] = Field(None, description="Best evolved pattern")
    convergence_generation: Optional[int] = Field(None, description="Generation where convergence occurred")
    evolution_time: float = Field(0.0, ge=0.0, description="Time taken for evolution in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if evolution failed")


class ThreatIntelResponse(BaseModel):
    """Response for threat intelligence scraping"""
    success: bool = Field(..., description="Whether the scraping was successful")
    patterns: List[AttackPattern] = Field(..., description="Scraped attack patterns")
    sources: List[str] = Field(..., description="Sources that were scraped")
    total_found: int = Field(0, ge=0, description="Total number of patterns found")
    scraping_time: float = Field(0.0, ge=0.0, description="Time taken for scraping in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if scraping failed")


class IncidentLearningResponse(BaseModel):
    """Response for incident learning"""
    success: bool = Field(..., description="Whether the learning was successful")
    learned_patterns: List[AttackPattern] = Field(..., description="Patterns learned from incidents")
    extraction_method: str = Field(..., description="Method used for pattern extraction")
    patterns_count: int = Field(0, ge=0, description="Number of patterns extracted")
    incidents_processed: int = Field(0, ge=0, description="Number of incidents processed")
    learning_time: float = Field(0.0, ge=0.0, description="Time taken for learning in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if learning failed")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: float = Field(..., description="Response timestamp")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")
    memory_usage: Optional[float] = Field(None, description="Memory usage in MB")
    cpu_usage: Optional[float] = Field(None, description="CPU usage percentage")


class PatternEvaluationResponse(BaseModel):
    """Response for pattern evaluation"""
    success: bool = Field(..., description="Whether the evaluation was successful")
    evaluation_results: List[Dict[str, Any]] = Field(..., description="Evaluation results for each pattern")
    average_metrics: Dict[str, float] = Field(..., description="Average metrics across all patterns")
    evaluation_time: float = Field(0.0, ge=0.0, description="Time taken for evaluation in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if evaluation failed")


class PatternFilterResponse(BaseModel):
    """Response for pattern filtering"""
    success: bool = Field(..., description="Whether the filtering was successful")
    patterns: List[AttackPattern] = Field(..., description="Filtered attack patterns")
    total_count: int = Field(0, ge=0, description="Total number of patterns matching filters")
    filtered_count: int = Field(0, ge=0, description="Number of patterns returned")
    filters_applied: Dict[str, Any] = Field(..., description="Filters that were applied")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if filtering failed")


class PatternUpdateResponse(BaseModel):
    """Response for pattern update"""
    success: bool = Field(..., description="Whether the update was successful")
    updated_pattern: Optional[AttackPattern] = Field(None, description="Updated pattern")
    update_time: float = Field(0.0, ge=0.0, description="Time taken for update in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if update failed")


class BatchGenerationResponse(BaseModel):
    """Response for batch pattern generation"""
    success: bool = Field(..., description="Whether the batch generation was successful")
    results: List[PatternGenerationResponse] = Field(..., description="Results for each generation request")
    total_patterns: int = Field(0, ge=0, description="Total number of patterns generated")
    successful_requests: int = Field(0, ge=0, description="Number of successful requests")
    failed_requests: int = Field(0, ge=0, description="Number of failed requests")
    total_time: float = Field(0.0, ge=0.0, description="Total time taken for batch generation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    errors: List[str] = Field(default_factory=list, description="Error messages for failed requests")


class QualityAssessmentResponse(BaseModel):
    """Response for quality assessment"""
    success: bool = Field(..., description="Whether the assessment was successful")
    assessment_results: List[Dict[str, Any]] = Field(..., description="Assessment results for each pattern")
    average_scores: Dict[str, float] = Field(..., description="Average scores across all patterns")
    quality_distribution: Dict[str, int] = Field(..., description="Distribution of quality scores")
    assessment_time: float = Field(0.0, ge=0.0, description="Time taken for assessment in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if assessment failed")


class GenerationStatsResponse(BaseModel):
    """Response for generation statistics"""
    success: bool = Field(..., description="Whether the stats retrieval was successful")
    stats: Dict[str, Any] = Field(..., description="Generation statistics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if stats retrieval failed")


class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = Field(False, description="Always false for error responses")
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
