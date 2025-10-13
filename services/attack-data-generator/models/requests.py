"""
Request Models for Attack Data Generator Service
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class PatternType(str, Enum):
    """Types of attack patterns"""
    JAILBREAK = "jailbreak"
    PROMPT_INJECTION = "prompt_injection"
    ADVERSARIAL = "adversarial"
    EVASION = "evasion"
    POISONING = "poisoning"
    EXTRACTION = "extraction"
    INFERENCE = "inference"
    BACKDOOR = "backdoor"


class AttackCategory(str, Enum):
    """OWASP LLM Top 10 attack categories"""
    PROMPT_INJECTION = "prompt_injection"
    INSEQURE_OUTPUT_HANDLING = "insecure_output_handling"
    TRAINING_DATA_POISONING = "training_data_poisoning"
    MODEL_DOS = "model_dos"
    SUPPLY_CHAIN_VULNERABILITIES = "supply_chain_vulnerabilities"
    SENSITIVE_INFORMATION_DISCLOSURE = "sensitive_information_disclosure"
    INSEQURE_PLUGIN_DESIGN = "insecure_plugin_design"
    EXCESSIVE_AGENCY = "excessive_agency"
    OVERRELIANCE = "overreliance"
    MODEL_THEFT = "model_theft"


class GenerationMethod(str, Enum):
    """Pattern generation methods"""
    RULE_BASED = "rule_based"
    TEMPLATE_EXPANSION = "template_expansion"
    LLM_BASED = "llm_based"
    EVOLUTIONARY = "evolutionary"
    THREAT_INTEL = "threat_intel"
    INCIDENT_LEARNING = "incident_learning"


class ComplexityLevel(str, Enum):
    """Pattern complexity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXPERT = "expert"


class CreativityLevel(str, Enum):
    """LLM creativity levels"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    CREATIVE = "creative"
    EXPERIMENTAL = "experimental"


class PatternGenerationRequest(BaseModel):
    """Request for generating attack patterns"""
    pattern_type: PatternType = Field(..., description="Type of attack pattern to generate")
    count: int = Field(10, ge=1, le=1000, description="Number of patterns to generate")
    complexity: ComplexityLevel = Field(ComplexityLevel.MEDIUM, description="Complexity level of patterns")
    target_model: Optional[str] = Field(None, description="Target model for pattern generation")
    attack_category: Optional[AttackCategory] = Field(None, description="OWASP LLM Top 10 category")
    prompt: Optional[str] = Field(None, description="Custom prompt for LLM-based generation")
    creativity_level: CreativityLevel = Field(CreativityLevel.BALANCED, description="Creativity level for LLM generation")
    variables: Optional[Dict[str, Any]] = Field(None, description="Variables for template expansion")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class EvolutionaryGenerationRequest(BaseModel):
    """Request for evolutionary pattern generation"""
    initial_population: Optional[List[Dict[str, Any]]] = Field(None, description="Initial population of patterns")
    generations: int = Field(50, ge=1, le=1000, description="Number of generations to evolve")
    population_size: int = Field(100, ge=10, le=10000, description="Size of population per generation")
    mutation_rate: float = Field(0.1, ge=0.0, le=1.0, description="Mutation rate for evolution")
    crossover_rate: float = Field(0.8, ge=0.0, le=1.0, description="Crossover rate for evolution")
    fitness_function: str = Field("attack_success", description="Fitness function to optimize")
    target_models: List[str] = Field(..., description="Target models for fitness evaluation")
    pattern_type: PatternType = Field(..., description="Type of patterns to evolve")
    attack_category: Optional[AttackCategory] = Field(None, description="Attack category to focus on")
    diversity_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for diversity in fitness")
    novelty_weight: float = Field(0.2, ge=0.0, le=1.0, description="Weight for novelty in fitness")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ThreatIntelRequest(BaseModel):
    """Request for threat intelligence scraping"""
    sources: List[str] = Field(..., description="Threat intelligence sources to scrape")
    attack_types: List[PatternType] = Field(..., description="Types of attacks to look for")
    time_range: Optional[str] = Field("30d", description="Time range for intelligence (e.g., '7d', '30d', '1y')")
    keywords: Optional[List[str]] = Field(None, description="Keywords to search for")
    max_results: int = Field(1000, ge=1, le=10000, description="Maximum number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters for scraping")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class IncidentLearningRequest(BaseModel):
    """Request for learning from production incidents"""
    incident_data: List[Dict[str, Any]] = Field(..., description="Incident data to learn from")
    learning_method: str = Field("pattern_extraction", description="Method for learning patterns")
    pattern_extraction: bool = Field(True, description="Whether to extract patterns from incidents")
    deduplication: bool = Field(True, description="Whether to deduplicate learned patterns")
    similarity_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Similarity threshold for deduplication")
    severity_filter: Optional[str] = Field(None, description="Filter by incident severity")
    time_filter: Optional[str] = Field(None, description="Filter by incident time range")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class PatternEvaluationRequest(BaseModel):
    """Request for evaluating attack patterns"""
    patterns: List[Dict[str, Any]] = Field(..., description="Patterns to evaluate")
    target_models: List[str] = Field(..., description="Target models for evaluation")
    evaluation_metrics: List[str] = Field(["attack_success", "semantic_preservation"], description="Metrics to evaluate")
    batch_size: int = Field(10, ge=1, le=100, description="Batch size for evaluation")
    timeout: int = Field(300, ge=1, le=3600, description="Timeout for evaluation in seconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class PatternFilterRequest(BaseModel):
    """Request for filtering patterns"""
    pattern_type: Optional[PatternType] = Field(None, description="Filter by pattern type")
    attack_category: Optional[AttackCategory] = Field(None, description="Filter by attack category")
    quality_score_min: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum quality score")
    success_rate_min: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum success rate")
    created_after: Optional[str] = Field(None, description="Filter patterns created after this date")
    created_before: Optional[str] = Field(None, description="Filter patterns created before this date")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    limit: int = Field(100, ge=1, le=10000, description="Maximum number of patterns to return")
    offset: int = Field(0, ge=0, description="Offset for pagination")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class PatternUpdateRequest(BaseModel):
    """Request for updating attack patterns"""
    pattern_id: str = Field(..., description="ID of pattern to update")
    updates: Dict[str, Any] = Field(..., description="Updates to apply to pattern")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class BatchGenerationRequest(BaseModel):
    """Request for batch pattern generation"""
    requests: List[PatternGenerationRequest] = Field(..., description="List of generation requests")
    parallel: bool = Field(True, description="Whether to run requests in parallel")
    max_workers: int = Field(10, ge=1, le=100, description="Maximum number of parallel workers")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class QualityAssessmentRequest(BaseModel):
    """Request for quality assessment of patterns"""
    patterns: List[Dict[str, Any]] = Field(..., description="Patterns to assess")
    assessment_criteria: List[str] = Field(["grammaticality", "semantic_preservation", "attack_effectiveness"], description="Criteria for assessment")
    reference_patterns: Optional[List[Dict[str, Any]]] = Field(None, description="Reference patterns for comparison")
    human_evaluation: bool = Field(False, description="Whether to include human evaluation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
