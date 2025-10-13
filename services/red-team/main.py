"""
Main FastAPI Application for Red Team Service
Comprehensive AI red teaming service with all attack modules and integrations
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn
import logging
import asyncio
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime
import traceback

# Import all service modules
from services.adversarial_ml import (
    GradientAttacks, WordLevelAttacks, MultiTurnAttacks
)
from services.agent_attacks import (
    ToolInjectionAttacks, PromptLeakingAttacks, 
    RecursiveAttacks, ChainOfThoughtAttacks
)
from services.traditional_ml_attacks import (
    EvasionAttacks, PoisoningAttacks, 
    ModelExtractionAttacks, MembershipInferenceAttacks
)
from services.compliance import (
    NISTAIRiskManagementFramework, EUAIActCompliance
)
from services.benchmarking import (
    HarmBenchEvaluator, StrongREJECTEvaluator, 
    SafetyBenchEvaluator, BenchmarkManager, PerformanceBenchmark
)
from services.pattern_evolution import (
    GeneticAlgorithm, MultiObjectiveOptimizer, PatternEvolutionEngine
)
from services.threat_intelligence import (
    MITREATLASScraper, CVEScraper, JailbreakScraper, ThreatIntelManager
)
from services.incident_learning import (
    IncidentProcessor, PatternExtractor, FeedbackLoop, IncidentLearningPipeline
)
from services.behavior_analysis import (
    ActivationAnalyzer, AttributionAnalyzer, CausalAnalyzer, 
    AnomalyDetector, BehaviorAnalyzer
)
from services.certification import (
    RandomizedSmoothing, IntervalBoundPropagation, CertificationManager
)
from services.privacy_attacks import (
    MembershipInference, ModelInversion, DataExtraction, PrivacyAttackManager
)
from services.distributed import (
    TaskQueue, WorkerManager, LoadBalancer, DistributedCoordinator
)
from services.optimization import (
    ResultCache, BatchOptimizer, GPUManager, OptimizationCoordinator
)
from services.monitoring import (
    PrometheusMetrics, WebSocketStreaming, AlertManager, 
    MonitoringCoordinator, PerformanceDashboard
)
from services.visualizations import (
    AttackTreeVisualizer, EmbeddingPlotter, HeatmapGenerator, 
    SecurityDashboard, VisualizationCoordinator
)
from services.reporting import (
    ExecutiveReporter, TechnicalReporter, ComplianceReporter,
    RiskAnalyzer, ROICalculator, ReportingCoordinator
)
from services.mlflow_integration import (
    ModelFetcher, VersionManager, MLflowCoordinator
)
from services.siem_integration import (
    SplunkIntegration, ElasticSecurityIntegration, 
    AzureSentinelIntegration, SIEMCoordinator
)
from services.fairness_testing import (
    DemographicParityTester, CounterfactualFairnessTester, 
    BiasDetector, FairnessCoordinator
)
from services.datasets import (
    OWASPDatasetGenerator, NISTDatasetGenerator, JailbreakDatasetGenerator,
    FairnessDatasetGenerator, PrivacyDatasetGenerator, DatasetManager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Red Team Service",
    description="Comprehensive AI red teaming service for security testing and compliance",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Global service instances
services = {}

@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup"""
    logger.info("Initializing AI Red Team Service...")
    
    try:
        # Initialize core services
        services["gradient_attacks"] = GradientAttacks()
        services["word_level_attacks"] = WordLevelAttacks()
        services["multi_turn_attacks"] = MultiTurnAttacks()
        
        # Agent attacks
        services["tool_injection"] = ToolInjectionAttacks()
        services["prompt_leaking"] = PromptLeakingAttacks()
        services["recursive_attacks"] = RecursiveAttacks()
        services["chain_of_thought"] = ChainOfThoughtAttacks()
        
        # Traditional ML attacks
        services["evasion_attacks"] = EvasionAttacks()
        services["poisoning_attacks"] = PoisoningAttacks()
        services["model_extraction"] = ModelExtractionAttacks()
        services["membership_inference"] = MembershipInferenceAttacks()
        
        # Compliance
        services["nist_rmf"] = NISTAIRiskManagementFramework()
        services["eu_ai_act"] = EUAIActCompliance()
        
        # Benchmarking
        services["harmbench"] = HarmBenchEvaluator()
        services["strongreject"] = StrongREJECTEvaluator()
        services["safetybench"] = SafetyBenchEvaluator()
        services["benchmark_manager"] = BenchmarkManager()
        services["performance_benchmark"] = PerformanceBenchmark()
        
        # Pattern evolution
        services["genetic_algorithm"] = GeneticAlgorithm()
        services["multi_objective_optimizer"] = MultiObjectiveOptimizer()
        services["pattern_evolution"] = PatternEvolutionEngine()
        
        # Threat intelligence
        services["mitre_atlas"] = MITREATLASScraper()
        services["cve_scraper"] = CVEScraper()
        services["jailbreak_scraper"] = JailbreakScraper()
        services["threat_intel_manager"] = ThreatIntelManager()
        
        # Incident learning
        services["incident_processor"] = IncidentProcessor()
        services["pattern_extractor"] = PatternExtractor()
        services["feedback_loop"] = FeedbackLoop()
        services["incident_learning"] = IncidentLearningPipeline()
        
        # Behavior analysis
        services["activation_analyzer"] = ActivationAnalyzer()
        services["attribution_analyzer"] = AttributionAnalyzer()
        services["causal_analyzer"] = CausalAnalyzer()
        services["anomaly_detector"] = AnomalyDetector()
        services["behavior_analyzer"] = BehaviorAnalyzer()
        
        # Certification
        services["randomized_smoothing"] = RandomizedSmoothing()
        services["interval_bound_propagation"] = IntervalBoundPropagation()
        services["certification_manager"] = CertificationManager()
        
        # Privacy attacks
        services["membership_inference_privacy"] = MembershipInference()
        services["model_inversion"] = ModelInversion()
        services["data_extraction"] = DataExtraction()
        services["privacy_attack_manager"] = PrivacyAttackManager()
        
        # Distributed infrastructure
        services["task_queue"] = TaskQueue()
        services["worker_manager"] = WorkerManager()
        services["load_balancer"] = LoadBalancer()
        services["distributed_coordinator"] = DistributedCoordinator()
        
        # Optimization
        services["result_cache"] = ResultCache()
        services["batch_optimizer"] = BatchOptimizer()
        services["gpu_manager"] = GPUManager()
        services["optimization_coordinator"] = OptimizationCoordinator()
        
        # Monitoring
        services["prometheus_metrics"] = PrometheusMetrics()
        services["websocket_streaming"] = WebSocketStreaming()
        services["alert_manager"] = AlertManager()
        services["monitoring_coordinator"] = MonitoringCoordinator()
        services["performance_dashboard"] = PerformanceDashboard()
        
        # Visualizations
        services["attack_tree_visualizer"] = AttackTreeVisualizer()
        services["embedding_plotter"] = EmbeddingPlotter()
        services["heatmap_generator"] = HeatmapGenerator()
        services["security_dashboard"] = SecurityDashboard()
        services["visualization_coordinator"] = VisualizationCoordinator()
        
        # Reporting
        services["executive_reporter"] = ExecutiveReporter()
        services["technical_reporter"] = TechnicalReporter()
        services["compliance_reporter"] = ComplianceReporter()
        services["risk_analyzer"] = RiskAnalyzer()
        services["roi_calculator"] = ROICalculator()
        services["reporting_coordinator"] = ReportingCoordinator()
        
        # MLflow integration
        services["model_fetcher"] = ModelFetcher()
        services["version_manager"] = VersionManager()
        services["mlflow_coordinator"] = MLflowCoordinator()
        
        # SIEM integration
        services["splunk_integration"] = SplunkIntegration()
        services["elastic_security"] = ElasticSecurityIntegration()
        services["azure_sentinel"] = AzureSentinelIntegration()
        services["siem_coordinator"] = SIEMCoordinator()
        
        # Fairness testing
        services["demographic_parity"] = DemographicParityTester()
        services["counterfactual_fairness"] = CounterfactualFairnessTester()
        services["bias_detector"] = BiasDetector()
        services["fairness_coordinator"] = FairnessCoordinator()
        
        # Datasets
        services["owasp_dataset"] = OWASPDatasetGenerator()
        services["nist_dataset"] = NISTDatasetGenerator()
        services["jailbreak_dataset"] = JailbreakDatasetGenerator()
        services["fairness_dataset"] = FairnessDatasetGenerator()
        services["privacy_dataset"] = PrivacyDatasetGenerator()
        services["dataset_manager"] = DatasetManager()
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        logger.error(traceback.format_exc())
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI Red Team Service...")
    
    # Cleanup any resources
    for service_name, service in services.items():
        try:
            if hasattr(service, 'cleanup'):
                await service.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up {service_name}: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": list(services.keys())
    }

# Service status endpoint
@app.get("/status")
async def service_status():
    """Get status of all services"""
    status = {}
    
    for service_name, service in services.items():
        try:
            if hasattr(service, 'get_status'):
                status[service_name] = service.get_status()
            else:
                status[service_name] = {"status": "unknown"}
        except Exception as e:
            status[service_name] = {"status": "error", "error": str(e)}
    
    return status

# Attack execution endpoints
@app.post("/attacks/gradient")
async def execute_gradient_attacks(request: Dict[str, Any]):
    """Execute gradient-based attacks"""
    try:
        service = services["gradient_attacks"]
        result = await service.execute_attacks(request)
        return result
    except Exception as e:
        logger.error(f"Error executing gradient attacks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/attacks/word-level")
async def execute_word_level_attacks(request: Dict[str, Any]):
    """Execute word-level attacks"""
    try:
        service = services["word_level_attacks"]
        result = await service.execute_attacks(request)
        return result
    except Exception as e:
        logger.error(f"Error executing word-level attacks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/attacks/multi-turn")
async def execute_multi_turn_attacks(request: Dict[str, Any]):
    """Execute multi-turn attacks"""
    try:
        service = services["multi_turn_attacks"]
        result = await service.execute_attacks(request)
        return result
    except Exception as e:
        logger.error(f"Error executing multi-turn attacks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/attacks/agent")
async def execute_agent_attacks(request: Dict[str, Any]):
    """Execute agent-specific attacks"""
    try:
        attack_type = request.get("attack_type", "tool_injection")
        service = services.get(attack_type)
        
        if not service:
            raise HTTPException(status_code=400, detail=f"Unknown attack type: {attack_type}")
        
        result = await service.execute_attacks(request)
        return result
    except Exception as e:
        logger.error(f"Error executing agent attacks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/attacks/traditional-ml")
async def execute_traditional_ml_attacks(request: Dict[str, Any]):
    """Execute traditional ML attacks"""
    try:
        attack_type = request.get("attack_type", "evasion")
        service = services.get(attack_type + "_attacks")
        
        if not service:
            raise HTTPException(status_code=400, detail=f"Unknown attack type: {attack_type}")
        
        result = await service.execute_attacks(request)
        return result
    except Exception as e:
        logger.error(f"Error executing traditional ML attacks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Compliance endpoints
@app.post("/compliance/nist-rmf")
async def check_nist_rmf_compliance(request: Dict[str, Any]):
    """Check NIST AI RMF compliance"""
    try:
        service = services["nist_rmf"]
        result = await service.assess_compliance(request)
        return result
    except Exception as e:
        logger.error(f"Error checking NIST RMF compliance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compliance/eu-ai-act")
async def check_eu_ai_act_compliance(request: Dict[str, Any]):
    """Check EU AI Act compliance"""
    try:
        service = services["eu_ai_act"]
        result = await service.assess_compliance(request)
        return result
    except Exception as e:
        logger.error(f"Error checking EU AI Act compliance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Benchmarking endpoints
@app.post("/benchmarking/execute")
async def execute_benchmarks(request: Dict[str, Any]):
    """Execute benchmarking tests"""
    try:
        service = services["benchmark_manager"]
        result = await service.execute_benchmarks(request)
        return result
    except Exception as e:
        logger.error(f"Error executing benchmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/benchmarking/performance")
async def execute_performance_benchmarks(request: Dict[str, Any]):
    """Execute performance benchmarks"""
    try:
        service = services["performance_benchmark"]
        result = await service.execute_benchmarks(request)
        return result
    except Exception as e:
        logger.error(f"Error executing performance benchmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Fairness testing endpoints
@app.post("/fairness/demographic-parity")
async def test_demographic_parity(request: Dict[str, Any]):
    """Test demographic parity"""
    try:
        service = services["demographic_parity"]
        result = await service.test_demographic_parity(
            request["predictions"],
            request["protected_attributes"],
            request["protected_groups"]
        )
        return result
    except Exception as e:
        logger.error(f"Error testing demographic parity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fairness/counterfactual")
async def test_counterfactual_fairness(request: Dict[str, Any]):
    """Test counterfactual fairness"""
    try:
        service = services["counterfactual_fairness"]
        result = await service.test_counterfactual_fairness(
            request["predictions"],
            request["protected_attributes"],
            request["features"],
            request["protected_groups"]
        )
        return result
    except Exception as e:
        logger.error(f"Error testing counterfactual fairness: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fairness/bias-detection")
async def detect_bias(request: Dict[str, Any]):
    """Detect various types of bias"""
    try:
        service = services["bias_detector"]
        result = await service.detect_bias(
            request["predictions"],
            request["protected_attributes"],
            request["features"],
            request["protected_groups"]
        )
        return result
    except Exception as e:
        logger.error(f"Error detecting bias: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fairness/comprehensive")
async def comprehensive_fairness_test(request: Dict[str, Any]):
    """Run comprehensive fairness tests"""
    try:
        service = services["fairness_coordinator"]
        result = await service.test_fairness(
            request["predictions"],
            request["protected_attributes"],
            request["features"],
            request["protected_groups"]
        )
        return result
    except Exception as e:
        logger.error(f"Error running comprehensive fairness test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Reporting endpoints
@app.post("/reports/executive")
async def generate_executive_report(request: Dict[str, Any]):
    """Generate executive report"""
    try:
        service = services["executive_reporter"]
        result = await service.generate_report(request)
        return result
    except Exception as e:
        logger.error(f"Error generating executive report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reports/technical")
async def generate_technical_report(request: Dict[str, Any]):
    """Generate technical report"""
    try:
        service = services["technical_reporter"]
        result = await service.generate_report(request)
        return result
    except Exception as e:
        logger.error(f"Error generating technical report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reports/compliance")
async def generate_compliance_report(request: Dict[str, Any]):
    """Generate compliance report"""
    try:
        service = services["compliance_reporter"]
        result = await service.generate_report(request)
        return result
    except Exception as e:
        logger.error(f"Error generating compliance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring endpoints
@app.get("/monitoring/metrics")
async def get_metrics():
    """Get Prometheus metrics"""
    try:
        service = services["prometheus_metrics"]
        result = await service.get_metrics()
        return result
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/dashboard")
async def get_dashboard():
    """Get performance dashboard data"""
    try:
        service = services["performance_dashboard"]
        result = await service.get_dashboard_data()
        return result
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Dataset generation endpoints
@app.post("/datasets/generate")
async def generate_datasets(request: Dict[str, Any]):
    """Generate training datasets"""
    try:
        service = services["dataset_manager"]
        result = await service.generate_datasets(request)
        return result
    except Exception as e:
        logger.error(f"Error generating datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Threat intelligence endpoints
@app.get("/threat-intel/mitre-atlas")
async def get_mitre_atlas_data():
    """Get MITRE ATLAS data"""
    try:
        service = services["mitre_atlas"]
        result = await service.scrape_data()
        return result
    except Exception as e:
        logger.error(f"Error getting MITRE ATLAS data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/threat-intel/cve")
async def get_cve_data():
    """Get CVE data"""
    try:
        service = services["cve_scraper"]
        result = await service.scrape_data()
        return result
    except Exception as e:
        logger.error(f"Error getting CVE data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/threat-intel/jailbreaks")
async def get_jailbreak_data():
    """Get jailbreak data"""
    try:
        service = services["jailbreak_scraper"]
        result = await service.scrape_data()
        return result
    except Exception as e:
        logger.error(f"Error getting jailbreak data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Visualization endpoints
@app.post("/visualizations/attack-tree")
async def generate_attack_tree(request: Dict[str, Any]):
    """Generate attack tree visualization"""
    try:
        service = services["attack_tree_visualizer"]
        result = await service.generate_visualization(request)
        return result
    except Exception as e:
        logger.error(f"Error generating attack tree: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visualizations/embedding-plot")
async def generate_embedding_plot(request: Dict[str, Any]):
    """Generate embedding plot visualization"""
    try:
        service = services["embedding_plotter"]
        result = await service.generate_visualization(request)
        return result
    except Exception as e:
        logger.error(f"Error generating embedding plot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visualizations/heatmap")
async def generate_heatmap(request: Dict[str, Any]):
    """Generate heatmap visualization"""
    try:
        service = services["heatmap_generator"]
        result = await service.generate_visualization(request)
        return result
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Comprehensive testing endpoint
@app.post("/test/comprehensive")
async def run_comprehensive_test(request: Dict[str, Any]):
    """Run comprehensive red team test"""
    try:
        # This would orchestrate all testing modules
        # For now, return a placeholder response
        return {
            "status": "success",
            "message": "Comprehensive test initiated",
            "test_id": "test_" + str(int(datetime.utcnow().timestamp())),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error running comprehensive test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="AI Red Team Service",
        version="1.0.0",
        description="Comprehensive AI red teaming service for security testing and compliance",
        routes=app.routes,
    )
    
    # Add custom tags
    openapi_schema["tags"] = [
        {"name": "Health", "description": "Health check and status endpoints"},
        {"name": "Attacks", "description": "Attack execution endpoints"},
        {"name": "Compliance", "description": "Compliance checking endpoints"},
        {"name": "Benchmarking", "description": "Benchmarking and performance testing"},
        {"name": "Fairness", "description": "Fairness testing endpoints"},
        {"name": "Reporting", "description": "Report generation endpoints"},
        {"name": "Monitoring", "description": "Monitoring and metrics endpoints"},
        {"name": "Datasets", "description": "Dataset generation endpoints"},
        {"name": "Threat Intelligence", "description": "Threat intelligence endpoints"},
        {"name": "Visualizations", "description": "Visualization generation endpoints"},
        {"name": "Testing", "description": "Comprehensive testing endpoints"}
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )