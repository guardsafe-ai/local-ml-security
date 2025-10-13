"""
Red Team Service - Testing Routes
Red team testing and management endpoints
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from models.requests import TestRequest, AdvancedTestRequest, ModelTestRequest, AdvancedSecurityTestRequest
from models.responses import TestResult, TestStatus, TestSession, SuccessResponse
from services.red_team_service import RedTeamService
from fastapi import Query

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize service
red_team_service = RedTeamService()


@router.get("/status")
async def get_status():
    """Get current red team service status"""
    try:
        return {
            "status": "running",
            "active_tests": len(red_team_service.active_tests),
            "learning_enabled": red_team_service.learning_enabled,
            "timestamp": "2025-09-26T17:40:00.000000"
        }
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start", response_model=TestResult)
async def start_testing(request: TestRequest = None, background_tasks: BackgroundTasks = None):
    """Start red team testing"""
    try:
        if request is None:
            request = TestRequest()
        
        test_id = await red_team_service.start_test(request)
        
        return TestResult(
            test_id=test_id,
            status="started",
            message=f"Red team test {test_id} started successfully",
            timestamp="2025-09-26T17:40:00.000000"
        )
    except Exception as e:
        logger.error(f"Failed to start testing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_testing():
    """Stop all active testing"""
    try:
        stopped_count = 0
        for test_id in list(red_team_service.active_tests.keys()):
            if await red_team_service.stop_test(test_id):
                stopped_count += 1
        
        return {
            "status": "success",
            "message": f"Stopped {stopped_count} active tests",
            "timestamp": "2025-09-26T17:40:00.000000"
        }
    except Exception as e:
        logger.error(f"Failed to stop testing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test", response_model=TestResult)
async def run_test(request: TestRequest = None, background_tasks: BackgroundTasks = None):
    """Run a single red team test"""
    try:
        if request is None:
            request = TestRequest()
        
        test_id = await red_team_service.start_test(request)
        
        return TestResult(
            test_id=test_id,
            status="started",
            message=f"Red team test {test_id} started successfully",
            timestamp="2025-09-26T17:40:00.000000"
        )
    except Exception as e:
        logger.error(f"Failed to run test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def get_available_models():
    """Get available models for testing"""
    try:
        models = await red_team_service.get_available_models()
        return {
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results")
async def get_latest_results():
    """Get latest test results"""
    try:
        results = await red_team_service.get_latest_results()
        return {
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Failed to get latest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test/{test_id}/status", response_model=TestStatus)
async def get_test_status(test_id: str):
    """Get status of a specific test"""
    try:
        status = await red_team_service.get_test_status(test_id)
        if not status:
            raise HTTPException(status_code=404, detail="Test not found")
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get test status for {test_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test/{test_id}/results", response_model=TestSession)
async def get_test_results(test_id: str):
    """Get results of a specific test"""
    try:
        results = await red_team_service.get_test_results(test_id)
        if not results:
            raise HTTPException(status_code=404, detail="Test results not found")
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get test results for {test_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics(time_range_hours: int = 24, model_name: str = None, category: str = None):
    """Get red team testing metrics"""
    try:
        metrics = await red_team_service.get_metrics(time_range_hours, model_name, category)
        return metrics
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced-test", response_model=TestResult)
async def run_advanced_test(request: AdvancedTestRequest, background_tasks: BackgroundTasks = None):
    """Run advanced red team test with custom configuration"""
    try:
        # Convert to basic test request
        test_request = TestRequest(
            config=request.config,
            test_id=request.test_id,
            models=request.models,
            attack_categories=request.config.attack_categories if request.config else None,
            num_attacks=request.config.num_attacks if request.config else None
        )
        
        test_id = await red_team_service.start_test(test_request)
        
        return TestResult(
            test_id=test_id,
            status="started",
            message=f"Advanced red team test {test_id} started successfully",
            timestamp="2025-09-26T17:40:00.000000"
        )
    except Exception as e:
        logger.error(f"Failed to run advanced test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-models", response_model=TestResult)
async def test_models(request: ModelTestRequest, background_tasks: BackgroundTasks = None):
    """Test specific models against attack patterns"""
    try:
        # Create test request for each model
        test_requests = []
        for model_name in request.model_names:
            test_request = TestRequest(
                models=[model_name],
                attack_categories=request.attack_categories,
                num_attacks=request.num_attacks
            )
            test_requests.append(test_request)
        
        # Start tests for all models
        test_ids = []
        for test_request in test_requests:
            test_id = await red_team_service.start_test(test_request)
            test_ids.append(test_id)
        
        return TestResult(
            test_id=",".join(test_ids),
            status="started",
            message=f"Started tests for {len(request.model_names)} models",
            timestamp="2025-09-26T17:40:00.000000"
        )
    except Exception as e:
        logger.error(f"Failed to test models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cleanup")
async def cleanup_old_data(days_old: int = 30):
    """Clean up old test data"""
    try:
        cleaned_count = await red_team_service.cleanup_old_data(days_old)
        return {
            "status": "success",
            "message": f"Cleaned up {cleaned_count} old records",
            "cleaned_count": cleaned_count,
            "days_old": days_old,
            "timestamp": "2025-09-26T17:40:00.000000"
        }
    except Exception as e:
        logger.error(f"Failed to cleanup old data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/security-metrics")
async def get_security_metrics():
    """Get security metrics for dashboard"""
    try:
        metrics = await red_team_service.get_security_metrics()
        return {
            "status": "success",
            "metrics": metrics,
            "timestamp": "2025-09-26T17:40:00.000000"
        }
    except Exception as e:
        logger.error(f"Failed to get security metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compliance-report/{test_id}")
async def get_compliance_report(test_id: str):
    """Generate compliance report for audit"""
    try:
        report = await red_team_service.generate_compliance_report(test_id)
        return {
            "status": "success",
            "report": report,
            "timestamp": "2025-09-26T17:40:00.000000"
        }
    except Exception as e:
        logger.error(f"Failed to generate compliance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/owasp-coverage")
async def get_owasp_coverage():
    """Get OWASP LLM Top 10 coverage status"""
    try:
        # Get all available attack categories
        categories = red_team_service.attack_generator.get_available_categories()
        
        # Get recent test results to see which categories were tested
        recent_results = await red_team_service.get_latest_results(limit=1000)
        tested_categories = set(r.get("attack_category", "unknown") for r in recent_results)
        
        # Map to OWASP categories
        owasp_mapping = {
            "prompt_injection": "LLM01",
            "jailbreak": "LLM01",
            "output_security": "LLM02",
            "data_poisoning": "LLM03",
            "dos_attack": "LLM04",
            "supply_chain": "LLM05",
            "system_extraction": "LLM06",
            "code_injection": "LLM07",
            "excessive_agency": "LLM08",
            "hallucination": "LLM09",
            "model_theft": "LLM10"
        }
        
        coverage = {}
        for category in categories:
            owasp_id = owasp_mapping.get(category, "UNKNOWN")
            tested = category in tested_categories
            coverage[owasp_id] = {
                "category": category,
                "tested": tested,
                "status": "COVERED" if tested else "NOT_COVERED"
            }
        
        total_covered = sum(1 for c in coverage.values() if c["tested"])
        coverage_percentage = (total_covered / len(owasp_mapping)) * 100
        
        return {
            "status": "success",
            "owasp_coverage": {
                "coverage_percentage": round(coverage_percentage, 2),
                "categories_covered": total_covered,
                "total_categories": len(owasp_mapping),
                "detailed_coverage": coverage
            },
            "timestamp": "2025-09-26T17:40:00.000000"
        }
    except Exception as e:
        logger.error(f"Failed to get OWASP coverage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vulnerabilities")
async def get_vulnerabilities(
    severity: Optional[str] = Query(None, description="Filter by severity (CRITICAL, HIGH, MEDIUM, LOW)"),
    category: Optional[str] = Query(None, description="Filter by attack category"),
    limit: int = Query(100, description="Maximum number of results")
):
    """Get vulnerability details with CVSS scoring"""
    try:
        # Get recent results
        results = await red_team_service.get_latest_results(limit=limit)
        
        # Filter vulnerabilities (where detected=False)
        vulnerabilities = [r for r in results if not r.get("detected", True)]
        
        # Apply filters
        if severity:
            vulnerabilities = [v for v in vulnerabilities if v.get("security_risk", "").upper() == severity.upper()]
        
        if category:
            vulnerabilities = [v for v in vulnerabilities if v.get("attack_category", "").lower() == category.lower()]
        
        # Add CVSS scoring for each vulnerability
        enhanced_vulnerabilities = []
        for vuln in vulnerabilities:
            cvss_assessment = red_team_service.cvss_calculator.assess_ml_vulnerability(
                attack_category=vuln.get("attack_category", "unknown"),
                detected=vuln.get("detected", False),
                confidence=vuln.get("confidence", 0.0),
                model_type="production"
            )
            
            enhanced_vuln = {
                **vuln,
                "cvss_score": cvss_assessment.get("overall_score", 0.0),
                "severity_rating": cvss_assessment.get("severity", "UNKNOWN"),
                "owasp_mapping": red_team_service._map_to_owasp_category(vuln.get("attack_category", "unknown"))
            }
            enhanced_vulnerabilities.append(enhanced_vuln)
        
        return {
            "status": "success",
            "vulnerabilities": enhanced_vulnerabilities,
            "total_count": len(enhanced_vulnerabilities),
            "filters_applied": {
                "severity": severity,
                "category": category,
                "limit": limit
            },
            "timestamp": "2025-09-26T17:40:00.000000"
        }
    except Exception as e:
        logger.error(f"Failed to get vulnerabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced-security-test")
async def advanced_security_test(request: AdvancedSecurityTestRequest):
    """
    Perform advanced ML security testing using world-class algorithms
    
    This endpoint provides state-of-the-art adversarial ML testing including:
    - Gradient-based attacks (FGSM, PGD, C&W)
    - Word-level attacks (TextFooler, BERT-Attack, HotFlip)
    - Universal adversarial triggers
    - Explainability analysis (SHAP, LIME, Integrated Gradients, Attention)
    - Research-grade evaluation metrics
    """
    try:
        # Perform advanced security test
        results = await red_team_service.advanced_security_test(
            text=request.text,
            attack_categories=request.attack_categories,
            explainability_methods=request.explainability_methods,
            evaluation_metrics=request.evaluation_metrics
        )
        
        return {
            "status": "success",
            "message": "Advanced security test completed",
            "results": results,
            "timestamp": "2025-09-26T17:40:00.000000"
        }
        
    except Exception as e:
        logger.error(f"Advanced security test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/initialize-advanced-platform")
async def initialize_advanced_platform():
    """
    Initialize the advanced ML security platform
    This should be called after the model is loaded
    """
    try:
        # Note: In a real implementation, you would get the model and tokenizer
        # from your model service or load them here
        # For now, we'll just return a success message
        
        return {
            "status": "success",
            "message": "Advanced ML Security Platform initialization requested",
            "note": "Model and tokenizer need to be provided for full initialization",
            "timestamp": "2025-09-26T17:40:00.000000"
        }
        
    except Exception as e:
        logger.error(f"Advanced platform initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
