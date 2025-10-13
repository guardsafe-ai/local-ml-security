"""
Ecosystem Integration Routes
Provides endpoints for integrating with all platform services
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from services.ecosystem_integration import EcosystemIntegrationService, ServiceType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ecosystem", tags=["ecosystem"])

# Initialize ecosystem integration service
ecosystem_service = EcosystemIntegrationService()


@router.get("/health")
async def get_ecosystem_health() -> Dict[str, Any]:
    """
    Get health status of all platform services
    
    Returns:
        Health status for all services
    """
    try:
        health_status = await ecosystem_service.check_all_services_health()
        return health_status
        
    except Exception as e:
        logger.error(f"Ecosystem health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/{service_type}")
async def get_service_health(service_type: str) -> Dict[str, Any]:
    """
    Get health status of a specific service
    
    Args:
        service_type: Type of service to check
        
    Returns:
        Health status for the service
    """
    try:
        try:
            service_enum = ServiceType(service_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown service type: {service_type}")
        
        health_status = await ecosystem_service.check_service_health(service_enum)
        return health_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Service health check failed for {service_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_ecosystem_status() -> Dict[str, Any]:
    """
    Get comprehensive ecosystem status
    
    Returns:
        Ecosystem status information
    """
    try:
        status = await ecosystem_service.get_ecosystem_status()
        return status
        
    except Exception as e:
        logger.error(f"Ecosystem status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities")
async def get_service_capabilities() -> Dict[str, Any]:
    """
    Get capabilities of all ecosystem services
    
    Returns:
        Service capabilities information
    """
    try:
        capabilities = await ecosystem_service.get_service_capabilities()
        return capabilities
        
    except Exception as e:
        logger.error(f"Service capabilities retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync")
async def sync_with_ecosystem(red_team_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sync red team data with all ecosystem services
    
    Args:
        red_team_data: Red team testing data
        
    Returns:
        Sync results
    """
    try:
        sync_results = await ecosystem_service.sync_with_ecosystem(red_team_data)
        return sync_results
        
    except Exception as e:
        logger.error(f"Ecosystem sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analytics/send")
async def send_analytics_data(analytics_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send data to analytics service
    
    Args:
        analytics_data: Data to send to analytics service
        
    Returns:
        Send status
    """
    try:
        success = await ecosystem_service.send_analytics_data(analytics_data)
        
        if success:
            return {
                "status": "success",
                "message": "Analytics data sent successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "message": "Failed to send analytics data",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Analytics data sending failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/send")
async def send_business_metrics(metrics_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send metrics to business metrics service
    
    Args:
        metrics_data: Metrics data to send
        
    Returns:
        Send status
    """
    try:
        success = await ecosystem_service.send_business_metrics(metrics_data)
        
        if success:
            return {
                "status": "success",
                "message": "Business metrics sent successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "message": "Failed to send business metrics",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Business metrics sending failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mlflow/models")
async def get_mlflow_models() -> Dict[str, Any]:
    """
    Get models from MLflow service
    
    Returns:
        List of models from MLflow
    """
    try:
        models = await ecosystem_service.get_mlflow_models()
        
        return {
            "status": "success",
            "models": models,
            "count": len(models),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"MLflow models retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model-api/predict/{model_name}")
async def get_model_predictions(
    model_name: str,
    input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get predictions from model API service
    
    Args:
        model_name: Name of the model
        input_data: Input data for prediction
        
    Returns:
        Prediction results
    """
    try:
        predictions = await ecosystem_service.get_model_predictions(model_name, input_data)
        
        return {
            "status": "success",
            "model_name": model_name,
            "predictions": predictions,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model prediction failed for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training/start")
async def trigger_model_training(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trigger model training via training service
    
    Args:
        training_config: Training configuration
        
    Returns:
        Training job information
    """
    try:
        training_result = await ecosystem_service.trigger_model_training(training_config)
        
        if "error" in training_result:
            return {
                "status": "error",
                "message": training_result["error"],
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "success",
                "training_job": training_result,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Model training trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data-privacy/classify")
async def classify_data_privacy(data: str) -> Dict[str, Any]:
    """
    Classify data for privacy using data privacy service
    
    Args:
        data: Data to classify
        
    Returns:
        Privacy classification results
    """
    try:
        classification = await ecosystem_service.classify_data_privacy(data)
        
        if "error" in classification:
            return {
                "status": "error",
                "message": classification["error"],
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "success",
                "classification": classification,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Data privacy classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model-cache/load/{model_name}")
async def load_model_to_cache(model_name: str) -> Dict[str, Any]:
    """
    Load model to cache via model cache service
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Load status
    """
    try:
        success = await ecosystem_service.load_model_to_cache(model_name)
        
        if success:
            return {
                "status": "success",
                "message": f"Model {model_name} loaded to cache successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to load model {model_name} to cache",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Model cache loading failed for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-flows")
async def get_data_flows() -> Dict[str, Any]:
    """
    Get data flow information for the ecosystem
    
    Returns:
        Data flow information
    """
    try:
        data_flows = {
            "security_testing": {
                "description": "Red team testing data flows through analytics and business metrics",
                "flow": "Red Team → Analytics → Business Metrics",
                "purpose": "Track security testing effectiveness and business impact"
            },
            "model_management": {
                "description": "Model lifecycle management across services",
                "flow": "Training → MLflow → Model Cache → Model API",
                "purpose": "Complete model lifecycle from training to deployment"
            },
            "privacy_protection": {
                "description": "Data privacy classification and protection",
                "flow": "Data Privacy → Analytics → Business Metrics",
                "purpose": "Ensure data privacy compliance and monitoring"
            },
            "compliance_reporting": {
                "description": "Compliance data aggregation and reporting",
                "flow": "All Services → Analytics → Reporting",
                "purpose": "Generate comprehensive compliance reports"
            },
            "real_time_monitoring": {
                "description": "Real-time monitoring and alerting",
                "flow": "All Services → Business Metrics → Dashboard",
                "purpose": "Provide real-time visibility into system health"
            }
        }
        
        return {
            "status": "success",
            "data_flows": data_flows,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Data flows retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/integration-status")
async def get_integration_status() -> Dict[str, Any]:
    """
    Get integration status for all services
    
    Returns:
        Integration status information
    """
    try:
        # Get health status
        health_status = await ecosystem_service.check_all_services_health()
        
        # Get capabilities
        capabilities = await ecosystem_service.get_service_capabilities()
        
        # Calculate integration score
        healthy_services = health_status.get("healthy_services", 0)
        total_services = health_status.get("total_services", 0)
        integration_score = (healthy_services / total_services) * 100 if total_services > 0 else 0
        
        integration_status = {
            "overall_score": round(integration_score, 2),
            "healthy_services": healthy_services,
            "total_services": total_services,
            "services_status": health_status.get("services", {}),
            "capabilities": capabilities,
            "integration_health": "excellent" if integration_score >= 90 else "good" if integration_score >= 70 else "needs_attention",
            "timestamp": datetime.now().isoformat()
        }
        
        return integration_status
        
    except Exception as e:
        logger.error(f"Integration status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-integration")
async def test_ecosystem_integration() -> Dict[str, Any]:
    """
    Test integration with all ecosystem services
    
    Returns:
        Integration test results
    """
    try:
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "overall_success": True
        }
        
        # Test each service
        for service_type in ServiceType:
            service_name = service_type.value
            
            try:
                # Test health check
                health_result = await ecosystem_service.check_service_health(service_type)
                
                test_results["tests"][service_name] = {
                    "health_check": health_result.get("status", "unknown"),
                    "response_time_ms": health_result.get("response_time_ms", 0),
                    "success": health_result.get("status") == "healthy"
                }
                
                if health_result.get("status") != "healthy":
                    test_results["overall_success"] = False
                    
            except Exception as e:
                test_results["tests"][service_name] = {
                    "health_check": "error",
                    "error": str(e),
                    "success": False
                }
                test_results["overall_success"] = False
        
        # Calculate test summary
        total_tests = len(test_results["tests"])
        successful_tests = sum(1 for test in test_results["tests"].values() if test.get("success", False))
        test_results["summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        }
        
        return test_results
        
    except Exception as e:
        logger.error(f"Ecosystem integration test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
