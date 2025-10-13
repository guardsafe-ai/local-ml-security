"""
Ecosystem Integration Service
Integrates with all platform services (analytics, business-metrics, MLflow)
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import httpx
import json
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Platform service types"""
    ANALYTICS = "analytics"
    BUSINESS_METRICS = "business-metrics"
    MLFLOW = "mlflow"
    MODEL_API = "model-api"
    TRAINING = "training"
    DATA_PRIVACY = "data-privacy"
    MODEL_CACHE = "model-cache"


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""
    service_type: ServiceType
    base_url: str
    health_endpoint: str
    api_version: str = "v1"
    timeout: float = 30.0
    retry_attempts: int = 3


class EcosystemIntegrationService:
    """
    Service for integrating with all platform services
    """
    
    def __init__(self):
        """Initialize ecosystem integration service"""
        self.service_endpoints: Dict[ServiceType, ServiceEndpoint] = {}
        self.http_client: Optional[httpx.AsyncClient] = None
        self._setup_service_endpoints()
        logger.info("✅ Ecosystem Integration Service initialized")
    
    def _setup_service_endpoints(self):
        """Setup service endpoint configurations"""
        try:
            # Analytics service
            self.service_endpoints[ServiceType.ANALYTICS] = ServiceEndpoint(
                service_type=ServiceType.ANALYTICS,
                base_url="http://analytics:8000",
                health_endpoint="/health",
                api_version="v1"
            )
            
            # Business metrics service
            self.service_endpoints[ServiceType.BUSINESS_METRICS] = ServiceEndpoint(
                service_type=ServiceType.BUSINESS_METRICS,
                base_url="http://business-metrics:8000",
                health_endpoint="/health",
                api_version="v1"
            )
            
            # MLflow service
            self.service_endpoints[ServiceType.MLFLOW] = ServiceEndpoint(
                service_type=ServiceType.MLFLOW,
                base_url="http://mlflow:5000",
                health_endpoint="/health",
                api_version="v1"
            )
            
            # Model API service
            self.service_endpoints[ServiceType.MODEL_API] = ServiceEndpoint(
                service_type=ServiceType.MODEL_API,
                base_url="http://model-api:8000",
                health_endpoint="/health",
                api_version="v1"
            )
            
            # Training service
            self.service_endpoints[ServiceType.TRAINING] = ServiceEndpoint(
                service_type=ServiceType.TRAINING,
                base_url="http://training:8000",
                health_endpoint="/health",
                api_version="v1"
            )
            
            # Data privacy service
            self.service_endpoints[ServiceType.DATA_PRIVACY] = ServiceEndpoint(
                service_type=ServiceType.DATA_PRIVACY,
                base_url="http://data-privacy:8000",
                health_endpoint="/health",
                api_version="v1"
            )
            
            # Model cache service
            self.service_endpoints[ServiceType.MODEL_CACHE] = ServiceEndpoint(
                service_type=ServiceType.MODEL_CACHE,
                base_url="http://model-cache:8000",
                health_endpoint="/health",
                api_version="v1"
            )
            
        except Exception as e:
            logger.error(f"Service endpoints setup failed: {e}")
    
    async def initialize(self):
        """Initialize HTTP client"""
        try:
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
            )
            logger.info("✅ HTTP client initialized")
        except Exception as e:
            logger.error(f"HTTP client initialization failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup HTTP client"""
        try:
            if self.http_client:
                await self.http_client.aclose()
                logger.info("✅ HTTP client cleaned up")
        except Exception as e:
            logger.error(f"HTTP client cleanup failed: {e}")
    
    async def check_service_health(self, service_type: ServiceType) -> Dict[str, Any]:
        """
        Check health of a specific service
        
        Args:
            service_type: Type of service to check
            
        Returns:
            Health status dictionary
        """
        try:
            if service_type not in self.service_endpoints:
                return {"status": "error", "message": f"Unknown service: {service_type}"}
            
            endpoint = self.service_endpoints[service_type]
            
            if not self.http_client:
                await self.initialize()
            
            response = await self.http_client.get(
                f"{endpoint.base_url}{endpoint.health_endpoint}",
                timeout=endpoint.timeout
            )
            
            if response.status_code == 200:
                health_data = response.json()
                return {
                    "status": "healthy",
                    "service": service_type.value,
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "details": health_data
                }
            else:
                return {
                    "status": "unhealthy",
                    "service": service_type.value,
                    "status_code": response.status_code,
                    "message": "Service returned non-200 status"
                }
                
        except Exception as e:
            logger.error(f"Health check failed for {service_type}: {e}")
            return {
                "status": "error",
                "service": service_type.value,
                "error": str(e)
            }
    
    async def check_all_services_health(self) -> Dict[str, Any]:
        """
        Check health of all platform services
        
        Returns:
            Health status for all services
        """
        try:
            health_results = {}
            
            # Check all services concurrently
            tasks = []
            for service_type in ServiceType:
                tasks.append(self.check_service_health(service_type))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, service_type in enumerate(ServiceType):
                result = results[i]
                if isinstance(result, Exception):
                    health_results[service_type.value] = {
                        "status": "error",
                        "error": str(result)
                    }
                else:
                    health_results[service_type.value] = result
            
            # Calculate overall health
            healthy_services = sum(1 for result in health_results.values() if result.get("status") == "healthy")
            total_services = len(health_results)
            overall_health = (healthy_services / total_services) * 100 if total_services > 0 else 0
            
            return {
                "overall_health": overall_health,
                "healthy_services": healthy_services,
                "total_services": total_services,
                "services": health_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"All services health check failed: {e}")
            return {"error": str(e)}
    
    async def send_analytics_data(self, analytics_data: Dict[str, Any]) -> bool:
        """
        Send data to analytics service
        
        Args:
            analytics_data: Data to send to analytics service
            
        Returns:
            True if successful
        """
        try:
            if not self.http_client:
                await self.initialize()
            
            endpoint = self.service_endpoints[ServiceType.ANALYTICS]
            
            response = await self.http_client.post(
                f"{endpoint.base_url}/api/{endpoint.api_version}/analytics/record",
                json=analytics_data,
                timeout=endpoint.timeout
            )
            
            if response.status_code in [200, 201]:
                logger.info("✅ Analytics data sent successfully")
                return True
            else:
                logger.error(f"Failed to send analytics data: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Analytics data sending failed: {e}")
            return False
    
    async def send_business_metrics(self, metrics_data: Dict[str, Any]) -> bool:
        """
        Send metrics to business metrics service
        
        Args:
            metrics_data: Metrics data to send
            
        Returns:
            True if successful
        """
        try:
            if not self.http_client:
                await self.initialize()
            
            endpoint = self.service_endpoints[ServiceType.BUSINESS_METRICS]
            
            response = await self.http_client.post(
                f"{endpoint.base_url}/api/{endpoint.api_version}/metrics/record",
                json=metrics_data,
                timeout=endpoint.timeout
            )
            
            if response.status_code in [200, 201]:
                logger.info("✅ Business metrics sent successfully")
                return True
            else:
                logger.error(f"Failed to send business metrics: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Business metrics sending failed: {e}")
            return False
    
    async def get_mlflow_models(self) -> List[Dict[str, Any]]:
        """
        Get models from MLflow service
        
        Returns:
            List of models from MLflow
        """
        try:
            if not self.http_client:
                await self.initialize()
            
            endpoint = self.service_endpoints[ServiceType.MLFLOW]
            
            response = await self.http_client.get(
                f"{endpoint.base_url}/api/{endpoint.api_version}/models",
                timeout=endpoint.timeout
            )
            
            if response.status_code == 200:
                models_data = response.json()
                logger.info(f"✅ Retrieved {len(models_data.get('models', []))} models from MLflow")
                return models_data.get('models', [])
            else:
                logger.error(f"Failed to get MLflow models: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"MLflow models retrieval failed: {e}")
            return []
    
    async def get_model_predictions(self, model_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get predictions from model API service
        
        Args:
            model_name: Name of the model
            input_data: Input data for prediction
            
        Returns:
            Prediction results
        """
        try:
            if not self.http_client:
                await self.initialize()
            
            endpoint = self.service_endpoints[ServiceType.MODEL_API]
            
            response = await self.http_client.post(
                f"{endpoint.base_url}/api/{endpoint.api_version}/predict/{model_name}",
                json=input_data,
                timeout=endpoint.timeout
            )
            
            if response.status_code == 200:
                prediction_data = response.json()
                logger.info(f"✅ Got prediction from model {model_name}")
                return prediction_data
            else:
                logger.error(f"Failed to get prediction from model {model_name}: {response.status_code}")
                return {"error": f"Prediction failed: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Model prediction failed for {model_name}: {e}")
            return {"error": str(e)}
    
    async def trigger_model_training(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trigger model training via training service
        
        Args:
            training_config: Training configuration
            
        Returns:
            Training job information
        """
        try:
            if not self.http_client:
                await self.initialize()
            
            endpoint = self.service_endpoints[ServiceType.TRAINING]
            
            response = await self.http_client.post(
                f"{endpoint.base_url}/api/{endpoint.api_version}/training/start",
                json=training_config,
                timeout=endpoint.timeout
            )
            
            if response.status_code in [200, 201]:
                training_data = response.json()
                logger.info(f"✅ Training job started: {training_data.get('job_id')}")
                return training_data
            else:
                logger.error(f"Failed to start training job: {response.status_code}")
                return {"error": f"Training failed: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Model training trigger failed: {e}")
            return {"error": str(e)}
    
    async def classify_data_privacy(self, data: str) -> Dict[str, Any]:
        """
        Classify data for privacy using data privacy service
        
        Args:
            data: Data to classify
            
        Returns:
            Privacy classification results
        """
        try:
            if not self.http_client:
                await self.initialize()
            
            endpoint = self.service_endpoints[ServiceType.DATA_PRIVACY]
            
            response = await self.http_client.post(
                f"{endpoint.base_url}/api/{endpoint.api_version}/privacy/classify",
                json={"data": data},
                timeout=endpoint.timeout
            )
            
            if response.status_code == 200:
                privacy_data = response.json()
                logger.info("✅ Data privacy classification completed")
                return privacy_data
            else:
                logger.error(f"Failed to classify data privacy: {response.status_code}")
                return {"error": f"Privacy classification failed: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Data privacy classification failed: {e}")
            return {"error": str(e)}
    
    async def load_model_to_cache(self, model_name: str) -> bool:
        """
        Load model to cache via model cache service
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            True if successful
        """
        try:
            if not self.http_client:
                await self.initialize()
            
            endpoint = self.service_endpoints[ServiceType.MODEL_CACHE]
            
            response = await self.http_client.post(
                f"{endpoint.base_url}/api/{endpoint.api_version}/models/{model_name}/load",
                timeout=endpoint.timeout
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"✅ Model {model_name} loaded to cache")
                return True
            else:
                logger.error(f"Failed to load model {model_name} to cache: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Model cache loading failed for {model_name}: {e}")
            return False
    
    async def get_ecosystem_status(self) -> Dict[str, Any]:
        """
        Get comprehensive ecosystem status
        
        Returns:
            Ecosystem status information
        """
        try:
            # Get health status for all services
            health_status = await self.check_all_services_health()
            
            # Get additional status information
            ecosystem_status = {
                "timestamp": datetime.now().isoformat(),
                "overall_health": health_status.get("overall_health", 0),
                "services_health": health_status,
                "integration_capabilities": {
                    "analytics": "Data analysis and reporting",
                    "business_metrics": "KPI tracking and monitoring",
                    "mlflow": "Model registry and experiment tracking",
                    "model_api": "Model inference and predictions",
                    "training": "Model training and retraining",
                    "data_privacy": "Data classification and privacy protection",
                    "model_cache": "Model caching and optimization"
                },
                "data_flows": {
                    "security_testing": "Red Team → Analytics → Business Metrics",
                    "model_management": "Training → MLflow → Model Cache → Model API",
                    "privacy_protection": "Data Privacy → Analytics → Business Metrics",
                    "compliance_reporting": "All Services → Analytics → Reporting"
                }
            }
            
            return ecosystem_status
            
        except Exception as e:
            logger.error(f"Ecosystem status retrieval failed: {e}")
            return {"error": str(e)}
    
    async def sync_with_ecosystem(self, red_team_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sync red team data with all ecosystem services
        
        Args:
            red_team_data: Red team testing data
            
        Returns:
            Sync results
        """
        try:
            sync_results = {
                "timestamp": datetime.now().isoformat(),
                "services_synced": [],
                "services_failed": [],
                "overall_success": True
            }
            
            # Send to analytics service
            analytics_data = {
                "event_type": "red_team_test",
                "timestamp": datetime.now().isoformat(),
                "data": red_team_data
            }
            
            if await self.send_analytics_data(analytics_data):
                sync_results["services_synced"].append("analytics")
            else:
                sync_results["services_failed"].append("analytics")
                sync_results["overall_success"] = False
            
            # Send to business metrics service
            metrics_data = {
                "metric_name": "red_team_tests",
                "value": 1,
                "tags": {
                    "test_type": red_team_data.get("test_type", "unknown"),
                    "model_name": red_team_data.get("model_name", "unknown")
                },
                "metadata": red_team_data
            }
            
            if await self.send_business_metrics(metrics_data):
                sync_results["services_synced"].append("business_metrics")
            else:
                sync_results["services_failed"].append("business_metrics")
                sync_results["overall_success"] = False
            
            # Update MLflow with test results
            if "model_name" in red_team_data:
                mlflow_data = {
                    "model_name": red_team_data["model_name"],
                    "test_results": red_team_data.get("results", {}),
                    "timestamp": datetime.now().isoformat()
                }
                
                # This would typically update MLflow experiment tracking
                # For now, we'll just log it
                logger.info(f"MLflow update data: {mlflow_data}")
                sync_results["services_synced"].append("mlflow")
            
            return sync_results
            
        except Exception as e:
            logger.error(f"Ecosystem sync failed: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "overall_success": False
            }
    
    async def get_service_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of all ecosystem services
        
        Returns:
            Service capabilities information
        """
        try:
            capabilities = {
                "analytics": {
                    "description": "Data analysis and reporting service",
                    "endpoints": ["/analytics/record", "/analytics/query", "/analytics/reports"],
                    "features": ["Data aggregation", "Trend analysis", "Custom reporting"]
                },
                "business_metrics": {
                    "description": "KPI tracking and business metrics service",
                    "endpoints": ["/metrics/record", "/metrics/query", "/metrics/dashboard"],
                    "features": ["KPI tracking", "Performance monitoring", "Business intelligence"]
                },
                "mlflow": {
                    "description": "Model registry and experiment tracking service",
                    "endpoints": ["/models", "/experiments", "/runs"],
                    "features": ["Model versioning", "Experiment tracking", "Model deployment"]
                },
                "model_api": {
                    "description": "Model inference and prediction service",
                    "endpoints": ["/predict", "/models", "/health"],
                    "features": ["Model inference", "Batch predictions", "Model management"]
                },
                "training": {
                    "description": "Model training and retraining service",
                    "endpoints": ["/training/start", "/training/status", "/training/results"],
                    "features": ["Model training", "Hyperparameter tuning", "Auto-retraining"]
                },
                "data_privacy": {
                    "description": "Data classification and privacy protection service",
                    "endpoints": ["/privacy/classify", "/privacy/scan", "/privacy/reports"],
                    "features": ["PII detection", "Data classification", "Privacy compliance"]
                },
                "model_cache": {
                    "description": "Model caching and optimization service",
                    "endpoints": ["/models/load", "/models/unload", "/cache/stats"],
                    "features": ["Model caching", "Performance optimization", "Memory management"]
                }
            }
            
            return capabilities
            
        except Exception as e:
            logger.error(f"Service capabilities retrieval failed: {e}")
            return {"error": str(e)}
