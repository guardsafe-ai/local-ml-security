"""
Enterprise Dashboard Backend - API Client
Handles communication with all ML Security services
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import httpx
import redis
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class APIClient:
    """Client for communicating with ML Security services"""
    
    def __init__(self):
        self.service_urls = {
            "training": "http://training:8002",
            "model_api": "http://model-api:8000", 
            "red_team": "http://red-team:8001",
            "analytics": "http://analytics:8006",
            "business_metrics": "http://business-metrics:8004",
            "data_privacy": "http://data-privacy:8005",
            "mlflow": "http://mlflow:5000",
            "minio": "http://minio:9000",
            "prometheus": "http://prometheus:9090",
            "grafana": "http://grafana:3000"
        }
        self.redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
        self.timeout = 30.0

    async def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service"""
        try:
            # Service-specific health endpoints
            if service_name == "mlflow":
                urls_to_try = [f"{self.service_urls[service_name]}/health"]
            elif service_name == "minio":
                urls_to_try = [
                    f"{self.service_urls[service_name]}/minio/health/live",
                    f"{self.service_urls[service_name]}/minio/health/ready"
                ]
            elif service_name == "prometheus":
                urls_to_try = [f"{self.service_urls[service_name]}/-/healthy"]
            elif service_name == "grafana":
                urls_to_try = [
                    f"{self.service_urls[service_name]}/api/health",
                    f"{self.service_urls[service_name]}/",
                    f"{self.service_urls[service_name]}/login"
                ]
            elif service_name == "data_privacy":
                # Data privacy service has health endpoint at /health/health
                urls_to_try = [
                    f"{self.service_urls[service_name]}/health/health",
                    f"{self.service_urls[service_name]}/health",
                    f"{self.service_urls[service_name]}/"
                ]
            else:
                # Try both /health and /health/ endpoints to handle redirects
                urls_to_try = [
                    f"{self.service_urls[service_name]}/health",
                    f"{self.service_urls[service_name]}/health/",
                    f"{self.service_urls[service_name]}/"
                ]
            
            for url in urls_to_try:
                try:
                    async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                        start_time = datetime.now()
                        response = await client.get(url)
                        response_time = (datetime.now() - start_time).total_seconds() * 1000
                        
                        if response.status_code in [200, 302]:
                            try:
                                data = response.json()
                            except:
                                # Handle non-JSON responses (like MLflow's "OK", Prometheus health)
                                if service_name == "prometheus":
                                    data = {"status": "healthy", "response": response.text.strip()}
                                elif service_name == "minio":
                                    data = {"status": "healthy", "response": response.text.strip()}
                                elif service_name == "grafana" and response.status_code == 302:
                                    data = {"status": "healthy", "response": "redirecting to login"}
                                else:
                                    data = {"response": response.text.strip()}
                            return {
                                "name": service_name,
                                "status": "healthy",
                                "response_time": response_time,
                                "last_check": datetime.now(),
                                "details": data
                            }
                except Exception:
                    continue
            
            # If all URLs failed, return unhealthy
            return {
                "name": service_name,
                "status": "unhealthy",
                "response_time": 0.0,
                "last_check": datetime.now(),
                "details": {"error": "All health check URLs failed"}
            }
        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            return {
                "name": service_name,
                "status": "unhealthy",
                "response_time": 0.0,
                "last_check": datetime.now(),
                "details": {"error": str(e)}
            }

    async def get_all_services_health(self) -> List[Dict[str, Any]]:
        """Check health of all services"""
        tasks = [self.check_service_health(service) for service in self.service_urls.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_status = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Health check failed: {result}")
                health_status.append({
                    "name": "unknown",
                    "status": "error",
                    "response_time": 0.0,
                    "last_check": datetime.now(),
                    "details": {"error": str(result)}
                })
            else:
                health_status.append(result)
        
        return health_status

    async def get_training_jobs(self) -> List[Dict[str, Any]]:
        """Get training jobs from training service"""
        try:
            url = f"{self.service_urls['training']}/training/jobs"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to get training jobs: {response.status_code}")
                    return []
        except Exception as e:
            logger.error(f"Error getting training jobs: {e}")
            return []

    async def get_job_logs(self, job_id: str) -> Dict[str, Any]:
        """Get logs for a specific training job"""
        try:
            url = f"{self.service_urls['training']}/training/jobs/{job_id}/logs"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to get job logs: {response.status_code}")
                    return {"job_id": job_id, "logs": []}
        except Exception as e:
            logger.error(f"Error getting job logs: {e}")
            return {"job_id": job_id, "logs": []}

    async def get_available_models(self) -> Dict[str, Any]:
        """Get available models from model-api service"""
        try:
            url = f"{self.service_urls['model_api']}/models"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to get available models: {response.status_code}")
                    return {"models": {}, "available_models": [], "mlflow_models": []}
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return {"models": {}, "available_models": [], "mlflow_models": []}

    async def get_model_registry(self) -> Dict[str, Any]:
        """Get model registry from training service"""
        try:
            url = f"{self.service_urls['training']}/models/model-registry"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to get model registry: {response.status_code}")
                    return {"models": [], "total_models": 0}
        except Exception as e:
            logger.error(f"Error getting model registry: {e}")
            return {"models": [], "total_models": 0}

    async def get_latest_models(self) -> Dict[str, Any]:
        """Get latest models from training service"""
        try:
            url = f"{self.service_urls['training']}/models/latest-models"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to get latest models: {response.status_code}")
                    return {"models": [], "count": 0}
        except Exception as e:
            logger.error(f"Error getting latest models: {e}")
            return {"models": [], "count": 0}

    async def get_best_models(self) -> Dict[str, Any]:
        """Get best models from training service"""
        try:
            url = f"{self.service_urls['training']}/models/best-models"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to get best models: {response.status_code}")
                    return {"models": [], "count": 0}
        except Exception as e:
            logger.error(f"Error getting best models: {e}")
            return {"models": [], "count": 0}

    async def get_red_team_results(self) -> List[Dict[str, Any]]:
        """Get red team results from red-team service"""
        try:
            url = f"{self.service_urls['red_team']}/red-team/results"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("results", [])
                else:
                    logger.error(f"Failed to get red team results: {response.status_code}")
                    return []
        except Exception as e:
            logger.error(f"Error getting red team results: {e}")
            return []

    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary from analytics service"""
        try:
            url = f"{self.service_urls['analytics']}/red-team/summary"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to get analytics summary: {response.status_code}")
                    return {"summary": []}
        except Exception as e:
            logger.error(f"Error getting analytics summary: {e}")
            return {"summary": []}

    async def get_mlflow_experiments(self) -> List[Dict[str, Any]]:
        """Get MLflow experiments"""
        try:
            url = f"{self.service_urls['mlflow']}/api/2.0/mlflow/experiments/list"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("experiments", [])
                else:
                    logger.error(f"Failed to get MLflow experiments: {response.status_code}")
                    return []
        except Exception as e:
            logger.error(f"Error getting MLflow experiments: {e}")
            return []

    async def start_training(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Start training job"""
        try:
            url = f"{self.service_urls['training']}/training/train"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=request_data)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to start training: {response.status_code}")
                    return {"status": "error", "message": "Failed to start training"}
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            return {"status": "error", "message": str(e)}

    async def train_model(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train a model"""
        try:
            url = f"{self.service_urls['training']}/training/train"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=request_data)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to train model: {response.status_code}")
                    return {"status": "error", "message": "Failed to train model"}
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {"status": "error", "message": str(e)}

    async def train_loaded_model(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train a loaded model"""
        try:
            url = f"{self.service_urls['training']}/training/train-loaded"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=request_data)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to train loaded model: {response.status_code}")
                    return {"status": "error", "message": "Failed to train loaded model"}
        except Exception as e:
            logger.error(f"Error training loaded model: {e}")
            return {"status": "error", "message": str(e)}

    async def predict_model(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make model prediction"""
        try:
            url = f"{self.service_urls['model_api']}/predict"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=request_data)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to make prediction: {response.status_code}")
                    return {"error": "Prediction failed"}
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {"error": str(e)}

    async def start_red_team_test(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Start red team test"""
        try:
            url = f"{self.service_urls['red_team']}/red-team/test"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=request_data)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to start red team test: {response.status_code}")
                    return {"status": "error", "message": "Failed to start test"}
        except Exception as e:
            logger.error(f"Error starting red team test: {e}")
            return {"status": "error", "message": str(e)}

    async def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a model"""
        try:
            url = f"{self.service_urls['model_api']}/load"
            request_data = {"model_name": model_name}
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=request_data)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to load model: {response.status_code}")
                    return {"status": "error", "message": "Failed to load model"}
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return {"status": "error", "message": str(e)}

    async def unload_model(self, model_name: str) -> Dict[str, Any]:
        """Unload a model"""
        try:
            url = f"{self.service_urls['model_api']}/unload"
            request_data = {"model_name": model_name}
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=request_data)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to unload model: {response.status_code}")
                    return {"status": "error", "message": "Failed to unload model"}
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return {"status": "error", "message": str(e)}

    async def get_monitoring_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics"""
        try:
            url = f"{self.service_urls['monitoring']}/metrics"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to get monitoring metrics: {response.status_code}")
                    return {}
        except Exception as e:
            logger.error(f"Error getting monitoring metrics: {e}")
            return {}

    def cache_data(self, key: str, data: Any, ttl: int = 300):
        """Cache data in Redis"""
        try:
            import json
            self.redis_client.setex(key, ttl, json.dumps(data, default=str))
        except Exception as e:
            logger.error(f"Failed to cache data: {e}")

    def get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data from Redis"""
        try:
            import json
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Failed to get cached data: {e}")
            return None

    async def get_business_metrics_summary(self) -> Dict[str, Any]:
        """Get business metrics summary from business-metrics service"""
        try:
            url = f"{self.service_urls['business_metrics']}/business-metrics/summary"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to get business metrics summary: {response.status_code}")
                    return {"summary": {}}
        except Exception as e:
            logger.error(f"Error getting business metrics summary: {e}")
            return {"summary": {}}

    async def get_data_privacy_summary(self) -> Dict[str, Any]:
        """Get data privacy summary from data-privacy service"""
        try:
            url = f"{self.service_urls['data_privacy']}/data-privacy/summary"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to get data privacy summary: {response.status_code}")
                    return {"summary": {}}
        except Exception as e:
            logger.error(f"Error getting data privacy summary: {e}")
            return {"summary": {}}

    # =============================================================================
    # DATA MANAGEMENT METHODS
    # =============================================================================

    async def upload_large_file(self, file, data_type: str, description: str, validation_rules: str) -> Dict[str, Any]:
        """Upload large file to training service"""
        try:
            url = f"{self.service_urls['training']}/data/efficient/upload-large-file"
            
            # Prepare form data
            files = {"file": (file.filename, file.file, file.content_type)}
            data = {
                "data_type": data_type,
                "description": description,
                "validation_rules": validation_rules
            }
            
            async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minutes timeout for uploads
                response = await client.post(url, files=files, data=data)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to upload large file: {response.status_code}")
                    raise HTTPException(status_code=response.status_code, detail="Upload failed")
        except Exception as e:
            logger.error(f"Error uploading large file: {e}")
            raise

    async def get_upload_progress(self, file_id: str) -> Dict[str, Any]:
        """Get upload progress for a file"""
        try:
            url = f"{self.service_urls['training']}/data/efficient/upload-progress/{file_id}"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to get upload progress: {response.status_code}")
                    return {"error": "Failed to get progress"}
        except Exception as e:
            logger.error(f"Error getting upload progress: {e}")
            return {"error": str(e)}

    async def get_staged_files(self, status: Optional[str] = None) -> Dict[str, Any]:
        """Get staged files from training service"""
        try:
            url = f"{self.service_urls['training']}/data/efficient/staged-files"
            params = {"status": status} if status else {}
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to get staged files: {response.status_code}")
                    return {"files": [], "count": 0}
        except Exception as e:
            logger.error(f"Error getting staged files: {e}")
            return {"files": [], "count": 0}

    async def process_file(self, file_id: str, validation_rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a file"""
        try:
            url = f"{self.service_urls['training']}/data/efficient/process-file/{file_id}"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=validation_rules or {})
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to process file: {response.status_code}")
                    return {"error": "Processing failed"}
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return {"error": str(e)}

    async def download_file(self, file_id: str) -> Any:
        """Download a file"""
        try:
            url = f"{self.service_urls['training']}/data/efficient/download-file/{file_id}"
            async with httpx.AsyncClient(timeout=60.0) as client:  # Longer timeout for downloads
                response = await client.get(url)
                if response.status_code == 200:
                    return response.content
                else:
                    logger.error(f"Failed to download file: {response.status_code}")
                    raise HTTPException(status_code=response.status_code, detail="Download failed")
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            raise

    async def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """Get file information"""
        try:
            url = f"{self.service_urls['training']}/data/efficient/file-info/{file_id}"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to get file info: {response.status_code}")
                    return {"error": "File not found"}
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {"error": str(e)}

    async def retry_failed_file(self, file_id: str) -> Dict[str, Any]:
        """Retry processing a failed file"""
        try:
            url = f"{self.service_urls['training']}/data/efficient/retry-failed-file/{file_id}"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to retry file: {response.status_code}")
                    return {"error": "Retry failed"}
        except Exception as e:
            logger.error(f"Error retrying file: {e}")
            return {"error": str(e)}

    async def cleanup_failed_uploads(self, hours_old: int = 24) -> Dict[str, Any]:
        """Cleanup failed uploads"""
        try:
            url = f"{self.service_urls['training']}/data/efficient/cleanup-failed-uploads"
            params = {"hours_old": hours_old}
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(url, params=params)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to cleanup uploads: {response.status_code}")
                    return {"error": "Cleanup failed"}
        except Exception as e:
            logger.error(f"Error cleaning up uploads: {e}")
            return {"error": str(e)}

    # =============================================================================
    # TRAINING CONFIGURATION METHODS
    # =============================================================================

    async def get_training_config(self, model_name: str) -> Dict[str, Any]:
        """Get training configuration for a specific model"""
        try:
            url = f"{self.service_urls['training']}/training/config/{model_name}"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    raise HTTPException(status_code=404, detail=f"No training configuration found for model: {model_name}")
                else:
                    logger.error(f"Failed to get training config: {response.status_code}")
                    raise HTTPException(status_code=response.status_code, detail="Failed to get training configuration")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting training config: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def update_training_config(self, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Save training configuration for a specific model"""
        try:
            url = f"{self.service_urls['training']}/training/config/{model_name}"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.put(url, json=config)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to save training config: {response.status_code}")
                    raise HTTPException(status_code=response.status_code, detail="Failed to save training configuration")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error saving training config: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def list_training_configs(self) -> Dict[str, Any]:
        """List all saved training configurations"""
        try:
            url = f"{self.service_urls['training']}/training/configs"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to list training configs: {response.status_code}")
                    return {"configurations": [], "count": 0}
        except Exception as e:
            logger.error(f"Error listing training configs: {e}")
            return {"configurations": [], "count": 0}

    async def delete_training_config(self, model_name: str) -> Dict[str, Any]:
        """Delete training configuration for a specific model"""
        try:
            url = f"{self.service_urls['training']}/training/config/{model_name}"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to delete training config: {response.status_code}")
                    raise HTTPException(status_code=response.status_code, detail="Failed to delete training configuration")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting training config: {e}")
            raise HTTPException(status_code=500, detail=str(e))
