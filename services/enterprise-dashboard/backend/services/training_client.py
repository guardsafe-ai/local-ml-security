"""
Training Service Client
100% API Coverage for Training Service (port 8002)
"""

import logging
from typing import Dict, List, Optional, Any
from .base_client import BaseServiceClient

logger = logging.getLogger(__name__)


class TrainingClient(BaseServiceClient):
    """Client for Training Service - 100% API Coverage"""
    
    def __init__(self, base_url: str, redis_client):
        super().__init__("training", base_url, redis_client)
    
    # =============================================================================
    # HEALTH ENDPOINTS
    # =============================================================================
    
    async def get_health(self) -> Dict[str, Any]:
        """GET /health - Basic health check"""
        return await self._make_request("GET", "/health", use_cache=True, cache_ttl=60)
    
    async def get_health_deep(self) -> Dict[str, Any]:
        """GET /health/deep - Deep health check with dependencies"""
        return await self._make_request("GET", "/health/deep", use_cache=True, cache_ttl=60)
    
    async def get_root(self) -> Dict[str, Any]:
        """GET / - Root endpoint with service status"""
        return await self._make_request("GET", "/", use_cache=True, cache_ttl=60)
    
    async def get_metrics(self) -> str:
        """GET /metrics - Prometheus metrics endpoint"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/metrics")
                return response.text
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return ""
    
    # =============================================================================
    # MODEL MANAGEMENT ENDPOINTS
    # =============================================================================
    
    async def get_models(self) -> Dict[str, Any]:
        """GET /models/models - Get available models for training"""
        return await self._make_request("GET", "/models/models", use_cache=True, cache_ttl=300)
    
    async def get_model_registry(self) -> Dict[str, Any]:
        """GET /models/model-registry - Get model registry information"""
        return await self._make_request("GET", "/models/model-registry", use_cache=True, cache_ttl=300)
    
    async def get_latest_models(self) -> Dict[str, Any]:
        """GET /models/latest-models - Get latest versions of all models"""
        return await self._make_request("GET", "/models/latest-models", use_cache=True, cache_ttl=300)
    
    async def get_best_models(self) -> Dict[str, Any]:
        """GET /models/best-models - Get best performing models"""
        return await self._make_request("GET", "/models/best-models", use_cache=True, cache_ttl=300)
    
    async def get_model_versions(self, model_name: str) -> Dict[str, Any]:
        """GET /models/{model_name}/versions - Get model versions"""
        return await self._make_request("GET", f"/models/{model_name}/versions", use_cache=True, cache_ttl=300)
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """GET /models/{model_name}/info - Get detailed information about a model"""
        return await self._make_request("GET", f"/models/{model_name}/info", use_cache=True, cache_ttl=300)
    
    async def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """GET /models/{model_name}/status - Get training status for a model"""
        return await self._make_request("GET", f"/models/{model_name}/status", use_cache=True, cache_ttl=300)
    
    async def get_model_loading_status(self) -> Dict[str, Any]:
        """GET /model-loading-status - Get model loading status"""
        return await self._make_request("GET", "/model-loading-status", use_cache=True, cache_ttl=60)
    
    async def get_model_stages(self, model_name: str) -> Dict[str, Any]:
        """GET /models/{model_name}/stages - Get model stages"""
        return await self._make_request("GET", f"/models/{model_name}/stages", use_cache=True, cache_ttl=300)
    
    async def promote_model(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """POST /models/{model_name}/promote - Promote model from Staging to Production"""
        params = {"version": version} if version else {}
        return await self._make_request("POST", f"/models/{model_name}/promote", params=params)
    
    async def rollback_model(self, model_name: str, version: str) -> Dict[str, Any]:
        """POST /models/{model_name}/rollback - Rollback to a previous Production version"""
        params = {"version": version}
        return await self._make_request("POST", f"/models/{model_name}/rollback", params=params)
    
    # =============================================================================
    # TRAINING ENDPOINTS
    # =============================================================================
    
    async def get_training_jobs(self, status: Optional[str] = None, 
                               model_name: Optional[str] = None,
                               limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """GET /training/jobs - List all training jobs with filtering"""
        params = {
            "status": status,
            "model_name": model_name,
            "limit": limit,
            "offset": offset
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/training/jobs", params=params, use_cache=True, cache_ttl=60)
    
    async def get_training_job(self, job_id: str) -> Dict[str, Any]:
        """GET /training/jobs/{job_id} - Get specific training job"""
        return await self._make_request("GET", f"/training/jobs/{job_id}", use_cache=True, cache_ttl=30)
    
    async def get_job_logs(self, job_id: str) -> Dict[str, Any]:
        """GET /training/jobs/{job_id}/logs - Get logs for a specific training job"""
        return await self._make_request("GET", f"/training/jobs/{job_id}/logs", use_cache=True, cache_ttl=30)
    
    async def start_training(self, model_name: str, training_data_path: str,
                           config: Optional[Dict[str, Any]] = None,
                           priority: str = "normal") -> Dict[str, Any]:
        """POST /training/train - Start a new training job"""
        data = {
            "model_name": model_name,
            "training_data_path": training_data_path,
            "config": config or {},
            "priority": priority
        }
        return await self._make_request("POST", "/training/train", data=data)
    
    async def train_loaded_model(self, model_name: str, training_data_path: str,
                               config: Optional[Dict[str, Any]] = None,
                               priority: str = "normal") -> Dict[str, Any]:
        """POST /training/train-loaded - Train a loaded model"""
        data = {
            "model_name": model_name,
            "training_data_path": training_data_path,
            "config": config or {},
            "priority": priority
        }
        return await self._make_request("POST", "/training/train-loaded", data=data)
    
    async def retrain_model(self, model_name: str, training_data_path: str,
                          config: Optional[Dict[str, Any]] = None,
                          retrain: bool = True) -> Dict[str, Any]:
        """POST /training/retrain - Retrain a model"""
        data = {
            "model_name": model_name,
            "training_data_path": training_data_path,
            "config": config or {},
            "retrain": retrain
        }
        return await self._make_request("POST", "/training/retrain", data=data)
    
    async def stop_training(self, job_id: str) -> Dict[str, Any]:
        """POST /training/stop/{job_id} - Stop a training job"""
        return await self._make_request("POST", f"/training/stop/{job_id}")
    
    async def start_model_loading(self) -> Dict[str, Any]:
        """POST /training/start-model-loading - Start model loading process"""
        return await self._make_request("POST", "/training/start-model-loading")
    
    async def load_specific_models(self, model_names: List[str]) -> Dict[str, Any]:
        """POST /training/load-models - Load specific models"""
        data = {"model_names": model_names}
        return await self._make_request("POST", "/training/load-models", data=data)
    
    async def load_single_model(self, model_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """POST /training/load-model - Load a single model"""
        data = {"model_name": model_name, "config": config or {}}
        return await self._make_request("POST", "/training/load-model", data=data)
    
    async def advanced_retrain(self, model_name: str, version: str, training_data_path: str,
                              config: Dict[str, Any], reason: str, priority: str = "normal") -> Dict[str, Any]:
        """POST /training/advanced-retrain - Advanced retraining with custom parameters"""
        data = {
            "model_name": model_name,
            "version": version,
            "training_data_path": training_data_path,
            "config": config,
            "reason": reason,
            "priority": priority
        }
        return await self._make_request("POST", "/training/advanced-retrain", data=data)
    
    # =============================================================================
    # TRAINING CONFIGURATION ENDPOINTS
    # =============================================================================
    
    async def get_training_config(self, model_name: str) -> Dict[str, Any]:
        """GET /training/config/{model_name} - Get training configuration for a specific model"""
        return await self._make_request("GET", f"/training/config/{model_name}", use_cache=True, cache_ttl=300)
    
    async def update_training_config(self, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """PUT /training/config/{model_name} - Save training configuration for a specific model"""
        return await self._make_request("PUT", f"/training/config/{model_name}", data=config)
    
    async def list_training_configs(self) -> Dict[str, Any]:
        """GET /training/configs - List all saved training configurations"""
        return await self._make_request("GET", "/training/configs", use_cache=True, cache_ttl=300)
    
    async def delete_training_config(self, model_name: str) -> Dict[str, Any]:
        """DELETE /training/config/{model_name} - Delete training configuration for a specific model"""
        return await self._make_request("DELETE", f"/training/config/{model_name}")
    
    # =============================================================================
    # DATA MANAGEMENT ENDPOINTS
    # =============================================================================
    
    async def upload_data(self, file, data_type: str = "training", 
                         description: str = "", validation_rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """POST /data/upload-data - Upload training data file to MinIO"""
        files = {"file": (file.filename, file.file, file.content_type)}
        data = {
            "data_type": data_type,
            "description": description,
            "validation_rules": validation_rules or {}
        }
        return await self._make_request("POST", "/data/upload-data", data=data, files=files, timeout=self.long_timeout)
    
    async def upload_large_file(self, file, data_type: str = "custom",
                               description: str = "", validation_rules: str = "{}") -> Dict[str, Any]:
        """POST /data/efficient/upload-large-file - Upload large file with chunked upload support"""
        files = {"file": (file.filename, file.file, file.content_type)}
        data = {
            "data_type": data_type,
            "description": description,
            "validation_rules": validation_rules
        }
        return await self._make_request("POST", "/data/efficient/upload-large-file", data=data, files=files, timeout=self.long_timeout)
    
    async def get_upload_progress(self, file_id: str) -> Dict[str, Any]:
        """GET /data/efficient/upload-progress/{file_id} - Get upload progress for a file"""
        return await self._make_request("GET", f"/data/efficient/upload-progress/{file_id}")
    
    async def get_staged_files(self, status: Optional[str] = None) -> Dict[str, Any]:
        """GET /data/efficient/staged-files - Get staged files from training service"""
        params = {"status": status} if status else {}
        return await self._make_request("GET", "/data/efficient/staged-files", params=params)
    
    async def process_file(self, file_id: str, validation_rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """POST /data/efficient/process-file/{file_id} - Process a file"""
        data = validation_rules or {}
        return await self._make_request("POST", f"/data/efficient/process-file/{file_id}", data=data)
    
    async def download_file(self, file_id: str) -> bytes:
        """GET /data/efficient/download-file/{file_id} - Download a file"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=self.long_timeout) as client:
                response = await client.get(f"{self.base_url}/data/efficient/download-file/{file_id}")
                return response.content
        except Exception as e:
            logger.error(f"Failed to download file {file_id}: {e}")
            raise
    
    async def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """GET /data/efficient/file-info/{file_id} - Get file information"""
        return await self._make_request("GET", f"/data/efficient/file-info/{file_id}")
    
    async def retry_failed_file(self, file_id: str) -> Dict[str, Any]:
        """POST /data/efficient/retry-failed-file/{file_id} - Retry processing a failed file"""
        return await self._make_request("POST", f"/data/efficient/retry-failed-file/{file_id}")
    
    async def cleanup_failed_uploads(self, hours_old: int = 24) -> Dict[str, Any]:
        """DELETE /data/efficient/cleanup-failed-uploads - Cleanup failed uploads"""
        params = {"hours_old": hours_old}
        return await self._make_request("DELETE", "/data/efficient/cleanup-failed-uploads", params=params)
    
    # =============================================================================
    # ADDITIONAL DATA MANAGEMENT ENDPOINTS
    # =============================================================================
    
    async def upload_multiple_data(self, files: List[Any], data_type: str = "training",
                                 description: str = "", validation_rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """POST /data/upload-multiple-data - Upload multiple training data files"""
        files_data = {}
        for i, file in enumerate(files):
            files_data[f"file_{i}"] = (file.filename, file.file, file.content_type)
        
        data = {
            "data_type": data_type,
            "description": description,
            "validation_rules": validation_rules or {}
        }
        return await self._make_request("POST", "/data/upload-multiple-data", data=data, files=files_data, timeout=self.long_timeout)
    
    async def get_fresh_data_files(self, data_type: Optional[str] = None) -> Dict[str, Any]:
        """GET /data/fresh-data - Get fresh data files"""
        params = {"data_type": data_type} if data_type else {}
        return await self._make_request("GET", "/data/fresh-data", params=params, use_cache=True, cache_ttl=60)
    
    async def get_used_data_files(self, data_type: Optional[str] = None) -> Dict[str, Any]:
        """GET /data/used-data - Get used data files"""
        params = {"data_type": data_type} if data_type else {}
        return await self._make_request("GET", "/data/used-data", params=params, use_cache=True, cache_ttl=60)
    
    async def get_data_statistics(self) -> Dict[str, Any]:
        """GET /data/data-statistics - Get data statistics"""
        return await self._make_request("GET", "/data/data-statistics", use_cache=True, cache_ttl=300)
    
    async def get_training_data_path(self, data_type: Optional[str] = None) -> Dict[str, Any]:
        """GET /data/training-data-path - Get training data path"""
        params = {"data_type": data_type} if data_type else {}
        return await self._make_request("GET", "/data/training-data-path", params=params, use_cache=True, cache_ttl=300)
    
    async def cleanup_old_data(self, days_old: int = 30) -> Dict[str, Any]:
        """DELETE /data/cleanup-old-data - Cleanup old data"""
        params = {"days_old": days_old}
        return await self._make_request("DELETE", "/data/cleanup-old-data", params=params)
    
    async def create_sample_data(self) -> Dict[str, Any]:
        """POST /data/create-sample-data - Create sample data"""
        return await self._make_request("POST", "/data/create-sample-data")
    
    async def validate_data_quality(self, data_path: str, model_type: str = "bert",
                                  custom_thresholds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """POST /data/validate-quality - Validate data quality"""
        data = {
            "data_path": data_path,
            "model_type": model_type,
            "custom_thresholds": custom_thresholds or {}
        }
        return await self._make_request("POST", "/data/validate-quality", data=data)
    
    async def get_quality_thresholds(self, model_type: str) -> Dict[str, Any]:
        """GET /data/quality-thresholds/{model_type} - Get quality thresholds for model type"""
        return await self._make_request("GET", f"/data/quality-thresholds/{model_type}", use_cache=True, cache_ttl=300)
    
    async def set_custom_quality_thresholds(self, thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """POST /data/quality-thresholds/custom - Set custom quality thresholds"""
        return await self._make_request("POST", "/data/quality-thresholds/custom", data=thresholds)
    
    # =============================================================================
    # TRAINING QUEUE ENDPOINTS
    # =============================================================================
    
    async def submit_training_job(self, model_name: str, training_data_path: str,
                                config: Optional[Dict[str, Any]] = None,
                                priority: str = "normal", description: Optional[str] = None) -> Dict[str, Any]:
        """POST /training/queue/submit - Submit training job to queue"""
        data = {
            "model_name": model_name,
            "training_data_path": training_data_path,
            "config": config or {},
            "priority": priority,
            "description": description
        }
        return await self._make_request("POST", "/training/queue/submit", data=data)
    
    async def get_queue_job_status(self, job_id: str) -> Dict[str, Any]:
        """GET /training/queue/job/{job_id} - Get job status from queue"""
        return await self._make_request("GET", f"/training/queue/job/{job_id}", use_cache=True, cache_ttl=30)
    
    async def cancel_queue_job(self, job_id: str) -> Dict[str, Any]:
        """DELETE /training/queue/job/{job_id} - Cancel job in queue"""
        return await self._make_request("DELETE", f"/training/queue/job/{job_id}")
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """GET /training/queue/stats - Get queue statistics"""
        return await self._make_request("GET", "/training/queue/stats", use_cache=True, cache_ttl=60)
    
    async def list_queue_jobs(self, status: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        """GET /training/queue/jobs - List jobs in queue"""
        params = {"status": status, "limit": limit}
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/training/queue/jobs", params=params, use_cache=True, cache_ttl=60)
    
    async def retry_queue_job(self, job_id: str) -> Dict[str, Any]:
        """POST /training/queue/retry/{job_id} - Retry failed job in queue"""
        return await self._make_request("POST", f"/training/queue/retry/{job_id}")
    
    # =============================================================================
    # DATA AUGMENTATION ENDPOINTS
    # =============================================================================
    
    async def augment_data(self, texts: List[str], labels: List[str],
                          augmentation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """POST /data/augment - Augment training data"""
        data = {
            "texts": texts,
            "labels": labels,
            "augmentation_config": augmentation_config or {}
        }
        return await self._make_request("POST", "/data/augment", data=data)
    
    async def augment_single_text(self, text: str, label: str, num_augmentations: int = 3) -> Dict[str, Any]:
        """POST /data/augment/single - Augment single text"""
        data = {
            "text": text,
            "label": label,
            "num_augmentations": num_augmentations
        }
        return await self._make_request("POST", "/data/augment/single", data=data)
    
    async def balance_dataset(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """POST /data/balance - Balance dataset"""
        data = {"texts": texts, "labels": labels}
        return await self._make_request("POST", "/data/balance", data=data)
    
    async def generate_synthetic_data(self, num_samples: int, target_distribution: Dict[str, int],
                                    model_type: str = "gpt2") -> Dict[str, Any]:
        """POST /data/synthetic - Generate synthetic data"""
        data = {
            "num_samples": num_samples,
            "target_distribution": target_distribution,
            "model_type": model_type
        }
        return await self._make_request("POST", "/data/synthetic", data=data)
    
    async def get_augmentation_config(self) -> Dict[str, Any]:
        """GET /data/augment/config - Get augmentation configuration"""
        return await self._make_request("GET", "/data/augment/config", use_cache=True, cache_ttl=300)
    
    async def update_augmentation_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """POST /data/augment/config - Update augmentation configuration"""
        return await self._make_request("POST", "/data/augment/config", data=config)
    
    async def get_available_techniques(self) -> Dict[str, Any]:
        """GET /data/augment/techniques - Get available augmentation techniques"""
        return await self._make_request("GET", "/data/augment/techniques", use_cache=True, cache_ttl=300)
    
    async def preview_augmentation(self, text: str, label: str, num_samples: int = 5) -> Dict[str, Any]:
        """POST /data/augment/preview - Preview augmentation results"""
        data = {
            "text": text,
            "label": label,
            "num_samples": num_samples
        }
        return await self._make_request("POST", "/data/augment/preview", data=data)
    
    # =============================================================================
    # CONVENIENCE METHODS
    # =============================================================================
    
    async def get_active_jobs(self) -> List[Dict[str, Any]]:
        """Get all active training jobs"""
        return await self.get_training_jobs(status="running")
    
    async def get_completed_jobs(self) -> List[Dict[str, Any]]:
        """Get all completed training jobs"""
        return await self.get_training_jobs(status="completed")
    
    async def get_failed_jobs(self) -> List[Dict[str, Any]]:
        """Get all failed training jobs"""
        return await self.get_training_jobs(status="failed")
    
    async def is_job_running(self, job_id: str) -> bool:
        """Check if a specific job is running"""
        try:
            job = await self.get_training_job(job_id)
            return job.get("status") == "running"
        except:
            return False
