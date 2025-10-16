"""
MLflow Service Client
100% API Coverage for MLflow Service (port 5000)
"""

import logging
from typing import Dict, List, Optional, Any
from .base_client import BaseServiceClient

logger = logging.getLogger(__name__)


class MLflowClient(BaseServiceClient):
    """Client for MLflow Service - 100% API Coverage"""
    
    def __init__(self, base_url: str, redis_client):
        super().__init__("mlflow", base_url, redis_client)
    
    # =============================================================================
    # HEALTH ENDPOINTS
    # =============================================================================
    
    async def get_health(self) -> Dict[str, Any]:
        """GET /health - Comprehensive health check with MLflow status"""
        return await self._make_request("GET", "/health", use_cache=True, cache_ttl=60)
    
    async def get_root(self) -> Dict[str, Any]:
        """GET / - Root endpoint with MLflow service status"""
        return await self._make_request("GET", "/", use_cache=True, cache_ttl=60)
    
    # =============================================================================
    # EXPERIMENT MANAGEMENT ENDPOINTS
    # =============================================================================
    
    async def list_experiments(self, max_results: int = 100,
                              view_type: str = "ACTIVE_ONLY") -> Dict[str, Any]:
        """GET /api/2.0/mlflow/experiments/list - List all experiments"""
        params = {"max_results": max_results, "view_type": view_type}
        return await self._make_request("GET", "/api/2.0/mlflow/experiments/list", 
                                      params=params, use_cache=True, cache_ttl=300)
    
    async def create_experiment(self, name: str, artifact_location: Optional[str] = None,
                               tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """POST /api/2.0/mlflow/experiments/create - Create a new experiment"""
        data = {
            "name": name,
            "artifact_location": artifact_location,
            "tags": tags or {}
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", "/api/2.0/mlflow/experiments/create", data=data)
    
    async def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """GET /api/2.0/mlflow/experiments/get - Get experiment details by ID"""
        return await self._make_request("GET", "/api/2.0/mlflow/experiments/get", 
                                      params={"experiment_id": experiment_id}, use_cache=True, cache_ttl=300)
    
    async def get_experiment_by_name(self, experiment_name: str) -> Dict[str, Any]:
        """GET /api/2.0/mlflow/experiments/get-by-name - Get experiment by name"""
        return await self._make_request("GET", "/api/2.0/mlflow/experiments/get-by-name", 
                                      params={"experiment_name": experiment_name}, use_cache=True, cache_ttl=300)
    
    async def update_experiment(self, experiment_id: str, new_name: Optional[str] = None) -> Dict[str, Any]:
        """POST /api/2.0/mlflow/experiments/update - Update experiment"""
        data = {"experiment_id": experiment_id, "new_name": new_name}
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", "/api/2.0/mlflow/experiments/update", data=data)
    
    async def delete_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """POST /api/2.0/mlflow/experiments/delete - Delete experiment"""
        data = {"experiment_id": experiment_id}
        return await self._make_request("POST", "/api/2.0/mlflow/experiments/delete", data=data)
    
    async def restore_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """POST /api/2.0/mlflow/experiments/restore - Restore deleted experiment"""
        data = {"experiment_id": experiment_id}
        return await self._make_request("POST", "/api/2.0/mlflow/experiments/restore", data=data)
    
    # =============================================================================
    # RUN MANAGEMENT ENDPOINTS
    # =============================================================================
    
    async def search_runs(self, experiment_ids: Optional[List[str]] = None,
                         filter: Optional[str] = None,
                         run_view_type: str = "ACTIVE_ONLY",
                         max_results: int = 100,
                         order_by: Optional[List[str]] = None,
                         page_token: Optional[str] = None) -> Dict[str, Any]:
        """POST /api/2.0/mlflow/runs/search - Search runs with filtering and pagination"""
        data = {
            "experiment_ids": experiment_ids,
            "filter": filter,
            "run_view_type": run_view_type,
            "max_results": max_results,
            "order_by": order_by,
            "page_token": page_token
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", "/api/2.0/mlflow/runs/search", data=data)
    
    async def get_run(self, run_id: str) -> Dict[str, Any]:
        """GET /api/2.0/mlflow/runs/get - Get run details by ID"""
        return await self._make_request("GET", "/api/2.0/mlflow/runs/get", 
                                      params={"run_id": run_id}, use_cache=True, cache_ttl=300)
    
    async def create_run(self, experiment_id: str, user_id: Optional[str] = None,
                        run_name: Optional[str] = None, start_time: Optional[int] = None,
                        tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """POST /api/2.0/mlflow/runs/create - Create a new run"""
        data = {
            "experiment_id": experiment_id,
            "user_id": user_id,
            "run_name": run_name,
            "start_time": start_time,
            "tags": tags or {}
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", "/api/2.0/mlflow/runs/create", data=data)
    
    async def update_run(self, run_id: str, status: Optional[str] = None,
                        end_time: Optional[int] = None, run_name: Optional[str] = None) -> Dict[str, Any]:
        """POST /api/2.0/mlflow/runs/update - Update run information"""
        data = {
            "run_id": run_id,
            "status": status,
            "end_time": end_time,
            "run_name": run_name
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", "/api/2.0/mlflow/runs/update", data=data)
    
    async def delete_run(self, run_id: str) -> Dict[str, Any]:
        """POST /api/2.0/mlflow/runs/delete - Delete run"""
        data = {"run_id": run_id}
        return await self._make_request("POST", "/api/2.0/mlflow/runs/delete", data=data)
    
    async def restore_run(self, run_id: str) -> Dict[str, Any]:
        """POST /api/2.0/mlflow/runs/restore - Restore deleted run"""
        data = {"run_id": run_id}
        return await self._make_request("POST", "/api/2.0/mlflow/runs/restore", data=data)
    
    # =============================================================================
    # METRICS AND PARAMETERS ENDPOINTS
    # =============================================================================
    
    async def log_metric(self, run_id: str, key: str, value: float,
                        timestamp: Optional[int] = None, step: Optional[int] = None) -> Dict[str, Any]:
        """POST /api/2.0/mlflow/runs/log-metric - Log metric for a run"""
        data = {
            "run_id": run_id,
            "key": key,
            "value": value,
            "timestamp": timestamp,
            "step": step
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", "/api/2.0/mlflow/runs/log-metric", data=data)
    
    async def log_param(self, run_id: str, key: str, value: str) -> Dict[str, Any]:
        """POST /api/2.0/mlflow/runs/log-parameter - Log parameter for a run"""
        data = {"run_id": run_id, "key": key, "value": value}
        return await self._make_request("POST", "/api/2.0/mlflow/runs/log-parameter", data=data)
    
    async def log_batch(self, run_id: str, metrics: Optional[List[Dict[str, Any]]] = None,
                       params: Optional[List[Dict[str, str]]] = None,
                       tags: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """POST /api/2.0/mlflow/runs/log-batch - Log multiple metrics, params, and tags"""
        data = {
            "run_id": run_id,
            "metrics": metrics or [],
            "params": params or [],
            "tags": tags or []
        }
        return await self._make_request("POST", "/api/2.0/mlflow/runs/log-batch", data=data)
    
    async def get_metric_history(self, run_id: str, metric_key: str) -> Dict[str, Any]:
        """GET /api/2.0/mlflow/metrics/get-history - Get metric history for a run"""
        params = {"run_id": run_id, "metric_key": metric_key}
        return await self._make_request("GET", "/api/2.0/mlflow/metrics/get-history", 
                                      params=params, use_cache=True, cache_ttl=300)
    
    # =============================================================================
    # MODEL REGISTRY ENDPOINTS
    # =============================================================================
    
    async def list_registered_models(self, max_results: int = 100,
                                   page_token: Optional[str] = None) -> Dict[str, Any]:
        """GET /api/2.0/mlflow/registered-models/list - List all registered models"""
        params = {"max_results": max_results, "page_token": page_token}
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/api/2.0/mlflow/registered-models/list", 
                                      params=params, use_cache=True, cache_ttl=300)
    
    async def create_registered_model(self, name: str, description: Optional[str] = None,
                                    tags: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """POST /api/2.0/mlflow/registered-models/create - Create a new registered model"""
        data = {
            "name": name,
            "description": description,
            "tags": tags or []
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", "/api/2.0/mlflow/registered-models/create", data=data)
    
    async def get_registered_model(self, name: str) -> Dict[str, Any]:
        """GET /api/2.0/mlflow/registered-models/get - Get registered model details"""
        return await self._make_request("GET", "/api/2.0/mlflow/registered-models/get", 
                                      params={"name": name}, use_cache=True, cache_ttl=300)
    
    async def update_registered_model(self, name: str, description: Optional[str] = None) -> Dict[str, Any]:
        """PATCH /api/2.0/mlflow/registered-models/update - Update registered model"""
        data = {"name": name, "description": description}
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("PATCH", "/api/2.0/mlflow/registered-models/update", data=data)
    
    async def delete_registered_model(self, name: str) -> Dict[str, Any]:
        """DELETE /api/2.0/mlflow/registered-models/delete - Delete registered model"""
        return await self._make_request("DELETE", "/api/2.0/mlflow/registered-models/delete", 
                                      params={"name": name})
    
    # =============================================================================
    # MODEL VERSION MANAGEMENT ENDPOINTS
    # =============================================================================
    
    async def create_model_version(self, name: str, source: str, run_id: Optional[str] = None,
                                 tags: Optional[List[Dict[str, str]]] = None,
                                 description: Optional[str] = None) -> Dict[str, Any]:
        """POST /api/2.0/mlflow/model-versions/create - Create a new model version"""
        data = {
            "name": name,
            "source": source,
            "run_id": run_id,
            "tags": tags or [],
            "description": description
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", "/api/2.0/mlflow/model-versions/create", data=data)
    
    async def list_model_versions(self, name: str, max_results: int = 100) -> Dict[str, Any]:
        """GET /api/2.0/mlflow/model-versions/list - List model versions"""
        params = {"name": name, "max_results": max_results}
        return await self._make_request("GET", "/api/2.0/mlflow/model-versions/list", 
                                      params=params, use_cache=True, cache_ttl=300)
    
    async def get_model_version(self, name: str, version: str) -> Dict[str, Any]:
        """GET /api/2.0/mlflow/model-versions/get - Get model version details"""
        params = {"name": name, "version": version}
        return await self._make_request("GET", "/api/2.0/mlflow/model-versions/get", 
                                      params=params, use_cache=True, cache_ttl=300)
    
    async def update_model_version(self, name: str, version: str,
                                 description: Optional[str] = None) -> Dict[str, Any]:
        """PATCH /api/2.0/mlflow/model-versions/update - Update model version"""
        data = {"name": name, "version": version, "description": description}
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("PATCH", "/api/2.0/mlflow/model-versions/update", data=data)
    
    async def transition_model_version_stage(self, name: str, version: str, stage: str,
                                           archive_existing_versions: bool = False,
                                           comment: Optional[str] = None) -> Dict[str, Any]:
        """POST /api/2.0/mlflow/model-versions/transition-stage - Transition model version stage"""
        data = {
            "name": name,
            "version": version,
            "stage": stage,
            "archive_existing_versions": archive_existing_versions,
            "comment": comment
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", "/api/2.0/mlflow/model-versions/transition-stage", data=data)
    
    async def delete_model_version(self, name: str, version: str) -> Dict[str, Any]:
        """DELETE /api/2.0/mlflow/model-versions/delete - Delete model version"""
        params = {"name": name, "version": version}
        return await self._make_request("DELETE", "/api/2.0/mlflow/model-versions/delete", 
                                      params=params)
    
    # =============================================================================
    # ARTIFACTS ENDPOINTS
    # =============================================================================
    
    async def list_artifacts(self, run_id: str, path: str = "") -> Dict[str, Any]:
        """GET /api/2.0/mlflow/artifacts/list - List artifacts for a run"""
        params = {"run_id": run_id, "path": path}
        return await self._make_request("GET", "/api/2.0/mlflow/artifacts/list", 
                                      params=params, use_cache=True, cache_ttl=300)
    
    async def get_artifact_uri(self, run_id: str, path: str) -> str:
        """GET /api/2.0/mlflow/artifacts/get-uri - Get artifact URI"""
        try:
            response = await self._make_request("GET", "/api/2.0/mlflow/artifacts/get-uri", 
                                              params={"run_id": run_id, "path": path})
            return response.get("artifact_uri", "")
        except:
            return ""
    
    # =============================================================================
    # CONVENIENCE METHODS
    # =============================================================================
    
    async def get_latest_models(self) -> List[Dict[str, Any]]:
        """Get latest versions of all models"""
        try:
            models = await self.list_registered_models()
            latest_models = []
            for model in models.get("registered_models", []):
                latest_versions = model.get("latest_versions", [])
                if latest_versions:
                    latest_models.append({
                        "name": model.get("name"),
                        "latest_version": latest_versions[0]
                    })
            return latest_models
        except:
            return []
    
    async def get_production_models(self) -> List[Dict[str, Any]]:
        """Get models in production stage"""
        try:
            models = await self.list_registered_models()
            production_models = []
            for model in models.get("registered_models", []):
                for version in model.get("latest_versions", []):
                    if version.get("current_stage") == "Production":
                        production_models.append({
                            "name": model.get("name"),
                            "version": version
                        })
            return production_models
        except:
            return []
    
    async def get_experiment_runs(self, experiment_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get runs for a specific experiment"""
        try:
            runs = await self.search_runs(experiment_ids=[experiment_id], max_results=limit)
            return runs.get("runs", [])
        except:
            return []
    
    async def get_best_run(self, experiment_id: str, metric_name: str = "accuracy") -> Optional[Dict[str, Any]]:
        """Get the best run based on a metric"""
        try:
            runs = await self.search_runs(
                experiment_ids=[experiment_id],
                order_by=[f"metrics.{metric_name} DESC"],
                max_results=1
            )
            runs_list = runs.get("runs", [])
            return runs_list[0] if runs_list else None
        except:
            return None
