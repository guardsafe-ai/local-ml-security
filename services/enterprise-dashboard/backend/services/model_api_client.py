"""
Model API Service Client
100% API Coverage for Model API Service (port 8000)
"""

import logging
from typing import Dict, List, Optional, Any
from .base_client import BaseServiceClient

logger = logging.getLogger(__name__)


class ModelAPIClient(BaseServiceClient):
    """Client for Model API Service - 100% API Coverage"""
    
    def __init__(self, base_url: str, redis_client):
        super().__init__("model_api", base_url, redis_client)
    
    # =============================================================================
    # HEALTH ENDPOINTS
    # =============================================================================
    
    async def get_health(self) -> Dict[str, Any]:
        """GET /health - Basic health check"""
        return await self._make_request("GET", "/health", use_cache=True, cache_ttl=60)
    
    async def get_health_deep(self) -> Dict[str, Any]:
        """GET /health/deep - Deep health check with dependencies"""
        return await self._make_request("GET", "/health/deep", use_cache=True, cache_ttl=60)
    
    async def get_health_ready(self) -> Dict[str, Any]:
        """GET /health/ready - Kubernetes readiness probe"""
        return await self._make_request("GET", "/health/ready", use_cache=True, cache_ttl=30)
    
    async def get_health_live(self) -> Dict[str, Any]:
        """GET /health/live - Kubernetes liveness probe"""
        return await self._make_request("GET", "/health/live", use_cache=True, cache_ttl=30)
    
    async def get_health_startup(self) -> Dict[str, Any]:
        """GET /health/startup - Kubernetes startup probe"""
        return await self._make_request("GET", "/health/startup", use_cache=True, cache_ttl=30)
    
    # =============================================================================
    # PREDICTION ENDPOINTS
    # =============================================================================
    
    async def predict(self, text: str, model_name: Optional[str] = None, 
                     ensemble: bool = False, confidence_threshold: Optional[float] = None,
                     return_probabilities: bool = True) -> Dict[str, Any]:
        """POST /predict - Make prediction on text"""
        data = {
            "text": text,
            "model_name": model_name,
            "ensemble": ensemble,
            "confidence_threshold": confidence_threshold,
            "return_probabilities": return_probabilities
        }
        return await self._make_request("POST", "/predict", data=data)
    
    async def predict_batch(self, texts: List[str], model_name: Optional[str] = None,
                           ensemble: bool = False, confidence_threshold: Optional[float] = None,
                           return_probabilities: bool = True) -> Dict[str, Any]:
        """POST /predict/batch - Make batch predictions"""
        data = {
            "texts": texts,
            "model_name": model_name,
            "ensemble": ensemble,
            "confidence_threshold": confidence_threshold,
            "return_probabilities": return_probabilities
        }
        return await self._make_request("POST", "/predict/batch", data=data)
    
    # =============================================================================
    # MODEL MANAGEMENT ENDPOINTS
    # =============================================================================
    
    async def get_models(self) -> Dict[str, Any]:
        """GET /models - List all available models"""
        return await self._make_request("GET", "/models", use_cache=True, cache_ttl=300)
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """GET /models/{model_name} - Get information about a specific model"""
        return await self._make_request("GET", f"/models/{model_name}", use_cache=True, cache_ttl=300)
    
    async def load_model(self, model_name: str, version: Optional[str] = None,
                        config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """POST /models/load - Load a model into memory"""
        data = {
            "model_name": model_name,
            "version": version,
            "config": config or {}
        }
        return await self._make_request("POST", "/models/load", data=data)
    
    async def unload_model(self, model_name: str) -> Dict[str, Any]:
        """POST /models/unload - Unload a model from memory"""
        data = {"model_name": model_name}
        return await self._make_request("POST", "/models/unload", data=data)
    
    async def batch_load_models(self, model_names: List[str]) -> Dict[str, Any]:
        """POST /models/batch-load - Batch load multiple models concurrently"""
        data = {"model_names": model_names}
        return await self._make_request("POST", "/models/batch-load", data=data)
    
    async def warm_cache(self, model_name: str) -> Dict[str, Any]:
        """POST /models/warm-cache/{model_name} - Warm up the cache for a specific model"""
        return await self._make_request("POST", f"/models/warm-cache/{model_name}")
    
    async def get_preload_status(self) -> Dict[str, Any]:
        """GET /models/preload-status - Get status of model preloading tasks"""
        return await self._make_request("GET", "/models/preload-status", use_cache=True, cache_ttl=60)
    
    # =============================================================================
    # METRICS ENDPOINTS
    # =============================================================================
    
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
    # CONVENIENCE METHODS
    # =============================================================================
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get available models (alias for get_models)"""
        return await self.get_models()
    
    async def is_model_loaded(self, model_name: str) -> bool:
        """Check if a specific model is loaded"""
        try:
            models = await self.get_models()
            model_info = models.get("models", {}).get(model_name, {})
            return model_info.get("loaded", False)
        except:
            return False
    
    async def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        try:
            models = await self.get_models()
            loaded_models = []
            for name, info in models.get("models", {}).items():
                if info.get("loaded", False):
                    loaded_models.append(name)
            return loaded_models
        except:
            return []
    
    async def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific model"""
        try:
            model_info = await self.get_model_info(model_name)
            return model_info.get("performance", {})
        except:
            return {}
