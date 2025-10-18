"""
Model Cache Service Client
100% API Coverage for Model Cache Service (port 8003)
"""

import logging
from typing import Dict, List, Optional, Any
from .base_client import BaseServiceClient

logger = logging.getLogger(__name__)


class ModelCacheClient(BaseServiceClient):
    """Client for Model Cache Service - 100% API Coverage"""
    
    def __init__(self, base_url: str, redis_client):
        super().__init__("model_cache", base_url, redis_client)
    
    # =============================================================================
    # HEALTH ENDPOINTS
    # =============================================================================
    
    async def get_health(self) -> Dict[str, Any]:
        """GET /health - Comprehensive health check with cache statistics"""
        return await self._make_request("GET", "/health", use_cache=True, cache_ttl=60)
    
    async def get_root(self) -> Dict[str, Any]:
        """GET / - Root endpoint with service status"""
        return await self._make_request("GET", "/", use_cache=True, cache_ttl=60)
    
    # =============================================================================
    # INFERENCE ENDPOINTS
    # =============================================================================
    
    async def predict(self, text: str, model_name: Optional[str] = None,
                     ensemble: bool = False, confidence_threshold: Optional[float] = None,
                     return_probabilities: bool = True) -> Dict[str, Any]:
        """POST /predict - Main prediction endpoint for security classification"""
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
        """POST /predict/batch - Batch prediction endpoint"""
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
        """GET /models - List all models in cache with their status"""
        return await self._make_request("GET", "/models", use_cache=True, cache_ttl=60)
    
    async def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """GET /models/{model_name}/status - Get detailed status of a specific model"""
        return await self._make_request("GET", f"/models/{model_name}/status", use_cache=True, cache_ttl=30)
    
    async def load_model(self, model_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """POST /models/{model_name}/load - Load a model into cache"""
        data = config or {}
        return await self._make_request("POST", f"/models/{model_name}/load", data=data)
    
    async def unload_model(self, model_name: str) -> Dict[str, Any]:
        """POST /models/{model_name}/unload - Unload a model from cache"""
        return await self._make_request("POST", f"/models/{model_name}/unload")
    
    async def reload_model(self, model_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """POST /models/{model_name}/reload - Reload a model in cache"""
        data = config or {}
        return await self._make_request("POST", f"/models/{model_name}/reload", data=data)
    
    async def preload_model(self, model_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """POST /models/{model_name}/preload - Preload a model into cache"""
        data = config or {}
        return await self._make_request("POST", f"/models/{model_name}/preload", data=data)
    
    async def warm_model(self, model_name: str, warmup_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """POST /models/{model_name}/warm - Warm up a model with sample data"""
        data = {"warmup_texts": warmup_texts or []}
        return await self._make_request("POST", f"/models/{model_name}/warm", data=data)
    
    # =============================================================================
    # CACHE MANAGEMENT ENDPOINTS
    # =============================================================================
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """GET /stats - Get comprehensive cache statistics"""
        return await self._make_request("GET", "/stats", use_cache=True, cache_ttl=30)
    
    async def clear_cache(self, confirm: bool = True) -> Dict[str, Any]:
        """POST /clear-cache - Clear all cached models and reset statistics"""
        params = {"confirm": confirm}
        return await self._make_request("POST", "/clear-cache", params=params)
    
    async def get_cache_config(self) -> Dict[str, Any]:
        """GET /config - Get cache configuration"""
        return await self._make_request("GET", "/config", use_cache=True, cache_ttl=300)
    
    async def update_cache_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """PUT /config - Update cache configuration"""
        return await self._make_request("PUT", "/config", data=config)
    
    # =============================================================================
    # LOGGING AND MONITORING ENDPOINTS
    # =============================================================================
    
    async def get_logs(self, level: Optional[str] = None, model_name: Optional[str] = None,
                      limit: int = 100, since: Optional[str] = None) -> Dict[str, Any]:
        """GET /logs - Get service logs for debugging and monitoring"""
        params = {
            "level": level,
            "model_name": model_name,
            "limit": limit,
            "since": since
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/logs", params=params)
    
    async def get_model_logs(self, model_name: str, level: Optional[str] = None,
                           limit: int = 50) -> Dict[str, Any]:
        """GET /models/{model_name}/logs - Get logs for a specific model"""
        params = {"level": level, "limit": limit}
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", f"/models/{model_name}/logs", params=params)
    
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
    # PERFORMANCE ENDPOINTS
    # =============================================================================
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """GET /performance - Get performance statistics"""
        return await self._make_request("GET", "/performance", use_cache=True, cache_ttl=30)
    
    async def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """GET /models/{model_name}/performance - Get performance stats for a specific model"""
        return await self._make_request("GET", f"/models/{model_name}/performance", use_cache=True, cache_ttl=30)
    
    async def benchmark_model(self, model_name: str, num_requests: int = 100) -> Dict[str, Any]:
        """POST /models/{model_name}/benchmark - Benchmark a model's performance"""
        data = {"num_requests": num_requests}
        return await self._make_request("POST", f"/models/{model_name}/benchmark", data=data)
    
    # =============================================================================
    # CONVENIENCE METHODS
    # =============================================================================
    
    async def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        try:
            models = await self.get_models()
            loaded_models = []
            for model in models.get("models", []):
                if model.get("status") == "loaded":
                    loaded_models.append(model.get("name"))
            return loaded_models
        except:
            return []
    
    async def is_model_loaded(self, model_name: str) -> bool:
        """Check if a specific model is loaded"""
        try:
            status = await self.get_model_status(model_name)
            return status.get("status") == "loaded"
        except:
            return False
    
    async def get_cache_utilization(self) -> float:
        """Get cache memory utilization percentage"""
        try:
            stats = await self.get_cache_stats()
            memory_used = stats.get("memory_usage_mb", 0)
            max_memory = stats.get("max_memory_mb", 1)
            return (memory_used / max_memory) * 100 if max_memory > 0 else 0
        except:
            return 0.0
    
    async def get_hit_rate(self) -> float:
        """Get cache hit rate percentage"""
        try:
            stats = await self.get_cache_stats()
            return stats.get("hit_rate", 0.0) * 100
        except:
            return 0.0
