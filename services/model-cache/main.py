"""
Model Cache Service - Modular Main
Modularized model cache service with clean architecture
"""

import asyncio
import signal
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import threading
from concurrent.futures import ThreadPoolExecutor
import redis
from collections import OrderedDict
import psutil
import os
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Import modular components
from models.requests import PredictionRequest, ModelLoadRequest, CacheStatsRequest
from models.responses import PredictionResponse, ModelStatus, CacheStats, HealthResponse, SuccessResponse

# Import tracing setup
import sys
import os
sys.path.append('/app')
from tracing_setup import setup_tracing, get_tracer, trace_request, add_span_attributes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CachedModel:
    """High-performance cached model for inference"""
    
    def __init__(self, model_name: str, model_path: str, model_source: str = "Hugging Face"):
        self.model_name = model_name
        self.model_path = model_path
        self.model_source = model_source
        self.model = None
        self.tokenizer = None
        self.labels = ["prompt_injection", "jailbreak", "system_extraction", "code_injection", "benign"]
        self.loaded = False
        self.loaded_at = None
        self.usage_count = 0
        self.last_health_check = None
        self.health_status = "unknown"
        self.loading_lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._executor_shutdown = False
    
    def __del__(self):
        """Cleanup ThreadPoolExecutor to prevent memory leaks"""
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if not self._executor_shutdown and self.executor:
            try:
                self.executor.shutdown(wait=False)
                self._executor_shutdown = True
                logger.debug(f"🧹 [CACHE] Cleaned up executor for {self.model_name}")
            except Exception as e:
                logger.warning(f"⚠️ [CACHE] Error cleaning up executor for {self.model_name}: {e}")
    
    async def load(self):
        """Load model and tokenizer asynchronously"""
        if self.loaded:
            logger.info(f"Model {self.model_name} already loaded")
            return True
            
        async with self.loading_lock:
            if self.loaded:  # Double-check after acquiring lock
                logger.info(f"Model {self.model_name} already loaded (double-check)")
                return True
                
            try:
                logger.info(f"🔄 [CACHE] Loading model {self.model_name} from {self.model_path}")
                
                # Run model loading in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                logger.info(f"Starting model load in thread pool for {self.model_name}")
                success = await loop.run_in_executor(
                    self.executor, 
                    self._load_model_sync
                )
                
                if success:
                    self.loaded = True
                    self.loaded_at = time.time()
                    self.health_status = "healthy"
                    logger.info(f"✅ [CACHE] Successfully loaded model {self.model_name}")
                    return True
                else:
                    self.health_status = "error"
                    logger.error(f"❌ [CACHE] Model load failed for {self.model_name}")
                    return False
                    
            except Exception as e:
                from utils.enhanced_logging import log_error_with_context
                log_error_with_context(
                    error=e,
                    operation="model_cache_load",
                    model_name=self.model_name,
                    additional_context={"model_source": self.model_source, "model_path": self.model_path}
                )
                self.health_status = "error"
                return False
    
    def _load_model_sync(self):
        """Synchronous model loading (runs in thread pool)"""
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=len(self.labels)
            )
            
            # Set to evaluation mode
            self.model.eval()
            return True
            
        except Exception as e:
            logger.error(f"❌ [CACHE] Sync load failed for {self.model_name}: {e}")
            return False
    
    async def predict(self, text: str) -> Dict[str, Any]:
        """Make prediction on text asynchronously"""
        if not self.loaded:
            raise ValueError(f"Model {self.model_name} not loaded")
        
        start_time = time.time()
        
        try:
            # Run prediction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._predict_sync,
                text
            )
            
            processing_time = (time.time() - start_time) * 1000
            self.usage_count += 1
            
            result.update({
                "processing_time_ms": float(processing_time),
                "from_cache": True,
                "model_name": self.model_name
            })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ [CACHE] Prediction error for {self.model_name}: {e}")
            raise
    
    def _predict_sync(self, text: str) -> Dict[str, Any]:
        """Synchronous prediction (runs in thread pool)"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = torch.max(probabilities, dim=-1)[0].item()
            
            # Get probabilities for all classes
            prob_dict = {
                self.labels[i]: probabilities[0][i].item() 
                for i in range(len(self.labels))
            }
            
            return {
                "prediction": str(self.labels[predicted_class]),
                "confidence": float(confidence),
                "probabilities": {str(k): float(v) for k, v in prob_dict.items()}
            }
            
        except Exception as e:
            logger.error(f"❌ [CACHE] Sync prediction error for {self.model_name}: {e}")
            raise


class CacheManager:
    """Intelligent cache management with TTL and event-based invalidation"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour
    
    async def set_with_ttl(self, key: str, value: Any, ttl: int = None):
        """Set cache value with TTL"""
        ttl = ttl or self.default_ttl
        await self.redis.setex(key, ttl, json.dumps(value))
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern"""
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)
            logger.info(f"Invalidated {len(keys)} cache entries matching '{pattern}'")
    
    async def invalidate_model_cache(self, model_name: str):
        """Invalidate all cache entries for a model"""
        patterns = [
            f"model:{model_name}:*",
            f"prediction:{model_name}:*",
            f"metadata:{model_name}:*"
        ]
        for pattern in patterns:
            await self.invalidate_pattern(pattern)

class ModelCache:
    """Model cache service for managing model predictions"""
    
    def __init__(self):
        self.model_api_url = "http://model-api:8000"
        self.training_api_url = "http://training:8002"
        self.models = {}  # Cache for CachedModel instances
        self.redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
        self.cache_manager = CacheManager(self.redis_client)
        self.model_configs = {
            "bert-base": {
                "path": "bert-base-uncased",
                "source": "Hugging Face"
            },
            "roberta-base": {
                "path": "roberta-base", 
                "source": "Hugging Face"
            },
            "distilbert": {
                "path": "distilbert-base-uncased",
                "source": "Hugging Face"
            },
            "deberta-v3-base": {
                "path": "microsoft/deberta-v3-base",
                "source": "Hugging Face"
            }
        }
        self.stats = {
            "total_predictions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "model_loads": 0,
            "start_time": time.time()
        }
        
        # LRU Cache configuration
        self.max_models = 3  # Maximum number of models to keep in memory
        self._shutdown = False
    
    def __del__(self):
        """Cleanup all resources"""
        self.cleanup()
    
    def cleanup(self):
        """Clean up all cached models and their resources"""
        if self._shutdown:
            return
            
        try:
            logger.info("🧹 [CACHE] Cleaning up ModelCache resources...")
            for model_name, cached_model in self.models.items():
                try:
                    cached_model.cleanup()
                except Exception as e:
                    logger.warning(f"⚠️ [CACHE] Error cleaning up model {model_name}: {e}")
            
            self.models.clear()
            self._shutdown = True
            logger.info("✅ [CACHE] ModelCache cleanup completed")
        except Exception as e:
            logger.error(f"❌ [CACHE] Error during ModelCache cleanup: {e}")
        self.max_memory_mb = 2048  # Maximum memory usage in MB
        self.model_access_order = OrderedDict()  # LRU tracking
        self.memory_monitor_enabled = True
        # In-memory log storage for real-time access
        self.logs = []
        self.max_logs = 1000  # Keep last 1000 log entries
    
    def _add_log(self, level: str, message: str, model_name: str = None, details: Dict[str, Any] = None):
        """Add a log entry to in-memory storage"""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            "model_name": model_name,
            "details": details or {}
        }
        self.logs.append(log_entry)
        
        # Keep only the last max_logs entries
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return 0.0
    
    def _update_model_access(self, model_name: str):
        """Update model access order for LRU tracking"""
        if model_name in self.model_access_order:
            # Move to end (most recently used)
            self.model_access_order.move_to_end(model_name)
        else:
            # Add to end
            self.model_access_order[model_name] = time.time()
    
    def _evict_lru_models(self):
        """Evict least recently used models to free memory"""
        try:
            current_memory = self._get_memory_usage_mb()
            logger.info(f"🔄 [LRU] Current memory usage: {current_memory:.2f} MB")
            
            # Evict models if we exceed limits
            while (len(self.models) > self.max_models or 
                   current_memory > self.max_memory_mb) and self.model_access_order:
                
                # Get least recently used model
                lru_model = next(iter(self.model_access_order))
                
                if lru_model in self.models:
                    logger.info(f"🗑️ [LRU] Evicting model: {lru_model}")
                    
                    # Unload the model
                    model = self.models[lru_model]
                    if hasattr(model, 'unload'):
                        model.unload()
                    
                    # Remove from cache
                    del self.models[lru_model]
                    del self.model_access_order[lru_model]
                    
                    self.stats["model_unloads"] += 1
                    self._add_log("INFO", f"Model evicted due to LRU policy", lru_model)
                    
                    # Update memory usage
                    current_memory = self._get_memory_usage_mb()
                    logger.info(f"🔄 [LRU] Memory after eviction: {current_memory:.2f} MB")
                else:
                    # Remove from access order if model not in cache
                    del self.model_access_order[lru_model]
            
        except Exception as e:
            logger.error(f"❌ [LRU] Error during model eviction: {e}")
    
    def _check_memory_limits(self):
        """Check if we need to evict models based on memory limits"""
        if not self.memory_monitor_enabled:
            return
        
        try:
            current_memory = self._get_memory_usage_mb()
            
            # Evict if we exceed memory limit
            if current_memory > self.max_memory_mb:
                logger.warning(f"⚠️ [LRU] Memory usage {current_memory:.2f} MB exceeds limit {self.max_memory_mb} MB")
                self._evict_lru_models()
            
            # Evict if we exceed model count limit
            if len(self.models) > self.max_models:
                logger.warning(f"⚠️ [LRU] Model count {len(self.models)} exceeds limit {self.max_models}")
                self._evict_lru_models()
                
        except Exception as e:
            logger.error(f"❌ [LRU] Error checking memory limits: {e}")
    
    def get_logs(self, model_name: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get logs, optionally filtered by model name"""
        logs = self.logs
        if model_name:
            logs = [log for log in logs if log.get("model_name") == model_name]
        
        # Return the most recent logs
        return logs[-limit:] if limit else logs
    
    async def _check_model_loaded(self, model_name: str) -> bool:
        """Check if model is loaded in model-api"""
        try:
            self._add_log("INFO", f"Checking if model {model_name} is loaded in model-api", model_name)
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.model_api_url}/models",
                    timeout=2.0
                )
                if response.status_code == 200:
                    models_data = response.json()
                    available_models = models_data.get("available_models", [])
                    is_loaded = model_name in available_models
                    self._add_log("INFO", f"Model {model_name} loaded status: {is_loaded}", model_name)
                    return is_loaded
                else:
                    self._add_log("WARNING", f"Failed to check model status: {response.status_code}", model_name)
                    return False
        except Exception as e:
            self._add_log("WARNING", f"Error checking model status: {e}", model_name)
            return False
    
    async def _load_model(self, model_name: str) -> bool:
        """Load model in model-api service"""
        try:
            self._add_log("INFO", f"Starting model load process for {model_name}", model_name, {
                "action": "model_load_start",
                "target_service": "model-api"
            })
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.model_api_url}/load",
                    json={"model_name": model_name},
                    timeout=30.0
                )
                if response.status_code == 200:
                    self.stats["model_loads"] += 1
                    self._add_log("INFO", f"Successfully loaded model: {model_name}", model_name, {
                        "action": "model_load_success",
                        "response_status": response.status_code
                    })
                    
                    # Invalidate cache for this model
                    await self.cache_manager.invalidate_model_cache(model_name)
                    
                    return True
                else:
                    self._add_log("ERROR", f"Failed to load model {model_name}: {response.status_code}", model_name, {
                        "action": "model_load_failed",
                        "response_status": response.status_code,
                        "error_details": response.text
                    })
                    return False
        except Exception as e:
            self._add_log("ERROR", f"Error loading model {model_name}: {e}", model_name, {
                "action": "model_load_error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            return False
    
    async def _load_model_directly(self, model_name: str) -> bool:
        """Load model directly into cache (primary inference engine)"""
        try:
            logger.info(f"Starting direct model load for: {model_name}")
            
            # Extract base model name from model_name
            base_name = model_name.replace("_pretrained", "").replace("_trained", "")
            logger.info(f"Base model name: {base_name}")
            
            if base_name not in self.model_configs:
                error_msg = f"Unknown model configuration: {base_name}"
                self._add_log("ERROR", error_msg, model_name)
                logger.error(error_msg)
                return False
            
            config = self.model_configs[base_name]
            model_path = config["path"]
            model_source = config["source"]
            
            logger.info(f"Model config: path={model_path}, source={model_source}")
            self._add_log("INFO", f"Loading model directly into cache: {model_name}", model_name)
            
            # Create CachedModel and load it
            logger.info(f"Creating CachedModel instance for {model_name}")
            cached_model = CachedModel(
                model_name=model_name,
                model_path=model_path,
                model_source=model_source
            )
            
            logger.info(f"Starting model load process for {model_name}")
            success = await cached_model.load()
            
            if success:
                self.models[model_name] = cached_model
                self.stats["model_loads"] += 1
                
                # Update LRU tracking
                self._update_model_access(model_name)
                
                # Check memory limits and evict if necessary
                self._check_memory_limits()
                
                success_msg = f"Successfully loaded model {model_name} into cache"
                self._add_log("INFO", success_msg, model_name, {
                    "memory_usage_mb": self._get_memory_usage_mb(),
                    "cached_models": len(self.models)
                })
                logger.info(success_msg)
                return True
            else:
                error_msg = f"Failed to load model {model_name} into cache"
                self._add_log("ERROR", error_msg, model_name)
                logger.error(error_msg)
                return False
                
        except Exception as e:
            from utils.enhanced_logging import log_error_with_context
            log_error_with_context(
                error=e,
                operation="model_cache_load_by_name",
                model_name=model_name,
                additional_context={"base_name": base_name, "model_type": model_type}
            )
            error_msg = f"Error loading model {model_name}: {e}"
            self._add_log("ERROR", error_msg, model_name)
            return False
    
    async def predict(self, text: str, models: List[str], ensemble: bool = False) -> PredictionResponse:
        """Direct inference using cached models - NO circular calls"""
        start_time = time.time()
        
        try:
            self._add_log("INFO", f"Starting direct inference for models: {models}", details={"text_length": len(text)})
            
            # Check if we have models cached locally
            cached_models = []
            uncached_models = []
            
            for model_name in models:
                if model_name in self.models and self.models[model_name].loaded:
                    cached_models.append(model_name)
                    # Update LRU tracking for cached models
                    self._update_model_access(model_name)
                    self._add_log("INFO", f"Model {model_name} available in local cache", model_name)
                else:
                    uncached_models.append(model_name)
                    self._add_log("INFO", f"Model {model_name} not in local cache", model_name)
            
            # If we have all models cached, do local inference
            if len(cached_models) == len(models):
                return await self._predict_locally(text, models, start_time)
            
            # If some models not cached, load them first
            for model_name in uncached_models:
                self._add_log("INFO", f"Loading model {model_name} into cache", model_name)
                success = await self._load_model_directly(model_name)
                if not success:
                    raise HTTPException(status_code=503, detail=f"Failed to load model {model_name}")
            
            # Now do local inference with all models loaded
            return await self._predict_locally(text, models, start_time)
                    
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self._add_log("ERROR", f"Direct inference failed: {e}", details={"traceback": error_details})
            logger.error(f"Direct inference failed: {e}")
            logger.error(f"Traceback: {error_details}")
            raise HTTPException(status_code=500, detail=f"Direct inference failed: {str(e)}")
    
    async def _predict_locally(self, text: str, models: List[str], start_time: float) -> PredictionResponse:
        """Predict using locally cached models"""
        try:
            model_predictions = {}
            predictions = []
            confidences = []
            
            for model_name in models:
                cached_model = self.models[model_name]
                result = await cached_model.predict(text)
                model_predictions[model_name] = result
                predictions.append(result["prediction"])
                confidences.append(result["confidence"])
                
                self.stats["cache_hits"] += 1
                self.models[model_name].usage_count += 1
            
            # Use first model's result (can be enhanced for ensemble)
            prediction = predictions[0]
            confidence = confidences[0]
            
            self.stats["total_predictions"] += 1
            processing_time = (time.time() - start_time) * 1000
            
            self._add_log("INFO", f"Local prediction completed", details={
                "prediction": prediction,
                "confidence": confidence,
                "processing_time_ms": processing_time
            })
            
            return PredictionResponse(
                text=text,
                prediction=prediction,
                confidence=confidence,
                probabilities=model_predictions[models[0]]["probabilities"],
                model_predictions=model_predictions,
                ensemble_used=False,
                processing_time_ms=processing_time,
                from_cache=True,
                timestamp=time.time()
            )
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self._add_log("ERROR", f"Local prediction failed: {e}", details={"traceback": error_details})
            logger.error(f"Local prediction failed: {e}")
            logger.error(f"Traceback: {error_details}")
            raise HTTPException(status_code=500, detail=f"Local prediction failed: {str(e)}")
    
    # Removed circular call methods - model-cache now does direct inference only
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        uptime = time.time() - self.stats["start_time"]
        hit_rate = (self.stats["cache_hits"] / self.stats["total_predictions"]) if self.stats["total_predictions"] > 0 else 0.0
        
        return CacheStats(
            total_predictions=self.stats["total_predictions"],
            cache_hits=self.stats["cache_hits"],
            cache_misses=self.stats["cache_misses"],
            hit_rate=hit_rate,
            model_loads=self.stats["model_loads"],
            uptime_seconds=uptime,
            models_loaded=list(self.models.keys()),
            timestamp=time.time()
        )


# Initialize model cache
model_cache = ModelCache()

# Create FastAPI application
app = FastAPI(
    title="ML Security Model Cache Service (Modular)",
    version="2.0.0",
    description="Model cache service for ML Security platform"
)

# Setup distributed tracing
setup_tracing("model-cache", app)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("🧹 [CACHE] Shutting down Model Cache Service...")
    try:
        model_cache.cleanup()
        logger.info("✅ [CACHE] Model Cache Service shutdown completed")
    except Exception as e:
        logger.error(f"❌ [CACHE] Error during shutdown: {e}")

# Prometheus metrics
CACHE_REQUEST_COUNT = Counter('model_cache_requests_total', 'Total number of cache requests', ['method', 'endpoint', 'status'])
CACHE_REQUEST_DURATION = Histogram('model_cache_request_duration_seconds', 'Cache request duration in seconds', ['method', 'endpoint'])
CACHE_HIT_COUNT = Counter('model_cache_hits_total', 'Total number of cache hits', ['model_name'])
CACHE_MISS_COUNT = Counter('model_cache_misses_total', 'Total number of cache misses', ['model_name'])
CACHE_SIZE = Gauge('model_cache_size_bytes', 'Current cache size in bytes')
CACHE_MODELS = Gauge('model_cache_models_count', 'Number of models in cache')

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "model-cache",
        "version": "2.0.0",
        "status": "running",
        "description": "Model cache service for ML Security platform"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        uptime = time.time() - model_cache.stats["start_time"]
        
        # Check dependencies
        dependencies = {
            "model_api": True,  # Would check actual connection
            "training_api": True,  # Would check actual connection
            "redis": True  # Would check actual connection
        }
        
        return HealthResponse(
            status="healthy",
            service="model-cache",
            timestamp=time.time(),
            uptime_seconds=uptime,
            dependencies=dependencies
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from fastapi import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Get prediction from cached models"""
    try:
        logger.info(f"Received prediction request: {request.text[:50]}... with models: {request.models}")
        result = await model_cache.predict(
            text=request.text,
            models=request.models,
            ensemble=request.ensemble
        )
        logger.info(f"Prediction successful: {result.prediction}")
        return result
    except HTTPException as he:
        logger.error(f"HTTP Exception in prediction: {he.detail}")
        raise he
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error in prediction: {e}")
        logger.error(f"Traceback: {error_details}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/stats", response_model=CacheStats)
async def get_stats():
    """Get cache statistics"""
    try:
        return model_cache.get_stats()
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs")
async def get_logs(model_name: str = None, limit: int = 100):
    """Get model cache service logs"""
    try:
        logs = model_cache.get_logs(model_name=model_name, limit=limit)
        return {
            "logs": logs,
            "total_logs": len(logs),
            "model_name": model_name,
            "limit": limit,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to get logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_name}/logs")
async def get_model_logs(model_name: str, limit: int = 100):
    """Get logs for a specific model"""
    try:
        logs = model_cache.get_logs(model_name=model_name, limit=limit)
        return {
            "model_name": model_name,
            "logs": logs,
            "total_logs": len(logs),
            "limit": limit,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to get logs for model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/{model_name}/notify")
async def notify_model_event(model_name: str, request: Dict[str, Any]):
    """Receive notifications about model events from model-api service"""
    try:
        action = request.get("action", "unknown")
        timestamp = request.get("timestamp", time.time())
        
        # Add log entry for the notification
        model_cache._add_log(
            level="INFO",
            message=f"Received notification from model-api: {action}",
            model_name=model_name,
            details={
                "action": action,
                "source": "model-api",
                "timestamp": timestamp
            }
        )
        
        return {
            "status": "success",
            "message": f"Notification received for {model_name}",
            "action": action,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to process notification for model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/{model_name}/preload")
async def preload_model(model_name: str):
    """Preload a model into cache for faster inference"""
    try:
        if model_name in model_cache.models and model_cache.models[model_name].loaded:
            return {
                "status": "success",
                "message": f"Model {model_name} already loaded in cache",
                "model_name": model_name,
                "timestamp": time.time()
            }
        
        success = await model_cache._load_model_directly(model_name)
        if success:
            return {
                "status": "success",
                "message": f"Model {model_name} preloaded successfully",
                "model_name": model_name,
                "timestamp": time.time()
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to preload model {model_name}")
            
    except Exception as e:
        logger.error(f"Failed to preload model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    from fastapi import Response
    
    # Basic metrics for model cache service
    metrics_data = f"""# HELP model_cache_service_up Model cache service status
# TYPE model_cache_service_up gauge
model_cache_service_up 1

# HELP model_cache_hits_total Total cache hits
# TYPE model_cache_hits_total counter
model_cache_hits_total 0

# HELP model_cache_misses_total Total cache misses
# TYPE model_cache_misses_total counter
model_cache_misses_total 0

# HELP model_cache_size Current cache size
# TYPE model_cache_size gauge
model_cache_size 0
"""
    
    return Response(content=metrics_data, media_type="text/plain")


@app.post("/clear-cache", response_model=SuccessResponse)
async def clear_cache():
    """Clear model cache"""
    try:
        # Clear in-memory models
        model_cache.models.clear()
        model_cache.stats["cache_hits"] = 0
        model_cache.stats["cache_misses"] = 0
        
        # Clear Redis cache
        await model_cache.cache_manager.invalidate_pattern("*")
        
        return SuccessResponse(
            status="success",
            message="Cache cleared successfully",
            timestamp=time.time()
        )
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/{model_name}/unload")
async def unload_model(model_name: str):
    """Unload a model from cache"""
    try:
        if model_name not in model_cache.models:
            return {
                "status": "success",
                "message": f"Model {model_name} not in cache",
                "model_name": model_name,
                "timestamp": time.time()
            }
        
        # Remove model from cache
        del model_cache.models[model_name]
        model_cache.stats["model_loads"] = max(0, model_cache.stats["model_loads"] - 1)
        
        model_cache._add_log("INFO", f"Model {model_name} unloaded from cache", model_name)
        
        return {
            "status": "success",
            "message": f"Model {model_name} unloaded successfully",
            "model_name": model_name,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to unload model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Graceful shutdown handler
class GracefulShutdown:
    """Handles graceful shutdown of the model-cache service"""
    
    def __init__(self, app):
        self.app = app
        self.is_shutting_down = False
        self.shutdown_timeout = 30  # seconds
        self.pending_tasks = set()
        logger.info("GracefulShutdown handler initialized for model-cache service.")
    
    def register_handlers(self):
        """Register signal handlers for graceful shutdown."""
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self.shutdown_handler)
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, self.shutdown_handler)
        logger.info("Registered SIGTERM and SIGINT handlers for model-cache service.")
        
        @self.app.on_event("shutdown")
        async def _on_shutdown():
            await self._perform_cleanup()
    
    async def shutdown_handler(self, signum, frame):
        """Handle shutdown signals."""
        if self.is_shutting_down:
            logger.warning("Model-cache service already in shutdown process, ignoring signal.")
            return
        
        self.is_shutting_down = True
        logger.info(f"Model-cache service received signal {signum}, initiating graceful shutdown...")
        
        # Trigger FastAPI's shutdown event
        await self.app.shutdown()
    
    async def _perform_cleanup(self):
        """Perform actual cleanup tasks."""
        logger.info("Performing graceful shutdown cleanup for model-cache service...")
        
        # Cancel any pending background tasks
        if self.pending_tasks:
            logger.info(f"Cancelling {len(self.pending_tasks)} pending background tasks...")
            for task in list(self.pending_tasks):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.debug(f"Task {task.get_name()} cancelled.")
                except Exception as e:
                    logger.error(f"Error during task cancellation: {e}")
            self.pending_tasks.clear()
            logger.info("All pending tasks cancelled.")
        
        # Clear model cache
        try:
            if hasattr(model_cache, 'models'):
                for model_name in list(model_cache.models.keys()):
                    try:
                        del model_cache.models[model_name]
                        logger.info(f"Unloaded model {model_name} during shutdown")
                    except Exception as e:
                        logger.error(f"Error unloading model {model_name}: {e}")
                logger.info("Model cache cleared.")
        except Exception as e:
            logger.error(f"Error clearing model cache: {e}")
        
        # Close Redis connections
        try:
            if hasattr(model_cache, 'redis_client') and model_cache.redis_client:
                await model_cache.redis_client.close()
                logger.info("Redis client closed.")
        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")
        
        logger.info("Model-cache service graceful shutdown cleanup complete.")

# Initialize graceful shutdown
shutdown_handler = GracefulShutdown(app)
shutdown_handler.register_handlers()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await shutdown_handler._perform_cleanup()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
