"""
Model Manager Service - Core model management logic
"""

import logging
import redis
import mlflow
from typing import Dict, List, Optional, Any
from datetime import datetime
from .model_wrappers import PyTorchModel, SklearnModel

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages all security models"""
    
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.model_cache_url = "http://model-cache:8003"
        self.redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
        self.cache_enabled = True  # Enable distributed caching
        self.cache_stats = {"hits": 0, "misses": 0, "errors": 0}
        
        # Setup MLflow
        mlflow.set_tracking_uri("http://mlflow:5000")
        
        # Model configurations
        self.model_configs = {
            "distilbert": {
                "type": "pytorch",
                "path": "distilbert-base-uncased",
                "priority": 1
            },
            "bert-base": {
                "type": "pytorch",
                "path": "bert-base-uncased", 
                "priority": 2
            },
            "roberta-base": {
                "type": "pytorch", 
                "path": "roberta-base",
                "priority": 3
            },
            "deberta-v3-base": {
                "type": "pytorch",
                "path": "microsoft/deberta-v3-base",
                "priority": 4
            }
        }
        
        # Check model-cache availability on init
        self._check_cache_health()
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return [name for name, model in self.models.items() if model.loaded]
    
    def get_mlflow_models(self) -> List[str]:
        """Get list of models available in MLflow/MinIO"""
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            registered_models = client.search_registered_models()
            return [model.name for model in registered_models if model.name.startswith("security_")]
        except Exception as e:
            logger.error(f"Error getting MLflow models: {e}")
            return []
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        return {
            "name": model.model_name,
            "type": "pytorch" if isinstance(model, PyTorchModel) else "sklearn",
            "loaded": model.loaded,
            "path": model.model_path,
            "labels": model.labels,
            "performance": None,  # Could be added later
            "model_source": model.model_source,
            "model_version": model.model_version
        }
    
    def get_models_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all models"""
        return {
            name: self.get_model_info(name) 
            for name in self.models.keys()
        }
    
    async def load_model(self, model_name: str, version: str = None) -> bool:
        """Load a model into memory"""
        try:
            # Check if it's a trained model request
            if model_name.endswith("_trained"):
                base_name = model_name.replace("_trained", "")
                return await self._load_trained_model(base_name, model_name, version)
            elif model_name.endswith("_pretrained"):
                base_name = model_name.replace("_pretrained", "")
                return self._load_pretrained_model(base_name, model_name)
            else:
                # Default to pre-trained
                return self._load_pretrained_model(model_name, f"{model_name}_pretrained")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory"""
        try:
            if model_name not in self.models:
                logger.warning(f"Model {model_name} is not loaded")
                return True
            
            self.models[model_name].unload()
            del self.models[model_name]
            logger.info(f"Successfully unloaded model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            return False
    
    def _load_pretrained_model(self, base_name: str, model_name: str) -> bool:
        """Load pre-trained model from Hugging Face"""
        try:
            if base_name not in self.model_configs:
                logger.error(f"Unknown model: {base_name}")
                return False
            
            config = self.model_configs[base_name]
            logger.info(f"🔄 [PRE-TRAINED] Loading {model_name} from Hugging Face...")
            
            if config["type"] == "pytorch":
                model = PyTorchModel(config["path"], model_name)
            else:
                model = SklearnModel(config["path"], model_name)
            
            model.load()
            model.model_source = "Hugging Face"
            model.model_version = "pre-trained"
            self.models[model_name] = model
            
            logger.info(f"✅ Successfully loaded pre-trained model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading pre-trained model {model_name}: {e}")
            return False
    
    async def _load_trained_model(self, base_name: str, model_name: str, version: str = None) -> bool:
        """Load trained model from MLflow"""
        try:
            logger.info(f"🔄 [TRAINED] Loading trained model {model_name} from MLflow...")
            
            # Get model info from training service or MLflow
            model_info = self.get_trained_model_info(base_name, version)
            if not model_info:
                logger.error(f"No trained model found for {base_name}")
                return False
            
            # Check cache first
            cached_model = await self._get_from_cache(model_name, "dummy_text_for_cache_check")
            if cached_model and cached_model.get("predictions"):
                logger.info(f"✅ [CACHE] Model {model_name} available in distributed cache")
                # For now, we still need to load the model locally for tokenizer access
                # In a full implementation, we'd return cached model directly
            
            # Load model from MLflow
            model_uri = model_info.get('mlflow_uri', f"models:/security_{base_name}/latest")
            logger.info(f"📥 [MLFLOW] Loading from: {model_uri}")
            
            import mlflow.pytorch
            import mlflow.artifacts
            import tempfile
            import os
            
            # Load model with signature validation
            model = mlflow.pytorch.load_model(model_uri)
            
            # Validate model signature if available
            try:
                from mlflow.models import get_model_info
                model_info_obj = get_model_info(model_uri)
                if hasattr(model_info_obj, 'signature') and model_info_obj.signature:
                    logger.info("✅ [MLFLOW] Model signature validated")
                else:
                    logger.warning("⚠️ [MLFLOW] No model signature found")
            except Exception as e:
                logger.warning(f"⚠️ [MLFLOW] Could not validate model signature: {e}")
            
            # Load tokenizer from MLflow artifacts
            try:
                # Download tokenizer artifacts
                tokenizer_path = mlflow.artifacts.download_artifacts(
                    run_id=model_info.get('run_id', ''),
                    artifact_path="tokenizer"
                )
                logger.info(f"📥 [MLFLOW] Loading tokenizer from: {tokenizer_path}")
                
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                logger.info("✅ [MLFLOW] Tokenizer loaded from MLflow artifacts")
                
            except Exception as e:
                logger.warning(f"⚠️ [MLFLOW] Failed to load tokenizer from MLflow: {e}")
                logger.info("🔄 [MLFLOW] Falling back to original config tokenizer")
                
                # Fallback to original config
                config = self.model_configs[base_name]
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(config["path"])
            
            # Create wrapper
            trained_model = PyTorchModel(model_uri, model_name)
            trained_model.model = model
            trained_model.tokenizer = tokenizer
            trained_model.loaded = True
            trained_model.model_source = "MLflow/MinIO"
            trained_model.model_version = model_info.get('version', 'unknown')
            
            self.models[model_name] = trained_model
            logger.info(f"✅ Successfully loaded trained model: {model_name}")
            
            # Store model in distributed cache for future use
            await self._preload_model_in_cache(model_name)
            await self._notify_cache_model_loaded(model_name)
            
            return True
        except Exception as e:
            logger.error(f"Error loading trained model {model_name}: {e}")
            return False
    
    def get_trained_model_info(self, model_name: str, version: str = None):
        """Get trained model info using hybrid approach"""
        try:
            # Try training service API first
            import requests
            response = requests.get("http://training:8002/models/latest", timeout=5)
            
            if response.status_code == 200:
                latest_models = response.json().get("latest_models", {})
                if model_name in latest_models:
                    return latest_models[model_name]
        except Exception as e:
            logger.warning(f"Training service unavailable: {e}")
        
        # Fallback to MLflow
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            target_version = version or "latest"
            
            if target_version == "latest":
                model_versions = client.get_latest_versions(f"security_{model_name}")
                if model_versions:
                    target_version_obj = model_versions[0]
                else:
                    return None
            else:
                target_version_obj = client.get_model_version(f"security_{model_name}", target_version)
            
            return {
                "model_name": model_name,
                "version": target_version_obj.version,
                "run_id": target_version_obj.run_id,
                "f1_score": 0.0,
                "accuracy": 0.0,
                "timestamp": datetime.fromtimestamp(target_version_obj.creation_timestamp / 1000).isoformat(),
                "mlflow_uri": f"models:/security_{model_name}/{target_version}"
            }
        except Exception as e:
            logger.warning(f"MLflow query failed: {e}")
            return None
    
    def _check_cache_health(self) -> bool:
        """Check if model-cache service is available"""
        try:
            import httpx
            response = httpx.get(f"{self.model_cache_url}/health", timeout=2.0)
            if response.status_code == 200:
                logger.info("✅ [CACHE] Model-cache service is available")
                self.cache_enabled = True
                return True
            else:
                logger.warning(f"⚠️ [CACHE] Model-cache service returned {response.status_code}")
                self.cache_enabled = False
                return False
        except Exception as e:
            logger.warning(f"⚠️ [CACHE] Model-cache service unavailable: {e}. Will use local cache only.")
            self.cache_enabled = False
            return False
    
    async def _get_from_cache(self, model_name: str, text: str) -> Optional[Dict[str, Any]]:
        """Get prediction from model-cache service"""
        if not self.cache_enabled:
            self.cache_stats["misses"] += 1
            return None
        
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.model_cache_url}/predict",
                    json={"text": text, "models": [model_name]},
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self.cache_stats["hits"] += 1
                    logger.debug(f"✅ [CACHE HIT] Got prediction for {model_name} from cache")
                    return result
                else:
                    self.cache_stats["misses"] += 1
                    logger.debug(f"⚠️ [CACHE MISS] Cache returned {response.status_code}")
                    return None
        except Exception as e:
            self.cache_stats["errors"] += 1
            logger.warning(f"⚠️ [CACHE ERROR] Failed to get from cache: {e}")
            return None
    
    async def _preload_model_in_cache(self, model_name: str) -> bool:
        """Preload a model into the distributed cache"""
        if not self.cache_enabled:
            return False
        
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.model_cache_url}/models/{model_name}/preload",
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ [CACHE] Preloaded model {model_name} into cache")
                    return True
                else:
                    logger.warning(f"⚠️ [CACHE] Failed to preload {model_name}: {response.status_code}")
                    return False
        except Exception as e:
            logger.warning(f"⚠️ [CACHE] Error preloading {model_name}: {e}")
            return False
    
    async def _notify_cache_model_loaded(self, model_name: str) -> None:
        """Notify cache service that a model has been loaded"""
        if not self.cache_enabled:
            return
        
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.model_cache_url}/models/{model_name}/notify",
                    json={
                        "action": "loaded",
                        "timestamp": datetime.now().timestamp(),
                        "source": "model-api"
                    },
                    timeout=2.0
                )
                logger.debug(f"📢 [CACHE] Notified cache about {model_name} load")
        except Exception as e:
            logger.debug(f"⚠️ [CACHE] Failed to notify cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total * 100) if total > 0 else 0.0
        
        return {
            "enabled": self.cache_enabled,
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "errors": self.cache_stats["errors"],
            "hit_rate": f"{hit_rate:.2f}%",
            "cache_url": self.model_cache_url if self.cache_enabled else None
        }
