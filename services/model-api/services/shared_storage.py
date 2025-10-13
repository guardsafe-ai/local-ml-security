"""
Shared Model Storage Service
Provides distributed model storage using Redis for caching
"""

import logging
import json
import pickle
import redis
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Metadata for stored models"""
    model_name: str
    model_type: str
    version: str
    size_mb: float
    created_at: float
    last_accessed: float
    access_count: int = 0
    quantization_type: Optional[str] = None

class SharedModelStorage:
    """Distributed model storage using Redis"""
    
    def __init__(self, redis_host: str = "redis", redis_port: int = 6379):
        self.redis_client = redis.Redis(
            host=redis_host, 
            port=redis_port, 
            decode_responses=False,  # We need binary data for models
            socket_connect_timeout=5,
            socket_timeout=5
        )
        self.metadata_key_prefix = "model_metadata:"
        self.model_key_prefix = "model_data:"
        
    def _get_metadata_key(self, model_name: str) -> str:
        """Get Redis key for model metadata"""
        return f"{self.metadata_key_prefix}{model_name}"
    
    def _get_model_key(self, model_name: str) -> str:
        """Get Redis key for model data"""
        return f"{self.model_key_prefix}{model_name}"
    
    async def store_model(self, 
                         model_name: str,
                         model_data: Any,
                         model_type: str = "pytorch",
                         version: str = "1.0",
                         size_mb: float = 0.0,
                         quantization_type: str = None) -> bool:
        """
        Store a model in shared storage
        
        Args:
            model_name: Name of the model
            model_data: Model data to store
            model_type: Type of model (pytorch, sklearn, etc.)
            version: Model version
            size_mb: Model size in MB
            quantization_type: Optional quantization type
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import time
            
            # Serialize model data
            if model_type == "pytorch":
                import torch
                serialized_data = pickle.dumps(model_data)
            elif model_type == "sklearn":
                serialized_data = pickle.dumps(model_data)
            else:
                serialized_data = pickle.dumps(model_data)
            
            # Store model data
            model_key = self._get_model_key(model_name)
            self.redis_client.setex(model_key, 86400, serialized_data)  # 24 hour TTL
            
            # Store metadata
            metadata = ModelMetadata(
                model_name=model_name,
                model_type=model_type,
                version=version,
                size_mb=size_mb,
                created_at=time.time(),
                last_accessed=time.time(),
                quantization_type=quantization_type
            )
            
            metadata_key = self._get_metadata_key(model_name)
            metadata_json = json.dumps({
                "model_name": metadata.model_name,
                "model_type": metadata.model_type,
                "version": metadata.version,
                "size_mb": metadata.size_mb,
                "created_at": metadata.created_at,
                "last_accessed": metadata.last_accessed,
                "access_count": metadata.access_count,
                "quantization_type": metadata.quantization_type
            })
            
            self.redis_client.setex(metadata_key, 86400, metadata_json)
            
            logger.info(f"✅ [SHARED STORAGE] Stored model {model_name} ({size_mb:.2f}MB)")
            return True
            
        except Exception as e:
            logger.error(f"❌ [SHARED STORAGE] Failed to store model {model_name}: {e}")
            return False
    
    async def get_model(self, model_name: str) -> Optional[Any]:
        """
        Retrieve a model from shared storage
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            Model data if found, None otherwise
        """
        try:
            # Get model data
            model_key = self._get_model_key(model_name)
            serialized_data = self.redis_client.get(model_key)
            
            if not serialized_data:
                logger.warning(f"⚠️ [SHARED STORAGE] Model {model_name} not found in cache")
                return None
            
            # Deserialize model data
            model_data = pickle.loads(serialized_data)
            
            # Update access metadata
            await self._update_access_metadata(model_name)
            
            logger.info(f"✅ [SHARED STORAGE] Retrieved model {model_name}")
            return model_data
            
        except Exception as e:
            logger.error(f"❌ [SHARED STORAGE] Failed to retrieve model {model_name}: {e}")
            return None
    
    async def _update_access_metadata(self, model_name: str) -> None:
        """Update access metadata for a model"""
        try:
            metadata_key = self._get_metadata_key(model_name)
            metadata_json = self.redis_client.get(metadata_key)
            
            if metadata_json:
                metadata = json.loads(metadata_json)
                metadata["last_accessed"] = time.time()
                metadata["access_count"] = metadata.get("access_count", 0) + 1
                
                self.redis_client.setex(metadata_key, 86400, json.dumps(metadata))
                
        except Exception as e:
            logger.warning(f"⚠️ [SHARED STORAGE] Failed to update access metadata for {model_name}: {e}")
    
    async def get_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """Get metadata for a model"""
        try:
            metadata_key = self._get_metadata_key(model_name)
            metadata_json = self.redis_client.get(metadata_key)
            
            if not metadata_json:
                return None
            
            metadata_dict = json.loads(metadata_json)
            return ModelMetadata(**metadata_dict)
            
        except Exception as e:
            logger.error(f"❌ [SHARED STORAGE] Failed to get metadata for {model_name}: {e}")
            return None
    
    async def list_models(self) -> list:
        """List all models in shared storage"""
        try:
            pattern = f"{self.metadata_key_prefix}*"
            keys = self.redis_client.keys(pattern)
            
            models = []
            for key in keys:
                model_name = key.decode().replace(self.metadata_key_prefix, "")
                metadata = await self.get_model_metadata(model_name)
                if metadata:
                    models.append(metadata)
            
            return models
            
        except Exception as e:
            logger.error(f"❌ [SHARED STORAGE] Failed to list models: {e}")
            return []
    
    async def delete_model(self, model_name: str) -> bool:
        """Delete a model from shared storage"""
        try:
            model_key = self._get_model_key(model_name)
            metadata_key = self._get_metadata_key(model_name)
            
            self.redis_client.delete(model_key)
            self.redis_client.delete(metadata_key)
            
            logger.info(f"✅ [SHARED STORAGE] Deleted model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [SHARED STORAGE] Failed to delete model {model_name}: {e}")
            return False
    
    async def cleanup_expired_models(self, max_age_hours: int = 24) -> int:
        """Clean up models that haven't been accessed recently"""
        try:
            import time
            cutoff_time = time.time() - (max_age_hours * 3600)
            cleaned_count = 0
            
            models = await self.list_models()
            for model in models:
                if model.last_accessed < cutoff_time:
                    await self.delete_model(model.model_name)
                    cleaned_count += 1
            
            logger.info(f"🧹 [SHARED STORAGE] Cleaned up {cleaned_count} expired models")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"❌ [SHARED STORAGE] Failed to cleanup expired models: {e}")
            return 0
