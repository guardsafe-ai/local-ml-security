"""
Shared Model Storage Service
Implements shared model storage using Redis for distributed caching
"""

import logging
import json
import pickle
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import redis.asyncio as redis
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Model metadata for shared storage"""
    model_name: str
    model_type: str
    version: str
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int
    checksum: str
    quantization_type: Optional[str] = None
    framework: str = "pytorch"

class SharedModelStorage:
    """Handles shared model storage across services"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://redis:6379/3")
        self.redis_client = None
        self.model_metadata = {}
        self.max_models = 10  # Maximum models in shared storage
        self.max_size_mb = 2048  # Maximum total size in MB
        
    async def get_redis_client(self) -> redis.Redis:
        """Get or create Redis client"""
        if self.redis_client is None:
            try:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=False)
                await self.redis_client.ping()
                logger.info("✅ [SHARED_STORAGE] Redis client connected")
            except Exception as e:
                logger.error(f"❌ [SHARED_STORAGE] Redis connection failed: {e}")
                self.redis_client = None
        return self.redis_client
    
    async def store_model(self, model_name: str, model_data: Any, 
                         model_type: str = "pytorch", version: str = "latest",
                         quantization_type: str = None) -> bool:
        """
        Store model in shared storage
        
        Args:
            model_name: Name of the model
            model_data: Model data to store
            model_type: Type of model (pytorch, sklearn, etc.)
            version: Model version
            quantization_type: Optional quantization type
            
        Returns:
            True if successful
        """
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                logger.error("❌ [SHARED_STORAGE] Redis not available")
                return False
            
            # Serialize model data
            if model_type == "pytorch":
                serialized_data = pickle.dumps(model_data)
            else:
                serialized_data = pickle.dumps(model_data)
            
            # Calculate metadata
            size_bytes = len(serialized_data)
            checksum = hashlib.md5(serialized_data).hexdigest()
            current_time = time.time()
            
            # Create metadata
            metadata = ModelMetadata(
                model_name=model_name,
                model_type=model_type,
                version=version,
                size_bytes=size_bytes,
                created_at=current_time,
                last_accessed=current_time,
                access_count=0,
                checksum=checksum,
                quantization_type=quantization_type
            )
            
            # Check storage limits
            if not await self._check_storage_limits(size_bytes):
                await self._evict_old_models()
            
            # Store model data
            model_key = f"model:{model_name}:{version}"
            await redis_client.set(model_key, serialized_data, ex=86400)  # 24 hour TTL
            
            # Store metadata
            metadata_key = f"metadata:{model_name}:{version}"
            metadata_dict = {
                "model_name": metadata.model_name,
                "model_type": metadata.model_type,
                "version": metadata.version,
                "size_bytes": metadata.size_bytes,
                "created_at": metadata.created_at,
                "last_accessed": metadata.last_accessed,
                "access_count": metadata.access_count,
                "checksum": metadata.checksum,
                "quantization_type": metadata.quantization_type,
                "framework": metadata.framework
            }
            await redis_client.set(metadata_key, json.dumps(metadata_dict), ex=86400)
            
            # Update index
            await redis_client.sadd("model_index", f"{model_name}:{version}")
            
            logger.info(f"✅ [SHARED_STORAGE] Stored model: {model_name}:{version} ({size_bytes} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"❌ [SHARED_STORAGE] Failed to store model {model_name}: {e}")
            return False
    
    async def load_model(self, model_name: str, version: str = "latest") -> Optional[Any]:
        """
        Load model from shared storage
        
        Args:
            model_name: Name of the model
            version: Model version
            
        Returns:
            Model data or None if not found
        """
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                logger.error("❌ [SHARED_STORAGE] Redis not available")
                return None
            
            model_key = f"model:{model_name}:{version}"
            metadata_key = f"metadata:{model_name}:{version}"
            
            # Check if model exists
            if not await redis_client.exists(model_key):
                logger.warning(f"⚠️ [SHARED_STORAGE] Model not found: {model_name}:{version}")
                return None
            
            # Load model data
            serialized_data = await redis_client.get(model_key)
            if not serialized_data:
                logger.error(f"❌ [SHARED_STORAGE] Failed to load model data: {model_name}:{version}")
                return None
            
            # Deserialize model
            model_data = pickle.loads(serialized_data)
            
            # Update metadata
            await self._update_model_metadata(model_name, version)
            
            logger.info(f"✅ [SHARED_STORAGE] Loaded model: {model_name}:{version}")
            return model_data
            
        except Exception as e:
            logger.error(f"❌ [SHARED_STORAGE] Failed to load model {model_name}:{version}: {e}")
            return None
    
    async def _update_model_metadata(self, model_name: str, version: str):
        """Update model metadata with access information"""
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return
            
            metadata_key = f"metadata:{model_name}:{version}"
            metadata_json = await redis_client.get(metadata_key)
            
            if metadata_json:
                metadata_dict = json.loads(metadata_json)
                metadata_dict["last_accessed"] = time.time()
                metadata_dict["access_count"] = metadata_dict.get("access_count", 0) + 1
                
                await redis_client.set(metadata_key, json.dumps(metadata_dict), ex=86400)
                
        except Exception as e:
            logger.error(f"❌ [SHARED_STORAGE] Failed to update metadata: {e}")
    
    async def _check_storage_limits(self, new_size_bytes: int) -> bool:
        """Check if adding new model would exceed storage limits"""
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return False
            
            # Get current storage usage
            model_index = await redis_client.smembers("model_index")
            total_size = 0
            model_count = len(model_index)
            
            for model_ref in model_index:
                model_name, version = model_ref.decode().split(":")
                metadata_key = f"metadata:{model_name}:{version}"
                metadata_json = await redis_client.get(metadata_key)
                
                if metadata_json:
                    metadata_dict = json.loads(metadata_json)
                    total_size += metadata_dict.get("size_bytes", 0)
            
            # Check limits
            total_size_mb = total_size / (1024 * 1024)
            new_size_mb = new_size_bytes / (1024 * 1024)
            
            if model_count >= self.max_models:
                logger.warning(f"⚠️ [SHARED_STORAGE] Model count limit reached: {model_count}/{self.max_models}")
                return False
            
            if total_size_mb + new_size_mb > self.max_size_mb:
                logger.warning(f"⚠️ [SHARED_STORAGE] Size limit would be exceeded: {total_size_mb + new_size_mb:.1f}MB/{self.max_size_mb}MB")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ [SHARED_STORAGE] Failed to check storage limits: {e}")
            return False
    
    async def _evict_old_models(self):
        """Evict old models to make space"""
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return
            
            # Get all models with metadata
            model_index = await redis_client.smembers("model_index")
            models_info = []
            
            for model_ref in model_index:
                model_name, version = model_ref.decode().split(":")
                metadata_key = f"metadata:{model_name}:{version}"
                metadata_json = await redis_client.get(metadata_key)
                
                if metadata_json:
                    metadata_dict = json.loads(metadata_json)
                    models_info.append((model_name, version, metadata_dict))
            
            # Sort by last accessed time (oldest first)
            models_info.sort(key=lambda x: x[2].get("last_accessed", 0))
            
            # Evict oldest models until we have space
            evicted_count = 0
            for model_name, version, metadata in models_info:
                if evicted_count >= 2:  # Evict at least 2 models
                    break
                
                await self._remove_model(model_name, version)
                evicted_count += 1
                logger.info(f"🗑️ [SHARED_STORAGE] Evicted model: {model_name}:{version}")
            
        except Exception as e:
            logger.error(f"❌ [SHARED_STORAGE] Failed to evict old models: {e}")
    
    async def _remove_model(self, model_name: str, version: str):
        """Remove model from shared storage"""
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return
            
            model_key = f"model:{model_name}:{version}"
            metadata_key = f"metadata:{model_name}:{version}"
            
            # Remove model data and metadata
            await redis_client.delete(model_key)
            await redis_client.delete(metadata_key)
            
            # Remove from index
            await redis_client.srem("model_index", f"{model_name}:{version}")
            
        except Exception as e:
            logger.error(f"❌ [SHARED_STORAGE] Failed to remove model: {e}")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all models in shared storage"""
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return []
            
            model_index = await redis_client.smembers("model_index")
            models = []
            
            for model_ref in model_index:
                model_name, version = model_ref.decode().split(":")
                metadata_key = f"metadata:{model_name}:{version}"
                metadata_json = await redis_client.get(metadata_key)
                
                if metadata_json:
                    metadata_dict = json.loads(metadata_json)
                    models.append(metadata_dict)
            
            return models
            
        except Exception as e:
            logger.error(f"❌ [SHARED_STORAGE] Failed to list models: {e}")
            return []
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return {"error": "Redis not available"}
            
            model_index = await redis_client.smembers("model_index")
            total_size = 0
            model_count = len(model_index)
            
            for model_ref in model_index:
                model_name, version = model_ref.decode().split(":")
                metadata_key = f"metadata:{model_name}:{version}"
                metadata_json = await redis_client.get(metadata_key)
                
                if metadata_json:
                    metadata_dict = json.loads(metadata_json)
                    total_size += metadata_dict.get("size_bytes", 0)
            
            return {
                "total_models": model_count,
                "total_size_mb": total_size / (1024 * 1024),
                "max_models": self.max_models,
                "max_size_mb": self.max_size_mb,
                "utilization_percent": (model_count / self.max_models) * 100,
                "size_utilization_percent": (total_size / (1024 * 1024)) / self.max_size_mb * 100
            }
            
        except Exception as e:
            logger.error(f"❌ [SHARED_STORAGE] Failed to get storage stats: {e}")
            return {"error": str(e)}
    
    async def cleanup_expired_models(self):
        """Clean up expired models"""
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return
            
            # Redis TTL will handle expiration automatically
            # This method can be used for additional cleanup logic
            
            logger.info("✅ [SHARED_STORAGE] Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ [SHARED_STORAGE] Cleanup failed: {e}")

# Global shared model storage
shared_model_storage = SharedModelStorage()
