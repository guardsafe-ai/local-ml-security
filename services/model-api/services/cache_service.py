"""
Cache Service - Redis caching logic
"""

import logging
import redis
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


class CacheService:
    """Handles Redis caching for predictions"""
    
    def __init__(self):
        self.redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
        self.cache_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for prediction"""
        import hashlib
        content = f"{text}:{model_name}"
        return f"prediction:{hashlib.md5(content.encode()).hexdigest()}"
    
    def get_cached_prediction(self, text: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction if available"""
        try:
            cache_key = self.get_cache_key(text, model_name)
            cached = self.redis_client.get(cache_key)
            
            if cached:
                self.cache_stats["cache_hits"] += 1
                logger.info(f"🎯 [CACHE HIT] Found cached prediction for {model_name}")
                return json.loads(cached)
            else:
                self.cache_stats["cache_misses"] += 1
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def cache_prediction(self, text: str, model_name: str, prediction: Dict[str, Any], ttl: int = 3600):
        """Cache prediction result"""
        try:
            cache_key = self.get_cache_key(text, model_name)
            self.redis_client.setex(cache_key, ttl, json.dumps(prediction))
            logger.info(f"💾 [CACHE SET] Cached prediction for {model_name}")
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.cache_stats["total_requests"]
        hits = self.cache_stats["cache_hits"]
        misses = self.cache_stats["cache_misses"]
        
        return {
            "total_requests": total,
            "cache_hits": hits,
            "cache_misses": misses,
            "hit_rate": hits / total if total > 0 else 0.0,
            "models_cached": len(self.redis_client.keys("prediction:*"))
        }
    
    def clear_cache(self):
        """Clear all cached predictions"""
        try:
            keys = self.redis_client.keys("prediction:*")
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"🗑️ [CACHE CLEAR] Cleared {len(keys)} cached predictions")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
