"""
Query result caching utilities
"""

import json
import hashlib
import logging
from typing import Any, Optional, Callable
from functools import wraps
import redis.asyncio as redis
import asyncio

logger = logging.getLogger(__name__)

class QueryCache:
    """Redis-based query result cache"""
    
    def __init__(self, redis_url: str = "redis://redis:6379/2"):
        self.redis_url = redis_url
        self.redis_client = None
        self.default_ttl = 300  # 5 minutes
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("✅ Query cache connected to Redis")
        except Exception as e:
            logger.error(f"❌ Failed to connect to query cache: {e}")
            self.redis_client = None
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("🔒 Query cache disconnected from Redis")
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a cache key from function arguments"""
        # Create a hash of the arguments
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis_client:
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"❌ Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        if not self.redis_client:
            return False
        
        try:
            ttl = ttl or self.default_ttl
            await self.redis_client.setex(key, ttl, json.dumps(value))
            return True
        except Exception as e:
            logger.error(f"❌ Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"❌ Cache delete error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        if not self.redis_client:
            return 0
        
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                return await self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"❌ Cache clear pattern error: {e}")
            return 0

# Global cache instance
query_cache = QueryCache()

def cached(prefix: str, ttl: int = 300):
    """
    Decorator for caching function results
    
    Args:
        prefix: Cache key prefix
        ttl: Time to live in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            cache_key = query_cache._generate_cache_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_result = await query_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"🎯 [CACHE HIT] {func.__name__}")
                return cached_result
            
            # Execute function
            logger.debug(f"🔄 [CACHE MISS] {func.__name__}")
            result = await func(*args, **kwargs)
            
            # Cache result
            await query_cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

# Specific cache decorators for common queries
def cached_drift_summary(ttl: int = 300):
    """Cache drift detection summary queries"""
    return cached("drift_summary", ttl)

def cached_metrics_summary(ttl: int = 300):
    """Cache business metrics summary queries"""
    return cached("metrics_summary", ttl)

def cached_model_performance(ttl: int = 600):
    """Cache model performance queries (longer TTL)"""
    return cached("model_performance", ttl)

def cached_analytics_data(ttl: int = 300):
    """Cache analytics data queries"""
    return cached("analytics_data", ttl)
