"""
Business Metrics Service - Cache Utilities
Simple caching decorators for business metrics
"""

import asyncio
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Simple in-memory cache
_cache: Dict[str, Dict[str, Any]] = {}

def cached_metrics_summary(ttl: int = 300):
    """
    Cache decorator for metrics summary with TTL
    
    Args:
        ttl: Time to live in seconds (default: 5 minutes)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Check if cached result exists and is still valid
            if cache_key in _cache:
                cached_data = _cache[cache_key]
                if time.time() - cached_data['timestamp'] < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_data['result']
                else:
                    # Remove expired cache entry
                    del _cache[cache_key]
            
            # Execute function and cache result
            try:
                result = await func(*args, **kwargs)
                _cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time()
                }
                logger.debug(f"Cached result for {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Error in cached function {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator

def clear_cache():
    """Clear all cached data"""
    global _cache
    _cache.clear()
    logger.info("Cache cleared")

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    return {
        'total_entries': len(_cache),
        'cache_keys': list(_cache.keys())
    }
