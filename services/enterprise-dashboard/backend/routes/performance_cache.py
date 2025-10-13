from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import redis
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
router = APIRouter()

# Redis connection
redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

# Cache keys
PERFORMANCE_DATA_KEY = "performance:analytics:data"
PERFORMANCE_METADATA_KEY = "performance:analytics:metadata"

class PerformanceCache:
    """Redis-based caching for performance analytics data"""
    
    @staticmethod
    def cache_performance_data(data: Dict[str, Any], ttl: int = 300) -> bool:
        """Cache performance data with TTL (default 5 minutes)"""
        try:
            # Store the data
            redis_client.setex(
                PERFORMANCE_DATA_KEY, 
                ttl, 
                json.dumps(data, default=str)
            )
            
            # Store metadata
            metadata = {
                "cached_at": datetime.now().isoformat(),
                "ttl": ttl,
                "data_count": len(data.get('jobs', [])),
                "jobs_with_metrics": len([job for job in data.get('jobs', []) if job.get('result', {}).get('metrics')])
            }
            redis_client.setex(
                PERFORMANCE_METADATA_KEY,
                ttl,
                json.dumps(metadata, default=str)
            )
            
            logger.info(f"✅ Cached performance data: {metadata['jobs_with_metrics']} jobs with metrics")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to cache performance data: {e}")
            return False
    
    @staticmethod
    def get_cached_performance_data() -> Optional[Dict[str, Any]]:
        """Get cached performance data"""
        try:
            data = redis_client.get(PERFORMANCE_DATA_KEY)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get cached performance data: {e}")
            return None
    
    @staticmethod
    def get_cache_metadata() -> Optional[Dict[str, Any]]:
        """Get cache metadata"""
        try:
            metadata = redis_client.get(PERFORMANCE_METADATA_KEY)
            if metadata:
                return json.loads(metadata)
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get cache metadata: {e}")
            return None
    
    @staticmethod
    def invalidate_cache() -> bool:
        """Invalidate performance cache"""
        try:
            redis_client.delete(PERFORMANCE_DATA_KEY, PERFORMANCE_METADATA_KEY)
            logger.info("🗑️ Performance cache invalidated")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to invalidate cache: {e}")
            return False
    
    @staticmethod
    def is_cache_valid() -> bool:
        """Check if cache is valid and not expired"""
        try:
            return redis_client.exists(PERFORMANCE_DATA_KEY) > 0
        except Exception as e:
            logger.error(f"❌ Failed to check cache validity: {e}")
            return False

@router.get("/performance/cache/status")
async def get_cache_status():
    """Get cache status and metadata"""
    try:
        metadata = PerformanceCache.get_cache_metadata()
        is_valid = PerformanceCache.is_cache_valid()
        
        return {
            "cache_valid": is_valid,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get cache status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/cache/data")
async def get_cached_performance_data():
    """Get cached performance data"""
    try:
        data = PerformanceCache.get_cached_performance_data()
        if data:
            return data
        else:
            return {"message": "No cached data available", "data": None}
    except Exception as e:
        logger.error(f"Failed to get cached data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/performance/cache/invalidate")
async def invalidate_performance_cache():
    """Invalidate performance cache"""
    try:
        success = PerformanceCache.invalidate_cache()
        return {"success": success, "message": "Cache invalidated" if success else "Failed to invalidate cache"}
    except Exception as e:
        logger.error(f"Failed to invalidate cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/performance/cache/refresh")
async def refresh_performance_cache():
    """Refresh performance cache with fresh data"""
    try:
        # This would typically fetch fresh data from the training service
        # For now, we'll just invalidate and let the next request populate it
        PerformanceCache.invalidate_cache()
        return {"success": True, "message": "Cache refresh initiated"}
    except Exception as e:
        logger.error(f"Failed to refresh cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))
