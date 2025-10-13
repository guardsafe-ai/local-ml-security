"""
Rate Limiting Service
Implements rate limiting per user/IP with Redis backend
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import redis.asyncio as redis
import json

logger = logging.getLogger(__name__)

class RateLimitType(Enum):
    """Types of rate limits"""
    PER_USER = "per_user"
    PER_IP = "per_ip"
    PER_API_KEY = "per_api_key"
    GLOBAL = "global"

@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10  # Allow burst of requests
    window_size: int = 60  # seconds

class RateLimiter:
    """Handles rate limiting for API requests"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://redis:6379/2")
        self.redis_client = None
        self.local_cache = {}  # Fallback local cache
        self.configs = {
            RateLimitType.PER_USER: RateLimitConfig(requests_per_minute=60, requests_per_hour=1000),
            RateLimitType.PER_IP: RateLimitConfig(requests_per_minute=100, requests_per_hour=5000),
            RateLimitType.PER_API_KEY: RateLimitConfig(requests_per_minute=200, requests_per_hour=10000),
            RateLimitType.GLOBAL: RateLimitConfig(requests_per_minute=1000, requests_per_hour=50000)
        }
        
    async def get_redis_client(self) -> redis.Redis:
        """Get or create Redis client"""
        if self.redis_client is None:
            try:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                await self.redis_client.ping()
                logger.info("✅ [RATE_LIMIT] Redis client connected")
            except Exception as e:
                logger.warning(f"⚠️ [RATE_LIMIT] Redis unavailable, using local cache: {e}")
                self.redis_client = None
        return self.redis_client
    
    async def check_rate_limit(self, identifier: str, limit_type: RateLimitType,
                              custom_config: RateLimitConfig = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limits
        
        Args:
            identifier: User ID, IP address, or API key
            limit_type: Type of rate limit
            custom_config: Custom rate limit configuration
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        try:
            config = custom_config or self.configs[limit_type]
            redis_client = await self.get_redis_client()
            
            if redis_client:
                return await self._check_redis_rate_limit(identifier, limit_type, config, redis_client)
            else:
                return await self._check_local_rate_limit(identifier, limit_type, config)
                
        except Exception as e:
            logger.error(f"❌ [RATE_LIMIT] Rate limit check failed: {e}")
            # Allow request on error (fail open)
            return True, {"error": str(e)}
    
    async def _check_redis_rate_limit(self, identifier: str, limit_type: RateLimitType,
                                    config: RateLimitConfig, redis_client: redis.Redis) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using Redis"""
        try:
            current_time = int(time.time())
            window_start = current_time - config.window_size
            
            # Create keys for different time windows
            minute_key = f"rate_limit:{limit_type.value}:{identifier}:minute:{current_time // 60}"
            hour_key = f"rate_limit:{limit_type.value}:{identifier}:hour:{current_time // 3600}"
            day_key = f"rate_limit:{limit_type.value}:{identifier}:day:{current_time // 86400}"
            
            # Use Redis pipeline for atomic operations
            pipe = redis_client.pipeline()
            
            # Increment counters
            pipe.incr(minute_key)
            pipe.expire(minute_key, 60)
            
            pipe.incr(hour_key)
            pipe.expire(hour_key, 3600)
            
            pipe.incr(day_key)
            pipe.expire(day_key, 86400)
            
            # Execute pipeline
            results = await pipe.execute()
            
            minute_count, hour_count, day_count = results[0], results[1], results[2]
            
            # Check limits
            is_allowed = True
            exceeded_limits = []
            
            if minute_count > config.requests_per_minute:
                is_allowed = False
                exceeded_limits.append(f"minute:{minute_count}/{config.requests_per_minute}")
            
            if hour_count > config.requests_per_hour:
                is_allowed = False
                exceeded_limits.append(f"hour:{hour_count}/{config.requests_per_hour}")
            
            if day_count > config.requests_per_day:
                is_allowed = False
                exceeded_limits.append(f"day:{day_count}/{config.requests_per_day}")
            
            # Calculate reset times
            minute_reset = ((current_time // 60) + 1) * 60
            hour_reset = ((current_time // 3600) + 1) * 3600
            day_reset = ((current_time // 86400) + 1) * 86400
            
            rate_limit_info = {
                "is_allowed": is_allowed,
                "identifier": identifier,
                "limit_type": limit_type.value,
                "current_counts": {
                    "minute": minute_count,
                    "hour": hour_count,
                    "day": day_count
                },
                "limits": {
                    "minute": config.requests_per_minute,
                    "hour": config.requests_per_hour,
                    "day": config.requests_per_day
                },
                "reset_times": {
                    "minute": minute_reset,
                    "hour": hour_reset,
                    "day": day_reset
                },
                "exceeded_limits": exceeded_limits
            }
            
            if not is_allowed:
                logger.warning(f"🚨 [RATE_LIMIT] Rate limit exceeded for {identifier}: {exceeded_limits}")
            
            return is_allowed, rate_limit_info
            
        except Exception as e:
            logger.error(f"❌ [RATE_LIMIT] Redis rate limit check failed: {e}")
            raise
    
    async def _check_local_rate_limit(self, identifier: str, limit_type: RateLimitType,
                                    config: RateLimitConfig) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using local cache (fallback)"""
        try:
            current_time = time.time()
            key = f"{limit_type.value}:{identifier}"
            
            # Initialize or get existing data
            if key not in self.local_cache:
                self.local_cache[key] = {
                    "requests": [],
                    "last_cleanup": current_time
                }
            
            data = self.local_cache[key]
            
            # Clean old requests
            cutoff_time = current_time - config.window_size
            data["requests"] = [req_time for req_time in data["requests"] if req_time > cutoff_time]
            
            # Add current request
            data["requests"].append(current_time)
            
            # Count requests in different windows
            minute_cutoff = current_time - 60
            hour_cutoff = current_time - 3600
            day_cutoff = current_time - 86400
            
            minute_count = len([req for req in data["requests"] if req > minute_cutoff])
            hour_count = len([req for req in data["requests"] if req > hour_cutoff])
            day_count = len([req for req in data["requests"] if req > day_cutoff])
            
            # Check limits
            is_allowed = True
            exceeded_limits = []
            
            if minute_count > config.requests_per_minute:
                is_allowed = False
                exceeded_limits.append(f"minute:{minute_count}/{config.requests_per_minute}")
            
            if hour_count > config.requests_per_hour:
                is_allowed = False
                exceeded_limits.append(f"hour:{hour_count}/{config.requests_per_hour}")
            
            if day_count > config.requests_per_day:
                is_allowed = False
                exceeded_limits.append(f"day:{day_count}/{config.requests_per_day}")
            
            rate_limit_info = {
                "is_allowed": is_allowed,
                "identifier": identifier,
                "limit_type": limit_type.value,
                "current_counts": {
                    "minute": minute_count,
                    "hour": hour_count,
                    "day": day_count
                },
                "limits": {
                    "minute": config.requests_per_minute,
                    "hour": config.requests_per_hour,
                    "day": config.requests_per_day
                },
                "exceeded_limits": exceeded_limits,
                "backend": "local_cache"
            }
            
            if not is_allowed:
                logger.warning(f"🚨 [RATE_LIMIT] Rate limit exceeded for {identifier}: {exceeded_limits}")
            
            return is_allowed, rate_limit_info
            
        except Exception as e:
            logger.error(f"❌ [RATE_LIMIT] Local rate limit check failed: {e}")
            raise
    
    async def get_rate_limit_status(self, identifier: str, limit_type: RateLimitType) -> Dict[str, Any]:
        """Get current rate limit status without incrementing counters"""
        try:
            config = self.configs[limit_type]
            redis_client = await self.get_redis_client()
            
            if redis_client:
                return await self._get_redis_status(identifier, limit_type, config, redis_client)
            else:
                return await self._get_local_status(identifier, limit_type, config)
                
        except Exception as e:
            logger.error(f"❌ [RATE_LIMIT] Failed to get rate limit status: {e}")
            return {"error": str(e)}
    
    async def _get_redis_status(self, identifier: str, limit_type: RateLimitType,
                              config: RateLimitConfig, redis_client: redis.Redis) -> Dict[str, Any]:
        """Get rate limit status from Redis"""
        try:
            current_time = int(time.time())
            
            minute_key = f"rate_limit:{limit_type.value}:{identifier}:minute:{current_time // 60}"
            hour_key = f"rate_limit:{limit_type.value}:{identifier}:hour:{current_time // 3600}"
            day_key = f"rate_limit:{limit_type.value}:{identifier}:day:{current_time // 86400}"
            
            # Get current counts
            minute_count = await redis_client.get(minute_key) or 0
            hour_count = await redis_client.get(hour_key) or 0
            day_count = await redis_client.get(day_key) or 0
            
            return {
                "identifier": identifier,
                "limit_type": limit_type.value,
                "current_counts": {
                    "minute": int(minute_count),
                    "hour": int(hour_count),
                    "day": int(day_count)
                },
                "limits": {
                    "minute": config.requests_per_minute,
                    "hour": config.requests_per_hour,
                    "day": config.requests_per_day
                },
                "remaining": {
                    "minute": max(0, config.requests_per_minute - int(minute_count)),
                    "hour": max(0, config.requests_per_hour - int(hour_count)),
                    "day": max(0, config.requests_per_day - int(day_count))
                }
            }
            
        except Exception as e:
            logger.error(f"❌ [RATE_LIMIT] Failed to get Redis status: {e}")
            raise
    
    async def _get_local_status(self, identifier: str, limit_type: RateLimitType,
                              config: RateLimitConfig) -> Dict[str, Any]:
        """Get rate limit status from local cache"""
        try:
            current_time = time.time()
            key = f"{limit_type.value}:{identifier}"
            
            if key not in self.local_cache:
                return {
                    "identifier": identifier,
                    "limit_type": limit_type.value,
                    "current_counts": {"minute": 0, "hour": 0, "day": 0},
                    "limits": {
                        "minute": config.requests_per_minute,
                        "hour": config.requests_per_hour,
                        "day": config.requests_per_day
                    },
                    "remaining": {
                        "minute": config.requests_per_minute,
                        "hour": config.requests_per_hour,
                        "day": config.requests_per_day
                    }
                }
            
            data = self.local_cache[key]
            
            # Count requests in different windows
            minute_cutoff = current_time - 60
            hour_cutoff = current_time - 3600
            day_cutoff = current_time - 86400
            
            minute_count = len([req for req in data["requests"] if req > minute_cutoff])
            hour_count = len([req for req in data["requests"] if req > hour_cutoff])
            day_count = len([req for req in data["requests"] if req > day_cutoff])
            
            return {
                "identifier": identifier,
                "limit_type": limit_type.value,
                "current_counts": {
                    "minute": minute_count,
                    "hour": hour_count,
                    "day": day_count
                },
                "limits": {
                    "minute": config.requests_per_minute,
                    "hour": config.requests_per_hour,
                    "day": config.requests_per_day
                },
                "remaining": {
                    "minute": max(0, config.requests_per_minute - minute_count),
                    "hour": max(0, config.requests_per_hour - hour_count),
                    "day": max(0, config.requests_per_day - day_count)
                },
                "backend": "local_cache"
            }
            
        except Exception as e:
            logger.error(f"❌ [RATE_LIMIT] Failed to get local status: {e}")
            raise
    
    async def reset_rate_limit(self, identifier: str, limit_type: RateLimitType) -> bool:
        """Reset rate limit for an identifier"""
        try:
            redis_client = await self.get_redis_client()
            
            if redis_client:
                # Delete Redis keys
                pattern = f"rate_limit:{limit_type.value}:{identifier}:*"
                keys = await redis_client.keys(pattern)
                if keys:
                    await redis_client.delete(*keys)
            else:
                # Clear local cache
                key = f"{limit_type.value}:{identifier}"
                if key in self.local_cache:
                    del self.local_cache[key]
            
            logger.info(f"✅ [RATE_LIMIT] Reset rate limit for {identifier}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [RATE_LIMIT] Failed to reset rate limit: {e}")
            return False

# Global rate limiter
rate_limiter = RateLimiter()
