"""
Result Cache
Implements intelligent caching for attack results and model outputs
"""

import asyncio
import logging
import hashlib
import pickle
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import redis
from collections import OrderedDict
import threading
import weakref

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategies"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class CacheLevel(Enum):
    """Cache levels"""
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"
    DISTRIBUTED = "distributed"


@dataclass
class CacheConfig:
    """Cache configuration"""
    max_size: int = 1000
    ttl_seconds: int = 3600
    strategy: CacheStrategy = CacheStrategy.LRU
    levels: List[CacheLevel] = None
    redis_url: str = "redis://localhost:6379/1"
    disk_path: str = "/tmp/red_team_cache"
    compression: bool = True
    serialization: str = "pickle"  # pickle, json
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.levels is None:
            self.levels = [CacheLevel.MEMORY, CacheLevel.REDIS]
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CacheEntry:
    """Cache entry"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ResultCache:
    """
    Result Cache
    Implements intelligent caching for attack results and model outputs
    """
    
    def __init__(self, config: CacheConfig = None):
        """Initialize result cache"""
        self.config = config or CacheConfig()
        
        # Memory cache
        self.memory_cache: OrderedDict = OrderedDict()
        self.memory_lock = threading.RLock()
        
        # Redis cache
        self.redis_client = None
        if CacheLevel.REDIS in self.config.levels:
            try:
                self.redis_client = redis.from_url(self.config.redis_url, decode_responses=False)
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
        
        # Disk cache
        self.disk_path = self.config.disk_path
        if CacheLevel.DISK in self.config.levels:
            import os
            os.makedirs(self.disk_path, exist_ok=True)
        
        # Statistics
        self.stats = CacheStats()
        
        # Cleanup task
        self.cleanup_task = None
        self.cache_active = False
        
        logger.info("✅ Initialized Result Cache")
    
    async def start_cache(self):
        """Start cache operations"""
        try:
            if self.cache_active:
                return
            
            self.cache_active = True
            
            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Result cache started")
            
        except Exception as e:
            logger.error(f"Failed to start cache: {e}")
            raise
    
    async def stop_cache(self):
        """Stop cache operations"""
        try:
            self.cache_active = False
            
            # Stop cleanup task
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Save cache to disk if configured
            if CacheLevel.DISK in self.config.levels:
                await self._save_to_disk()
            
            logger.info("Result cache stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop cache: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            # Try memory cache first
            if CacheLevel.MEMORY in self.config.levels:
                value = await self._get_from_memory(key)
                if value is not None:
                    self.stats.hits += 1
                    self._update_hit_rate()
                    return value
            
            # Try Redis cache
            if CacheLevel.REDIS in self.config.levels and self.redis_client:
                value = await self._get_from_redis(key)
                if value is not None:
                    # Store in memory cache for faster access
                    if CacheLevel.MEMORY in self.config.levels:
                        await self._store_in_memory(key, value)
                    self.stats.hits += 1
                    self._update_hit_rate()
                    return value
            
            # Try disk cache
            if CacheLevel.DISK in self.config.levels:
                value = await self._get_from_disk(key)
                if value is not None:
                    # Store in memory and Redis for faster access
                    if CacheLevel.MEMORY in self.config.levels:
                        await self._store_in_memory(key, value)
                    if CacheLevel.REDIS in self.config.levels and self.redis_client:
                        await self._store_in_redis(key, value)
                    self.stats.hits += 1
                    self._update_hit_rate()
                    return value
            
            self.stats.misses += 1
            self._update_hit_rate()
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cache entry for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            # Calculate TTL
            if ttl is None:
                ttl = self.config.ttl_seconds
            
            # Store in all configured levels
            success = True
            
            if CacheLevel.MEMORY in self.config.levels:
                success &= await self._store_in_memory(key, value, ttl)
            
            if CacheLevel.REDIS in self.config.levels and self.redis_client:
                success &= await self._store_in_redis(key, value, ttl)
            
            if CacheLevel.DISK in self.config.levels:
                success &= await self._store_in_disk(key, value, ttl)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to set cache entry for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            success = True
            
            # Delete from all levels
            if CacheLevel.MEMORY in self.config.levels:
                success &= await self._delete_from_memory(key)
            
            if CacheLevel.REDIS in self.config.levels and self.redis_client:
                success &= await self._delete_from_redis(key)
            
            if CacheLevel.DISK in self.config.levels:
                success &= await self._delete_from_disk(key)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete cache entry for key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            success = True
            
            # Clear all levels
            if CacheLevel.MEMORY in self.config.levels:
                with self.memory_lock:
                    self.memory_cache.clear()
                    self.stats.entry_count = 0
                    self.stats.total_size_bytes = 0
            
            if CacheLevel.REDIS in self.config.levels and self.redis_client:
                try:
                    self.redis_client.flushdb()
                except Exception as e:
                    logger.warning(f"Failed to clear Redis cache: {e}")
                    success = False
            
            if CacheLevel.DISK in self.config.levels:
                import os
                import shutil
                try:
                    if os.path.exists(self.disk_path):
                        shutil.rmtree(self.disk_path)
                        os.makedirs(self.disk_path, exist_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to clear disk cache: {e}")
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    async def get_or_set(self, key: str, factory: Callable, ttl: Optional[int] = None) -> Any:
        """Get value from cache or set it using factory function"""
        try:
            # Try to get from cache
            value = await self.get(key)
            if value is not None:
                return value
            
            # Generate value using factory
            value = await factory() if asyncio.iscoroutinefunction(factory) else factory()
            
            # Store in cache
            await self.set(key, value, ttl)
            
            return value
            
        except Exception as e:
            logger.error(f"Failed to get or set cache entry for key {key}: {e}")
            raise
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        try:
            # Create a hashable representation
            key_data = {
                "args": args,
                "kwargs": sorted(kwargs.items()) if kwargs else {}
            }
            
            # Serialize and hash
            key_str = json.dumps(key_data, sort_keys=True, default=str)
            key_hash = hashlib.sha256(key_str.encode()).hexdigest()
            
            return key_hash
            
        except Exception as e:
            logger.error(f"Failed to generate cache key: {e}")
            return str(hash(str(args) + str(kwargs)))
    
    async def _get_from_memory(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        try:
            with self.memory_lock:
                if key in self.memory_cache:
                    entry = self.memory_cache[key]
                    
                    # Check TTL
                    if entry.ttl and datetime.utcnow() > entry.ttl:
                        del self.memory_cache[key]
                        self.stats.evictions += 1
                        return None
                    
                    # Update access info
                    entry.last_accessed = datetime.utcnow()
                    entry.access_count += 1
                    
                    # Move to end (LRU)
                    self.memory_cache.move_to_end(key)
                    
                    return entry.value
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get from memory cache: {e}")
            return None
    
    async def _store_in_memory(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in memory cache"""
        try:
            with self.memory_lock:
                # Calculate size
                size_bytes = self._calculate_size(value)
                
                # Create entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.utcnow(),
                    last_accessed=datetime.utcnow(),
                    size_bytes=size_bytes,
                    ttl=datetime.utcnow() + timedelta(seconds=ttl) if ttl else None
                )
                
                # Check if we need to evict
                while (len(self.memory_cache) >= self.config.max_size or 
                       self.stats.total_size_bytes + size_bytes > self.config.max_size * 1024 * 1024):  # 1MB per entry
                    await self._evict_from_memory()
                
                # Store entry
                self.memory_cache[key] = entry
                self.stats.entry_count = len(self.memory_cache)
                self.stats.total_size_bytes += size_bytes
                
                return True
            
        except Exception as e:
            logger.error(f"Failed to store in memory cache: {e}")
            return False
    
    async def _delete_from_memory(self, key: str) -> bool:
        """Delete value from memory cache"""
        try:
            with self.memory_lock:
                if key in self.memory_cache:
                    entry = self.memory_cache[key]
                    self.stats.total_size_bytes -= entry.size_bytes
                    del self.memory_cache[key]
                    self.stats.entry_count = len(self.memory_cache)
                    return True
                return False
            
        except Exception as e:
            logger.error(f"Failed to delete from memory cache: {e}")
            return False
    
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        try:
            if not self.redis_client:
                return None
            
            data = self.redis_client.get(f"cache:{key}")
            if data:
                if self.config.serialization == "pickle":
                    return pickle.loads(data)
                else:
                    return json.loads(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get from Redis cache: {e}")
            return None
    
    async def _store_in_redis(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in Redis cache"""
        try:
            if not self.redis_client:
                return False
            
            if self.config.serialization == "pickle":
                data = pickle.dumps(value)
            else:
                data = json.dumps(value, default=str).encode()
            
            if ttl:
                self.redis_client.setex(f"cache:{key}", ttl, data)
            else:
                self.redis_client.set(f"cache:{key}", data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store in Redis cache: {e}")
            return False
    
    async def _delete_from_redis(self, key: str) -> bool:
        """Delete value from Redis cache"""
        try:
            if not self.redis_client:
                return False
            
            result = self.redis_client.delete(f"cache:{key}")
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete from Redis cache: {e}")
            return False
    
    async def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        try:
            import os
            file_path = os.path.join(self.disk_path, f"{key}.cache")
            
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'rb') as f:
                if self.config.serialization == "pickle":
                    return pickle.load(f)
                else:
                    return json.load(f)
            
        except Exception as e:
            logger.error(f"Failed to get from disk cache: {e}")
            return None
    
    async def _store_in_disk(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in disk cache"""
        try:
            import os
            file_path = os.path.join(self.disk_path, f"{key}.cache")
            
            with open(file_path, 'wb') as f:
                if self.config.serialization == "pickle":
                    pickle.dump(value, f)
                else:
                    json.dump(value, f, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store in disk cache: {e}")
            return False
    
    async def _delete_from_disk(self, key: str) -> bool:
        """Delete value from disk cache"""
        try:
            import os
            file_path = os.path.join(self.disk_path, f"{key}.cache")
            
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete from disk cache: {e}")
            return False
    
    async def _evict_from_memory(self):
        """Evict entry from memory cache based on strategy"""
        try:
            with self.memory_lock:
                if not self.memory_cache:
                    return
                
                if self.config.strategy == CacheStrategy.LRU:
                    # Remove least recently used
                    key, entry = self.memory_cache.popitem(last=False)
                elif self.config.strategy == CacheStrategy.LFU:
                    # Remove least frequently used
                    key = min(self.memory_cache.keys(), key=lambda k: self.memory_cache[k].access_count)
                    entry = self.memory_cache.pop(key)
                else:
                    # Default to LRU
                    key, entry = self.memory_cache.popitem(last=False)
                
                self.stats.total_size_bytes -= entry.size_bytes
                self.stats.evictions += 1
                self.stats.entry_count = len(self.memory_cache)
                
        except Exception as e:
            logger.error(f"Failed to evict from memory cache: {e}")
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes"""
        try:
            if self.config.serialization == "pickle":
                return len(pickle.dumps(value))
            else:
                return len(json.dumps(value, default=str).encode())
        except Exception:
            return 1024  # Default size
    
    def _update_hit_rate(self):
        """Update hit rate statistics"""
        total = self.stats.hits + self.stats.misses
        if total > 0:
            self.stats.hit_rate = self.stats.hits / total
    
    async def _cleanup_loop(self):
        """Cleanup expired entries"""
        while self.cache_active:
            try:
                await self._cleanup_expired_entries()
                await asyncio.sleep(60)  # Cleanup every minute
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_expired_entries(self):
        """Clean up expired entries from memory cache"""
        try:
            with self.memory_lock:
                current_time = datetime.utcnow()
                expired_keys = []
                
                for key, entry in self.memory_cache.items():
                    if entry.ttl and current_time > entry.ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    entry = self.memory_cache.pop(key)
                    self.stats.total_size_bytes -= entry.size_bytes
                    self.stats.evictions += 1
                
                self.stats.entry_count = len(self.memory_cache)
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired entries: {e}")
    
    async def _save_to_disk(self):
        """Save memory cache to disk"""
        try:
            with self.memory_lock:
                for key, entry in self.memory_cache.items():
                    await self._store_in_disk(key, entry.value)
                
        except Exception as e:
            logger.error(f"Failed to save to disk: {e}")
    
    async def get_cache_stats(self) -> CacheStats:
        """Get cache statistics"""
        try:
            # Update memory usage
            import psutil
            process = psutil.Process()
            self.stats.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return self.stats
    
    async def export_cache_data(self, format: str = "json") -> str:
        """Export cache data"""
        try:
            if format.lower() == "json":
                data = {
                    "config": asdict(self.config),
                    "stats": asdict(await self.get_cache_stats()),
                    "memory_entries": len(self.memory_cache),
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Cache data export failed: {e}")
            return ""
