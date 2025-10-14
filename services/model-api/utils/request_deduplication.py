"""
Request Deduplication System
Prevents duplicate processing of identical requests within a time window
"""

import hashlib
import time
import logging
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import json

logger = logging.getLogger(__name__)

@dataclass
class DeduplicationEntry:
    """Represents a deduplication cache entry"""
    request_hash: str
    response: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    request_count: int = 1

class RequestDeduplicationManager:
    """Manages request deduplication to prevent duplicate processing"""
    
    def __init__(self, default_ttl_seconds: int = 300, max_cache_size: int = 10000):
        self.default_ttl_seconds = default_ttl_seconds
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, DeduplicationEntry] = {}
        self.active_requests: Set[str] = set()  # Track currently processing requests
        
    def _generate_request_hash(self, request_data: Dict[str, Any], 
                             model_name: str, version: str = None) -> str:
        """Generate a hash for request deduplication"""
        try:
            # Create a normalized request representation
            normalized_data = {
                "model_name": model_name,
                "version": version,
                "text": request_data.get("text", "").strip().lower(),
                "max_length": request_data.get("max_length", 512),
                "temperature": request_data.get("temperature", 0.7),
                "top_p": request_data.get("top_p", 0.9),
                "do_sample": request_data.get("do_sample", True)
            }
            
            # Convert to JSON string and hash
            json_str = json.dumps(normalized_data, sort_keys=True, separators=(',', ':'))
            return hashlib.sha256(json_str.encode()).hexdigest()[:16]
            
        except Exception as e:
            logger.error(f"❌ [DEDUPLICATION] Failed to generate request hash: {e}")
            return hashlib.sha256(str(request_data).encode()).hexdigest()[:16]
    
    def _is_request_processing(self, request_hash: str) -> bool:
        """Check if request is currently being processed"""
        return request_hash in self.active_requests
    
    def _mark_request_processing(self, request_hash: str):
        """Mark request as being processed"""
        self.active_requests.add(request_hash)
    
    def _unmark_request_processing(self, request_hash: str):
        """Unmark request as being processed"""
        self.active_requests.discard(request_hash)
    
    def _cleanup_expired_entries(self):
        """Remove expired entries from cache"""
        try:
            current_time = datetime.now(timezone.utc)
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.expires_at < current_time
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                logger.debug(f"🧹 [DEDUPLICATION] Cleaned up {len(expired_keys)} expired entries")
                
        except Exception as e:
            logger.error(f"❌ [DEDUPLICATION] Failed to cleanup expired entries: {e}")
    
    def _trim_cache_if_needed(self):
        """Trim cache if it exceeds max size"""
        try:
            if len(self.cache) > self.max_cache_size:
                # Remove oldest entries (by created_at)
                sorted_entries = sorted(
                    self.cache.items(),
                    key=lambda x: x[1].created_at
                )
                
                entries_to_remove = len(self.cache) - self.max_cache_size
                for key, _ in sorted_entries[:entries_to_remove]:
                    del self.cache[key]
                
                logger.info(f"🧹 [DEDUPLICATION] Trimmed cache, removed {entries_to_remove} entries")
                
        except Exception as e:
            logger.error(f"❌ [DEDUPLICATION] Failed to trim cache: {e}")
    
    def check_duplicate(self, request_data: Dict[str, Any], model_name: str, 
                       version: str = None, ttl_seconds: int = None) -> Optional[Dict[str, Any]]:
        """Check if request is a duplicate and return cached response if so"""
        try:
            # Cleanup expired entries first
            self._cleanup_expired_entries()
            
            # Generate request hash
            request_hash = self._generate_request_hash(request_data, model_name, version)
            
            # Check if request is currently being processed
            if self._is_request_processing(request_hash):
                logger.info(f"⏳ [DEDUPLICATION] Request already being processed: {request_hash}")
                return None  # Let the current request complete
            
            # Check if we have a cached response
            if request_hash in self.cache:
                entry = self.cache[request_hash]
                current_time = datetime.now(timezone.utc)
                
                if entry.expires_at > current_time:
                    # Update request count and return cached response
                    entry.request_count += 1
                    logger.info(f"✅ [DEDUPLICATION] Returning cached response for {request_hash} (count: {entry.request_count})")
                    return entry.response
                else:
                    # Entry expired, remove it
                    del self.cache[request_hash]
            
            return None
            
        except Exception as e:
            logger.error(f"❌ [DEDUPLICATION] Failed to check duplicate: {e}")
            return None
    
    def store_response(self, request_data: Dict[str, Any], model_name: str, 
                      response: Dict[str, Any], version: str = None, 
                      ttl_seconds: int = None) -> str:
        """Store response for future deduplication"""
        try:
            # Generate request hash
            request_hash = self._generate_request_hash(request_data, model_name, version)
            
            # Unmark as processing
            self._unmark_request_processing(request_hash)
            
            # Calculate TTL
            ttl = ttl_seconds or self.default_ttl_seconds
            current_time = datetime.now(timezone.utc)
            expires_at = current_time + timedelta(seconds=ttl)
            
            # Store in cache
            entry = DeduplicationEntry(
                request_hash=request_hash,
                response=response,
                created_at=current_time,
                expires_at=expires_at,
                request_count=1
            )
            
            self.cache[request_hash] = entry
            
            # Trim cache if needed
            self._trim_cache_if_needed()
            
            logger.info(f"💾 [DEDUPLICATION] Stored response for {request_hash} (TTL: {ttl}s)")
            return request_hash
            
        except Exception as e:
            logger.error(f"❌ [DEDUPLICATION] Failed to store response: {e}")
            return ""
    
    def mark_request_processing(self, request_data: Dict[str, Any], 
                               model_name: str, version: str = None) -> str:
        """Mark request as being processed"""
        try:
            request_hash = self._generate_request_hash(request_data, model_name, version)
            self._mark_request_processing(request_hash)
            logger.debug(f"🔄 [DEDUPLICATION] Marked request as processing: {request_hash}")
            return request_hash
        except Exception as e:
            logger.error(f"❌ [DEDUPLICATION] Failed to mark request processing: {e}")
            return ""
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get deduplication cache statistics"""
        try:
            current_time = datetime.now(timezone.utc)
            active_entries = len([e for e in self.cache.values() if e.expires_at > current_time])
            expired_entries = len(self.cache) - active_entries
            processing_requests = len(self.active_requests)
            
            total_requests = sum(entry.request_count for entry in self.cache.values())
            
            return {
                "cache_size": len(self.cache),
                "active_entries": active_entries,
                "expired_entries": expired_entries,
                "processing_requests": processing_requests,
                "total_requests_served": total_requests,
                "deduplication_rate": (total_requests - len(self.cache)) / max(total_requests, 1) * 100,
                "max_cache_size": self.max_cache_size,
                "default_ttl_seconds": self.default_ttl_seconds
            }
        except Exception as e:
            logger.error(f"❌ [DEDUPLICATION] Failed to get cache stats: {e}")
            return {}
    
    def clear_cache(self):
        """Clear all deduplication cache"""
        try:
            self.cache.clear()
            self.active_requests.clear()
            logger.info("🧹 [DEDUPLICATION] Cache cleared")
        except Exception as e:
            logger.error(f"❌ [DEDUPLICATION] Failed to clear cache: {e}")

# Global instance
_deduplication_manager = None

def get_deduplication_manager() -> RequestDeduplicationManager:
    """Get global deduplication manager instance"""
    global _deduplication_manager
    if _deduplication_manager is None:
        _deduplication_manager = RequestDeduplicationManager()
    return _deduplication_manager

def check_request_duplicate(request_data: Dict[str, Any], model_name: str, 
                           version: str = None, ttl_seconds: int = None) -> Optional[Dict[str, Any]]:
    """Check if request is duplicate (convenience function)"""
    manager = get_deduplication_manager()
    return manager.check_duplicate(request_data, model_name, version, ttl_seconds)

def store_request_response(request_data: Dict[str, Any], model_name: str, 
                          response: Dict[str, Any], version: str = None, 
                          ttl_seconds: int = None) -> str:
    """Store request response (convenience function)"""
    manager = get_deduplication_manager()
    return manager.store_response(request_data, model_name, response, version, ttl_seconds)

def mark_request_processing(request_data: Dict[str, Any], model_name: str, 
                           version: str = None) -> str:
    """Mark request as processing (convenience function)"""
    manager = get_deduplication_manager()
    return manager.mark_request_processing(request_data, model_name, version)