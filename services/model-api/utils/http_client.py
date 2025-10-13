"""
Shared HTTP Client Manager
Manages HTTP client connections with proper pooling and cleanup
"""

import asyncio
import logging
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


class HTTPClientManager:
    """Singleton HTTP client manager with connection pooling"""
    
    _instance: Optional['HTTPClientManager'] = None
    _lock = asyncio.Lock()
    
    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None
        self._initialized = False
    
    @classmethod
    async def get_instance(cls) -> 'HTTPClientManager':
        """Get singleton instance with thread-safe initialization"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    async def initialize(self) -> None:
        """Initialize HTTP client with connection pooling"""
        if self._initialized:
            return
            
        try:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(
                    max_keepalive_connections=20,
                    max_connections=100,
                    keepalive_expiry=30.0
                ),
                http2=True,  # Enable HTTP/2 for better performance
                follow_redirects=True
            )
            self._initialized = True
            logger.info("✅ HTTP client manager initialized with connection pooling")
        except Exception as e:
            logger.error(f"❌ Failed to initialize HTTP client manager: {e}")
            raise
    
    async def close(self) -> None:
        """Close HTTP client and cleanup resources"""
        if self.client:
            try:
                await self.client.aclose()
                logger.info("🔒 HTTP client manager closed")
            except Exception as e:
                logger.error(f"❌ Error closing HTTP client manager: {e}")
            finally:
                self.client = None
                self._initialized = False
    
    def get_client(self) -> httpx.AsyncClient:
        """Get the HTTP client instance"""
        if not self._initialized or not self.client:
            raise RuntimeError("HTTP client manager not initialized. Call initialize() first.")
        return self.client
    
    async def health_check(self) -> bool:
        """Check if HTTP client is healthy"""
        if not self.client:
            return False
        
        try:
            # Simple health check - try to make a request
            response = await self.client.get("http://httpbin.org/status/200", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False


# Global instance for easy access
_http_manager: Optional[HTTPClientManager] = None


async def get_http_client() -> httpx.AsyncClient:
    """Get the shared HTTP client instance"""
    global _http_manager
    if _http_manager is None:
        _http_manager = await HTTPClientManager.get_instance()
        await _http_manager.initialize()
    return _http_manager.get_client()


async def close_http_client() -> None:
    """Close the shared HTTP client"""
    global _http_manager
    if _http_manager:
        await _http_manager.close()
        _http_manager = None
