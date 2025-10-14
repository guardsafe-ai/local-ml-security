"""
Resilient HTTP Client with Circuit Breaker Integration
Provides robust HTTP communication between services with circuit breaker protection
"""

import httpx
import logging
from typing import Dict, Any, Optional
from utils.circuit_breaker import get_external_api_breaker
from middleware.tracing import inject_trace_context

logger = logging.getLogger(__name__)

# Global HTTP client instance for backward compatibility
_http_client = None

async def get_http_client() -> httpx.AsyncClient:
    """Get or create the global HTTP client (backward compatibility)"""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client

async def close_http_client():
    """Close the global HTTP client (backward compatibility)"""
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None

class ResilientHTTPClient:
    """HTTP client with circuit breaker integration"""
    
    def __init__(self, service_name: str, timeout: float = 30.0):
        self.service_name = service_name
        self.timeout = timeout
        self.breaker = get_external_api_breaker(service_name)
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def post(self, url: str, **kwargs) -> httpx.Response:
        """POST request with circuit breaker and trace context"""
        async def _call():
            # Inject trace context into headers
            headers = kwargs.get('headers', {})
            inject_trace_context(headers)
            kwargs['headers'] = headers
            
            response = await self.client.post(url, **kwargs)
            response.raise_for_status()
            return response
        
        return await self.breaker.call(_call)
    
    async def get(self, url: str, **kwargs) -> httpx.Response:
        """GET request with circuit breaker and trace context"""
        async def _call():
            # Inject trace context into headers
            headers = kwargs.get('headers', {})
            inject_trace_context(headers)
            kwargs['headers'] = headers
            
            response = await self.client.get(url, **kwargs)
            response.raise_for_status()
            return response
        
        return await self.breaker.call(_call)
    
    async def put(self, url: str, **kwargs) -> httpx.Response:
        """PUT request with circuit breaker"""
        async def _call():
            response = await self.client.put(url, **kwargs)
            response.raise_for_status()
            return response
        
        return await self.breaker.call(_call)
    
    async def delete(self, url: str, **kwargs) -> httpx.Response:
        """DELETE request with circuit breaker"""
        async def _call():
            response = await self.client.delete(url, **kwargs)
            response.raise_for_status()
            return response
        
        return await self.breaker.call(_call)
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# Service-specific HTTP clients
business_metrics_client = ResilientHTTPClient("business_metrics", timeout=10.0)
analytics_client = ResilientHTTPClient("analytics", timeout=15.0)
training_client = ResilientHTTPClient("training", timeout=30.0)
data_privacy_client = ResilientHTTPClient("data_privacy", timeout=10.0)
model_cache_client = ResilientHTTPClient("model_cache", timeout=5.0)