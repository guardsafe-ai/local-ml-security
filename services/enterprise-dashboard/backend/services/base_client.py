"""
Base Service Client
Common functionality for all service clients
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import httpx
import redis
from fastapi import HTTPException
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker implementation for service calls"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def can_execute(self) -> bool:
        """Check if the circuit breaker allows execution"""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN


class BaseServiceClient:
    """Base class for all service clients with common functionality"""
    
    def __init__(self, service_name: str, base_url: str, redis_client: redis.Redis):
        self.service_name = service_name
        self.base_url = base_url
        self.redis_client = redis_client
        self.timeout = 30.0
        self.long_timeout = 300.0  # For file uploads and long operations
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        self.max_retries = 3
        self.retry_delay = 1.0  # Initial delay in seconds
        
        # Connection pooling for better performance
        self._http_client = None
        self._long_timeout_client = None
        self._client_lock = asyncio.Lock()
    
    async def _get_http_client(self, timeout: float = None) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling"""
        timeout = timeout or self.timeout
        
        async with self._client_lock:
            if timeout == self.long_timeout:
                if self._long_timeout_client is None:
                    self._long_timeout_client = httpx.AsyncClient(
                        base_url=self.base_url,
                        timeout=timeout,
                        follow_redirects=True,
                        limits=httpx.Limits(
                            max_keepalive_connections=20,
                            max_connections=100,
                            keepalive_expiry=30.0
                        )
                    )
                return self._long_timeout_client
            else:
                if self._http_client is None:
                    self._http_client = httpx.AsyncClient(
                        base_url=self.base_url,
                        timeout=timeout,
                        follow_redirects=True,
                        limits=httpx.Limits(
                            max_keepalive_connections=20,
                            max_connections=100,
                            keepalive_expiry=30.0
                        )
                    )
                return self._http_client
    
    async def close_connections(self):
        """Close HTTP connections"""
        async with self._client_lock:
            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None
            if self._long_timeout_client:
                await self._long_timeout_client.aclose()
                self._long_timeout_client = None
        
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        use_cache: bool = False,
        cache_ttl: int = 300,
        retry_on_failure: bool = True
    ) -> Dict[str, Any]:
        """Make HTTP request with error handling, caching, retry logic, and circuit breaker"""
        url = f"{self.base_url}{endpoint}"
        timeout = timeout or self.timeout
        
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker OPEN for {self.service_name}:{endpoint}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Service temporarily unavailable",
                    "service": self.service_name,
                    "endpoint": endpoint,
                    "message": f"Circuit breaker is OPEN for {self.service_name} service",
                    "circuit_state": self.circuit_breaker.state.value
                }
            )
        
        # Check cache first if enabled
        if use_cache and method.upper() == "GET":
            cache_key = f"{self.service_name}:{endpoint}:{str(params or {})}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                logger.info(f"📊 Returning cached data for {self.service_name}:{endpoint}")
                return cached_data
        
        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                # Use connection pool for better performance
                client = await self._get_http_client(timeout)
                
                if method.upper() == "GET":
                    response = await client.get(endpoint, params=params)
                elif method.upper() == "POST":
                    if files:
                        response = await client.post(endpoint, data=data, files=files)
                    else:
                        response = await client.post(endpoint, json=data, params=params)
                elif method.upper() == "PUT":
                    response = await client.put(endpoint, json=data, params=params)
                elif method.upper() == "DELETE":
                    response = await client.delete(endpoint, params=params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                
                # Handle both JSON and plain text responses
                try:
                    result = response.json()
                except (ValueError, TypeError):
                    # Handle plain text responses (like MLflow's "OK")
                    result = {
                        "status": "ok",
                        "message": response.text,
                        "content_type": response.headers.get("content-type", "text/plain")
                    }
                
                # Cache successful GET requests
                if use_cache and method.upper() == "GET" and response.status_code == 200:
                    self._cache_data(cache_key, result, cache_ttl)
                
                # Success - reset circuit breaker
                self.circuit_breaker.on_success()
                return result
                    
            except Exception as e:
                last_exception = e
                
                # Check if we should retry
                if not retry_on_failure or attempt >= self.max_retries:
                    break
                
                # Check if error is retryable
                if not self._is_retryable_error(e):
                    break
                
                # Calculate delay with exponential backoff
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(f"Request failed for {self.service_name}:{endpoint} (attempt {attempt + 1}/{self.max_retries + 1}), retrying in {delay}s: {str(e)}")
                await asyncio.sleep(delay)
        
        # All retries failed - update circuit breaker and raise exception
        self.circuit_breaker.on_failure()
        self._handle_request_error(last_exception, url, endpoint, timeout)
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable"""
        if isinstance(error, httpx.TimeoutException):
            return True
        elif isinstance(error, httpx.ConnectError):
            return True
        elif isinstance(error, httpx.RequestError):
            return True
        elif isinstance(error, httpx.HTTPStatusError):
            # Retry on 5xx errors, not on 4xx errors
            return 500 <= error.response.status_code < 600
        return False
    
    def _handle_request_error(self, error: Exception, url: str, endpoint: str, timeout: float):
        """Handle request errors with detailed error information"""
        if isinstance(error, httpx.TimeoutException):
            logger.error(f"Timeout error for {self.service_name}:{endpoint} - {str(error)}")
            raise HTTPException(
                status_code=504, 
                detail={
                    "error": "Service timeout",
                    "service": self.service_name,
                    "endpoint": endpoint,
                    "timeout": timeout,
                    "message": f"Service {self.service_name} did not respond within {timeout} seconds"
                }
            )
        elif isinstance(error, httpx.HTTPStatusError):
            error_detail = {
                "error": "HTTP error",
                "service": self.service_name,
                "endpoint": endpoint,
                "status_code": error.response.status_code,
                "message": f"Service {self.service_name} returned {error.response.status_code}"
            }
            
            # Try to get error details from response
            try:
                error_response = error.response.json()
                error_detail["response"] = error_response
            except:
                error_detail["response"] = error.response.text
            
            logger.error(f"HTTP error for {self.service_name}:{endpoint} - {error.response.status_code}: {error_detail}")
            raise HTTPException(status_code=error.response.status_code, detail=error_detail)
        elif isinstance(error, httpx.ConnectError):
            logger.error(f"Connection error for {self.service_name}:{endpoint} - {str(error)}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Service unavailable",
                    "service": self.service_name,
                    "endpoint": endpoint,
                    "message": f"Could not connect to {self.service_name} service",
                    "url": url
                }
            )
        elif isinstance(error, httpx.RequestError):
            logger.error(f"Request error for {self.service_name}:{endpoint} - {str(error)}")
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "Bad gateway",
                    "service": self.service_name,
                    "endpoint": endpoint,
                    "message": f"Request to {self.service_name} failed",
                    "error_type": type(error).__name__
                }
            )
        else:
            logger.error(f"Unexpected error for {self.service_name}:{endpoint} - {str(error)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Internal server error",
                    "service": self.service_name,
                    "endpoint": endpoint,
                    "message": f"An unexpected error occurred with {self.service_name} service",
                    "error_type": type(error).__name__
                }
            )
    
    def _cache_data(self, key: str, data: Any, ttl: int = 300):
        """Cache data in Redis"""
        try:
            import json
            self.redis_client.setex(key, ttl, json.dumps(data, default=str))
        except Exception as e:
            logger.error(f"Failed to cache data: {e}")
    
    def _get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data from Redis"""
        try:
            import json
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Failed to get cached data: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            # Try multiple health endpoints
            health_endpoints = ["/health", "/health/", "/", "/status"]
            
            for endpoint in health_endpoints:
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(f"{self.base_url}{endpoint}")
                        if response.status_code in [200, 302]:
                            try:
                                data = response.json()
                            except:
                                data = {"response": response.text.strip()}
                            
                            return {
                                "name": self.service_name,
                                "status": "healthy",
                                "response_time": 0.0,  # Could add timing
                                "last_check": datetime.now(),
                                "details": data
                            }
                except:
                    continue
            
            return {
                "name": self.service_name,
                "status": "unhealthy",
                "response_time": 0.0,
                "last_check": datetime.now(),
                "details": {"error": "All health check endpoints failed"}
            }
        except Exception as e:
            logger.error(f"Health check failed for {self.service_name}: {e}")
            return {
                "name": self.service_name,
                "status": "unhealthy",
                "response_time": 0.0,
                "last_check": datetime.now(),
                "details": {"error": str(e)}
            }
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for monitoring"""
        return {
            "service": self.service_name,
            "state": self.circuit_breaker.state.value,
            "failure_count": self.circuit_breaker.failure_count,
            "last_failure_time": self.circuit_breaker.last_failure_time,
            "can_execute": self.circuit_breaker.can_execute()
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this service client"""
        try:
            return {
                "service": self.service_name,
                "circuit_breaker": self.get_circuit_breaker_status(),
                "connection_pool": {
                    "has_http_client": self._http_client is not None,
                    "has_long_timeout_client": self._long_timeout_client is not None
                },
                "timeout_settings": {
                    "default_timeout": self.timeout,
                    "long_timeout": self.long_timeout
                },
                "retry_settings": {
                    "max_retries": self.max_retries,
                    "retry_delay": self.retry_delay
                },
                "cache": {
                    "total_cached_items": 0,
                    "estimated_memory_usage_bytes": 0,
                    "cache_pattern": f"{self.service_name}:*"
                }
            }
        except Exception as e:
            logger.error(f"Failed to get performance metrics for {self.service_name}: {e}")
            return {
                "service": self.service_name,
                "error": str(e)
            }
    
    def _get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from Redis"""
        try:
            # Get cache keys for this service
            pattern = f"{self.service_name}:*"
            keys = self.redis_client.keys(pattern)
            
            total_keys = len(keys)
            total_memory = 0
            
            for key in keys:
                try:
                    memory_usage = self.redis_client.memory_usage(key)
                    total_memory += memory_usage
                except:
                    pass
            
            return {
                "total_cached_items": total_keys,
                "estimated_memory_usage_bytes": total_memory,
                "cache_pattern": pattern
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats for {self.service_name}: {e}")
            return {
                "total_cached_items": 0,
                "estimated_memory_usage_bytes": 0,
                "error": str(e)
            }
