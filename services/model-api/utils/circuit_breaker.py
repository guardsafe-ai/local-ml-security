"""
Model API Service - Circuit Breaker Implementation
Service-specific circuit breaker for graceful degradation
"""

import asyncio
import time
import logging
from typing import Callable, Any, Optional, Dict
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import functools

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is back

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5          # Number of failures before opening
    recovery_timeout: int = 60          # Seconds to wait before half-open
    success_threshold: int = 3          # Successes needed to close from half-open
    timeout: float = 30.0               # Request timeout in seconds
    expected_exception: type = Exception  # Exception type to count as failure
    name: str = "model_api"             # Circuit breaker name for logging

class CircuitBreaker:
    """Circuit breaker implementation with async support"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self._lock = asyncio.Lock()
        
        logger.info(f"🔌 [MODEL_API_CIRCUIT] Initialized circuit breaker '{config.name}'")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            # Check if circuit should be opened
            if self.state == CircuitState.CLOSED and self._should_open():
                self._open_circuit()
            
            # Check if circuit should be half-opened
            elif self.state == CircuitState.OPEN and self._should_half_open():
                self._half_open_circuit()
            
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerOpenException(
                    f"Model API circuit breaker '{self.config.name}' is OPEN. "
                    f"Last failure: {self.last_failure_time}"
                )
        
        # Execute the function with timeout
        try:
            result = await asyncio.wait_for(
                self._execute_function(func, *args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Record success
            await self._record_success()
            return result
            
        except asyncio.TimeoutError:
            await self._record_failure(CircuitBreakerTimeoutException("Request timed out"))
            raise CircuitBreakerTimeoutException(
                f"Model API circuit breaker '{self.config.name}' request timed out after {self.config.timeout}s"
            )
        except Exception as e:
            # Check if this exception should be counted as failure
            if isinstance(e, self.config.expected_exception):
                await self._record_failure(e)
            else:
                # Unexpected exception, don't count as failure
                logger.warning(f"⚠️ [MODEL_API_CIRCUIT] Unexpected exception in '{self.config.name}': {e}")
            
            raise
    
    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Execute the function, handling both sync and async functions"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)
    
    def _should_open(self) -> bool:
        """Check if circuit should be opened"""
        return self.failure_count >= self.config.failure_threshold
    
    def _should_half_open(self) -> bool:
        """Check if circuit should be half-opened"""
        if not self.last_failure_time:
            return False
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.recovery_timeout
    
    def _open_circuit(self):
        """Open the circuit"""
        self.state = CircuitState.OPEN
        self.last_failure_time = time.time()
        logger.warning(f"🔴 [MODEL_API_CIRCUIT] Circuit breaker '{self.config.name}' OPENED after {self.failure_count} failures")
    
    def _half_open_circuit(self):
        """Half-open the circuit"""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        logger.info(f"🟡 [MODEL_API_CIRCUIT] Circuit breaker '{self.config.name}' HALF-OPENED for testing")
    
    async def _record_success(self):
        """Record a successful call"""
        async with self._lock:
            self.success_count += 1
            self.last_success_time = time.time()
            
            # Reset failure count on success
            if self.failure_count > 0:
                self.failure_count = 0
            
            # Close circuit if we're in half-open and have enough successes
            if self.state == CircuitState.HALF_OPEN and self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                logger.info(f"🟢 [MODEL_API_CIRCUIT] Circuit breaker '{self.config.name}' CLOSED after {self.success_count} successes")
    
    async def _record_failure(self, exception: Exception):
        """Record a failed call"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            logger.debug(f"❌ [MODEL_API_CIRCUIT] Failure #{self.failure_count} in '{self.config.name}': {exception}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout
            }
        }
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        logger.info(f"🔄 [MODEL_API_CIRCUIT] Circuit breaker '{self.config.name}' RESET")

class CircuitBreakerOpenException(Exception):
    """Raised when circuit breaker is open"""
    pass

class CircuitBreakerTimeoutException(Exception):
    """Raised when circuit breaker request times out"""
    pass

# Pre-configured circuit breakers for Model API service
def get_redis_breaker() -> CircuitBreaker:
    """Get circuit breaker for Redis operations"""
    config = CircuitBreakerConfig(
        name="model_api_redis",
        failure_threshold=5,
        recovery_timeout=30,
        success_threshold=2,
        timeout=5.0,
        expected_exception=Exception
    )
    return CircuitBreaker(config)

def get_mlflow_breaker() -> CircuitBreaker:
    """Get circuit breaker for MLflow operations"""
    config = CircuitBreakerConfig(
        name="model_api_mlflow",
        failure_threshold=3,
        recovery_timeout=60,
        success_threshold=2,
        timeout=30.0,
        expected_exception=Exception
    )
    return CircuitBreaker(config)

def get_external_api_breaker(service_name: str) -> CircuitBreaker:
    """Get circuit breaker for external API calls"""
    config = CircuitBreakerConfig(
        name=f"model_api_external_{service_name}",
        failure_threshold=5,
        recovery_timeout=120,
        success_threshold=3,
        timeout=30.0,
        expected_exception=Exception
    )
    return CircuitBreaker(config)
