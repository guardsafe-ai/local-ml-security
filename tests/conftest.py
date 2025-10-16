"""
Comprehensive Testing Suite for ML Security Platform
Pytest configuration and fixtures
"""

import pytest
import asyncio
import httpx
import json
from typing import Dict, Any, List
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_CONFIG = {
    "base_url": "http://localhost:8007",
    "services": {
        "model_api": "http://localhost:8000",
        "training": "http://localhost:8002", 
        "model_cache": "http://localhost:8003",
        "business_metrics": "http://localhost:8004",
        "analytics": "http://localhost:8006",
        "data_privacy": "http://localhost:8008",
        "tracing": "http://localhost:8009",
        "mlflow": "http://localhost:5000"
    },
    "timeout": 30.0,
    "retry_attempts": 3,
    "retry_delay": 1.0
}

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def http_client():
    """Create HTTP client for testing."""
    async with httpx.AsyncClient(timeout=TEST_CONFIG["timeout"]) as client:
        yield client

@pytest.fixture(scope="session")
async def backend_client():
    """Create backend gateway client for testing."""
    async with httpx.AsyncClient(
        base_url=TEST_CONFIG["base_url"],
        timeout=TEST_CONFIG["timeout"]
    ) as client:
        yield client

@pytest.fixture(scope="session")
async def service_clients():
    """Create clients for all services."""
    clients = {}
    for service_name, base_url in TEST_CONFIG["services"].items():
        clients[service_name] = httpx.AsyncClient(
            base_url=base_url,
            timeout=TEST_CONFIG["timeout"]
        )
    yield clients
    # Cleanup
    for client in clients.values():
        await client.aclose()

@pytest.fixture
def test_data():
    """Test data for various test scenarios."""
    return {
        "model_name": "distilbert",
        "model_type": "pytorch",
        "text_sample": "This is a test message for security analysis",
        "training_config": {
            "batch_size": 4,
            "learning_rate": 0.0001,
            "num_epochs": 2,
            "max_length": 128
        },
        "metric_data": {
            "metric_name": "test_metric",
            "value": 85.5,
            "unit": "percentage",
            "category": "performance"
        }
    }

class TestHelper:
    """Helper class for common test operations."""
    
    @staticmethod
    async def make_request(client: httpx.AsyncClient, method: str, url: str, 
                          data: Dict = None, params: Dict = None, 
                          retries: int = None) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        retries = retries or TEST_CONFIG["retry_attempts"]
        last_exception = None
        
        for attempt in range(retries):
            try:
                if method.upper() == "GET":
                    response = await client.get(url, params=params)
                elif method.upper() == "POST":
                    response = await client.post(url, json=data, params=params)
                elif method.upper() == "PUT":
                    response = await client.put(url, json=data, params=params)
                elif method.upper() == "DELETE":
                    response = await client.delete(url, params=params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response.json()
                
            except Exception as e:
                last_exception = e
                if attempt < retries - 1:
                    await asyncio.sleep(TEST_CONFIG["retry_delay"])
                    continue
                else:
                    raise last_exception
    
    @staticmethod
    def assert_response_structure(response: Dict[str, Any], required_fields: List[str]):
        """Assert that response has required fields."""
        for field in required_fields:
            assert field in response, f"Missing required field: {field}"
    
    @staticmethod
    def assert_health_response(response: Dict[str, Any]):
        """Assert health response structure."""
        required_fields = ["status", "service", "timestamp"]
        TestHelper.assert_response_structure(response, required_fields)
        assert response["status"] in ["healthy", "unhealthy", "degraded"]
    
    @staticmethod
    def assert_error_response(response: Dict[str, Any]):
        """Assert error response structure."""
        required_fields = ["error", "status_code", "message", "timestamp"]
        TestHelper.assert_response_structure(response, required_fields)
        assert response["error"] is True
        assert isinstance(response["status_code"], int)

@pytest.fixture
def test_helper():
    """Test helper instance."""
    return TestHelper()
