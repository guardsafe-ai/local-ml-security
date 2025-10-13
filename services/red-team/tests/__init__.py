"""
Test Suite for AI Red Team Security Service
Comprehensive unit, integration, and performance tests.
"""

import pytest
import unittest
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from datetime import datetime

# Test configuration
TEST_CONFIG = {
    "test_data_dir": "tests/data",
    "test_models_dir": "tests/models",
    "test_results_dir": "tests/results",
    "timeout": 300,  # 5 minutes
    "max_memory": "8GB",
    "parallel_workers": 4
}

# Test data generators
def generate_test_data(n_samples: int = 100, n_features: int = 10) -> np.ndarray:
    """Generate test data for testing"""
    return np.random.randn(n_samples, n_features)

def generate_test_labels(n_samples: int = 100, n_classes: int = 2) -> np.ndarray:
    """Generate test labels for testing"""
    return np.random.randint(0, n_classes, n_samples)

def generate_test_text(n_samples: int = 100) -> List[str]:
    """Generate test text data for testing"""
    texts = [
        "This is a test sentence for security testing.",
        "Another example text for adversarial attacks.",
        "Sample text data for model evaluation."
    ]
    return [texts[i % len(texts)] for i in range(n_samples)]

# Test utilities
class TestUtils:
    """Utility functions for testing"""
    
    @staticmethod
    def assert_attack_result(result: Dict[str, Any]):
        """Assert attack result structure"""
        assert "success" in result
        assert "confidence" in result
        assert "perturbation" in result
        assert isinstance(result["success"], bool)
        assert 0 <= result["confidence"] <= 1
        assert result["perturbation"] >= 0
    
    @staticmethod
    def assert_compliance_result(result: Dict[str, Any]):
        """Assert compliance result structure"""
        assert "framework" in result
        assert "score" in result
        assert "violations" in result
        assert 0 <= result["score"] <= 1
        assert isinstance(result["violations"], list)
    
    @staticmethod
    def assert_metrics_result(result: Dict[str, Any]):
        """Assert metrics result structure"""
        assert "total_attacks" in result
        assert "successful_attacks" in result
        assert "success_rate" in result
        assert result["total_attacks"] >= 0
        assert result["successful_attacks"] >= 0
        assert 0 <= result["success_rate"] <= 1

# Test fixtures
@pytest.fixture
def sample_data():
    """Sample data for testing"""
    return generate_test_data(100, 10)

@pytest.fixture
def sample_labels():
    """Sample labels for testing"""
    return generate_test_labels(100, 2)

@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return generate_test_text(100)

@pytest.fixture
def sample_model():
    """Sample model for testing"""
    from sklearn.linear_model import LogisticRegression
    X = generate_test_data(100, 10)
    y = generate_test_labels(100, 2)
    model = LogisticRegression()
    model.fit(X, y)
    return model

# Test markers
pytestmark = [
    pytest.mark.unit,
    pytest.mark.integration,
    pytest.mark.performance
]
