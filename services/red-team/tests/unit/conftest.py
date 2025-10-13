"""
Pytest configuration and fixtures for unit tests
"""

import pytest
import numpy as np
from unittest.mock import Mock

@pytest.fixture
def sample_model():
    """Sample model for testing"""
    model = Mock()
    model.predict.return_value = np.random.rand(10)
    model.predict_proba.return_value = np.random.rand(10, 2)
    model.score.return_value = 0.8
    return model

@pytest.fixture
def sample_data():
    """Sample data for testing"""
    return np.random.rand(100, 10)

@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return "This is a sample text for testing purposes."

@pytest.fixture
def sample_incident():
    """Sample incident for testing"""
    return {
        "id": "INC001",
        "timestamp": "2023-01-01T00:00:00Z",
        "type": "adversarial_attack",
        "severity": "high",
        "description": "A test incident",
        "model_id": "model_123",
        "attack_type": "prompt_injection",
        "success": True,
        "confidence": 0.8,
        "perturbation": 0.1,
        "metadata": {
            "input_text": "Test input",
            "output_text": "Test output",
            "model_version": "1.0"
        }
    }

@pytest.fixture
def sample_attack_data():
    """Sample attack data for testing"""
    return {
        "type": "adversarial",
        "text": "This is a test attack",
        "label": "adversarial",
        "confidence": 0.8,
        "perturbation": 0.1
    }

@pytest.fixture
def sample_compliance_data():
    """Sample compliance data for testing"""
    return {
        "soc2": {"score": 0.9, "controls": []},
        "iso27001": {"score": 0.8, "controls": []},
        "owasp_llm": {"score": 0.7, "controls": []},
        "nist": {"score": 0.85, "controls": []}
    }

@pytest.fixture
def sample_risk_data():
    """Sample risk data for testing"""
    return {
        "adversarial_risk": 0.8,
        "privacy_risk": 0.7,
        "bias_risk": 0.6,
        "robustness_risk": 0.9
    }

@pytest.fixture
def sample_roi_data():
    """Sample ROI data for testing"""
    return {
        "cost_savings": 100000,
        "risk_reduction": 0.7,
        "implementation_cost": 50000
    }

@pytest.fixture
def sample_visualization_data():
    """Sample visualization data for testing"""
    return {
        "type": "attack_tree",
        "data": {
            "root": "Adversarial Attack",
            "children": [
                {
                    "name": "Gradient-based",
                    "children": [
                        {"name": "FGSM", "success_rate": 0.8},
                        {"name": "PGD", "success_rate": 0.9}
                    ]
                },
                {
                    "name": "Word-level",
                    "children": [
                        {"name": "TextFooler", "success_rate": 0.7},
                        {"name": "BERT-Attack", "success_rate": 0.6}
                    ]
                }
            ]
        }
    }

@pytest.fixture
def sample_embedding_data():
    """Sample embedding data for testing"""
    return {
        "embeddings": np.random.rand(100, 128),
        "labels": np.random.randint(0, 5, 100)
    }

@pytest.fixture
def sample_heatmap_data():
    """Sample heatmap data for testing"""
    return {
        "data": np.random.rand(10, 10),
        "labels": ["Label1", "Label2", "Label3", "Label4", "Label5", "Label6", "Label7", "Label8", "Label9", "Label10"]
    }

@pytest.fixture
def sample_dashboard_data():
    """Sample dashboard data for testing"""
    return {
        "metrics": {"cpu_usage": 0.8, "memory_usage": 0.7},
        "alerts": [{"type": "high_cpu", "severity": "warning"}],
        "visualizations": [{"type": "attack_tree", "data": {}}]
    }

@pytest.fixture
def sample_mlflow_data():
    """Sample MLflow data for testing"""
    return {
        "model_name": "test_model",
        "model_version": "1.0",
        "model_data": {"model": Mock(), "metadata": {"version": "1.0", "stage": "Production"}}
    }

@pytest.fixture
def sample_siem_data():
    """Sample SIEM data for testing"""
    return {
        "timestamp": "2023-01-01T00:00:00Z",
        "event_type": "adversarial_attack",
        "severity": "high",
        "message": "Adversarial attack detected"
    }

@pytest.fixture
def sample_threat_intelligence_data():
    """Sample threat intelligence data for testing"""
    return {
        "mitre_atlas": [{"id": "T1234", "name": "Test Attack Pattern", "tactics": ["TA0001"]}],
        "cves": [{"id": "CVE-2023-1234", "description": "Test CVE", "severity": "HIGH"}],
        "jailbreaks": [{"id": "JB001", "name": "Test Jailbreak", "success_rate": 0.8}]
    }

@pytest.fixture
def sample_pattern_data():
    """Sample pattern data for testing"""
    return {
        "pattern_type": "text",
        "num_patterns": 50,
        "complexity": "high",
        "patterns": [
            {"pattern": "Pattern 1", "type": "text", "complexity": "high"},
            {"pattern": "Pattern 2", "type": "text", "complexity": "high"}
        ]
    }

@pytest.fixture
def sample_evolutionary_data():
    """Sample evolutionary data for testing"""
    return {
        "population_size": 100,
        "generations": 50,
        "mutation_rate": 0.1,
        "optimized_data": [
            {"text": "Optimized attack 1", "label": "adversarial", "fitness": 0.9},
            {"text": "Optimized attack 2", "label": "adversarial", "fitness": 0.95}
        ]
    }

@pytest.fixture
def sample_attack_data_generator_data():
    """Sample attack data generator data for testing"""
    return {
        "attack_type": "adversarial",
        "num_samples": 100,
        "complexity": "medium",
        "generated_data": [
            {"text": "Sample attack 1", "label": "adversarial", "confidence": 0.8},
            {"text": "Sample attack 2", "label": "adversarial", "confidence": 0.9}
        ]
    }

@pytest.fixture
def sample_benchmark_data():
    """Sample benchmark data for testing"""
    return {
        "harmbench": {"overall_score": 0.8, "category_scores": {"harmful_content": 0.9, "bias": 0.7}},
        "strongreject": {"overall_score": 0.85, "rejection_rate": 0.9},
        "safetybench": {"overall_score": 0.9, "safety_scores": {"safety_concern": 0.95, "risk": 0.85}}
    }

@pytest.fixture
def sample_certification_data():
    """Sample certification data for testing"""
    return {
        "randomized_smoothing": {"certified": True, "radius": 0.1, "confidence": 0.95, "robustness_score": 0.8},
        "interval_bound_propagation": {"certified": True, "bounds": {"lower": np.random.rand(10), "upper": np.random.rand(10)}, "robustness_score": 0.9}
    }

@pytest.fixture
def sample_privacy_attack_data():
    """Sample privacy attack data for testing"""
    return {
        "membership_inference": {"membership_scores": np.random.rand(100), "attack_accuracy": 0.8, "attack_auc": 0.85},
        "model_inversion": {"reconstructed_data": np.random.rand(100, 10), "reconstruction_quality": 0.8, "privacy_risk": 0.7},
        "data_extraction": {"extracted_data": ["data1", "data2", "data3"], "extraction_success_rate": 0.6, "privacy_risk": 0.8}
    }

@pytest.fixture
def sample_distributed_data():
    """Sample distributed data for testing"""
    return {
        "task_queue": {"task_id": "task_123", "status": "SUCCESS", "result": {"success": True, "confidence": 0.8}},
        "worker_manager": {"worker_id": "worker_123", "status": "active", "health": {"cpu_usage": 0.5, "memory_usage": 0.6}},
        "load_balancer": {"selected_worker": "worker_123", "load": 0.5}
    }

@pytest.fixture
def sample_optimization_data():
    """Sample optimization data for testing"""
    return {
        "result_cache": {"key": "test_key", "value": {"result": "test_result", "confidence": 0.8}, "hit": True},
        "batch_optimizer": {"batch_size": 32, "optimized_batch": [{"result": "result_1"}, {"result": "result_2"}]},
        "gpu_manager": {"memory_info": {"total_memory": 8192, "used_memory": 4096, "free_memory": 4096, "memory_usage": 0.5}}
    }

@pytest.fixture
def sample_monitoring_data():
    """Sample monitoring data for testing"""
    return {
        "prometheus_metrics": {"metric_name": "test_metric", "value": 1.0, "labels": {"label1": "value1"}},
        "websocket_streaming": {"connection_id": "conn_123", "data": {"type": "metric", "value": 1.0}},
        "alert_manager": {"alert_id": "alert_123", "type": "high_cpu_usage", "severity": "warning", "message": "CPU usage is above 80%"}
    }

@pytest.fixture
def sample_reporting_data():
    """Sample reporting data for testing"""
    return {
        "executive": {"ai_risk_score": 0.8, "compliance_posture": 0.9, "roi_analysis": {"cost_savings": 100000, "risk_reduction": 0.7}},
        "technical": {"attack_reproduction": {"steps": ["Step 1", "Step 2", "Step 3"]}, "vulnerability_fingerprinting": {"fingerprint": "abc123", "confidence": 0.9}},
        "compliance": {"soc2": {"score": 0.9, "controls": []}, "iso27001": {"score": 0.8, "controls": []}},
        "risk": {"adversarial_risk": 0.8, "privacy_risk": 0.7, "bias_risk": 0.6, "robustness_risk": 0.9},
        "roi": {"cost_savings": 100000, "implementation_cost": 50000, "roi_percentage": 100, "payback_period": 1, "net_present_value": 50000}
    }
