"""
Red Team Service - Configuration
Service configuration management
"""

import os
from typing import Dict, Any


def get_config() -> Dict[str, Any]:
    """Get service configuration"""
    return {
        "model_api_url": os.getenv("MODEL_API_URL", "http://model-api:8000"),
        "redis_url": os.getenv("REDIS_URL", "redis://redis:6379"),
        "database_url": os.getenv("DATABASE_URL", "postgresql://mlflow:password@postgres:5432/ml_security_consolidated"),
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        "max_concurrent_tests": int(os.getenv("MAX_CONCURRENT_TESTS", "5")),
        "default_num_attacks": int(os.getenv("DEFAULT_NUM_ATTACKS", "50")),
        "default_severity_threshold": float(os.getenv("DEFAULT_SEVERITY_THRESHOLD", "0.7")),
        "test_timeout_seconds": int(os.getenv("TEST_TIMEOUT_SECONDS", "300")),
        "learning_enabled": os.getenv("LEARNING_ENABLED", "true").lower() == "true",
        "learning_update_frequency_hours": int(os.getenv("LEARNING_UPDATE_FREQUENCY_HOURS", "24")),
        "data_retention_days": int(os.getenv("DATA_RETENTION_DAYS", "30")),
        "attack_categories": os.getenv("ATTACK_CATEGORIES", "prompt_injection,jailbreak,system_extraction,code_injection").split(",")
    }
