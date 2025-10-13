"""
Configuration Management
"""

import os
from typing import Dict, Any


def get_config() -> Dict[str, Any]:
    """Get service configuration"""
    return {
        "database": {
            "host": os.getenv("POSTGRES_HOST", "postgres"),
            "port": os.getenv("POSTGRES_PORT", "5432"),
            "database": os.getenv("POSTGRES_DB", "ml_security_consolidated"),
            "user": os.getenv("POSTGRES_USER", "mlflow"),
            "password": os.getenv("POSTGRES_PASSWORD", "password")
        },
        "service": {
            "name": "model-api",
            "version": "1.0.0",
            "host": "0.0.0.0",
            "port": 8000
        },
        "redis": {
            "host": os.getenv("REDIS_HOST", "redis"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
            "db": int(os.getenv("REDIS_DB", "0"))
        },
        "mlflow": {
            "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        }
    }
