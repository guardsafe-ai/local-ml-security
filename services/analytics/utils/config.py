"""
Configuration Management - Development Safe Version
"""

import os
from typing import Dict, Any
from .config_validation import validate_production_config


def get_config() -> Dict[str, Any]:
    """Get service configuration with development defaults"""
    config = {
        "database": {
            "host": os.getenv("POSTGRES_HOST", "postgres"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "database": os.getenv("POSTGRES_DB", "ml_security_consolidated"),
            "user": os.getenv("POSTGRES_USER", "mlflow"),
            "password": os.getenv("POSTGRES_PASSWORD", "password")
        },
        "service": {
            "name": "analytics",
            "version": "1.0.0",
            "host": "0.0.0.0",
            "port": 8006
        },
        "redis": {
            "host": os.getenv("REDIS_HOST", "redis"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
            "db": int(os.getenv("REDIS_DB", "0"))
        }
    }
    
    # Only validate in production environment
    if os.getenv("ENVIRONMENT") == "production":
        validate_production_config(config, "analytics")
    
    return config
