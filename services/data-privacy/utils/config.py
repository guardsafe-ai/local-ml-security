"""
Data Privacy Service - Configuration - Development Safe Version
Service configuration and environment variables with development defaults
"""

import os
from typing import Dict, Any
from .config_validation import validate_production_config


def get_config() -> Dict[str, Any]:
    """Get service configuration with development defaults"""
    config = {
        "service_name": "data-privacy",
        "version": "1.0.0",
        "port": int(os.getenv("PORT", 8005)),
        "host": os.getenv("HOST", "0.0.0.0"),
        "redis_url": os.getenv("REDIS_URL", "redis://redis:6379"),
        "postgres_url": os.getenv("POSTGRES_URL", "postgresql://mlflow:password@postgres:5432/ml_security_consolidated"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "encryption_key": os.getenv("ENCRYPTION_KEY", "dev-encryption-key-32-chars-minimum"),
        "default_retention_days": int(os.getenv("DEFAULT_RETENTION_DAYS", "365")),
        "audit_log_retention_days": int(os.getenv("AUDIT_LOG_RETENTION_DAYS", "2555")),  # 7 years
        "anonymization_levels": ["low", "medium", "high"],
        "supported_data_categories": [
            "personal_data",
            "sensitive_data",
            "biometric_data",
            "financial_data",
            "health_data",
            "location_data"
        ]
    }
    
    # Only validate in production environment
    if os.getenv("ENVIRONMENT") == "production":
        validate_production_config(config, "data-privacy")
    
    return config
