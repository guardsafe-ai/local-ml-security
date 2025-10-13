"""
Business Metrics Service - Configuration
Service configuration and environment variables
"""

import os
from typing import Dict, Any


def get_config() -> Dict[str, Any]:
    """Get service configuration"""
    return {
        "service_name": "business-metrics",
        "version": "1.0.0",
        "port": int(os.getenv("PORT", 8004)),
        "host": os.getenv("HOST", "0.0.0.0"),
        "redis_url": os.getenv("REDIS_URL", "redis://redis:6379"),
        "postgres_url": os.getenv("POSTGRES_URL", "postgresql://mlflow:password@postgres:5432/ml_security_consolidated"),
        "analytics_url": os.getenv("ANALYTICS_URL", "http://analytics:8006"),
        "model_api_url": os.getenv("MODEL_API_URL", "http://model-api:8000"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "cost_per_hour": float(os.getenv("COST_PER_HOUR", "0.5")),
        "storage_cost_per_gb": float(os.getenv("STORAGE_COST_PER_GB", "0.023")),
        "api_cost_per_1k": float(os.getenv("API_COST_PER_1K", "0.002")),
        "drift_threshold": float(os.getenv("DRIFT_THRESHOLD", "0.1")),
        "time_range_days": int(os.getenv("DEFAULT_TIME_RANGE_DAYS", "30"))
    }
