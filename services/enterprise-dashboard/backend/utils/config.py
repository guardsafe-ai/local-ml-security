"""
Enterprise Dashboard Backend - Configuration
Service configuration management
"""

import os
from typing import Dict, Any


def get_config() -> Dict[str, Any]:
    """Get service configuration"""
    return {
        "service_urls": {
            "training": os.getenv("TRAINING_URL", "http://training:8002"),
            "model_api": os.getenv("MODEL_API_URL", "http://model-api:8000"),
            "red_team": os.getenv("RED_TEAM_URL", "http://red-team:8001"),
            "analytics": os.getenv("ANALYTICS_URL", "http://analytics:8006"),
            "mlflow": os.getenv("MLFLOW_URL", "http://mlflow:5000"),
            "minio": os.getenv("MINIO_URL", "http://minio:9000"),
            "monitoring": os.getenv("MONITORING_URL", "http://monitoring:8501")
        },
        "redis_url": os.getenv("REDIS_URL", "redis://redis:6379"),
        "api_timeout": float(os.getenv("API_TIMEOUT", "30.0")),
        "cache_ttl": int(os.getenv("CACHE_TTL", "300")),
        "websocket_ping_interval": int(os.getenv("WEBSOCKET_PING_INTERVAL", "30")),
        "max_websocket_connections": int(os.getenv("MAX_WEBSOCKET_CONNECTIONS", "100")),
        "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
        "enable_websocket": os.getenv("ENABLE_WEBSOCKET", "true").lower() == "true",
        "enable_caching": os.getenv("ENABLE_CACHING", "true").lower() == "true",
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "service_name": os.getenv("SERVICE_NAME", "enterprise-dashboard-backend"),
        "version": os.getenv("SERVICE_VERSION", "1.0.0")
    }
