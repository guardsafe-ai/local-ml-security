"""
Monitoring Service - Configuration
Service configuration and environment variables
"""

import os
from typing import Dict, Any


def get_config() -> Dict[str, Any]:
    """Get service configuration"""
    return {
        "service_name": "monitoring",
        "version": "1.0.0",
        "port": int(os.getenv("PORT", 8008)),
        "host": os.getenv("HOST", "0.0.0.0"),
        "redis_url": os.getenv("REDIS_URL", "redis://redis:6379"),
        "postgres_url": os.getenv("POSTGRES_URL", "postgresql://mlflow:password@postgres:5432/ml_security_consolidated"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "refresh_interval": int(os.getenv("REFRESH_INTERVAL", "5")),  # seconds
        "cache_ttl": int(os.getenv("CACHE_TTL", "2")),  # seconds
        "timeout": int(os.getenv("TIMEOUT", "10")),  # seconds
        "service_urls": {
            "training": os.getenv("TRAINING_URL", "http://training:8002"),
            "model-api": os.getenv("MODEL_API_URL", "http://model-api:8000"),
            "red-team": os.getenv("RED_TEAM_URL", "http://red-team:8001"),
            "analytics": os.getenv("ANALYTICS_URL", "http://analytics:8006"),
            "business-metrics": os.getenv("BUSINESS_METRICS_URL", "http://business-metrics:8004"),
            "data-privacy": os.getenv("DATA_PRIVACY_URL", "http://data-privacy:8005"),
            "enterprise-dashboard": os.getenv("ENTERPRISE_DASHBOARD_URL", "http://enterprise-dashboard-backend:8007")
        },
        "chart_config": {
            "color_palette": {
                "primary": "#1f77b4",
                "secondary": "#ff7f0e",
                "success": "#2ca02c",
                "warning": "#d62728",
                "info": "#9467bd",
                "light": "#8c564b"
            },
            "default_height": 400,
            "chart_theme": "plotly_white"
        }
    }
