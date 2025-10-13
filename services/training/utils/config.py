"""
Training Service - Configuration - Development Safe Version
Service configuration management with development defaults
"""

import os
from typing import Dict, Any
from .config_validation import validate_production_config


def get_config() -> Dict[str, Any]:
    """Get service configuration with development defaults"""
    config = {
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        "redis_url": os.getenv("REDIS_URL", "redis://redis:6379"),
        "database_url": os.getenv("DATABASE_URL", "postgresql://ml_security_user:ml_security_password@postgres:5432/ml_security_consolidated"),
        "minio_endpoint": os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        "training_data_dir": os.getenv("TRAINING_DATA_DIR", "/app/training_data"),
        "models_dir": os.getenv("MODELS_DIR", "/app/models"),
        "logs_dir": os.getenv("LOGS_DIR", "/app/logs"),
        "max_training_jobs": int(os.getenv("MAX_TRAINING_JOBS", "5")),
        "default_batch_size": int(os.getenv("DEFAULT_BATCH_SIZE", "8")),
        "default_epochs": int(os.getenv("DEFAULT_EPOCHS", "2")),
        "default_learning_rate": float(os.getenv("DEFAULT_LEARNING_RATE", "2e-5")),
        "data_cleanup_days": int(os.getenv("DATA_CLEANUP_DAYS", "30")),
        "mlflow_cleanup_days": int(os.getenv("MLFLOW_CLEANUP_DAYS", "30"))
    }
    
    # Only validate in production environment
    if os.getenv("ENVIRONMENT") == "production":
        validate_production_config(config, "training")
    
    return config
