"""
Configuration Validation Utilities
Validates configuration for production environment only
"""

import os
import re
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def validate_production_config(config: Dict[str, Any], service_name: str) -> None:
    """Validate configuration for production environment only"""
    
    # Define required environment variables for each service
    required_env_vars = {
        "analytics": ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"],
        "training": ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD", "MLFLOW_TRACKING_URI", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        "model-api": ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"],
        "business-metrics": ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"],
        "data-privacy": ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD", "ENCRYPTION_KEY"],
        "red-team": ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"],
        "model-cache": ["REDIS_HOST", "REDIS_PORT", "REDIS_DB"],
    }
    
    # Check required environment variables
    missing_vars = []
    for var in required_env_vars.get(service_name, []):
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables for {service_name} in production: {', '.join(missing_vars)}")
    
    # Check for hardcoded credentials in config values (production only)
    hardcoded_patterns = [
        r"password\s*=\s*['\"]password['\"]",  # password="password"
        r"password\s*:\s*['\"]password['\"]",  # password: "password"
        r"minioadmin",  # MinIO default credentials
        r"ml_security_password",  # Default password
        r"dev-encryption-key",  # Development encryption key
    ]
    
    config_str = str(config)
    for pattern in hardcoded_patterns:
        if re.search(pattern, config_str, re.IGNORECASE):
            raise ValueError(f"Default credentials detected in {service_name} production config: {pattern}")
    
    # Validate encryption key if present
    if "encryption_key" in config:
        encryption_key = config["encryption_key"]
        if not encryption_key or len(encryption_key) < 32:
            raise ValueError(f"Encryption key must be at least 32 characters for {service_name}")
    
    logger.info(f"✅ Production configuration validation passed for {service_name}")


def validate_config(config: Dict[str, Any], service_name: str) -> None:
    """Legacy function - now calls production validation"""
    validate_production_config(config, service_name)


def get_required_env_var(var_name: str, service_name: str) -> str:
    """Get required environment variable or raise error"""
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"Required environment variable {var_name} not set for {service_name}")
    return value


def get_optional_env_var(var_name: str, default: str = None) -> str:
    """Get optional environment variable with default"""
    return os.getenv(var_name, default) if default else os.getenv(var_name)
