"""
Training Service - Database Timeout Configuration
Service-specific timeout settings for Training
"""

from dataclasses import dataclass
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class TrainingTimeoutConfig:
    """Database timeout configuration for Training service"""
    
    # Connection pool settings - optimized for ML training operations
    min_size: int = 10
    max_size: int = 30
    max_queries: int = 50000
    max_inactive_connection_lifetime: int = 300  # 5 min idle timeout
    
    # Timeout settings - longer for ML operations
    connection_timeout: int = 30  # 30s connection acquire timeout
    command_timeout: int = 60  # 60s query timeout (longer for ML operations)
    statement_timeout: int = 60000  # 60s statement timeout (milliseconds)
    
    # Performance settings
    jit_enabled: bool = False
    tcp_keepalives_idle: int = 600
    tcp_keepalives_interval: int = 30
    tcp_keepalives_count: int = 3
    
    def get_pool_config(self) -> Dict[str, Any]:
        """Get asyncpg pool configuration with timeouts"""
        return {
            "min_size": self.min_size,
            "max_size": self.max_size,
            "max_queries": self.max_queries,
            "max_inactive_connection_lifetime": self.max_inactive_connection_lifetime,
            "timeout": self.connection_timeout,
            "command_timeout": self.command_timeout,
            "server_settings": {
                'jit': 'off' if not self.jit_enabled else 'on',
                'statement_timeout': str(self.statement_timeout),
                'tcp_keepalives_idle': str(self.tcp_keepalives_idle),
                'tcp_keepalives_interval': str(self.tcp_keepalives_interval),
                'tcp_keepalives_count': str(self.tcp_keepalives_count),
                'application_name': 'training_service',
                'log_statement': 'none',
                'log_min_duration_statement': '-1'
            }
        }

def get_training_timeout_config() -> TrainingTimeoutConfig:
    """Get timeout configuration for Training service"""
    return TrainingTimeoutConfig()

def log_timeout_config(config: TrainingTimeoutConfig):
    """Log timeout configuration for monitoring"""
    logger.info(f"🔧 [TRAINING] Database timeout configuration:")
    logger.info(f"   Connection timeout: {config.connection_timeout}s")
    logger.info(f"   Command timeout: {config.command_timeout}s")
    logger.info(f"   Statement timeout: {config.statement_timeout}ms")
    logger.info(f"   Pool size: {config.min_size}-{config.max_size}")
    logger.info(f"   JIT enabled: {config.jit_enabled}")
