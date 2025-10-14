"""
Analytics Service - Database Timeout Configuration
Service-specific timeout settings for Analytics
"""

from dataclasses import dataclass
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class AnalyticsTimeoutConfig:
    """Database timeout configuration for Analytics service"""
    
    # Connection pool settings - optimized for analytics queries
    min_size: int = 5
    max_size: int = 20
    max_queries: int = 50000
    max_inactive_connection_lifetime: int = 300  # 5 min idle timeout
    
    # Timeout settings - longer for complex analytics
    connection_timeout: int = 30  # 30s connection acquire timeout
    command_timeout: int = 120  # 120s query timeout (longer for analytics)
    statement_timeout: int = 120000  # 120s statement timeout (milliseconds)
    
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
                'application_name': 'analytics_service',
                'log_statement': 'none',
                'log_min_duration_statement': '-1'
            }
        }

def get_analytics_timeout_config() -> AnalyticsTimeoutConfig:
    """Get timeout configuration for Analytics service"""
    return AnalyticsTimeoutConfig()

def log_timeout_config(config: AnalyticsTimeoutConfig):
    """Log timeout configuration for monitoring"""
    logger.info(f"🔧 [ANALYTICS] Database timeout configuration:")
    logger.info(f"   Connection timeout: {config.connection_timeout}s")
    logger.info(f"   Command timeout: {config.command_timeout}s")
    logger.info(f"   Statement timeout: {config.statement_timeout}ms")
    logger.info(f"   Pool size: {config.min_size}-{config.max_size}")
    logger.info(f"   JIT enabled: {config.jit_enabled}")
