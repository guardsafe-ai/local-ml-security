"""
Database Connection Management - Async Version
"""

import os
import asyncio
import asyncpg
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import logging
from utils.query_monitoring import monitor_query, get_analytics_query_monitor

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Async database connection manager with connection pooling"""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.connection_params = {
            "host": os.getenv("POSTGRES_HOST", "postgres"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "database": os.getenv("POSTGRES_DB", "ml_security_consolidated"),
            "user": os.getenv("POSTGRES_USER", "mlflow"),
            "password": os.getenv("POSTGRES_PASSWORD", "password")
        }
    
    async def connect(self) -> None:
        """Create database connection pool with retry logic and standardized timeouts"""
        try:
            # Import service-specific timeout configuration
            from utils.database_timeouts import get_analytics_timeout_config, log_timeout_config
            
            # Get timeout configuration for analytics service
            timeout_config = get_analytics_timeout_config()
            log_timeout_config(timeout_config)
            
            self.pool = await asyncpg.create_pool(
                host=self.connection_params["host"],
                port=self.connection_params["port"],
                database=self.connection_params["database"],
                user=self.connection_params["user"],
                password=self.connection_params["password"],
                **timeout_config.get_pool_config()
            )
            logger.info("✅ Analytics database connection pool created successfully")
        except Exception as e:
            logger.error(f"❌ Failed to create analytics database connection pool: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("🔒 Analytics database connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Async context manager for database connections with query monitoring"""
        if not self.pool:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        # Create a monitored connection wrapper
        class MonitoredConnection:
            def __init__(self, connection, monitor):
                self.connection = connection
                self.monitor = monitor
            
            def __getattr__(self, name):
                # Delegate all other attributes to the original connection
                return getattr(self.connection, name)
            
            async def execute(self, query, *args, **kwargs):
                """Monitored execute method"""
                async with monitor_query("connection_execute", self.connection.execute, query, *args, **kwargs) as result:
                    return result
            
            async def fetch(self, query, *args, **kwargs):
                """Monitored fetch method"""
                async with monitor_query("connection_fetch", self.connection.fetch, query, *args, **kwargs) as result:
                    return result
            
            async def fetchrow(self, query, *args, **kwargs):
                """Monitored fetchrow method"""
                async with monitor_query("connection_fetchrow", self.connection.fetchrow, query, *args, **kwargs) as result:
                    return result
            
            async def fetchval(self, query, *args, **kwargs):
                """Monitored fetchval method"""
                async with monitor_query("connection_fetchval", self.connection.fetchval, query, *args, **kwargs) as result:
                    return result
        
        async with self.pool.acquire() as connection:
            try:
                # Wrap the connection with monitoring
                monitored_conn = MonitoredConnection(connection, get_analytics_query_monitor())
                yield monitored_conn
            except Exception as e:
                logger.error(f"Database operation failed: {e}")
                raise
    
    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results as list of dicts"""
        async with self.get_connection() as conn:
            try:
                rows = await conn.fetch(query, *args)
                return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise
    
    async def execute_command(self, command: str, *args) -> str:
        """Execute an INSERT/UPDATE/DELETE command and return status with query monitoring"""
        async with monitor_query("execute_command", self._execute_command_impl, command, *args) as result:
            return result
    
    async def _execute_command_impl(self, command: str, *args) -> str:
        """Internal implementation of execute_command"""
        async with self.get_connection() as conn:
            try:
                result = await conn.execute(command, *args)
                return result
            except Exception as e:
                logger.error(f"Command execution failed: {e}")
                raise
    
    async def fetch_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Fetch a single row as dict with query monitoring"""
        async with monitor_query("fetch_one", self._fetch_one_impl, query, *args) as result:
            return result
    
    async def _fetch_one_impl(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Internal implementation of fetch_one"""
        async with self.get_connection() as conn:
            try:
                row = await conn.fetchrow(query, *args)
                return dict(row) if row else None
            except Exception as e:
                logger.error(f"Fetch one failed: {e}")
                raise
    
    async def fetch_many(self, query: str, *args, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetch multiple rows as list of dicts with limit and query monitoring"""
        async with monitor_query("fetch_many", self._fetch_many_impl, query, *args, limit=limit) as result:
            return result
    
    async def _fetch_many_impl(self, query: str, *args, limit: int = 1000) -> List[Dict[str, Any]]:
        """Internal implementation of fetch_many"""
        async with self.get_connection() as conn:
            try:
                rows = await conn.fetch(query, *args)
                return [dict(row) for row in rows[:limit]]
            except Exception as e:
                logger.error(f"Fetch many failed: {e}")
                raise
    
    @asynccontextmanager
    async def transaction(self):
        """Database transaction context manager with automatic rollback on failure and query monitoring"""
        if not self.pool:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        # Create a monitored connection wrapper
        class MonitoredConnection:
            def __init__(self, connection, monitor):
                self.connection = connection
                self.monitor = monitor
            
            def __getattr__(self, name):
                # Delegate all other attributes to the original connection
                return getattr(self.connection, name)
            
            async def execute(self, query, *args, **kwargs):
                """Monitored execute method"""
                async with monitor_query("transaction_execute", self.connection.execute, query, *args, **kwargs) as result:
                    return result
            
            async def fetch(self, query, *args, **kwargs):
                """Monitored fetch method"""
                async with monitor_query("transaction_fetch", self.connection.fetch, query, *args, **kwargs) as result:
                    return result
            
            async def fetchrow(self, query, *args, **kwargs):
                """Monitored fetchrow method"""
                async with monitor_query("transaction_fetchrow", self.connection.fetchrow, query, *args, **kwargs) as result:
                    return result
            
            async def fetchval(self, query, *args, **kwargs):
                """Monitored fetchval method"""
                async with monitor_query("transaction_fetchval", self.connection.fetchval, query, *args, **kwargs) as result:
                    return result
        
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                try:
                    # Wrap the connection with monitoring
                    monitored_conn = MonitoredConnection(connection, get_analytics_query_monitor())
                    yield monitored_conn
                except Exception as e:
                    logger.error(f"Transaction failed, rolling back: {e}")
                    raise

    async def initialize_schema(self):
        """Initialize analytics database schema"""
        try:
            async with self.get_connection() as conn:
                # Create analytics schema if it doesn't exist
                await conn.execute("CREATE SCHEMA IF NOT EXISTS analytics")
                
                # Create drift_detections table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS analytics.drift_detections (
                        id SERIAL PRIMARY KEY,
                        model_name VARCHAR(255) NOT NULL,
                        drift_score FLOAT NOT NULL,
                        drift_type VARCHAR(50) NOT NULL,
                        detected_at TIMESTAMP NOT NULL,
                        details JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create retrain_jobs table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS analytics.retrain_jobs (
                        id SERIAL PRIMARY KEY,
                        drift_id INTEGER REFERENCES analytics.drift_detections(id),
                        job_id VARCHAR(255) NOT NULL,
                        model_name VARCHAR(255) NOT NULL,
                        status VARCHAR(50) NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create drift_monitoring_log table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS analytics.drift_monitoring_log (
                        id SERIAL PRIMARY KEY,
                        model_name VARCHAR(255) NOT NULL,
                        drift_score FLOAT NOT NULL,
                        drift_type VARCHAR(50) NOT NULL,
                        detected BOOLEAN NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create drift_daily_reports table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS analytics.drift_daily_reports (
                        id SERIAL PRIMARY KEY,
                        report_date DATE NOT NULL,
                        report_data JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(report_date)
                    )
                """)
                
                logger.info("✅ Analytics database schema initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize analytics schema: {e}")
            raise


# Global database manager instance
db_manager = DatabaseManager()


# Legacy compatibility functions (deprecated - use db_manager directly)
async def get_db_connection():
    """Get database connection (legacy function for compatibility)"""
    logger.warning("⚠️ get_db_connection() is deprecated. Use db_manager.get_connection() context manager instead.")
    return db_manager.get_connection()
