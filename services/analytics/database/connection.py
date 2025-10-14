"""
Database Connection Management - Async Version
"""

import os
import asyncio
import asyncpg
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import logging

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
        """Create database connection pool with retry logic"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.connection_params["host"],
                port=self.connection_params["port"],
                database=self.connection_params["database"],
                user=self.connection_params["user"],
                password=self.connection_params["password"],
                min_size=5,
                max_size=20,
                max_queries=50000,  # Recycle connections after 50k queries
                max_inactive_connection_lifetime=300,  # 5 min idle timeout
                timeout=30,  # 30s connection acquire timeout
                command_timeout=60,  # 60s query timeout
                server_settings={
                    'jit': 'off',
                    'statement_timeout': '60000'  # 60s statement timeout
                }
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
        """Async context manager for database connections"""
        if not self.pool:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        async with self.pool.acquire() as connection:
            try:
                yield connection
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
        """Execute an INSERT/UPDATE/DELETE command and return status"""
        async with self.get_connection() as conn:
            try:
                result = await conn.execute(command, *args)
                return result
            except Exception as e:
                logger.error(f"Command execution failed: {e}")
                raise
    
    async def fetch_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Fetch a single row as dict"""
        async with self.get_connection() as conn:
            try:
                row = await conn.fetchrow(query, *args)
                return dict(row) if row else None
            except Exception as e:
                logger.error(f"Fetch one failed: {e}")
                raise
    
    async def fetch_many(self, query: str, *args, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetch multiple rows as list of dicts with limit"""
        async with self.get_connection() as conn:
            try:
                rows = await conn.fetch(query, *args)
                return [dict(row) for row in rows[:limit]]
            except Exception as e:
                logger.error(f"Fetch many failed: {e}")
                raise
    
    @asynccontextmanager
    async def transaction(self):
        """Database transaction context manager with automatic rollback on failure"""
        if not self.pool:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                try:
                    yield connection
                except Exception as e:
                    logger.error(f"Transaction failed, rolling back: {e}")
                    raise


# Global database manager instance
db_manager = DatabaseManager()


# Legacy compatibility functions (deprecated - use db_manager directly)
async def get_db_connection():
    """Get database connection (legacy function for compatibility)"""
    logger.warning("⚠️ get_db_connection() is deprecated. Use db_manager.get_connection() context manager instead.")
    return db_manager.get_connection()
