"""
Database service for enterprise dashboard
Handles database connections and operations
"""

import logging
import asyncpg
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class DatabaseService:
    """Database service for handling PostgreSQL operations"""
    
    def __init__(self):
        self.connection_pool = None
        self.database_url = os.getenv(
            'DATABASE_URL', 
            'postgresql://mlflow:password@postgres:5432/ml_security_consolidated'
        )
    
    async def initialize(self):
        """Initialize database connection pool"""
        try:
            self.connection_pool = await asyncpg.create_pool(
                self.database_url,
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
            logger.info("✅ Database connection pool initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize database connection pool: {e}")
            raise
    
    async def close(self):
        """Close database connection pool"""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("🔒 Database connection pool closed")
    
    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results"""
        try:
            async with self.connection_pool.acquire() as connection:
                rows = await connection.fetch(query, *args)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"❌ Database query failed: {e}")
            raise
    
    async def execute_command(self, command: str, *args) -> str:
        """Execute an INSERT/UPDATE/DELETE command"""
        try:
            async with self.connection_pool.acquire() as connection:
                result = await connection.execute(command, *args)
                return result
        except Exception as e:
            logger.error(f"❌ Database command failed: {e}")
            raise
    
    async def execute_fetchone(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Execute a query and return single result"""
        try:
            async with self.connection_pool.acquire() as connection:
                row = await connection.fetchrow(query, *args)
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"❌ Database fetchone failed: {e}")
            raise
    
    @asynccontextmanager
    async def transaction(self):
        """
        Database transaction context manager with automatic rollback
        
        Usage:
            async with db.transaction() as conn:
                await conn.execute("UPDATE ...")
                await conn.execute("INSERT ...")
                # Auto-commits on success, auto-rolls back on exception
        """
        if not self.connection_pool:
            raise RuntimeError("Database not connected")
        
        async with self.connection_pool.acquire() as conn:
            async with conn.transaction():
                try:
                    yield conn
                    logger.debug("Transaction committed successfully")
                except Exception as e:
                    logger.error(f"Transaction failed, rolling back: {e}")
                    raise  # Re-raise to maintain exception flow

# Global database service instance
db_service = DatabaseService()
