"""
Database service for enterprise dashboard
Handles database connections and operations
"""

import logging
import asyncpg
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

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
                min_size=1,
                max_size=10,
                command_timeout=60
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

# Global database service instance
db_service = DatabaseService()
