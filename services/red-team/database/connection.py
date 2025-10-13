"""
Red Team Service - Database Connection
Manages database connections and schema initialization
"""

import logging
import os
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import asyncpg
from asyncpg import Pool

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self):
        self.pool: Optional[Pool] = None
        self.connection_params = {
            "host": os.getenv("POSTGRES_HOST", "postgres"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "database": os.getenv("POSTGRES_DB", "ml_security_consolidated"),
            "user": os.getenv("POSTGRES_USER", "mlflow"),
            "password": os.getenv("POSTGRES_PASSWORD", "password")
        }

    async def connect(self):
        """Create database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.connection_params["host"],
                port=self.connection_params["port"],
                database=self.connection_params["database"],
                user=self.connection_params["user"],
                password=self.connection_params["password"],
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("✅ Red team database connection pool created successfully")
        except Exception as e:
            logger.error(f"❌ Failed to create red team database connection pool: {e}")
            raise

    async def disconnect(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("🔒 Red team database connection pool closed")

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

    async def initialize_schema(self):
        """Initialize database schema for red team service"""
        try:
            # Create red team test results table
            await self.execute_command("""
                CREATE TABLE IF NOT EXISTS red_team_test_results (
                    id SERIAL PRIMARY KEY,
                    test_id VARCHAR(255) NOT NULL,
                    model_name VARCHAR(255) NOT NULL,
                    model_type VARCHAR(50) NOT NULL,
                    model_version VARCHAR(100),
                    attack_category VARCHAR(100) NOT NULL,
                    attack_pattern TEXT NOT NULL,
                    attack_severity FLOAT NOT NULL,
                    detected BOOLEAN NOT NULL,
                    confidence FLOAT,
                    response_time_ms FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    test_duration_ms FLOAT,
                    vulnerability_score FLOAT,
                    security_risk VARCHAR(20),
                    pass_fail BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create red team test sessions table
            await self.execute_command("""
                CREATE TABLE IF NOT EXISTS red_team_test_sessions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    model_name VARCHAR(255) NOT NULL,
                    model_type VARCHAR(50) NOT NULL,
                    total_attacks INTEGER NOT NULL,
                    detected_attacks INTEGER NOT NULL,
                    detection_rate FLOAT NOT NULL,
                    pass_rate FLOAT NOT NULL,
                    overall_status VARCHAR(20) NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    duration_ms FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            await self.execute_command("CREATE INDEX IF NOT EXISTS idx_red_team_model ON red_team_test_results(model_name)")
            await self.execute_command("CREATE INDEX IF NOT EXISTS idx_red_team_timestamp ON red_team_test_results(timestamp)")
            await self.execute_command("CREATE INDEX IF NOT EXISTS idx_red_team_category ON red_team_test_results(attack_category)")
            await self.execute_command("CREATE INDEX IF NOT EXISTS idx_red_team_detected ON red_team_test_results(detected)")
            
            logger.info("✅ Red team database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize red team database schema: {e}")
            raise


# Global database manager instance
db_manager = DatabaseManager()
