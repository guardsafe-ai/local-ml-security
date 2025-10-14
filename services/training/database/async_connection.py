"""
Training Service - Async Database Connection
Manages async database connections and schema initialization
"""

import logging
import os
import asyncio
from typing import Optional
from contextlib import asynccontextmanager
import asyncpg
from asyncpg import Pool

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self):
        self.pool: Optional[Pool] = None
        self.connection_params = {
            "host": os.getenv("DB_HOST", "postgres"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "database": os.getenv("DB_NAME", "ml_security_consolidated"),
            "user": os.getenv("DB_USER", "mlflow"),
            "password": os.getenv("DB_PASSWORD", "password")
        }

    async def connect(self):
        """Create database connection pool with retry logic"""
        from utils.retry import retry_async, DB_RETRY_CONFIG
        
        @retry_async(DB_RETRY_CONFIG)
        async def _create_pool():
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
            logger.info("Database connection pool created successfully")
            return True
        
        try:
            await _create_pool()
        except Exception as e:
            logger.error(f"Failed to create database connection pool after retries: {e}")
            raise

    async def disconnect(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

    async def execute_query(self, query: str, *args):
        """Execute a query and return results"""
        if not self.pool:
            raise RuntimeError("Database connection pool not initialized")
        
        async with self.pool.acquire() as connection:
            try:
                rows = await connection.fetch(query, *args)
                return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise

    async def execute_command(self, command: str, *args):
        """Execute a command (INSERT, UPDATE, DELETE)"""
        if not self.pool:
            raise RuntimeError("Database connection pool not initialized")
        
        async with self.pool.acquire() as connection:
            try:
                result = await connection.execute(command, *args)
                return result
            except Exception as e:
                logger.error(f"Command execution failed: {e}")
                raise

    async def execute_fetchone(self, query: str, *args):
        """Execute a query and return single result"""
        if not self.pool:
            raise RuntimeError("Database connection pool not initialized")
        
        async with self.pool.acquire() as connection:
            try:
                row = await connection.fetchrow(query, *args)
                return dict(row) if row else None
            except Exception as e:
                logger.error(f"Query fetchone failed: {e}")
                raise

    def is_connected(self) -> bool:
        """Check if database connection pool is active"""
        return self.pool is not None and not self.pool.is_closed()
    
    @asynccontextmanager
    async def transaction(self):
        """Database transaction context manager with automatic rollback on failure"""
        if not self.pool:
            raise RuntimeError("Database connection pool not initialized")
        
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                try:
                    yield connection
                except Exception as e:
                    logger.error(f"Transaction failed, rolling back: {e}")
                    raise

    async def initialize_schema(self):
        """Initialize database schema for training service"""
        try:
            # Create training schema if it doesn't exist
            await self.execute_command("""
                CREATE SCHEMA IF NOT EXISTS training;
            """)
            
            # Create training jobs table
            await self.execute_command("""
                CREATE TABLE IF NOT EXISTS training.training_jobs (
                    job_id VARCHAR(255) PRIMARY KEY,
                    model_name VARCHAR(255) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    progress FLOAT DEFAULT 0.0,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    error_message TEXT,
                    result JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Add new columns if they don't exist
            try:
                await self.execute_command("""
                    ALTER TABLE training.training_jobs 
                    ADD COLUMN IF NOT EXISTS training_data_path VARCHAR(500);
                """)
                await self.execute_command("""
                    ALTER TABLE training.training_jobs 
                    ADD COLUMN IF NOT EXISTS learning_rate FLOAT;
                """)
                await self.execute_command("""
                    ALTER TABLE training.training_jobs 
                    ADD COLUMN IF NOT EXISTS batch_size INTEGER;
                """)
                await self.execute_command("""
                    ALTER TABLE training.training_jobs 
                    ADD COLUMN IF NOT EXISTS num_epochs INTEGER;
                """)
                await self.execute_command("""
                    ALTER TABLE training.training_jobs 
                    ADD COLUMN IF NOT EXISTS max_length INTEGER;
                """)
                await self.execute_command("""
                    ALTER TABLE training.training_jobs 
                    ADD COLUMN IF NOT EXISTS config JSONB;
                """)
                logger.info("Added new configuration columns to training_jobs table")
            except Exception as e:
                logger.warning(f"Could not add new columns to training_jobs table: {e}")
            
            # Create model performance table
            await self.execute_command("""
                CREATE TABLE IF NOT EXISTS training.model_performance (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(255) NOT NULL,
                    version VARCHAR(100),
                    metrics JSONB NOT NULL,
                    test_data_path VARCHAR(500),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create retraining history table
            await self.execute_command("""
                CREATE TABLE IF NOT EXISTS training.retraining_history (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(255) NOT NULL,
                    trigger_type VARCHAR(100),
                    performance_before JSONB,
                    performance_after JSONB,
                    retraining_data_path VARCHAR(500),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create training logs table for detailed job logging
            await self.execute_command("""
                CREATE TABLE IF NOT EXISTS training.training_logs (
                    id SERIAL PRIMARY KEY,
                    job_id VARCHAR(255) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    level VARCHAR(20) NOT NULL,
                    source VARCHAR(100) NOT NULL,
                    message TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create indexes for efficient log queries
            await self.execute_command("""
                CREATE INDEX IF NOT EXISTS idx_training_logs_job_id 
                ON training.training_logs(job_id);
            """)
            
            await self.execute_command("""
                CREATE INDEX IF NOT EXISTS idx_training_logs_timestamp 
                ON training.training_logs(timestamp);
            """)
            
            logger.info("Training database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize training database schema: {e}")
            raise


# Global database manager instance
db_manager = DatabaseManager()
