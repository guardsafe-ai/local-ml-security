"""
Business Metrics Service - Database Connection
Database connection and management
"""

import logging
import asyncio
import asyncpg
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from utils.config import get_config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self):
        self.config = get_config()
        self.pool: Optional[asyncpg.Pool] = None
        self.connection_string = self.config["postgres_url"]
    
    async def connect(self) -> None:
        """Establish database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
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
        except Exception as e:
            logger.error(f"Failed to create database connection pool: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    async def execute_query(self, query: str, *args) -> list:
        """Execute a query and return results"""
        if not self.pool:
            raise Exception("Database not connected")
        
        async with self.pool.acquire() as connection:
            try:
                results = await connection.fetch(query, *args)
                return [dict(row) for row in results]
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
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
        if not self.pool:
            raise RuntimeError("Database not connected")
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                try:
                    yield conn
                    logger.debug("Transaction committed successfully")
                except Exception as e:
                    logger.error(f"Transaction failed, rolling back: {e}")
                    raise  # Re-raise to maintain exception flow

    async def execute_command(self, command: str, *args) -> str:
        """Execute a command and return status"""
        if not self.pool:
            raise Exception("Database not connected")
        
        async with self.pool.acquire() as connection:
            try:
                result = await connection.execute(command, *args)
                return result
            except Exception as e:
                logger.error(f"Command execution failed: {e}")
                raise
    
    async def initialize_schema(self) -> None:
        """Initialize database schema for business metrics"""
        try:
            # Create business_metrics schema if it doesn't exist
            await self.execute_command("""
                CREATE SCHEMA IF NOT EXISTS business_metrics;
            """)
            
            # Create tables for storing metrics
            await self.execute_command("""
                CREATE TABLE IF NOT EXISTS business_metrics.attack_success_rates (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_attacks INTEGER NOT NULL,
                    successful_attacks INTEGER NOT NULL,
                    success_rate FLOAT NOT NULL,
                    by_category JSONB,
                    by_model JSONB,
                    trend_7d FLOAT,
                    trend_30d FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            await self.execute_command("""
                CREATE TABLE IF NOT EXISTS business_metrics.cost_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_cost_usd FLOAT NOT NULL,
                    compute_cost FLOAT NOT NULL,
                    storage_cost FLOAT NOT NULL,
                    api_calls_cost FLOAT NOT NULL,
                    model_training_cost FLOAT NOT NULL,
                    cost_per_prediction FLOAT NOT NULL,
                    cost_trend_7d FLOAT,
                    cost_trend_30d FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            await self.execute_command("""
                CREATE TABLE IF NOT EXISTS business_metrics.system_effectiveness (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    overall_effectiveness FLOAT NOT NULL,
                    detection_accuracy FLOAT NOT NULL,
                    false_positive_rate FLOAT NOT NULL,
                    false_negative_rate FLOAT NOT NULL,
                    response_time_p95 FLOAT NOT NULL,
                    availability_percent FLOAT NOT NULL,
                    user_satisfaction_score FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            await self.execute_command("""
                CREATE TABLE IF NOT EXISTS business_metrics.model_drift (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(255) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    drift_detected BOOLEAN NOT NULL,
                    drift_score FLOAT NOT NULL,
                    confidence_interval_lower FLOAT NOT NULL,
                    confidence_interval_upper FLOAT NOT NULL,
                    features_drifted JSONB,
                    severity VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            logger.info("Business metrics database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise


# Global database manager instance
db_manager = DatabaseManager()
