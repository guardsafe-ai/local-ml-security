"""
Data Privacy Service - Database Connection
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
        """Execute a query and return results with query monitoring"""
        if not self.pool:
            raise Exception("Database not connected")
        
        from utils.query_monitoring import monitor_query
        
        async with monitor_query("execute_query", self._execute_query_impl, query, *args) as result:
            return result
    
    async def _execute_query_impl(self, query: str, *args) -> list:
        """Internal implementation of execute_query without monitoring"""
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
        """Execute a command and return status with query monitoring"""
        if not self.pool:
            raise Exception("Database not connected")
        
        from utils.query_monitoring import monitor_query
        
        async with monitor_query("execute_command", self._execute_command_impl, command, *args) as result:
            return result
    
    async def _execute_command_impl(self, command: str, *args) -> str:
        """Internal implementation of execute_command without monitoring"""
        async with self.pool.acquire() as connection:
            try:
                result = await connection.execute(command, *args)
                return result
            except Exception as e:
                logger.error(f"Command execution failed: {e}")
                raise
    
    async def initialize_schema(self) -> None:
        """Initialize database schema for data privacy"""
        try:
            # Create data_privacy schema if it doesn't exist
            await self.execute_command("""
                CREATE SCHEMA IF NOT EXISTS data_privacy;
            """)
            
            # Create data subjects table
            await self.execute_command("""
                CREATE TABLE IF NOT EXISTS data_privacy.data_subjects (
                    subject_id VARCHAR(255) PRIMARY KEY,
                    email VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data_categories JSONB NOT NULL,
                    retention_until TIMESTAMP NOT NULL,
                    consent_given BOOLEAN NOT NULL DEFAULT TRUE,
                    consent_withdrawn BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create audit logs table
            await self.execute_command("""
                CREATE TABLE IF NOT EXISTS data_privacy.audit_logs (
                    log_id VARCHAR(255) PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    action VARCHAR(255) NOT NULL,
                    subject_id VARCHAR(255),
                    user_id VARCHAR(255),
                    details JSONB,
                    ip_address VARCHAR(45),
                    user_agent TEXT,
                    created_at_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create retention policies table
            await self.execute_command("""
                CREATE TABLE IF NOT EXISTS data_privacy.retention_policies (
                    policy_id VARCHAR(255) PRIMARY KEY,
                    data_category VARCHAR(255) NOT NULL,
                    retention_days INTEGER NOT NULL,
                    auto_delete BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create anonymization records table
            await self.execute_command("""
                CREATE TABLE IF NOT EXISTS data_privacy.anonymization_records (
                    record_id VARCHAR(255) PRIMARY KEY,
                    original_text TEXT NOT NULL,
                    anonymized_text TEXT NOT NULL,
                    anonymization_method VARCHAR(255) NOT NULL,
                    pii_detected JSONB,
                    confidence_scores JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            logger.info("Data privacy database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise


# Global database manager instance
db_manager = DatabaseManager()
