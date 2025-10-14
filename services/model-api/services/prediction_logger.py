"""
Prediction Logging Service
Logs all predictions to database for monitoring and analytics
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import asyncpg
import uuid

# Import data encryption
from services.data_encryption import DataEncryption

logger = logging.getLogger(__name__)

@dataclass
class PredictionLog:
    """Prediction log entry"""
    prediction_id: str
    timestamp: datetime
    model_name: str
    input_text: str
    prediction: str
    confidence: float
    processing_time_ms: float
    from_cache: bool
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class PredictionLogger:
    """Handles prediction logging to database with encryption for sensitive data"""
    
    def __init__(self, database_url: str = "postgresql://mlflow:password@postgres:5432/ml_security_consolidated"):
        self.database_url = database_url
        self.pool = None
        self.batch_size = 100
        self.batch_timeout = 5.0  # seconds
        self.pending_logs: List[PredictionLog] = []
        self.batch_task = None
        # Initialize data encryption
        self.encryption = DataEncryption()
        
    async def initialize(self):
        """Initialize database connection pool"""
        try:
            # Import service-specific timeout configuration
            from utils.database_timeouts import get_model_api_timeout_config, log_timeout_config
            
            # Get timeout configuration for model-api service
            timeout_config = get_model_api_timeout_config()
            log_timeout_config(timeout_config)
            
            self.pool = await asyncpg.create_pool(
                self.database_url,
                **timeout_config.get_pool_config()
            )
            
            # Create predictions table if it doesn't exist
            await self._create_predictions_table()
            logger.info("✅ Prediction logger initialized with database pool")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize prediction logger: {e}")
            raise
    
    async def _create_predictions_table(self):
        """Create predictions table for logging with query monitoring"""
        from utils.query_monitoring import monitor_query
        
        async with monitor_query("create_predictions_table", self._create_predictions_table_impl) as result:
            return result
    
    async def _create_predictions_table_impl(self):
        """Internal implementation of create_predictions_table without monitoring"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id UUID PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    model_name VARCHAR(100) NOT NULL,
                    input_text TEXT NOT NULL,
                    prediction VARCHAR(100) NOT NULL,
                    confidence FLOAT NOT NULL,
                    processing_time_ms FLOAT NOT NULL,
                    from_cache BOOLEAN NOT NULL DEFAULT FALSE,
                    user_id VARCHAR(100),
                    session_id VARCHAR(100),
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
                CREATE INDEX IF NOT EXISTS idx_predictions_model_name ON predictions(model_name);
                CREATE INDEX IF NOT EXISTS idx_predictions_prediction ON predictions(prediction);
                CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id);
                CREATE INDEX IF NOT EXISTS idx_predictions_session_id ON predictions(session_id);
            """)
            logger.info("✅ Predictions table created/verified")
    
    async def log_prediction(
        self,
        model_name: str,
        input_text: str,
        prediction: str,
        confidence: float,
        processing_time_ms: float,
        from_cache: bool = False,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a single prediction (async, non-blocking)"""
        try:
            prediction_log = PredictionLog(
                prediction_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                model_name=model_name,
                input_text=input_text,
                prediction=prediction,
                confidence=confidence,
                processing_time_ms=processing_time_ms,
                from_cache=from_cache,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata
            )
            
            # Add to batch
            self.pending_logs.append(prediction_log)
            
            # Start batch processing if not already running
            if self.batch_task is None or self.batch_task.done():
                self.batch_task = asyncio.create_task(self._process_batch())
            
            logger.debug(f"📝 Prediction logged: {prediction_log.prediction_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to log prediction: {e}")
    
    async def _process_batch(self):
        """Process batch of prediction logs"""
        try:
            # Wait for batch timeout or batch size
            await asyncio.sleep(self.batch_timeout)
            
            if not self.pending_logs:
                return
            
            # Process current batch
            batch = self.pending_logs.copy()
            self.pending_logs.clear()
            
            if not batch:
                return
            
            # Insert batch to database
            await self._insert_batch(batch)
            logger.info(f"📊 Logged {len(batch)} predictions to database")
            
        except Exception as e:
            logger.error(f"❌ Failed to process prediction batch: {e}")
            # Re-add failed logs to pending (with limit to prevent memory issues)
            if len(self.pending_logs) < 1000:
                self.pending_logs.extend(batch)
    
    async def _insert_batch(self, batch: List[PredictionLog]):
        """Insert batch of predictions to database with encryption for sensitive data and query monitoring"""
        from utils.query_monitoring import monitor_query
        
        async with monitor_query("insert_batch", self._insert_batch_impl, batch) as result:
            return result
    
    async def _insert_batch_impl(self, batch: List[PredictionLog]):
        """Internal implementation of insert_batch without monitoring"""
        async with self.pool.acquire() as conn:
            # Prepare encrypted data for each log entry
            encrypted_logs = []
            for log in batch:
                try:
                    # Encrypt sensitive input text
                    encrypted_input = self.encryption.encrypt_data(log.input_text)
                    
                    # Encrypt metadata if it contains sensitive information
                    encrypted_metadata = None
                    if log.metadata:
                        # Check if metadata contains sensitive data
                        sensitive_keys = ['user_id', 'session_id', 'ip_address', 'user_agent']
                        has_sensitive = any(key in log.metadata for key in sensitive_keys)
                        
                        if has_sensitive:
                            encrypted_metadata = self.encryption.encrypt_data(str(log.metadata))
                        else:
                            encrypted_metadata = {"encrypted_data": json.dumps(log.metadata), "encrypted": False}
                    
                    encrypted_logs.append((
                        log.prediction_id,
                        log.timestamp,
                        log.model_name,
                        json.dumps(encrypted_input),  # Store encrypted input as JSON
                        log.prediction,  # Prediction result is not sensitive
                        log.confidence,
                        log.processing_time_ms,
                        log.from_cache,
                        log.user_id,
                        log.session_id,
                        json.dumps(encrypted_metadata) if encrypted_metadata else None
                    ))
                    
                except Exception as e:
                    logger.error(f"❌ Failed to encrypt prediction log {log.prediction_id}: {e}")
                    # Fallback: store without encryption
                    encrypted_logs.append((
                        log.prediction_id,
                        log.timestamp,
                        log.model_name,
                        log.input_text,
                        log.prediction,
                        log.confidence,
                        log.processing_time_ms,
                        log.from_cache,
                        log.user_id,
                        log.session_id,
                        json.dumps(log.metadata) if log.metadata else None
                    ))
            
            await conn.executemany("""
                INSERT INTO predictions (
                    prediction_id, timestamp, model_name, input_text, prediction,
                    confidence, processing_time_ms, from_cache, user_id, session_id, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """, encrypted_logs)
    
    async def get_prediction_stats(
        self,
        model_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """Get prediction statistics with query monitoring"""
        from utils.query_monitoring import monitor_query
        
        async with monitor_query("get_prediction_stats", self._get_prediction_stats_impl, model_name, start_time, end_time, limit) as result:
            return result
    
    async def _get_prediction_stats_impl(
        self,
        model_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """Internal implementation of get_prediction_stats without monitoring"""
        try:
            async with self.pool.acquire() as conn:
                # Build query conditions
                conditions = []
                params = []
                param_count = 0
                
                if model_name:
                    param_count += 1
                    conditions.append(f"model_name = ${param_count}")
                    params.append(model_name)
                
                if start_time:
                    param_count += 1
                    conditions.append(f"timestamp >= ${param_count}")
                    params.append(start_time)
                
                if end_time:
                    param_count += 1
                    conditions.append(f"timestamp <= ${param_count}")
                    params.append(end_time)
                
                where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
                
                # Get basic stats
                stats_query = f"""
                    SELECT 
                        COUNT(*) as total_predictions,
                        COUNT(DISTINCT model_name) as unique_models,
                        AVG(confidence) as avg_confidence,
                        AVG(processing_time_ms) as avg_processing_time_ms,
                        COUNT(CASE WHEN from_cache THEN 1 END) as cache_hits,
                        COUNT(CASE WHEN NOT from_cache THEN 1 END) as cache_misses
                    FROM predictions {where_clause}
                """
                
                stats = await conn.fetchrow(stats_query, *params)
                
                # Get prediction distribution
                distribution_query = f"""
                    SELECT prediction, COUNT(*) as count
                    FROM predictions {where_clause}
                    GROUP BY prediction
                    ORDER BY count DESC
                    LIMIT 10
                """
                
                distribution = await conn.fetch(distribution_query, *params)
                
                # Get recent predictions
                recent_query = f"""
                    SELECT prediction_id, timestamp, model_name, prediction, confidence, processing_time_ms, from_cache
                    FROM predictions {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT $1
                """
                
                recent = await conn.fetch(recent_query, limit, *params)
                
                return {
                    "total_predictions": stats["total_predictions"],
                    "unique_models": stats["unique_models"],
                    "avg_confidence": float(stats["avg_confidence"]) if stats["avg_confidence"] else 0.0,
                    "avg_processing_time_ms": float(stats["avg_processing_time_ms"]) if stats["avg_processing_time_ms"] else 0.0,
                    "cache_hits": stats["cache_hits"],
                    "cache_misses": stats["cache_misses"],
                    "cache_hit_rate": stats["cache_hits"] / max(stats["total_predictions"], 1),
                    "prediction_distribution": [dict(row) for row in distribution],
                    "recent_predictions": [dict(row) for row in recent]
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get prediction stats: {e}")
            return {}
    
    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
            logger.info("✅ Prediction logger closed")

# Global instance
prediction_logger = PredictionLogger()
