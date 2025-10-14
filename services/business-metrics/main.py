from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
"""
Business Metrics Service
Enterprise-grade metrics collection and analytics for ML operations
"""

import asyncio
import logging
import os
import signal
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncpg
import json
import redis.asyncio as redis
from dataclasses import dataclass

# Import tracing setup
import sys
import os
sys.path.append('/app')
from tracing_setup import setup_tracing, get_tracer, trace_request, add_span_attributes
from utils.circuit_breaker import get_database_breaker, get_redis_breaker, get_external_api_breaker
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from database.connection import DatabaseManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class MetricData(BaseModel):
    metric_name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = {}
    metadata: Dict[str, Any] = {}

class BusinessMetricsResponse(BaseModel):
    total_metrics: int
    metrics_by_type: Dict[str, int]
    time_range: str
    data_points: List[Dict[str, Any]]

class SystemHealthResponse(BaseModel):
    status: str
    services: Dict[str, bool]
    uptime_seconds: float
    last_updated: datetime

# Data classes
@dataclass
class MetricPoint:
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    metadata: Dict[str, Any]

class BusinessMetricsService:
    """Enterprise business metrics collection and analytics"""
    
    def __init__(self):
        self.db_url = os.getenv(
            "DATABASE_URL", 
            "postgresql://mlflow:password@postgres:5432/ml_security_consolidated"
        )
        self.redis_url = os.getenv("REDIS_URL", "redis://redis:6379/1")
        self.db_manager = DatabaseManager()
        self.redis_client = None
        self.metrics_buffer = []
        self.buffer_size = 100
        self.start_time = time.time()
        
    async def initialize(self):
        """Initialize database and Redis connections with timeout configuration"""
        try:
            # Initialize database manager with timeout configuration
            await self.db_manager.connect()
            await self._create_tables()
            
            # Initialize Redis client
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            logger.info("✅ Business Metrics Service initialized")
            
        except Exception as e:
            from utils.enhanced_logging import log_error_with_context
            log_error_with_context(
                error=e,
                operation="business_metrics_initialization",
                additional_context={"service": "business-metrics"}
            )
            raise
    
    async def close(self):
        """Close connections"""
        if self.db_manager:
            await self.db_manager.disconnect()
        if self.redis_client:
            await self.redis_client.close()
        logger.info("✅ Business Metrics Service closed")
    
    async def _create_tables(self):
        """Create metrics tables with query monitoring"""
        # Business metrics table
        await self.db_manager.execute_command("""
            CREATE TABLE IF NOT EXISTS business_metrics (
                id SERIAL PRIMARY KEY,
                metric_name VARCHAR(255) NOT NULL,
                value DOUBLE PRECISION NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                tags JSONB,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Metrics summary table for aggregation
        await self.db_manager.execute_command("""
            CREATE TABLE IF NOT EXISTS metrics_summary (
                metric_name VARCHAR(255) NOT NULL,
                time_bucket TIMESTAMPTZ NOT NULL,
                count_metrics INTEGER NOT NULL,
                sum_value DOUBLE PRECISION NOT NULL,
                avg_value DOUBLE PRECISION NOT NULL,
                min_value DOUBLE PRECISION NOT NULL,
                max_value DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (metric_name, time_bucket)
            );
        """)
        
        # Create indexes
        await self.db_manager.execute_command("""
            CREATE INDEX IF NOT EXISTS idx_business_metrics_name_time 
            ON business_metrics(metric_name, timestamp);
        """)
        
        await self.db_manager.execute_command("""
            CREATE INDEX IF NOT EXISTS idx_business_metrics_timestamp 
            ON business_metrics(timestamp);
        """)
        
        logger.info("✅ Business metrics tables created")
    
    async def record_metric(self, metric: MetricData):
        """Record a business metric"""
        try:
            # Add to buffer
            self.metrics_buffer.append(metric)
            
            # Flush buffer if full
            if len(self.metrics_buffer) >= self.buffer_size:
                await self._flush_metrics()
            
            # Also store in Redis for real-time access
            await self._store_in_redis(metric)
            
        except Exception as e:
            from utils.enhanced_logging import log_error_with_context
            log_error_with_context(
                error=e,
                operation="record_business_metric",
                additional_context={"metric_name": metric.metric_name, "value": metric.value}
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((asyncpg.PostgresError, asyncpg.InterfaceError, asyncpg.ConnectionDoesNotExistError))
    )
    async def _flush_metrics(self):
        """Flush metrics buffer to database with transaction management and retry logic"""
        if not self.metrics_buffer:
            return
        
        try:
            # Use database manager transaction for batch metric insertion
            async with self.db_manager.transaction() as conn:
                # Insert all metrics in a single transaction
                for metric in self.metrics_buffer:
                    await conn.execute(
                        """
                        INSERT INTO business_metrics 
                        (metric_name, value, timestamp, tags, metadata)
                        VALUES ($1, $2, $3, $4, $5)
                        """,
                        metric.metric_name,
                        metric.value,
                        metric.timestamp,
                        json.dumps(metric.tags),
                        json.dumps(metric.metadata)
                    )
                
                # Update metrics summary in the same transaction
                await self._update_metrics_summary(conn)
            
            # Clear buffer after successful transaction
            self.metrics_buffer.clear()
            logger.debug(f"✅ Flushed {len(self.metrics_buffer)} metrics to database")
            
        except Exception as e:
            from utils.enhanced_logging import log_error_with_context
            log_error_with_context(
                error=e,
                operation="flush_business_metrics",
                additional_context={"buffer_size": len(self.metrics_buffer)}
            )
            raise
    
    async def _update_metrics_summary(self, conn):
        """Update metrics summary table within transaction"""
        try:
            # Get current time bucket (hourly)
            current_time = datetime.now()
            time_bucket = current_time.replace(minute=0, second=0, microsecond=0)
            
            # Update summary for each unique metric in buffer
            metric_names = list(set(metric.metric_name for metric in self.metrics_buffer))
            
            for metric_name in metric_names:
                # Calculate summary statistics
                metric_values = [m.value for m in self.metrics_buffer if m.metric_name == metric_name]
                
                summary_data = {
                    'count': len(metric_values),
                    'sum': sum(metric_values),
                    'avg': sum(metric_values) / len(metric_values),
                    'min': min(metric_values),
                    'max': max(metric_values)
                }
                
                # Upsert summary
                await conn.execute(
                    """
                    INSERT INTO metrics_summary 
                    (metric_name, time_bucket, count_metrics, sum_value, avg_value, min_value, max_value)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (metric_name, time_bucket)
                    DO UPDATE SET
                        count_metrics = metrics_summary.count_metrics + $3,
                        sum_value = metrics_summary.sum_value + $4,
                        avg_value = (metrics_summary.sum_value + $4) / (metrics_summary.count_metrics + $3),
                        min_value = LEAST(metrics_summary.min_value, $6),
                        max_value = GREATEST(metrics_summary.max_value, $7)
                    """,
                    metric_name, time_bucket, summary_data['count'], 
                    summary_data['sum'], summary_data['avg'], 
                    summary_data['min'], summary_data['max']
                )
                
        except Exception as e:
            logger.error(f"Error updating metrics summary: {e}")
            raise
    
    async def _store_in_redis(self, metric: MetricData):
        """Store metric in Redis for real-time access"""
        try:
            key = f"metrics:{metric.metric_name}:{int(metric.timestamp.timestamp())}"
            data = {
                "value": metric.value,
                "tags": json.dumps(metric.tags),
                "metadata": json.dumps(metric.metadata)
            }
            await self.redis_client.hset(key, mapping=data)
            await self.redis_client.expire(key, 86400)  # 24 hours TTL
            
        except Exception as e:
            logger.error(f"❌ Failed to store metric in Redis: {e}")
    
    async def get_metrics(
        self, 
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> BusinessMetricsResponse:
        """Get business metrics with filtering"""
        try:
            # Build query
            query = "SELECT * FROM business_metrics WHERE 1=1"
            params = []
            param_idx = 1
            
            if metric_name:
                query += f" AND metric_name = ${param_idx}"
                params.append(metric_name)
                param_idx += 1
            
            if start_time:
                query += f" AND timestamp >= ${param_idx}"
                params.append(start_time)
                param_idx += 1
            
            if end_time:
                query += f" AND timestamp <= ${param_idx}"
                params.append(end_time)
                param_idx += 1
            
            # Build optimized single-query CTE for better performance
            cte_query = f"""
                WITH data_cte AS (
                    {query}
                    ORDER BY timestamp DESC
                    LIMIT ${param_idx}
                )
                SELECT 
                    COALESCE(json_agg(
                        json_build_object(
                            'id', id,
                            'metric_name', metric_name,
                            'value', value,
                            'timestamp', timestamp,
                            'tags', tags,
                            'metadata', metadata
                        )
                    ), '[]'::json) as data_points,
                    COALESCE(
                        (SELECT json_object_agg(metric_name, metric_count)
                         FROM (
                             SELECT metric_name, COUNT(*) as metric_count 
                             FROM data_cte 
                             GROUP BY metric_name
                         ) counts),
                        '{{}}'::json
                    ) as metrics_by_type,
                    COUNT(*) as total_count
                FROM data_cte;
            """
            
            params.append(limit)
            
            # Execute single optimized query with query monitoring
            result = await self.db_manager.execute_query(cte_query, *params)
            if result:
                result = result[0]  # Get first row from results
                
                # Parse JSON results from database
                data_points_raw = json.loads(result['data_points']) if result['data_points'] else []
                metrics_by_type = json.loads(result['metrics_by_type']) if result['metrics_by_type'] else {}
                
                # Format data points (convert timestamp objects to ISO strings)
                data_points = []
                for row in data_points_raw:
                    data_point = {
                        "id": row['id'],
                        "metric_name": row['metric_name'],
                        "value": row['value'],
                        "timestamp": row['timestamp'] if isinstance(row['timestamp'], str) else row['timestamp'].isoformat(),
                        "tags": json.loads(row['tags']) if isinstance(row['tags'], str) and row['tags'] else (row.get('tags') or {}),
                        "metadata": json.loads(row['metadata']) if isinstance(row['metadata'], str) and row['metadata'] else (row.get('metadata') or {})
                    }
                    data_points.append(data_point)
                
                # Determine time range
                time_range = "all"
                if start_time and end_time:
                    time_range = f"{start_time.isoformat()} to {end_time.isoformat()}"
                elif start_time:
                    time_range = f"from {start_time.isoformat()}"
                elif end_time:
                    time_range = f"until {end_time.isoformat()}"
                
                return BusinessMetricsResponse(
                    total_metrics=len(data_points),
                    metrics_by_type=metrics_by_type,
                    time_range=time_range,
                    data_points=data_points
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to get metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_system_health(self) -> SystemHealthResponse:
        """Get system health status"""
        try:
            # Check database connection
            db_healthy = False
            try:
                result = await self.db_manager.execute_query("SELECT 1")
                db_healthy = True
            except:
                pass
            
            # Check Redis connection
            redis_healthy = False
            try:
                await self.redis_client.ping()
                redis_healthy = True
            except:
                pass
            
            # Calculate uptime
            uptime = time.time() - self.start_time
            
            return SystemHealthResponse(
                status="healthy" if db_healthy and redis_healthy else "degraded",
                services={
                    "database": db_healthy,
                    "redis": redis_healthy
                },
                uptime_seconds=uptime,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get system health: {e}")
            return SystemHealthResponse(
                status="unhealthy",
                services={"database": False, "redis": False},
                uptime_seconds=0,
                last_updated=datetime.now()
            )

# Initialize service
metrics_service = BusinessMetricsService()

# FastAPI app
app = FastAPI(title="Business Metrics Service", version="1.0.0")

# Setup distributed tracing
setup_tracing("business-metrics", app)

# Initialize circuit breakers
circuit_breakers = {
    "database": get_database_breaker(),
    "redis": get_redis_breaker(),
    "analytics": get_external_api_breaker("analytics"),
    "model_api": get_external_api_breaker("model_api"),
    "training": get_external_api_breaker("training")
}

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    await metrics_service.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down Business Metrics Service...")
    try:
        await metrics_service.close()
        logger.info("✅ Business Metrics Service shutdown completed")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# API Routes
@app.post("/metrics")
async def record_metric(metric: MetricData):
    """Record a business metric"""
    await metrics_service.record_metric(metric)
    return {"status": "success", "message": "Metric recorded"}

@app.get("/metrics", response_model=BusinessMetricsResponse)
async def get_metrics(
    metric_name: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000
):
    """Get business metrics with filtering"""
    return await metrics_service.get_metrics(metric_name, start_time, end_time, limit)

@app.get("/health", response_model=SystemHealthResponse)
async def health_check():
    """Health check endpoint"""
    return await metrics_service.get_system_health()

@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    from fastapi import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Query performance monitoring endpoints
@app.get("/query-performance")
async def get_query_performance():
    """Get query performance metrics"""
    try:
        from utils.query_monitoring import get_business_metrics_query_monitor
        monitor = await get_business_metrics_query_monitor()
        summary = monitor.get_performance_summary()
        
        # Calculate overall statistics
        total_queries = sum(metrics["total_calls"] for metrics in summary.values())
        successful_queries = sum(metrics["successful_calls"] for metrics in summary.values())
        failed_queries = sum(metrics["failed_calls"] for metrics in summary.values())
        timeout_violations = sum(metrics["timeout_violations"] for metrics in summary.values())
        slow_queries = sum(metrics["slow_queries"] for metrics in summary.values())
        
        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "timeout_violations": timeout_violations,
            "slow_queries": slow_queries,
            "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
            "avg_duration_ms": sum(float(metrics["avg_duration_ms"]) for metrics in summary.values()) / len(summary) if summary else 0,
            "max_duration_ms": max(float(metrics["max_duration_ms"]) for metrics in summary.values()) if summary else 0,
            "timeout_threshold_ms": 5000,
            "slow_query_threshold_ms": 1000,
            "query_details": summary
        }
    except Exception as e:
        logger.error(f"Error getting query performance: {e}")
        return {"error": str(e)}

@app.post("/query-performance/log")
async def log_query_performance_endpoint():
    """Log current query performance summary"""
    try:
        from utils.query_monitoring import log_business_metrics_query_performance
        await log_business_metrics_query_performance()
        return {"message": "Query performance logged successfully"}
    except Exception as e:
        logger.error(f"Error logging query performance: {e}")
        return {"error": str(e)}

@app.post("/query-performance/clear")
async def clear_query_metrics():
    """Clear query performance metrics"""
    try:
        from utils.query_monitoring import get_business_metrics_query_monitor
        monitor = await get_business_metrics_query_monitor()
        monitor.clear_metrics()
        return {"message": "Query metrics cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing query metrics: {e}")
        return {"error": str(e)}

@app.get("/metrics/summary")
async def get_metrics_summary():
    """Get metrics summary for dashboard with caching"""
    try:
        # Use caching for this expensive query
        from utils.cache import cached_metrics_summary
        
        @cached_metrics_summary(ttl=300)  # Cache for 5 minutes
        async def _get_metrics_summary():
            # Get last 24 hours of data
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            
            metrics = await metrics_service.get_metrics(
                start_time=start_time,
                end_time=end_time,
                limit=10000
            )
            
            # Calculate summary statistics
            summary = {
                "total_metrics_24h": metrics.total_metrics,
                "metrics_by_type": metrics.metrics_by_type,
                "time_range": metrics.time_range,
                "top_metrics": sorted(
                    metrics.metrics_by_type.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]
            }
            
            return summary
        
        return await _get_metrics_summary()
        
    except Exception as e:
        logger.error(f"❌ Failed to get metrics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Graceful shutdown handler
class GracefulShutdown:
    """Handles graceful shutdown of the business-metrics service"""
    
    def __init__(self, app):
        self.app = app
        self.is_shutting_down = False
        self.shutdown_timeout = 30  # seconds
        self.pending_tasks = set()
        logger.info("GracefulShutdown handler initialized for business-metrics service.")
    
    def register_handlers(self):
        """Register signal handlers for graceful shutdown."""
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self.shutdown_handler)
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, self.shutdown_handler)
        logger.info("Registered SIGTERM and SIGINT handlers for business-metrics service.")
        
        @self.app.on_event("shutdown")
        async def _on_shutdown():
            await self._perform_cleanup()
    
    async def shutdown_handler(self, signum, frame):
        """Handle shutdown signals."""
        if self.is_shutting_down:
            logger.warning("Business-metrics service already in shutdown process, ignoring signal.")
            return
        
        self.is_shutting_down = True
        logger.info(f"Business-metrics service received signal {signum}, initiating graceful shutdown...")
        
        # Trigger FastAPI's shutdown event
        await self.app.shutdown()
    
    async def _perform_cleanup(self):
        """Perform actual cleanup tasks."""
        logger.info("Performing graceful shutdown cleanup for business-metrics service...")
        
        # Cancel any pending background tasks
        if self.pending_tasks:
            logger.info(f"Cancelling {len(self.pending_tasks)} pending background tasks...")
            for task in list(self.pending_tasks):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.debug(f"Task {task.get_name()} cancelled.")
                except Exception as e:
                    logger.error(f"Error during task cancellation: {e}")
            self.pending_tasks.clear()
            logger.info("All pending tasks cancelled.")
        
        # Close database connections
        try:
            if hasattr(metrics_service, 'conn_pool') and metrics_service.conn_pool:
                await metrics_service.conn_pool.close()
                logger.info("Database connection pool closed.")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
        
        # Close Redis connections
        try:
            if hasattr(metrics_service, 'redis_client') and metrics_service.redis_client:
                await metrics_service.redis_client.close()
                logger.info("Redis client closed.")
        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")
        
        logger.info("Business-metrics service graceful shutdown cleanup complete.")

# Initialize graceful shutdown
shutdown_handler = GracefulShutdown(app)
shutdown_handler.register_handlers()

# Circuit breaker management endpoints
@app.get("/circuit-breaker/status")
async def get_circuit_breaker_status():
    """Get circuit breaker status for business metrics service"""
    try:
        states = {}
        for name, breaker in circuit_breakers.items():
            states[name] = breaker.get_state()
        return states
    except Exception as e:
        logger.error(f"Error getting circuit breaker status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/circuit-breaker/reset/{breaker_name}")
async def reset_circuit_breaker(breaker_name: str):
    """Reset a specific circuit breaker"""
    try:
        if breaker_name in circuit_breakers:
            circuit_breakers[breaker_name].reset()
            return {"message": f"Circuit breaker '{breaker_name}' reset successfully"}
        else:
            raise ValueError(f"Circuit breaker '{breaker_name}' not found")
    except Exception as e:
        logger.error(f"Error resetting circuit breaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/circuit-breaker/reset-all")
async def reset_all_circuit_breakers():
    """Reset all circuit breakers"""
    try:
        for breaker in circuit_breakers.values():
            breaker.reset()
        return {"message": "All circuit breakers reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting circuit breakers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await shutdown_handler._perform_cleanup()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)