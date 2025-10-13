from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
"""
Business Metrics Service
Enterprise-grade metrics collection and analytics for ML operations
"""

import asyncio
import logging
import os
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
        self.conn_pool = None
        self.redis_client = None
        self.metrics_buffer = []
        self.buffer_size = 100
        self.start_time = time.time()
        
    async def initialize(self):
        """Initialize database and Redis connections"""
        try:
            # Initialize PostgreSQL connection pool
            self.conn_pool = await asyncpg.create_pool(self.db_url)
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
        if self.conn_pool:
            await self.conn_pool.close()
        if self.redis_client:
            await self.redis_client.close()
        logger.info("✅ Business Metrics Service closed")
    
    async def _create_tables(self):
        """Create metrics tables"""
        async with self.conn_pool.acquire() as conn:
            # Business metrics table
            await conn.execute("""
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
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_business_metrics_name_time 
                ON business_metrics(metric_name, timestamp);
            """)
            
            await conn.execute("""
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
    
    async def _flush_metrics(self):
        """Flush metrics buffer to database"""
        if not self.metrics_buffer:
            return
        
        try:
            async with self.conn_pool.acquire() as conn:
                await conn.executemany("""
                    INSERT INTO business_metrics (metric_name, value, timestamp, tags, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                """, [
                    (
                        metric.metric_name,
                        metric.value,
                        metric.timestamp,
                        json.dumps(metric.tags),
                        json.dumps(metric.metadata)
                    ) for metric in self.metrics_buffer
                ])
            
            logger.info(f"✅ Flushed {len(self.metrics_buffer)} metrics to database")
            self.metrics_buffer.clear()
            
        except Exception as e:
            logger.error(f"❌ Failed to flush metrics: {e}")
    
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
            
            # Execute single optimized query
            async with self.conn_pool.acquire() as conn:
                result = await conn.fetchrow(cte_query, *params)
                
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
                async with self.conn_pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)