"""
Enterprise Dashboard Backend - Monitoring Routes
System monitoring and alerting endpoints
"""

import logging
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from models.responses import MonitoringAlert, MonitoringMetrics
from services.api_client import APIClient

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize API client
api_client = APIClient()


@router.get("/alerts")
async def get_monitoring_alerts():
    """Get system monitoring alerts"""
    try:
        # This would typically get alerts from monitoring service
        return {
            "alerts": [],
            "count": 0,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get monitoring alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def get_monitoring_logs():
    """Get system monitoring logs"""
    try:
        # This would typically get logs from monitoring service
        return {
            "logs": [],
            "count": 0,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get monitoring logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_monitoring_metrics():
    """Get system monitoring metrics"""
    try:
        metrics = await api_client.get_monitoring_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to get monitoring metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/circuit-breakers")
async def get_circuit_breaker_status():
    """Get circuit breaker status for all services"""
    try:
        # Get circuit breaker status from all service clients
        circuit_breakers = {}
        
        # Get status from each service client
        for service_name, client in api_client.services.items():
            if hasattr(client, 'get_circuit_breaker_status'):
                circuit_breakers[service_name] = client.get_circuit_breaker_status()
        
        return {
            "circuit_breakers": circuit_breakers,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get circuit breaker status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/detailed")
async def get_detailed_health():
    """Get detailed health status including circuit breakers"""
    try:
        # Get basic health status
        health_data = await api_client.get_all_services_health()
        
        # Get circuit breaker status
        circuit_breakers = {}
        for service_name, client in api_client.services.items():
            if hasattr(client, 'get_circuit_breaker_status'):
                circuit_breakers[service_name] = client.get_circuit_breaker_status()
        
        return {
            "health": health_data,
            "circuit_breakers": circuit_breakers,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get detailed health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_performance_metrics():
    """Get performance metrics for all services"""
    try:
        # Use the main API client's performance metrics method
        result = await api_client.get_performance_metrics()
        return result
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def get_cache_statistics():
    """Get cache statistics for all services"""
    try:
        # Get performance metrics from main API client
        perf_metrics = await api_client.get_performance_metrics()
        service_metrics = perf_metrics.get('performance_metrics', {})
        
        # Extract cache statistics
        cache_stats = {}
        total_items = 0
        total_memory = 0
        
        for service_name, metrics in service_metrics.items():
            cache_info = metrics.get('cache', {})
            cache_stats[service_name] = cache_info
            total_items += cache_info.get('total_cached_items', 0)
            total_memory += cache_info.get('estimated_memory_usage_bytes', 0)
        
        return {
            "service_cache_stats": cache_stats,
            "total_cached_items": total_items,
            "total_memory_usage_bytes": total_memory,
            "total_memory_usage_mb": round(total_memory / (1024 * 1024), 2),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get cache statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
