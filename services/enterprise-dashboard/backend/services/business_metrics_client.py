"""
Business Metrics Service Client
100% API Coverage for Business Metrics Service (port 8004)
"""

import logging
from typing import Dict, List, Optional, Any
from .base_client import BaseServiceClient

logger = logging.getLogger(__name__)


class BusinessMetricsClient(BaseServiceClient):
    """Client for Business Metrics Service - 100% API Coverage"""
    
    def __init__(self, base_url: str, redis_client):
        super().__init__("business_metrics", base_url, redis_client)
    
    # =============================================================================
    # HEALTH ENDPOINTS
    # =============================================================================
    
    async def get_health(self) -> Dict[str, Any]:
        """GET /health - Comprehensive health check with system status"""
        return await self._make_request("GET", "/health", use_cache=True, cache_ttl=60)
    
    async def get_root(self) -> Dict[str, Any]:
        """GET / - Root endpoint with service status"""
        return await self._make_request("GET", "/", use_cache=True, cache_ttl=60)
    
    # =============================================================================
    # METRICS COLLECTION ENDPOINTS
    # =============================================================================
    
    async def record_metric(self, metric_name: str, value: float, unit: str = "count",
                          category: str = "performance", service: str = "unknown",
                          model_name: Optional[str] = None, user_id: Optional[str] = None,
                          session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                          timestamp: Optional[str] = None) -> Dict[str, Any]:
        """POST /metrics - Record a single business metric"""
        data = {
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            "category": category,
            "service": service,
            "model_name": model_name,
            "user_id": user_id,
            "session_id": session_id,
            "metadata": metadata or {},
            "timestamp": timestamp
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", "/metrics", data=data)
    
    async def record_batch_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """POST /metrics/batch - Record multiple metrics in batch"""
        data = {"metrics": metrics}
        return await self._make_request("POST", "/metrics/batch", data=data)
    
    # =============================================================================
    # METRICS RETRIEVAL ENDPOINTS
    # =============================================================================
    
    async def get_metrics(self, metric_name: Optional[str] = None,
                         category: Optional[str] = None, service: Optional[str] = None,
                         model_name: Optional[str] = None, start_time: Optional[str] = None,
                         end_time: Optional[str] = None, aggregation: Optional[str] = None,
                         group_by: Optional[str] = None, limit: int = 1000,
                         offset: int = 0) -> Dict[str, Any]:
        """GET /metrics - Retrieve metrics with filtering and aggregation"""
        params = {
            "metric_name": metric_name,
            "category": category,
            "service": service,
            "model_name": model_name,
            "start_time": start_time,
            "end_time": end_time,
            "aggregation": aggregation,
            "group_by": group_by,
            "limit": limit,
            "offset": offset
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/metrics", params=params, use_cache=True, cache_ttl=60)
    
    async def get_metric_summary(self, metric_name: str, time_range: str = "24h") -> Dict[str, Any]:
        """GET /metrics/summary/{metric_name} - Get summary statistics for a specific metric"""
        return await self._make_request("GET", f"/metrics/summary/{metric_name}", 
                                      params={"time_range": time_range}, use_cache=True, cache_ttl=60)
    
    async def get_metrics_by_category(self, category: str, time_range: str = "24h") -> Dict[str, Any]:
        """GET /metrics/category/{category} - Get metrics by category"""
        return await self._make_request("GET", f"/metrics/category/{category}",
                                      params={"time_range": time_range}, use_cache=True, cache_ttl=60)
    
    async def get_metrics_by_service(self, service: str, time_range: str = "24h") -> Dict[str, Any]:
        """GET /metrics/service/{service} - Get metrics by service"""
        return await self._make_request("GET", f"/metrics/service/{service}",
                                      params={"time_range": time_range}, use_cache=True, cache_ttl=60)
    
    # =============================================================================
    # KPI ENDPOINTS
    # =============================================================================
    
    async def get_business_kpis(self, time_range_days: int = 30,
                               include_recommendations: bool = True) -> Dict[str, Any]:
        """GET /kpis - Get business KPIs and key performance indicators"""
        params = {
            "time_range_days": time_range_days,
            "include_recommendations": include_recommendations
        }
        return await self._make_request("GET", "/api/v1/kpis", params=params, use_cache=True, cache_ttl=300)
    
    async def get_cost_metrics(self, time_range: str = "30d") -> Dict[str, Any]:
        """GET /costs - Get cost analysis and financial metrics"""
        return await self._make_request("GET", "/costs", params={"time_range": time_range}, 
                                      use_cache=True, cache_ttl=300)
    
    async def get_revenue_metrics(self, time_range: str = "30d") -> Dict[str, Any]:
        """GET /revenue - Get revenue and business value metrics"""
        return await self._make_request("GET", "/revenue", params={"time_range": time_range},
                                      use_cache=True, cache_ttl=300)
    
    async def get_efficiency_metrics(self, time_range: str = "30d") -> Dict[str, Any]:
        """GET /efficiency - Get operational efficiency metrics"""
        return await self._make_request("GET", "/efficiency", params={"time_range": time_range},
                                      use_cache=True, cache_ttl=300)
    
    # =============================================================================
    # PERFORMANCE METRICS ENDPOINTS
    # =============================================================================
    
    async def get_performance_metrics(self, time_range: str = "24h") -> Dict[str, Any]:
        """GET /performance - Get system performance metrics"""
        return await self._make_request("GET", "/performance", params={"time_range": time_range},
                                      use_cache=True, cache_ttl=60)
    
    async def get_throughput_metrics(self, time_range: str = "24h") -> Dict[str, Any]:
        """GET /throughput - Get throughput and capacity metrics"""
        return await self._make_request("GET", "/throughput", params={"time_range": time_range},
                                      use_cache=True, cache_ttl=60)
    
    async def get_latency_metrics(self, time_range: str = "24h") -> Dict[str, Any]:
        """GET /latency - Get latency and response time metrics"""
        return await self._make_request("GET", "/latency", params={"time_range": time_range},
                                      use_cache=True, cache_ttl=60)
    
    # =============================================================================
    # SECURITY METRICS ENDPOINTS
    # =============================================================================
    
    async def get_security_metrics(self, time_range: str = "24h") -> Dict[str, Any]:
        """GET /security - Get security-related metrics"""
        return await self._make_request("GET", "/security", params={"time_range": time_range},
                                      use_cache=True, cache_ttl=60)
    
    async def get_threat_metrics(self, time_range: str = "24h") -> Dict[str, Any]:
        """GET /threats - Get threat detection and response metrics"""
        return await self._make_request("GET", "/threats", params={"time_range": time_range},
                                      use_cache=True, cache_ttl=60)
    
    async def get_compliance_metrics(self, time_range: str = "24h") -> Dict[str, Any]:
        """GET /compliance - Get compliance and regulatory metrics"""
        return await self._make_request("GET", "/compliance", params={"time_range": time_range},
                                      use_cache=True, cache_ttl=60)
    
    # =============================================================================
    # ANALYTICS ENDPOINTS
    # =============================================================================
    
    async def get_trend_analysis(self, metric_name: str, time_range: str = "30d") -> Dict[str, Any]:
        """GET /analytics/trends/{metric_name} - Get trend analysis for a metric"""
        return await self._make_request("GET", f"/analytics/trends/{metric_name}",
                                      params={"time_range": time_range}, use_cache=True, cache_ttl=300)
    
    async def get_correlation_analysis(self, metric1: str, metric2: str,
                                     time_range: str = "30d") -> Dict[str, Any]:
        """GET /analytics/correlation - Get correlation analysis between metrics"""
        params = {"metric1": metric1, "metric2": metric2, "time_range": time_range}
        return await self._make_request("GET", "/analytics/correlation", params=params,
                                      use_cache=True, cache_ttl=300)
    
    async def get_anomaly_detection(self, metric_name: str, time_range: str = "7d") -> Dict[str, Any]:
        """GET /analytics/anomalies/{metric_name} - Get anomaly detection results"""
        return await self._make_request("GET", f"/analytics/anomalies/{metric_name}",
                                      params={"time_range": time_range}, use_cache=True, cache_ttl=60)
    
    # =============================================================================
    # REPORTING ENDPOINTS
    # =============================================================================
    
    async def generate_report(self, report_type: str, time_range: str = "30d",
                            format: str = "json", include_charts: bool = True) -> Dict[str, Any]:
        """POST /reports/generate - Generate comprehensive business reports"""
        data = {
            "report_type": report_type,
            "time_range": time_range,
            "format": format,
            "include_charts": include_charts
        }
        return await self._make_request("POST", "/reports/generate", data=data)
    
    async def get_report_status(self, report_id: str) -> Dict[str, Any]:
        """GET /reports/{report_id}/status - Get report generation status"""
        return await self._make_request("GET", f"/reports/{report_id}/status")
    
    async def download_report(self, report_id: str) -> bytes:
        """GET /reports/{report_id}/download - Download generated report"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=self.long_timeout) as client:
                response = await client.get(f"{self.base_url}/reports/{report_id}/download")
                return response.content
        except Exception as e:
            logger.error(f"Failed to download report {report_id}: {e}")
            raise
    
    # =============================================================================
    # EXPORT ENDPOINTS
    # =============================================================================
    
    async def export_metrics(self, metric_names: List[str], start_time: str, end_time: str,
                           format: str = "csv") -> bytes:
        """POST /metrics/export - Export metrics data"""
        data = {
            "metric_names": metric_names,
            "start_time": start_time,
            "end_time": end_time,
            "format": format
        }
        try:
            import httpx
            async with httpx.AsyncClient(timeout=self.long_timeout) as client:
                response = await client.post(f"{self.base_url}/metrics/export", json=data)
                return response.content
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            raise
    
    # =============================================================================
    # CONVENIENCE METHODS
    # =============================================================================
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get overall metrics summary"""
        return await self._make_request("GET", "/summary", use_cache=True, cache_ttl=60)
    
    async def get_system_health_score(self) -> float:
        """Get overall system health score"""
        try:
            health = await self.get_health()
            return health.get("performance_metrics", {}).get("health_score", 0.0)
        except:
            return 0.0
    
    async def get_total_metrics_count(self) -> int:
        """Get total number of metrics collected"""
        try:
            summary = await self.get_metrics_summary()
            return summary.get("total_metrics", 0)
        except:
            return 0
