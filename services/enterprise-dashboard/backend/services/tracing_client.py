"""
Tracing Service Client
100% API Coverage for Tracing Service (port 8009)
"""

import logging
from typing import Dict, List, Optional, Any
from .base_client import BaseServiceClient

logger = logging.getLogger(__name__)


class TracingClient(BaseServiceClient):
    """Client for Tracing Service - 100% API Coverage"""
    
    def __init__(self, base_url: str, redis_client):
        super().__init__("tracing", base_url, redis_client)
    
    # =============================================================================
    # HEALTH ENDPOINTS
    # =============================================================================
    
    async def get_health(self) -> Dict[str, Any]:
        """GET /health - Health check for tracing service"""
        return await self._make_request("GET", "/health", use_cache=True, cache_ttl=60)
    
    async def get_root(self) -> Dict[str, Any]:
        """GET / - Root endpoint with service status"""
        return await self._make_request("GET", "/", use_cache=True, cache_ttl=60)
    
    # =============================================================================
    # TRACE MANAGEMENT ENDPOINTS
    # =============================================================================
    
    async def create_trace(self, trace_id: str, operation_name: str,
                         service_name: str, start_time: Optional[str] = None,
                         tags: Optional[Dict[str, Any]] = None,
                         parent_trace_id: Optional[str] = None) -> Dict[str, Any]:
        """POST /traces - Create a new trace"""
        data = {
            "trace_id": trace_id,
            "operation_name": operation_name,
            "service_name": service_name,
            "start_time": start_time,
            "tags": tags or {},
            "parent_trace_id": parent_trace_id
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", "/traces", data=data)
    
    async def get_trace(self, trace_id: str) -> Dict[str, Any]:
        """GET /traces/{trace_id} - Get trace by ID"""
        return await self._make_request("GET", f"/traces/{trace_id}", use_cache=True, cache_ttl=300)
    
    async def get_traces(self, service_name: Optional[str] = None,
                        operation_name: Optional[str] = None,
                        start_time: Optional[str] = None,
                        end_time: Optional[str] = None,
                        limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """GET /traces - Get traces with filtering"""
        params = {
            "service_name": service_name,
            "operation_name": operation_name,
            "start_time": start_time,
            "end_time": end_time,
            "limit": limit,
            "offset": offset
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/traces", 
                                      params=params, use_cache=True, cache_ttl=60)
    
    async def update_trace(self, trace_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """PUT /traces/{trace_id} - Update trace information"""
        return await self._make_request("PUT", f"/traces/{trace_id}", data=updates)
    
    async def finish_trace(self, trace_id: str, end_time: Optional[str] = None,
                          status: str = "success", error_message: Optional[str] = None) -> Dict[str, Any]:
        """POST /traces/{trace_id}/finish - Finish a trace"""
        data = {
            "end_time": end_time,
            "status": status,
            "error_message": error_message
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", f"/traces/{trace_id}/finish", data=data)
    
    # =============================================================================
    # SPAN MANAGEMENT ENDPOINTS
    # =============================================================================
    
    async def create_span(self, trace_id: str, span_id: str, operation_name: str,
                         start_time: Optional[str] = None, tags: Optional[Dict[str, Any]] = None,
                         parent_span_id: Optional[str] = None) -> Dict[str, Any]:
        """POST /traces/{trace_id}/spans - Create a new span"""
        data = {
            "span_id": span_id,
            "operation_name": operation_name,
            "start_time": start_time,
            "tags": tags or {},
            "parent_span_id": parent_span_id
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", f"/traces/{trace_id}/spans", data=data)
    
    async def get_span(self, trace_id: str, span_id: str) -> Dict[str, Any]:
        """GET /traces/{trace_id}/spans/{span_id} - Get span by ID"""
        return await self._make_request("GET", f"/traces/{trace_id}/spans/{span_id}", 
                                      use_cache=True, cache_ttl=300)
    
    async def get_spans(self, trace_id: str) -> Dict[str, Any]:
        """GET /traces/{trace_id}/spans - Get all spans for a trace"""
        return await self._make_request("GET", f"/traces/{trace_id}/spans", 
                                      use_cache=True, cache_ttl=300)
    
    async def update_span(self, trace_id: str, span_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """PUT /traces/{trace_id}/spans/{span_id} - Update span information"""
        return await self._make_request("PUT", f"/traces/{trace_id}/spans/{span_id}", data=updates)
    
    async def finish_span(self, trace_id: str, span_id: str, end_time: Optional[str] = None,
                         status: str = "success", error_message: Optional[str] = None) -> Dict[str, Any]:
        """POST /traces/{trace_id}/spans/{span_id}/finish - Finish a span"""
        data = {
            "end_time": end_time,
            "status": status,
            "error_message": error_message
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", f"/traces/{trace_id}/spans/{span_id}/finish", data=data)
    
    # =============================================================================
    # METRICS AND ANALYTICS ENDPOINTS
    # =============================================================================
    
    async def get_trace_metrics(self, time_range: str = "24h",
                               service_name: Optional[str] = None) -> Dict[str, Any]:
        """GET /metrics/traces - Get trace metrics and statistics"""
        params = {"time_range": time_range, "service_name": service_name}
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/metrics/traces", 
                                      params=params, use_cache=True, cache_ttl=60)
    
    async def get_performance_metrics(self, time_range: str = "24h",
                                    service_name: Optional[str] = None) -> Dict[str, Any]:
        """GET /metrics/performance - Get performance metrics"""
        params = {"time_range": time_range, "service_name": service_name}
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/metrics/performance", 
                                      params=params, use_cache=True, cache_ttl=60)
    
    async def get_error_metrics(self, time_range: str = "24h",
                              service_name: Optional[str] = None) -> Dict[str, Any]:
        """GET /metrics/errors - Get error metrics and statistics"""
        params = {"time_range": time_range, "service_name": service_name}
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/metrics/errors", 
                                      params=params, use_cache=True, cache_ttl=60)
    
    async def get_latency_metrics(self, time_range: str = "24h",
                                service_name: Optional[str] = None) -> Dict[str, Any]:
        """GET /metrics/latency - Get latency metrics"""
        params = {"time_range": time_range, "service_name": service_name}
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/metrics/latency", 
                                      params=params, use_cache=True, cache_ttl=60)
    
    # =============================================================================
    # SERVICE DISCOVERY ENDPOINTS
    # =============================================================================
    
    async def get_services(self) -> Dict[str, Any]:
        """GET /services - Get list of traced services"""
        return await self._make_request("GET", "/services", use_cache=True, cache_ttl=300)
    
    async def get_service_operations(self, service_name: str) -> Dict[str, Any]:
        """GET /services/{service_name}/operations - Get operations for a service"""
        return await self._make_request("GET", f"/services/{service_name}/operations", 
                                      use_cache=True, cache_ttl=300)
    
    async def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """GET /services/{service_name}/health - Get health status for a service"""
        return await self._make_request("GET", f"/services/{service_name}/health", 
                                      use_cache=True, cache_ttl=60)
    
    # =============================================================================
    # DASHBOARD AND VISUALIZATION ENDPOINTS
    # =============================================================================
    
    async def get_dashboard_data(self, time_range: str = "24h") -> Dict[str, Any]:
        """GET /dashboard - Get dashboard data for tracing visualization"""
        return await self._make_request("GET", "/dashboard", 
                                      params={"time_range": time_range}, use_cache=True, cache_ttl=60)
    
    async def get_service_map(self, time_range: str = "24h") -> Dict[str, Any]:
        """GET /service-map - Get service dependency map"""
        return await self._make_request("GET", "/service-map", 
                                      params={"time_range": time_range}, use_cache=True, cache_ttl=300)
    
    async def get_trace_timeline(self, trace_id: str) -> Dict[str, Any]:
        """GET /traces/{trace_id}/timeline - Get timeline view of a trace"""
        return await self._make_request("GET", f"/traces/{trace_id}/timeline", 
                                      use_cache=True, cache_ttl=300)
    
    # =============================================================================
    # SEARCH AND FILTERING ENDPOINTS
    # =============================================================================
    
    async def search_traces(self, query: str, service_name: Optional[str] = None,
                          operation_name: Optional[str] = None,
                          start_time: Optional[str] = None,
                          end_time: Optional[str] = None,
                          limit: int = 100) -> Dict[str, Any]:
        """POST /search/traces - Search traces with complex queries"""
        data = {
            "query": query,
            "service_name": service_name,
            "operation_name": operation_name,
            "start_time": start_time,
            "end_time": end_time,
            "limit": limit
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", "/search/traces", data=data)
    
    async def get_trace_dependencies(self, trace_id: str) -> Dict[str, Any]:
        """GET /traces/{trace_id}/dependencies - Get trace dependency graph"""
        return await self._make_request("GET", f"/traces/{trace_id}/dependencies", 
                                      use_cache=True, cache_ttl=300)
    
    # =============================================================================
    # CONVENIENCE METHODS
    # =============================================================================
    
    async def get_active_traces(self) -> List[Dict[str, Any]]:
        """Get currently active traces"""
        try:
            traces = await self.get_traces()
            return [trace for trace in traces.get("traces", []) if trace.get("status") == "active"]
        except:
            return []
    
    async def get_error_traces(self, time_range: str = "24h") -> List[Dict[str, Any]]:
        """Get traces with errors"""
        try:
            traces = await self.get_traces()
            return [trace for trace in traces.get("traces", []) if trace.get("status") == "error"]
        except:
            return []
    
    async def get_service_performance_score(self, service_name: str) -> float:
        """Get performance score for a service"""
        try:
            metrics = await self.get_performance_metrics(service_name=service_name)
            return metrics.get("performance_score", 0.0)
        except:
            return 0.0
    
    async def get_error_rate(self, service_name: Optional[str] = None) -> float:
        """Get error rate for a service or overall"""
        try:
            metrics = await self.get_error_metrics(service_name=service_name)
            return metrics.get("error_rate", 0.0)
        except:
            return 0.0
