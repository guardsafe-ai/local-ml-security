"""
Analytics Service Client
100% API Coverage for Analytics Service (port 8006)
"""

import logging
from typing import Dict, List, Optional, Any
from .base_client import BaseServiceClient

logger = logging.getLogger(__name__)


class AnalyticsClient(BaseServiceClient):
    """Client for Analytics Service - 100% API Coverage"""
    
    def __init__(self, base_url: str, redis_client):
        super().__init__("analytics", base_url, redis_client)
    
    # =============================================================================
    # HEALTH ENDPOINTS
    # =============================================================================
    
    async def get_health(self) -> Dict[str, Any]:
        """GET /health - Comprehensive health check with dependencies"""
        return await self._make_request("GET", "/health", use_cache=True, cache_ttl=60)
    
    async def get_root(self) -> Dict[str, Any]:
        """GET / - Root endpoint with service status"""
        return await self._make_request("GET", "/", use_cache=True, cache_ttl=60)
    
    # =============================================================================
    # ANALYTICS ENDPOINTS
    # =============================================================================
    
    async def get_performance_trends(self, days: int = 30, model_name: Optional[str] = None,
                                   metric: Optional[str] = None) -> Dict[str, Any]:
        """GET /analytics/trends - Get performance trends over time"""
        params = {
            "days": days,
            "model_name": model_name,
            "metric": metric
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/analytics/trends", params=params, use_cache=True, cache_ttl=300)
    
    async def get_model_comparison(self, model_name: str, days: int = 30) -> Dict[str, Any]:
        """GET /analytics/comparison/{model_name} - Get model comparison data"""
        params = {"days": days}
        return await self._make_request("GET", f"/analytics/comparison/{model_name}", 
                                      params=params, use_cache=True, cache_ttl=300)
    
    async def get_performance_summary(self, time_range: str = "7d") -> Dict[str, Any]:
        """GET /analytics/trends - Get performance trends (used as summary)"""
        # Convert time_range to days for the trends endpoint
        days = 7 if time_range == "7d" else 30 if time_range == "30d" else 7
        return await self._make_request("GET", "/analytics/trends", 
                                      params={"days": days}, use_cache=True, cache_ttl=300)
    
    async def get_drift_analysis(self, model_name: Optional[str] = None,
                               time_range: str = "7d") -> Dict[str, Any]:
        """GET /analytics/drift - Get data and model drift analysis"""
        params = {"model_name": model_name, "time_range": time_range}
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/analytics/drift", params=params, use_cache=True, cache_ttl=300)
    
    # =============================================================================
    # RED TEAM ANALYTICS ENDPOINTS
    # =============================================================================
    
    async def get_red_team_summary(self, days: int = 7) -> Dict[str, Any]:
        """GET /red-team/summary - Get red team testing summary"""
        return await self._make_request("GET", "/red-team/summary", 
                                      params={"days": days}, use_cache=True, cache_ttl=300)
    
    async def get_red_team_trends(self, days: int = 30) -> Dict[str, Any]:
        """GET /red-team/trends - Get red team testing trends"""
        return await self._make_request("GET", "/red-team/trends", 
                                      params={"days": days}, use_cache=True, cache_ttl=300)
    
    async def get_attack_patterns(self, time_range: str = "7d") -> Dict[str, Any]:
        """GET /red-team/patterns - Get attack pattern analysis"""
        return await self._make_request("GET", "/red-team/patterns", 
                                      params={"time_range": time_range}, use_cache=True, cache_ttl=300)
    
    async def get_vulnerability_analysis(self, severity: Optional[str] = None,
                                       time_range: str = "7d") -> Dict[str, Any]:
        """GET /red-team/vulnerabilities - Get vulnerability analysis"""
        params = {"severity": severity, "time_range": time_range}
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/red-team/vulnerabilities", 
                                      params=params, use_cache=True, cache_ttl=300)
    
    # =============================================================================
    # MODEL PERFORMANCE ENDPOINTS
    # =============================================================================
    
    async def get_model_performance(self, model_name: str, time_range: str = "7d") -> Dict[str, Any]:
        """GET /model-performance/{model_name} - Get model performance metrics"""
        return await self._make_request("GET", f"/model-performance/{model_name}",
                                      params={"time_range": time_range}, use_cache=True, cache_ttl=300)
    
    async def get_performance_metrics(self, model_name: Optional[str] = None,
                                    time_range: str = "7d") -> Dict[str, Any]:
        """GET /performance/metrics - Get detailed performance metrics"""
        params = {"model_name": model_name, "time_range": time_range}
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/performance/metrics", 
                                      params=params, use_cache=True, cache_ttl=300)
    
    async def get_accuracy_metrics(self, model_name: Optional[str] = None,
                                 time_range: str = "7d") -> Dict[str, Any]:
        """GET /performance/accuracy - Get accuracy metrics"""
        params = {"model_name": model_name, "time_range": time_range}
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/performance/accuracy", 
                                      params=params, use_cache=True, cache_ttl=300)
    
    async def get_latency_metrics(self, model_name: Optional[str] = None,
                                time_range: str = "7d") -> Dict[str, Any]:
        """GET /performance/latency - Get latency metrics"""
        params = {"model_name": model_name, "time_range": time_range}
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/performance/latency", 
                                      params=params, use_cache=True, cache_ttl=300)
    
    # =============================================================================
    # DRIFT DETECTION ENDPOINTS
    # =============================================================================
    
    async def get_drift_status(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """GET /drift/status - Get drift detection status"""
        params = {"model_name": model_name} if model_name else {}
        return await self._make_request("GET", "/drift/status", params=params, use_cache=True, cache_ttl=60)
    
    async def get_drift_history(self, model_name: Optional[str] = None,
                              time_range: str = "30d") -> Dict[str, Any]:
        """GET /drift/history - Get drift detection history"""
        params = {"model_name": model_name, "time_range": time_range}
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/drift/history", 
                                      params=params, use_cache=True, cache_ttl=300)
    
    async def get_drift_alerts(self, model_name: Optional[str] = None,
                             severity: Optional[str] = None) -> Dict[str, Any]:
        """GET /drift/alerts - Get drift detection alerts"""
        params = {"model_name": model_name, "severity": severity}
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/drift/alerts", 
                                      params=params, use_cache=True, cache_ttl=60)
    
    async def trigger_drift_check(self, model_name: str) -> Dict[str, Any]:
        """POST /drift/check/{model_name} - Manually trigger drift check"""
        return await self._make_request("POST", f"/drift/check/{model_name}")
    
    # =============================================================================
    # AUTO-RETRAIN ENDPOINTS
    # =============================================================================
    
    async def get_auto_retrain_config(self) -> Dict[str, Any]:
        """GET /auto-retrain/config - Get auto-retrain configuration"""
        return await self._make_request("GET", "/auto-retrain/config", use_cache=True, cache_ttl=300)
    
    async def update_auto_retrain_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """PUT /auto-retrain/config - Update auto-retrain configuration"""
        return await self._make_request("PUT", "/auto-retrain/config", data=config)
    
    async def trigger_retrain(self, model_name: str) -> Dict[str, Any]:
        """POST /auto-retrain/trigger/{model_name} - Manually trigger retraining"""
        return await self._make_request("POST", f"/auto-retrain/trigger/{model_name}")
    
    async def get_retrain_status(self, model_name: str) -> Dict[str, Any]:
        """GET /auto-retrain/status/{model_name} - Get retraining status"""
        return await self._make_request("GET", f"/auto-retrain/status/{model_name}")
    
    # =============================================================================
    # COMPREHENSIVE METRICS ENDPOINTS
    # =============================================================================
    
    async def calculate_comprehensive_metrics(self, y_true: List[str], y_pred: List[str],
                                            y_prob: Optional[List[float]] = None,
                                            model_name: str = "unknown") -> Dict[str, Any]:
        """POST /drift/comprehensive-metrics - Calculate comprehensive performance metrics"""
        data = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "model_name": model_name
        }
        return await self._make_request("POST", "/drift/comprehensive-metrics", data=data)
    
    async def get_production_inference_data(self, hours: int = 24,
                                          model_name: Optional[str] = None) -> Dict[str, Any]:
        """GET /drift/data/production-inference - Get production inference data for retraining"""
        params = {"hours": hours, "model_name": model_name}
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/drift/data/production-inference", 
                                      params=params, use_cache=True, cache_ttl=60)
    
    async def detect_production_drift(self, model_name: str, hours: int = 24) -> Dict[str, Any]:
        """POST /drift/drift/production-data - Detect drift using actual production inference data"""
        params = {"model_name": model_name, "hours": hours}
        return await self._make_request("POST", "/drift/drift/production-data", params=params)
    
    # =============================================================================
    # CONVENIENCE METHODS
    # =============================================================================
    
    async def get_overall_health_score(self) -> float:
        """Get overall system health score from analytics"""
        try:
            health = await self.get_health()
            return health.get("drift_detection", {}).get("health_score", 0.0)
        except:
            return 0.0
    
    async def get_drift_detection_rate(self) -> float:
        """Get drift detection rate"""
        try:
            drift_status = await self.get_drift_status()
            return drift_status.get("detection_rate", 0.0)
        except:
            return 0.0
    
    async def get_model_performance_score(self, model_name: str) -> float:
        """Get performance score for a specific model"""
        try:
            performance = await self.get_model_performance(model_name)
            return performance.get("overall_score", 0.0)
        except:
            return 0.0
