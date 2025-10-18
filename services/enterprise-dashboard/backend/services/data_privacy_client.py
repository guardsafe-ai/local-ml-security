"""
Data Privacy Service Client
100% API Coverage for Data Privacy Service (port 8008)
"""

import logging
import httpx
from datetime import datetime
from typing import Dict, List, Optional, Any
from .base_client import BaseServiceClient

logger = logging.getLogger(__name__)


class DataPrivacyClient(BaseServiceClient):
    """Client for Data Privacy Service - 100% API Coverage"""
    
    def __init__(self, base_url: str, redis_client):
        super().__init__("data_privacy", base_url, redis_client)
    
    # =============================================================================
    # HEALTH ENDPOINTS
    # =============================================================================
    
    async def get_health(self) -> Dict[str, Any]:
        """GET /health - Comprehensive health check with privacy metrics"""
        return await self._make_request("GET", "/health", use_cache=True, cache_ttl=60)
    
    async def get_health_health(self) -> Dict[str, Any]:
        """GET /health - Alternative health endpoint (same as main health)"""
        return await self._make_request("GET", "/health", use_cache=True, cache_ttl=60)
    
    async def health_check(self) -> Dict[str, Any]:
        """Custom health check for data privacy service"""
        try:
            # Try the main health endpoint
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "name": self.service_name,
                        "status": "healthy",
                        "response_time": 0.0,
                        "last_check": datetime.now(),
                        "details": data
                    }
                else:
                    return {
                        "name": self.service_name,
                        "status": "unhealthy",
                        "response_time": 0.0,
                        "last_check": datetime.now(),
                        "details": {"error": f"Health check returned status {response.status_code}"}
                    }
        except Exception as e:
            logger.error(f"Health check failed for {self.service_name}: {e}")
            return {
                "name": self.service_name,
                "status": "unhealthy",
                "response_time": 0.0,
                "last_check": datetime.now(),
                "details": {"error": str(e)}
            }
    
    async def get_root(self) -> Dict[str, Any]:
        """GET / - Root endpoint with service status"""
        return await self._make_request("GET", "/", use_cache=True, cache_ttl=60)
    
    # =============================================================================
    # DATA CLASSIFICATION ENDPOINTS
    # =============================================================================
    
    async def classify_data(self, data: Dict[str, Any], data_id: str,
                          context: Optional[str] = None, user_id: Optional[str] = None,
                          session_id: Optional[str] = None) -> Dict[str, Any]:
        """POST /classify - Classify data for privacy compliance"""
        request_data = {
            "data": data,
            "data_id": data_id,
            "context": context,
            "user_id": user_id,
            "session_id": session_id
        }
        # Remove None values
        request_data = {k: v for k, v in request_data.items() if v is not None}
        return await self._make_request("POST", "/classify", data=request_data)
    
    async def get_classification_history(self, data_id: Optional[str] = None,
                                       user_id: Optional[str] = None,
                                       time_range: str = "7d") -> Dict[str, Any]:
        """GET /classify/history - Get data classification history"""
        params = {"data_id": data_id, "user_id": user_id, "time_range": time_range}
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/classify/history", 
                                      params=params, use_cache=True, cache_ttl=300)
    
    async def get_classification_summary(self, time_range: str = "7d") -> Dict[str, Any]:
        """GET /compliance-report - Get data privacy compliance report as summary"""
        return await self._make_request("GET", "/compliance-report", 
                                      params={"time_range": time_range}, use_cache=True, cache_ttl=300)
    
    # =============================================================================
    # DATA ANONYMIZATION ENDPOINTS
    # =============================================================================
    
    async def anonymize_data(self, text: str, anonymization_level: str = "medium",
                           preserve_format: bool = True) -> Dict[str, Any]:
        """POST /anonymize - Anonymize text data"""
        data = {
            "text": text,
            "anonymization_level": anonymization_level,
            "preserve_format": preserve_format
        }
        return await self._make_request("POST", "/anonymize", data=data)
    
    async def batch_anonymize(self, texts: List[str], anonymization_level: str = "medium",
                            preserve_format: bool = True) -> Dict[str, Any]:
        """POST /anonymize/batch - Batch anonymize multiple texts"""
        data = {
            "texts": texts,
            "anonymization_level": anonymization_level,
            "preserve_format": preserve_format
        }
        return await self._make_request("POST", "/anonymize/batch", data=data)
    
    async def get_anonymization_stats(self, time_range: str = "7d") -> Dict[str, Any]:
        """GET /anonymize/stats - Get anonymization statistics"""
        return await self._make_request("GET", "/anonymize/stats", 
                                      params={"time_range": time_range}, use_cache=True, cache_ttl=300)
    
    # =============================================================================
    # DATA SUBJECT MANAGEMENT ENDPOINTS
    # =============================================================================
    
    async def register_data_subject(self, subject_id: str, email: Optional[str] = None,
                                  data_categories: Optional[List[str]] = None,
                                  retention_days: Optional[int] = None) -> Dict[str, Any]:
        """POST /data-subjects - Register a new data subject"""
        data = {
            "subject_id": subject_id,
            "email": email,
            "data_categories": data_categories or [],
            "retention_days": retention_days
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", "/data-subjects", data=data)
    
    async def get_data_subjects(self, subject_id: Optional[str] = None,
                              email: Optional[str] = None, limit: int = 100,
                              offset: int = 0) -> Dict[str, Any]:
        """GET /data-subjects - Get data subjects with filtering"""
        params = {
            "subject_id": subject_id,
            "email": email,
            "limit": limit,
            "offset": offset
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/data-subjects", 
                                      params=params, use_cache=True, cache_ttl=300)
    
    async def get_data_subject(self, subject_id: str) -> Dict[str, Any]:
        """GET /data-subjects/{subject_id} - Get specific data subject"""
        return await self._make_request("GET", f"/data-subjects/{subject_id}", use_cache=True, cache_ttl=300)
    
    async def update_data_subject(self, subject_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """PUT /data-subjects/{subject_id} - Update data subject information"""
        return await self._make_request("PUT", f"/data-subjects/{subject_id}", data=updates)
    
    async def delete_data_subject(self, subject_id: str) -> Dict[str, Any]:
        """DELETE /data-subjects/{subject_id} - Delete data subject (right to be forgotten)"""
        return await self._make_request("DELETE", f"/data-subjects/{subject_id}")
    
    # =============================================================================
    # CONSENT MANAGEMENT ENDPOINTS
    # =============================================================================
    
    async def record_consent(self, user_id: str, data_types: List[str], purposes: List[str],
                           consent_type: str = "explicit", consent_method: str = "web_form",
                           ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """POST /consent/record - Record user consent"""
        data = {
            "user_id": user_id,
            "data_types": data_types,
            "purposes": purposes,
            "consent_type": consent_type,
            "consent_method": consent_method,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "metadata": metadata or {}
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", "/consent/record", data=data)
    
    async def get_consent_status(self, user_id: str) -> Dict[str, Any]:
        """GET /consent/status/{user_id} - Get consent status for a specific user"""
        return await self._make_request("GET", f"/consent/status/{user_id}", use_cache=True, cache_ttl=300)
    
    async def withdraw_consent(self, user_id: str, data_types: Optional[List[str]] = None,
                             purposes: Optional[List[str]] = None, reason: Optional[str] = None) -> Dict[str, Any]:
        """POST /consent/withdraw - Withdraw user consent"""
        data = {
            "user_id": user_id,
            "data_types": data_types,
            "purposes": purposes,
            "reason": reason
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", "/consent/withdraw", data=data)
    
    async def get_consent_statistics(self, time_range: str = "30d") -> Dict[str, Any]:
        """GET /consent/statistics - Get consent statistics and compliance metrics"""
        return await self._make_request("GET", "/consent/statistics", 
                                      params={"time_range": time_range}, use_cache=True, cache_ttl=300)
    
    # =============================================================================
    # DATA RETENTION ENDPOINTS
    # =============================================================================
    
    async def get_retention_policies(self) -> Dict[str, Any]:
        """GET /retention/policies - Get data retention policies"""
        return await self._make_request("GET", "/retention/policies", use_cache=True, cache_ttl=300)
    
    async def create_retention_policy(self, data_type: str, retention_days: int,
                                    auto_delete: bool = True, description: Optional[str] = None) -> Dict[str, Any]:
        """POST /retention/policies - Create a new retention policy"""
        data = {
            "data_type": data_type,
            "retention_days": retention_days,
            "auto_delete": auto_delete,
            "description": description
        }
        return await self._make_request("POST", "/retention/policies", data=data)
    
    async def get_retention_status(self, data_type: Optional[str] = None) -> Dict[str, Any]:
        """GET /retention/status - Get data retention status"""
        params = {"data_type": data_type} if data_type else {}
        return await self._make_request("GET", "/retention/status", 
                                      params=params, use_cache=True, cache_ttl=300)
    
    async def trigger_cleanup(self, data_type: Optional[str] = None,
                            dry_run: bool = True) -> Dict[str, Any]:
        """POST /retention/cleanup - Trigger data cleanup based on retention policies"""
        data = {"data_type": data_type, "dry_run": dry_run}
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", "/retention/cleanup", data=data)
    
    # =============================================================================
    # AUDIT LOGGING ENDPOINTS
    # =============================================================================
    
    async def get_audit_logs(self, user_id: Optional[str] = None, action: Optional[str] = None,
                           start_date: Optional[str] = None, end_date: Optional[str] = None,
                           limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """GET /audit/logs - Get audit logs with filtering"""
        params = {
            "user_id": user_id,
            "action": action,
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
            "offset": offset
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/audit/logs", 
                                      params=params, use_cache=True, cache_ttl=60)
    
    async def create_audit_log(self, action: str, resource: str, user_id: Optional[str] = None,
                             details: Optional[Dict[str, Any]] = None,
                             ip_address: Optional[str] = None,
                             user_agent: Optional[str] = None) -> Dict[str, Any]:
        """POST /audit/logs - Create a new audit log entry"""
        data = {
            "action": action,
            "resource": resource,
            "user_id": user_id,
            "details": details or {},
            "ip_address": ip_address,
            "user_agent": user_agent
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", "/audit/logs", data=data)
    
    # =============================================================================
    # COMPLIANCE ENDPOINTS
    # =============================================================================
    
    async def get_compliance_status(self, regulation: str = "GDPR") -> Dict[str, Any]:
        """GET /compliance/status - Get compliance status for regulations"""
        return await self._make_request("GET", "/compliance/status", 
                                      params={"regulation": regulation}, use_cache=True, cache_ttl=300)
    
    async def get_compliance_report(self, regulation: str = "GDPR",
                                  time_range: str = "30d") -> Dict[str, Any]:
        """GET /compliance/report - Generate compliance report"""
        params = {"regulation": regulation, "time_range": time_range}
        return await self._make_request("GET", "/compliance/report", 
                                      params=params, use_cache=True, cache_ttl=300)
    
    async def get_violations(self, severity: Optional[str] = None,
                           status: Optional[str] = None, time_range: str = "7d") -> Dict[str, Any]:
        """GET /compliance/violations - Get compliance violations"""
        params = {"severity": severity, "status": status, "time_range": time_range}
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request("GET", "/compliance/violations", 
                                      params=params, use_cache=True, cache_ttl=60)
    
    async def resolve_violation(self, violation_id: str, resolution: str,
                              resolved_by: Optional[str] = None) -> Dict[str, Any]:
        """POST /compliance/violations/{violation_id}/resolve - Resolve a compliance violation"""
        data = {"resolution": resolution, "resolved_by": resolved_by}
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self._make_request("POST", f"/compliance/violations/{violation_id}/resolve", data=data)
    
    # =============================================================================
    # PRIVACY METRICS ENDPOINTS
    # =============================================================================
    
    async def get_privacy_metrics(self, time_range: str = "7d") -> Dict[str, Any]:
        """GET /metrics - Get privacy metrics and statistics"""
        return await self._make_request("GET", "/metrics", 
                                      params={"time_range": time_range}, use_cache=True, cache_ttl=300)
    
    async def get_pii_detection_stats(self, time_range: str = "7d") -> Dict[str, Any]:
        """GET /metrics/pii - Get PII detection statistics"""
        return await self._make_request("GET", "/metrics/pii", 
                                      params={"time_range": time_range}, use_cache=True, cache_ttl=300)
    
    async def get_anonymization_stats(self, time_range: str = "7d") -> Dict[str, Any]:
        """GET /metrics/anonymization - Get anonymization statistics"""
        return await self._make_request("GET", "/metrics/anonymization", 
                                      params={"time_range": time_range}, use_cache=True, cache_ttl=300)
    
    async def get_consent_metrics(self, time_range: str = "7d") -> Dict[str, Any]:
        """GET /metrics/consent - Get consent management metrics"""
        return await self._make_request("GET", "/metrics/consent", 
                                      params={"time_range": time_range}, use_cache=True, cache_ttl=300)
    
    # =============================================================================
    # CONVENIENCE METHODS
    # =============================================================================
    
    async def get_compliance_score(self) -> float:
        """Get overall compliance score"""
        try:
            status = await self.get_compliance_status()
            return status.get("compliance_score", 0.0)
        except:
            return 0.0
    
    async def get_pii_detection_rate(self) -> float:
        """Get PII detection rate"""
        try:
            metrics = await self.get_pii_detection_stats()
            return metrics.get("detection_rate", 0.0)
        except:
            return 0.0
    
    async def get_consent_rate(self) -> float:
        """Get consent rate percentage"""
        try:
            metrics = await self.get_consent_metrics()
            return metrics.get("consent_rate", 0.0)
        except:
            return 0.0
