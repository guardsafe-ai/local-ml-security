"""
Monitoring Service - Data Collector
Collects monitoring data from various services
"""

import logging
import requests
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects monitoring data from various services"""
    
    def __init__(self):
        self.redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
        self.service_urls = {
            "training": "http://training:8002",
            "model-api": "http://model-api:8000",
            "red-team": "http://red-team:8001",
            "analytics": "http://analytics:8006",
            "business-metrics": "http://business-metrics:8004",
            "data-privacy": "http://data-privacy:8005",
            "enterprise-dashboard": "http://enterprise-dashboard-backend:8007"
        }
        self.timeout = 10
    
    def get_model_loading_status(self) -> List[Dict[str, Any]]:
        """Get model loading status from training service"""
        try:
            response = requests.get(
                f"{self.service_urls['training']}/model-loading/status",
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get model loading status: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting model loading status: {e}")
            return []
    
    def get_training_status(self) -> List[Dict[str, Any]]:
        """Get training status from training service"""
        try:
            response = requests.get(
                f"{self.service_urls['training']}/jobs",
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get training status: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting training status: {e}")
            return []
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics from Redis cache"""
        try:
            metrics = self.redis_client.hgetall("system_metrics")
            if metrics:
                return {
                    "timestamp": datetime.now(),
                    "cpu_usage": float(metrics.get("cpu_usage", 0.0)),
                    "memory_usage": float(metrics.get("memory_usage", 0.0)),
                    "disk_usage": float(metrics.get("disk_usage", 0.0)),
                    "network_io": float(metrics.get("network_io", 0.0)),
                    "active_connections": int(metrics.get("active_connections", 0))
                }
            else:
                # Return default metrics if not available
                return {
                    "timestamp": datetime.now(),
                    "cpu_usage": 0.0,
                    "memory_usage": 0.0,
                    "disk_usage": 0.0,
                    "network_io": 0.0,
                    "active_connections": 0
                }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {
                "timestamp": datetime.now(),
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "disk_usage": 0.0,
                "network_io": 0.0,
                "active_connections": 0
            }
    
    def get_service_health(self) -> List[Dict[str, Any]]:
        """Get health status of all services"""
        health_data = []
        
        for service_name, url in self.service_urls.items():
            try:
                start_time = datetime.now()
                response = requests.get(f"{url}/health", timeout=self.timeout)
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    health_data.append({
                        "service_name": service_name,
                        "status": "healthy",
                        "response_time": response_time,
                        "last_check": datetime.now(),
                        "uptime": data.get("uptime_seconds", 0.0),
                        "error_count": 0
                    })
                else:
                    health_data.append({
                        "service_name": service_name,
                        "status": "unhealthy",
                        "response_time": response_time,
                        "last_check": datetime.now(),
                        "uptime": 0.0,
                        "error_count": 1
                    })
            except Exception as e:
                logger.error(f"Error checking health for {service_name}: {e}")
                health_data.append({
                    "service_name": service_name,
                    "status": "unhealthy",
                    "response_time": 0.0,
                    "last_check": datetime.now(),
                    "uptime": 0.0,
                    "error_count": 1
                })
        
        return health_data
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get alerts from Redis cache"""
        try:
            alerts = self.redis_client.lrange("alerts", 0, -1)
            alert_list = []
            
            for alert_json in alerts:
                try:
                    alert_data = json.loads(alert_json)
                    alert_list.append(alert_data)
                except json.JSONDecodeError:
                    continue
            
            return alert_list
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data"""
        try:
            return {
                "model_loading_status": self.get_model_loading_status(),
                "training_status": self.get_training_status(),
                "system_metrics": self.get_system_metrics(),
                "service_health": self.get_service_health(),
                "alerts": self.get_alerts(),
                "last_updated": datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {
                "model_loading_status": [],
                "training_status": [],
                "system_metrics": {
                    "timestamp": datetime.now(),
                    "cpu_usage": 0.0,
                    "memory_usage": 0.0,
                    "disk_usage": 0.0,
                    "network_io": 0.0,
                    "active_connections": 0
                },
                "service_health": [],
                "alerts": [],
                "last_updated": datetime.now()
            }
