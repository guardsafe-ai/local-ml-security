"""
Enterprise Dashboard Backend - Dashboard Service
Core business logic for dashboard operations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from services.main_api_client import MainAPIClient
from models.responses import DashboardMetrics, ServiceHealth, TrainingJob, AttackResult

logger = logging.getLogger(__name__)


class DashboardService:
    """Core dashboard business logic"""
    
    def __init__(self):
        self.api_client = MainAPIClient()

    async def get_dashboard_metrics(self) -> DashboardMetrics:
        """Get comprehensive dashboard metrics"""
        try:
            # Get data from all services in parallel
            tasks = [
                self.api_client.get_available_models(),
                self.api_client.get_training_jobs(),
                self.api_client.get_red_team_results(),
                self.api_client.get_analytics_summary()
            ]
            
            models_data, training_jobs, red_team_results, analytics_summary = await asyncio.gather(*tasks)
            
            # Calculate metrics
            total_models = len(models_data.get("models", {}))
            active_jobs = len([job for job in training_jobs if job.get("status") in ["running", "pending"]])
            total_attacks = len(red_team_results)
            
            # Calculate detection rate from analytics
            detection_rate = 0.0
            if analytics_summary and "summary" in analytics_summary:
                summary = analytics_summary["summary"]
                if summary and "detection_rate" in summary:
                    detection_rate = summary["detection_rate"]
            
            # Calculate system health (simplified) - optimize with generator expression
            health_status = await self.api_client.get_all_services_health()
            if health_status:
                healthy_services = sum(1 for service in health_status if service.get("status") == "healthy")
                system_health = (healthy_services / len(health_status)) * 100
            else:
                system_health = 0.0
            
            return DashboardMetrics(
                total_models=total_models,
                active_jobs=active_jobs,
                total_attacks=total_attacks,
                detection_rate=detection_rate,
                system_health=system_health,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to get dashboard metrics: {e}")
            return DashboardMetrics(
                total_models=0,
                active_jobs=0,
                total_attacks=0,
                detection_rate=0.0,
                system_health=0.0,
                last_updated=datetime.now()
            )

    async def get_models_overview(self) -> Dict[str, Any]:
        """Get comprehensive models overview"""
        try:
            models_data = await self.api_client.get_available_models()
            
            # Process models data
            models = models_data.get("models", {})
            available_models = models_data.get("available_models", [])
            mlflow_models = models_data.get("mlflow_models", [])
            
            # Count by type
            pretrained_count = sum(1 for model in models.values() if model.get("model_source") == "Hugging Face")
            trained_count = sum(1 for model in models.values() if model.get("model_source") == "MLflow")
            loaded_count = sum(1 for model in models.values() if model.get("loaded", False))
            
            return {
                "total_models": len(models),
                "pretrained_models": pretrained_count,
                "trained_models": trained_count,
                "loaded_models": loaded_count,
                "available_models": available_models,
                "mlflow_models": mlflow_models,
                "models": models,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get models overview: {e}")
            return {
                "total_models": 0,
                "pretrained_models": 0,
                "trained_models": 0,
                "loaded_models": 0,
                "available_models": [],
                "mlflow_models": [],
                "models": {},
                "last_updated": datetime.now().isoformat()
            }

    async def get_training_overview(self) -> Dict[str, Any]:
        """Get training jobs overview"""
        try:
            training_jobs = await self.api_client.get_training_jobs()
            
            # Categorize jobs by status
            running_jobs = [job for job in training_jobs if job.get("status") == "running"]
            completed_jobs = [job for job in training_jobs if job.get("status") == "completed"]
            failed_jobs = [job for job in training_jobs if job.get("status") == "failed"]
            pending_jobs = [job for job in training_jobs if job.get("status") == "pending"]
            
            # Calculate average progress
            avg_progress = 0.0
            if running_jobs:
                avg_progress = sum(job.get("progress", 0) for job in running_jobs) / len(running_jobs)
            
            return {
                "total_jobs": len(training_jobs),
                "running_jobs": len(running_jobs),
                "completed_jobs": len(completed_jobs),
                "failed_jobs": len(failed_jobs),
                "pending_jobs": len(pending_jobs),
                "average_progress": avg_progress,
                "jobs": training_jobs,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get training overview: {e}")
            return {
                "total_jobs": 0,
                "running_jobs": 0,
                "completed_jobs": 0,
                "failed_jobs": 0,
                "pending_jobs": 0,
                "average_progress": 0.0,
                "jobs": [],
                "last_updated": datetime.now().isoformat()
            }

    async def get_red_team_overview(self) -> Dict[str, Any]:
        """Get red team testing overview"""
        try:
            red_team_results = await self.api_client.get_red_team_results()
            analytics_summary = await self.api_client.get_analytics_summary()
            
            # Process results
            total_attacks = len(red_team_results)
            successful_attacks = sum(1 for result in red_team_results if result.get("success", False))
            detection_rate = (successful_attacks / total_attacks * 100) if total_attacks > 0 else 0.0
            
            # Categorize by attack type
            attack_categories = {}
            for result in red_team_results:
                category = result.get("category", "unknown")
                attack_categories[category] = attack_categories.get(category, 0) + 1
            
            return {
                "total_attacks": total_attacks,
                "successful_attacks": successful_attacks,
                "detection_rate": detection_rate,
                "attack_categories": attack_categories,
                "results": red_team_results,
                "analytics_summary": analytics_summary,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get red team overview: {e}")
            return {
                "total_attacks": 0,
                "successful_attacks": 0,
                "detection_rate": 0.0,
                "attack_categories": {},
                "results": [],
                "analytics_summary": {},
                "last_updated": datetime.now().isoformat()
            }

    async def get_system_health_overview(self) -> Dict[str, Any]:
        """Get system health overview"""
        try:
            health_status = await self.api_client.get_all_services_health()
            
            # Calculate overall health
            healthy_services = sum(1 for service in health_status if service["status"] == "healthy")
            total_services = len(health_status)
            overall_health = (healthy_services / total_services * 100) if total_services > 0 else 0.0
            
            # Calculate average response time
            avg_response_time = 0.0
            if health_status:
                response_times = [service["response_time"] for service in health_status if service["response_time"] > 0]
                avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
            
            return {
                "overall_health": overall_health,
                "healthy_services": healthy_services,
                "total_services": total_services,
                "average_response_time": avg_response_time,
                "services": health_status,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system health overview: {e}")
            return {
                "overall_health": 0.0,
                "healthy_services": 0,
                "total_services": 0,
                "average_response_time": 0.0,
                "services": [],
                "last_updated": datetime.now().isoformat()
            }

    async def get_recent_activity(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent system activity"""
        try:
            activities = []
            
            # Get recent training jobs
            training_jobs = await self.api_client.get_training_jobs()
            for job in training_jobs[:5]:  # Last 5 jobs
                activities.append({
                    "type": "training",
                    "id": job.get("job_id", "unknown"),
                    "description": f"Training job {job.get('model_name', 'unknown')} - {job.get('status', 'unknown')}",
                    "timestamp": job.get("start_time", datetime.now().isoformat()),
                    "status": job.get("status", "unknown")
                })
            
            # Get recent red team results
            red_team_results = await self.api_client.get_red_team_results()
            for result in red_team_results[:5]:  # Last 5 results
                activities.append({
                    "type": "red_team",
                    "id": result.get("attack_id", "unknown"),
                    "description": f"Red team test - {result.get('category', 'unknown')} attack",
                    "timestamp": result.get("timestamp", datetime.now().isoformat()),
                    "status": "success" if result.get("success", False) else "failed"
                })
            
            # Sort by timestamp and limit
            activities.sort(key=lambda x: x["timestamp"], reverse=True)
            return activities[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get recent activity: {e}")
            return []

    async def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends over time"""
        try:
            # This would typically query time-series data
            # For now, return mock data structure
            return {
                "time_range_hours": hours,
                "detection_rate_trend": [],
                "model_performance_trend": [],
                "system_health_trend": [],
                "attack_frequency_trend": [],
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance trends: {e}")
            return {
                "time_range_hours": hours,
                "detection_rate_trend": [],
                "model_performance_trend": [],
                "system_health_trend": [],
                "attack_frequency_trend": [],
                "last_updated": datetime.now().isoformat()
            }

    async def get_analytics_overview(self) -> Dict[str, Any]:
        """Get analytics overview"""
        try:
            analytics_summary = await self.api_client.get_analytics_summary()
            
            return {
                "total_analyses": analytics_summary.get("summary", {}).get("total_analyses", 0),
                "successful_analyses": analytics_summary.get("summary", {}).get("successful_analyses", 0),
                "failed_analyses": analytics_summary.get("summary", {}).get("failed_analyses", 0),
                "average_processing_time": analytics_summary.get("summary", {}).get("average_processing_time", 0.0),
                "detection_rate": analytics_summary.get("summary", {}).get("detection_rate", 0.0),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get analytics overview: {e}")
            return {
                "total_analyses": 0,
                "successful_analyses": 0,
                "failed_analyses": 0,
                "average_processing_time": 0.0,
                "detection_rate": 0.0,
                "last_updated": datetime.now().isoformat()
            }

    async def get_business_metrics_overview(self) -> Dict[str, Any]:
        """Get business metrics overview"""
        try:
            business_metrics_summary = await self.api_client.get_business_metrics_summary()
            
            return {
                "total_models": business_metrics_summary.get("summary", {}).get("total_models", 0),
                "active_models": business_metrics_summary.get("summary", {}).get("active_models", 0),
                "inactive_models": business_metrics_summary.get("summary", {}).get("inactive_models", 0),
                "total_predictions": business_metrics_summary.get("summary", {}).get("total_predictions", 0),
                "successful_predictions": business_metrics_summary.get("summary", {}).get("successful_predictions", 0),
                "failed_predictions": business_metrics_summary.get("summary", {}).get("failed_predictions", 0),
                "average_response_time": business_metrics_summary.get("summary", {}).get("average_response_time", 0.0),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get business metrics overview: {e}")
            return {
                "total_models": 0,
                "active_models": 0,
                "inactive_models": 0,
                "total_predictions": 0,
                "successful_predictions": 0,
                "failed_predictions": 0,
                "average_response_time": 0.0,
                "last_updated": datetime.now().isoformat()
            }

    async def get_data_privacy_overview(self) -> Dict[str, Any]:
        """Get data privacy overview"""
        try:
            data_privacy_summary = await self.api_client.get_data_privacy_summary()
            
            return {
                "total_checks": data_privacy_summary.get("summary", {}).get("total_checks", 0),
                "passed_checks": data_privacy_summary.get("summary", {}).get("passed_checks", 0),
                "failed_checks": data_privacy_summary.get("summary", {}).get("failed_checks", 0),
                "compliance_score": data_privacy_summary.get("summary", {}).get("compliance_score", 0.0),
                "privacy_violations": data_privacy_summary.get("summary", {}).get("privacy_violations", 0),
                "data_anonymization_rate": data_privacy_summary.get("summary", {}).get("data_anonymization_rate", 0.0),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get data privacy overview: {e}")
            return {
                "total_checks": 0,
                "passed_checks": 0,
                "failed_checks": 0,
                "compliance_score": 0.0,
                "privacy_violations": 0,
                "data_anonymization_rate": 0.0,
                "last_updated": datetime.now().isoformat()
            }
