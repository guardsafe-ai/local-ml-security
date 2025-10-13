"""
Training Service - MLflow Integration
Handles MLflow experiment tracking and model registry operations
"""

import logging
from typing import List, Dict, Any, Optional
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class MLflowService:
    """Service for MLflow operations"""
    
    def __init__(self):
        self.client = MlflowClient()
        self.tracking_uri = "http://mlflow:5000"
        mlflow.set_tracking_uri(self.tracking_uri)

    async def get_experiments(self) -> List[Dict[str, Any]]:
        """Get all MLflow experiments"""
        try:
            experiments = self.client.search_experiments()
            return [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "creation_time": exp.creation_time,
                    "last_update_time": exp.last_update_time
                }
                for exp in experiments
            ]
        except Exception as e:
            logger.error(f"Failed to get experiments: {e}")
            return []

    async def get_model_registry(self) -> List[Dict[str, Any]]:
        """Get registered models from MLflow"""
        try:
            models = self.client.search_registered_models()
            return [
                {
                    "name": model.name,
                    "latest_versions": [
                        {
                            "version": version.version,
                            "stage": version.current_stage,
                            "creation_timestamp": version.creation_timestamp,
                            "last_updated_timestamp": version.last_updated_timestamp
                        }
                        for version in model.latest_versions
                    ],
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp
                }
                for model in models
            ]
        except Exception as e:
            logger.error(f"Failed to get model registry: {e}")
            return []

    async def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get versions for a specific model"""
        try:
            versions = self.client.get_latest_versions(model_name)
            return [
                {
                    "version": version.version,
                    "stage": version.current_stage,
                    "creation_timestamp": version.creation_timestamp,
                    "last_updated_timestamp": version.last_updated_timestamp,
                    "description": version.description
                }
                for version in versions
            ]
        except Exception as e:
            logger.error(f"Failed to get versions for {model_name}: {e}")
            return []

    async def get_latest_models(self) -> List[Dict[str, Any]]:
        """Get latest versions of all models"""
        try:
            models = self.client.search_registered_models()
            latest_models = []
            
            for model in models:
                latest_version = self.client.get_latest_versions(model.name, stages=["None"])[0]
                latest_models.append({
                    "name": model.name,
                    "version": latest_version.version,
                    "creation_timestamp": latest_version.creation_timestamp,
                    "last_updated_timestamp": latest_version.last_updated_timestamp
                })
            
            return latest_models
        except Exception as e:
            logger.error(f"Failed to get latest models: {e}")
            return []

    async def get_best_models(self) -> List[Dict[str, Any]]:
        """Get best performing models"""
        try:
            models = self.client.search_registered_models()
            best_models = []
            
            for model in models:
                try:
                    # Get production stage models
                    prod_versions = self.client.get_latest_versions(model.name, stages=["Production"])
                    if prod_versions:
                        best_models.append({
                            "name": model.name,
                            "version": prod_versions[0].version,
                            "stage": "Production",
                            "creation_timestamp": prod_versions[0].creation_timestamp
                        })
                except Exception:
                    # If no production models, get latest
                    latest_versions = self.client.get_latest_versions(model.name, stages=["None"])
                    if latest_versions:
                        best_models.append({
                            "name": model.name,
                            "version": latest_versions[0].version,
                            "stage": "None",
                            "creation_timestamp": latest_versions[0].creation_timestamp
                        })
            
            return best_models
        except Exception as e:
            logger.error(f"Failed to get best models: {e}")
            return []

    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model"""
        try:
            model = self.client.get_registered_model(model_name)
            return {
                "name": model.name,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
                "description": model.description,
                "latest_versions": [
                    {
                        "version": version.version,
                        "stage": version.current_stage,
                        "creation_timestamp": version.creation_timestamp,
                        "last_updated_timestamp": version.last_updated_timestamp
                    }
                    for version in model.latest_versions
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return None

    async def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments"""
        try:
            experiments = await self.get_experiments()
            models = await self.get_model_registry()
            
            total_experiments = len(experiments)
            active_experiments = len([exp for exp in experiments if exp["lifecycle_stage"] == "active"])
            completed_experiments = len([exp for exp in experiments if exp["lifecycle_stage"] == "deleted"])
            failed_experiments = 0  # This would need to be calculated from runs
            
            total_models = len(models)
            best_model = None
            if models:
                # Find model with most recent update
                best_model = max(models, key=lambda x: x["last_updated_timestamp"])["name"]
            
            return {
                "total_experiments": total_experiments,
                "active_experiments": active_experiments,
                "completed_experiments": completed_experiments,
                "failed_experiments": failed_experiments,
                "total_models": total_models,
                "best_model": best_model
            }
        except Exception as e:
            logger.error(f"Failed to get experiment summary: {e}")
            return {
                "total_experiments": 0,
                "active_experiments": 0,
                "completed_experiments": 0,
                "failed_experiments": 0,
                "total_models": 0,
                "best_model": None
            }

    async def cleanup_old_runs(self, days_old: int = 30) -> int:
        """Clean up old MLflow runs"""
        try:
            # This is a simplified implementation
            # In practice, you'd need to query runs and delete old ones
            logger.info(f"Cleanup of runs older than {days_old} days requested")
            return 0
        except Exception as e:
            logger.error(f"Failed to cleanup old runs: {e}")
            return 0
