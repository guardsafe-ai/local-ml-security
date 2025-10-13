"""
Version Manager
Manages model versions and handles version-based testing strategies.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.entities import ModelVersion
except ImportError:
    # Fallback for environments without MLflow
    class mlflow:
        @staticmethod
        def search_runs(*args, **kwargs): return []
        @staticmethod
        def get_experiment(*args, **kwargs): return None
    class MlflowClient:
        def __init__(self, *args, **kwargs): pass
        def get_model_version(self, *args, **kwargs): return None
        def get_latest_versions(self, *args, **kwargs): return []
        def search_model_versions(self, *args, **kwargs): return []
        def set_model_version_tag(self, *args, **kwargs): pass
        def delete_model_version_tag(self, *args, **kwargs): pass
    class ModelVersion:
        def __init__(self, *args, **kwargs): pass

logger = logging.getLogger(__name__)

class VersionStrategy(Enum):
    """Version testing strategies"""
    LATEST = "latest"
    ALL_VERSIONS = "all_versions"
    STAGING_ONLY = "staging_only"
    PRODUCTION_ONLY = "production_only"
    BY_DATE_RANGE = "by_date_range"
    BY_PERFORMANCE = "by_performance"
    CUSTOM = "custom"

class VersionStatus(Enum):
    """Version status"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    TESTING = "testing"
    FAILED = "failed"
    ARCHIVED = "archived"

@dataclass
class VersionInfo:
    """Version information"""
    model_name: str
    version: str
    stage: str
    status: VersionStatus
    created_at: datetime
    updated_at: datetime
    description: str
    tags: Dict[str, str]
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    experiment_id: str = ""

@dataclass
class VersionTestConfig:
    """Version testing configuration"""
    strategy: VersionStrategy
    max_versions: int = 10
    date_range_days: int = 30
    performance_threshold: float = 0.8
    custom_versions: List[str] = field(default_factory=list)
    include_archived: bool = False
    test_stages: List[str] = field(default_factory=lambda: ["None", "Staging", "Production"])

class VersionManager:
    """Manages model versions and version-based testing"""
    
    def __init__(self, tracking_uri: str = "file:///tmp/mlflow"):
        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.version_cache: Dict[str, List[VersionInfo]] = {}
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)
    
    def get_model_versions(self, model_name: str, 
                          config: Optional[VersionTestConfig] = None) -> List[VersionInfo]:
        """Get model versions based on strategy"""
        try:
            config = config or VersionTestConfig(VersionStrategy.LATEST)
            
            if config.strategy == VersionStrategy.LATEST:
                return self._get_latest_versions(model_name, config)
            elif config.strategy == VersionStrategy.ALL_VERSIONS:
                return self._get_all_versions(model_name, config)
            elif config.strategy == VersionStrategy.STAGING_ONLY:
                return self._get_versions_by_stage(model_name, ["Staging"], config)
            elif config.strategy == VersionStrategy.PRODUCTION_ONLY:
                return self._get_versions_by_stage(model_name, ["Production"], config)
            elif config.strategy == VersionStrategy.BY_DATE_RANGE:
                return self._get_versions_by_date_range(model_name, config)
            elif config.strategy == VersionStrategy.BY_PERFORMANCE:
                return self._get_versions_by_performance(model_name, config)
            elif config.strategy == VersionStrategy.CUSTOM:
                return self._get_custom_versions(model_name, config)
            else:
                return self._get_latest_versions(model_name, config)
                
        except Exception as e:
            logger.error(f"Failed to get model versions for {model_name}: {e}")
            return []
    
    def _get_latest_versions(self, model_name: str, config: VersionTestConfig) -> List[VersionInfo]:
        """Get latest versions of model"""
        try:
            versions = self.client.get_latest_versions(
                model_name, 
                stages=config.test_stages
            )
            
            version_infos = []
            for version in versions[:config.max_versions]:
                version_info = self._create_version_info(version)
                if version_info:
                    version_infos.append(version_info)
            
            return version_infos
            
        except Exception as e:
            logger.error(f"Failed to get latest versions: {e}")
            return []
    
    def _get_all_versions(self, model_name: str, config: VersionTestConfig) -> List[VersionInfo]:
        """Get all versions of model"""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            version_infos = []
            for version in versions[:config.max_versions]:
                if self._should_include_version(version, config):
                    version_info = self._create_version_info(version)
                    if version_info:
                        version_infos.append(version_info)
            
            # Sort by creation time (newest first)
            version_infos.sort(key=lambda x: x.created_at, reverse=True)
            
            return version_infos
            
        except Exception as e:
            logger.error(f"Failed to get all versions: {e}")
            return []
    
    def _get_versions_by_stage(self, model_name: str, stages: List[str], 
                              config: VersionTestConfig) -> List[VersionInfo]:
        """Get versions by stage"""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            version_infos = []
            for version in versions:
                if version.current_stage in stages and self._should_include_version(version, config):
                    version_info = self._create_version_info(version)
                    if version_info:
                        version_infos.append(version_info)
            
            # Sort by creation time (newest first)
            version_infos.sort(key=lambda x: x.created_at, reverse=True)
            
            return version_infos[:config.max_versions]
            
        except Exception as e:
            logger.error(f"Failed to get versions by stage: {e}")
            return []
    
    def _get_versions_by_date_range(self, model_name: str, config: VersionTestConfig) -> List[VersionInfo]:
        """Get versions by date range"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=config.date_range_days)
            
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            version_infos = []
            for version in versions:
                version_date = datetime.fromtimestamp(version.creation_timestamp / 1000)
                
                if (start_date <= version_date <= end_date and 
                    self._should_include_version(version, config)):
                    version_info = self._create_version_info(version)
                    if version_info:
                        version_infos.append(version_info)
            
            # Sort by creation time (newest first)
            version_infos.sort(key=lambda x: x.created_at, reverse=True)
            
            return version_infos[:config.max_versions]
            
        except Exception as e:
            logger.error(f"Failed to get versions by date range: {e}")
            return []
    
    def _get_versions_by_performance(self, model_name: str, config: VersionTestConfig) -> List[VersionInfo]:
        """Get versions by performance metrics"""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            version_infos = []
            for version in versions:
                if self._should_include_version(version, config):
                    version_info = self._create_version_info(version)
                    if version_info and self._meets_performance_threshold(version_info, config):
                        version_infos.append(version_info)
            
            # Sort by performance metric (highest first)
            version_infos.sort(key=lambda x: x.metrics.get('accuracy', 0), reverse=True)
            
            return version_infos[:config.max_versions]
            
        except Exception as e:
            logger.error(f"Failed to get versions by performance: {e}")
            return []
    
    def _get_custom_versions(self, model_name: str, config: VersionTestConfig) -> List[VersionInfo]:
        """Get custom specified versions"""
        try:
            version_infos = []
            
            for version_str in config.custom_versions:
                try:
                    version = self.client.get_model_version(model_name, version_str)
                    if version and self._should_include_version(version, config):
                        version_info = self._create_version_info(version)
                        if version_info:
                            version_infos.append(version_info)
                except Exception as e:
                    logger.warning(f"Failed to get version {version_str}: {e}")
                    continue
            
            return version_infos
            
        except Exception as e:
            logger.error(f"Failed to get custom versions: {e}")
            return []
    
    def _should_include_version(self, version, config: VersionTestConfig) -> bool:
        """Check if version should be included"""
        try:
            # Check stage filter
            if version.current_stage not in config.test_stages:
                return False
            
            # Check archived filter
            if not config.include_archived and version.current_stage == "Archived":
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check version inclusion: {e}")
            return False
    
    def _meets_performance_threshold(self, version_info: VersionInfo, 
                                   config: VersionTestConfig) -> bool:
        """Check if version meets performance threshold"""
        try:
            # Check if version has performance metrics
            if not version_info.metrics:
                return False
            
            # Check against threshold (using accuracy as default metric)
            accuracy = version_info.metrics.get('accuracy', 0)
            return accuracy >= config.performance_threshold
            
        except Exception as e:
            logger.error(f"Failed to check performance threshold: {e}")
            return False
    
    def _create_version_info(self, version) -> Optional[VersionInfo]:
        """Create VersionInfo from MLflow version"""
        try:
            # Get run information for metrics and parameters
            run = mlflow.get_run(version.run_id)
            metrics = run.data.metrics if run else {}
            parameters = run.data.params if run else {}
            
            return VersionInfo(
                model_name=version.name,
                version=version.version,
                stage=version.current_stage,
                status=self._map_stage_to_status(version.current_stage),
                created_at=datetime.fromtimestamp(version.creation_timestamp / 1000),
                updated_at=datetime.fromtimestamp(version.last_updated_timestamp / 1000),
                description=version.description or "",
                tags=version.tags or {},
                metrics=metrics,
                parameters=parameters,
                run_id=version.run_id,
                experiment_id=run.info.experiment_id if run else ""
            )
            
        except Exception as e:
            logger.error(f"Failed to create version info: {e}")
            return None
    
    def _map_stage_to_status(self, stage: str) -> VersionStatus:
        """Map MLflow stage to version status"""
        stage_mapping = {
            "None": VersionStatus.ACTIVE,
            "Staging": VersionStatus.TESTING,
            "Production": VersionStatus.ACTIVE,
            "Archived": VersionStatus.ARCHIVED
        }
        return stage_mapping.get(stage, VersionStatus.ACTIVE)
    
    def update_version_status(self, model_name: str, version: str, 
                            status: VersionStatus, tags: Dict[str, str] = None):
        """Update version status and tags"""
        try:
            # Update tags
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(model_name, version, key, value)
            
            # Update status tag
            self.client.set_model_version_tag(
                model_name, version, "security_status", status.value
            )
            
            logger.info(f"Updated version {model_name}:{version} status to {status.value}")
            
        except Exception as e:
            logger.error(f"Failed to update version status: {e}")
    
    def get_version_comparison(self, model_name: str, 
                             versions: List[str]) -> Dict[str, Any]:
        """Compare multiple versions of a model"""
        try:
            comparison = {
                "model_name": model_name,
                "versions": [],
                "comparison_metrics": {},
                "created_at": datetime.now().isoformat()
            }
            
            version_infos = []
            for version_str in versions:
                version_info = self._get_version_info(model_name, version_str)
                if version_info:
                    version_infos.append(version_info)
                    comparison["versions"].append({
                        "version": version_info.version,
                        "stage": version_info.stage,
                        "status": version_info.status.value,
                        "created_at": version_info.created_at.isoformat(),
                        "metrics": version_info.metrics,
                        "tags": version_info.tags
                    })
            
            # Compare metrics across versions
            if version_infos:
                comparison["comparison_metrics"] = self._compare_metrics(version_infos)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to get version comparison: {e}")
            return {}
    
    def _get_version_info(self, model_name: str, version: str) -> Optional[VersionInfo]:
        """Get version info for specific version"""
        try:
            version_obj = self.client.get_model_version(model_name, version)
            if version_obj:
                return self._create_version_info(version_obj)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get version info: {e}")
            return None
    
    def _compare_metrics(self, version_infos: List[VersionInfo]) -> Dict[str, Any]:
        """Compare metrics across versions"""
        try:
            comparison = {}
            
            # Get all unique metrics
            all_metrics = set()
            for version_info in version_infos:
                all_metrics.update(version_info.metrics.keys())
            
            # Compare each metric
            for metric in all_metrics:
                values = []
                for version_info in version_infos:
                    if metric in version_info.metrics:
                        values.append(version_info.metrics[metric])
                
                if values:
                    comparison[metric] = {
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "count": len(values),
                        "values": values
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare metrics: {e}")
            return {}
    
    def get_version_history(self, model_name: str, 
                          days: int = 30) -> List[Dict[str, Any]]:
        """Get version history for a model"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            history = []
            for version in versions:
                version_date = datetime.fromtimestamp(version.creation_timestamp / 1000)
                
                if start_date <= version_date <= end_date:
                    version_info = self._create_version_info(version)
                    if version_info:
                        history.append({
                            "version": version_info.version,
                            "stage": version_info.stage,
                            "status": version_info.status.value,
                            "created_at": version_info.created_at.isoformat(),
                            "updated_at": version_info.updated_at.isoformat(),
                            "description": version_info.description,
                            "tags": version_info.tags,
                            "metrics": version_info.metrics
                        })
            
            # Sort by creation time (newest first)
            history.sort(key=lambda x: x["created_at"], reverse=True)
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get version history: {e}")
            return []
    
    def get_model_statistics(self, model_name: str) -> Dict[str, Any]:
        """Get statistics for a model"""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            stats = {
                "total_versions": len(versions),
                "stages": {},
                "statuses": {},
                "date_range": {},
                "metrics_summary": {}
            }
            
            if not versions:
                return stats
            
            # Count by stage
            for version in versions:
                stage = version.current_stage
                stats["stages"][stage] = stats["stages"].get(stage, 0) + 1
            
            # Count by status
            for version in versions:
                status = self._map_stage_to_status(version.current_stage)
                status_str = status.value
                stats["statuses"][status_str] = stats["statuses"].get(status_str, 0) + 1
            
            # Date range
            creation_times = [v.creation_timestamp for v in versions]
            if creation_times:
                stats["date_range"] = {
                    "earliest": datetime.fromtimestamp(min(creation_times) / 1000).isoformat(),
                    "latest": datetime.fromtimestamp(max(creation_times) / 1000).isoformat()
                }
            
            # Metrics summary
            all_metrics = {}
            for version in versions:
                try:
                    run = mlflow.get_run(version.run_id)
                    if run:
                        for metric, value in run.data.metrics.items():
                            if metric not in all_metrics:
                                all_metrics[metric] = []
                            all_metrics[metric].append(value)
                except:
                    continue
            
            for metric, values in all_metrics.items():
                if values:
                    stats["metrics_summary"][metric] = {
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "count": len(values)
                    }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get model statistics: {e}")
            return {}
    
    def export_version_data(self, model_name: str, format_type: str = "json") -> str:
        """Export version data"""
        try:
            versions = self.get_model_versions(model_name)
            
            data = {
                "model_name": model_name,
                "exported_at": datetime.now().isoformat(),
                "total_versions": len(versions),
                "versions": [
                    {
                        "version": v.version,
                        "stage": v.stage,
                        "status": v.status.value,
                        "created_at": v.created_at.isoformat(),
                        "updated_at": v.updated_at.isoformat(),
                        "description": v.description,
                        "tags": v.tags,
                        "metrics": v.metrics,
                        "parameters": v.parameters,
                        "run_id": v.run_id,
                        "experiment_id": v.experiment_id
                    }
                    for v in versions
                ],
                "statistics": self.get_model_statistics(model_name)
            }
            
            if format_type == "json":
                return json.dumps(data, indent=2)
            else:
                return str(data)
                
        except Exception as e:
            logger.error(f"Failed to export version data: {e}")
            return "{}"
