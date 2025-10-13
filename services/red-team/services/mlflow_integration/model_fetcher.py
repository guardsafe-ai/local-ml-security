"""
Model Fetcher
Automatically fetches models from MLflow registry for security testing.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import os
import tempfile
import shutil

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    import mlflow.tensorflow
    import mlflow.xgboost
    import mlflow.lightgbm
    from mlflow.tracking import MlflowClient
    from mlflow.entities import Run, Experiment
except ImportError:
    # Fallback for environments without MLflow
    class mlflow:
        class sklearn: pass
        class pytorch: pass
        class tensorflow: pass
        class xgboost: pass
        class lightgbm: pass
        @staticmethod
        def log_model(*args, **kwargs): pass
        @staticmethod
        def load_model(*args, **kwargs): return None
        @staticmethod
        def search_runs(*args, **kwargs): return []
        @staticmethod
        def get_experiment(*args, **kwargs): return None
        @staticmethod
        def search_experiments(*args, **kwargs): return []
    class MlflowClient:
        def __init__(self, *args, **kwargs): pass
        def search_registered_models(self, *args, **kwargs): return []
        def get_latest_versions(self, *args, **kwargs): return []
        def get_model_version(self, *args, **kwargs): return None
        def download_artifacts(self, *args, **kwargs): return ""

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported model types"""
    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CUSTOM = "custom"

class ModelStatus(Enum):
    """Model status"""
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"
    NONE = "None"

@dataclass
class ModelInfo:
    """Model information"""
    name: str
    version: str
    model_type: ModelType
    status: ModelStatus
    stage: str
    description: str
    tags: Dict[str, str]
    created_at: datetime
    updated_at: datetime
    run_id: str
    experiment_id: str
    artifact_path: str
    model_uri: str
    metadata: Dict[str, Any] = None

@dataclass
class FetchConfig:
    """Model fetching configuration"""
    registry_uri: str = "sqlite:///mlflow.db"
    tracking_uri: str = "file:///tmp/mlflow"
    model_stages: List[str] = None
    model_types: List[ModelType] = None
    max_models: int = 100
    include_archived: bool = False
    local_cache_dir: str = "/tmp/mlflow_models"
    auto_download: bool = True

class ModelFetcher:
    """Fetches models from MLflow registry for security testing"""
    
    def __init__(self, config: Optional[FetchConfig] = None):
        self.config = config or FetchConfig()
        self.client = MlflowClient(tracking_uri=self.config.tracking_uri)
        self.cached_models: Dict[str, ModelInfo] = {}
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.config.tracking_uri)
        
        # Create local cache directory
        os.makedirs(self.config.local_cache_dir, exist_ok=True)
    
    def fetch_all_models(self) -> List[ModelInfo]:
        """Fetch all available models from registry"""
        try:
            models = []
            
            # Search for registered models
            registered_models = self.client.search_registered_models()
            
            for model in registered_models:
                try:
                    # Get latest versions
                    versions = self.client.get_latest_versions(
                        model.name, 
                        stages=self.config.model_stages or ["None", "Staging", "Production"]
                    )
                    
                    for version in versions:
                        if self._should_include_model(version):
                            model_info = self._create_model_info(model, version)
                            models.append(model_info)
                            
                            # Cache model info
                            cache_key = f"{model.name}:{version.version}"
                            self.cached_models[cache_key] = model_info
                            
                except Exception as e:
                    logger.error(f"Failed to fetch versions for model {model.name}: {e}")
                    continue
            
            logger.info(f"Fetched {len(models)} models from MLflow registry")
            return models
            
        except Exception as e:
            logger.error(f"Failed to fetch models from MLflow: {e}")
            return []
    
    def fetch_model_by_name(self, model_name: str, version: Optional[str] = None) -> Optional[ModelInfo]:
        """Fetch specific model by name and version"""
        try:
            if version:
                # Fetch specific version
                model_version = self.client.get_model_version(model_name, version)
                if model_version:
                    return self._create_model_info_from_version(model_name, model_version)
            else:
                # Fetch latest version
                versions = self.client.get_latest_versions(model_name)
                if versions:
                    latest_version = versions[0]
                    return self._create_model_info_from_version(model_name, latest_version)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch model {model_name}:{version}: {e}")
            return None
    
    def fetch_models_by_experiment(self, experiment_id: str) -> List[ModelInfo]:
        """Fetch models from specific experiment"""
        try:
            models = []
            
            # Get experiment
            experiment = mlflow.get_experiment(experiment_id)
            if not experiment:
                logger.warning(f"Experiment {experiment_id} not found")
                return models
            
            # Search runs in experiment
            runs = mlflow.search_runs(experiment_ids=[experiment_id])
            
            for _, run in runs.iterrows():
                try:
                    # Check if run has logged model
                    artifacts = self.client.list_artifacts(run.run_id)
                    model_artifacts = [a for a in artifacts if a.path.endswith('model')]
                    
                    if model_artifacts:
                        model_info = self._create_model_info_from_run(run)
                        if model_info and self._should_include_model(model_info):
                            models.append(model_info)
                            
                except Exception as e:
                    logger.error(f"Failed to process run {run.run_id}: {e}")
                    continue
            
            logger.info(f"Fetched {len(models)} models from experiment {experiment_id}")
            return models
            
        except Exception as e:
            logger.error(f"Failed to fetch models from experiment {experiment_id}: {e}")
            return []
    
    def fetch_models_by_tags(self, tags: Dict[str, str]) -> List[ModelInfo]:
        """Fetch models by tags"""
        try:
            models = []
            
            # Search for registered models with specific tags
            registered_models = self.client.search_registered_models()
            
            for model in registered_models:
                try:
                    # Check if model has required tags
                    if self._has_required_tags(model, tags):
                        versions = self.client.get_latest_versions(model.name)
                        
                        for version in versions:
                            if self._should_include_model(version):
                                model_info = self._create_model_info(model, version)
                                models.append(model_info)
                                
                except Exception as e:
                    logger.error(f"Failed to process model {model.name}: {e}")
                    continue
            
            logger.info(f"Fetched {len(models)} models with tags {tags}")
            return models
            
        except Exception as e:
            logger.error(f"Failed to fetch models by tags: {e}")
            return []
    
    def download_model(self, model_info: ModelInfo) -> str:
        """Download model to local cache"""
        try:
            # Create model cache directory
            model_cache_dir = os.path.join(
                self.config.local_cache_dir,
                f"{model_info.name}_{model_info.version}"
            )
            os.makedirs(model_cache_dir, exist_ok=True)
            
            # Download model artifacts
            local_path = self.client.download_artifacts(
                model_info.run_id,
                model_info.artifact_path,
                model_cache_dir
            )
            
            logger.info(f"Downloaded model {model_info.name}:{model_info.version} to {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download model {model_info.name}:{model_info.version}: {e}")
            return ""
    
    def load_model(self, model_info: ModelInfo) -> Any:
        """Load model for testing"""
        try:
            # Download model if not cached
            if self.config.auto_download:
                local_path = self.download_model(model_info)
                if not local_path:
                    return None
            
            # Load model based on type
            model_uri = f"runs:/{model_info.run_id}/{model_info.artifact_path}"
            
            if model_info.model_type == ModelType.SKLEARN:
                return mlflow.sklearn.load_model(model_uri)
            elif model_info.model_type == ModelType.PYTORCH:
                return mlflow.pytorch.load_model(model_uri)
            elif model_info.model_type == ModelType.TENSORFLOW:
                return mlflow.tensorflow.load_model(model_uri)
            elif model_info.model_type == ModelType.XGBOOST:
                return mlflow.xgboost.load_model(model_uri)
            elif model_info.model_type == ModelType.LIGHTGBM:
                return mlflow.lightgbm.load_model(model_uri)
            else:
                # Generic model loading
                return mlflow.load_model(model_uri)
                
        except Exception as e:
            logger.error(f"Failed to load model {model_info.name}:{model_info.version}: {e}")
            return None
    
    def get_model_metadata(self, model_info: ModelInfo) -> Dict[str, Any]:
        """Get model metadata"""
        try:
            metadata = {
                "name": model_info.name,
                "version": model_info.version,
                "model_type": model_info.model_type.value,
                "status": model_info.status.value,
                "stage": model_info.stage,
                "description": model_info.description,
                "tags": model_info.tags,
                "created_at": model_info.created_at.isoformat(),
                "updated_at": model_info.updated_at.isoformat(),
                "run_id": model_info.run_id,
                "experiment_id": model_info.experiment_id,
                "artifact_path": model_info.artifact_path,
                "model_uri": model_info.model_uri
            }
            
            # Add custom metadata if available
            if model_info.metadata:
                metadata.update(model_info.metadata)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get model metadata: {e}")
            return {}
    
    def _should_include_model(self, model_version) -> bool:
        """Check if model should be included based on filters"""
        try:
            # Check model type filter
            if self.config.model_types:
                model_type = self._detect_model_type(model_version)
                if model_type not in self.config.model_types:
                    return False
            
            # Check stage filter
            if self.config.model_stages:
                if model_version.current_stage not in self.config.model_stages:
                    return False
            
            # Check archived filter
            if not self.config.include_archived:
                if model_version.current_stage == "Archived":
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check model inclusion: {e}")
            return False
    
    def _detect_model_type(self, model_version) -> ModelType:
        """Detect model type from version info"""
        try:
            # Check tags for model type
            tags = model_version.tags or {}
            model_type_tag = tags.get("model_type", "").lower()
            
            if model_type_tag in ["sklearn", "scikit-learn"]:
                return ModelType.SKLEARN
            elif model_type_tag in ["pytorch", "torch"]:
                return ModelType.PYTORCH
            elif model_type_tag in ["tensorflow", "tf"]:
                return ModelType.TENSORFLOW
            elif model_type_tag == "xgboost":
                return ModelType.XGBOOST
            elif model_type_tag == "lightgbm":
                return ModelType.LIGHTGBM
            else:
                # Try to detect from artifact path
                artifact_path = model_version.source
                if "sklearn" in artifact_path.lower():
                    return ModelType.SKLEARN
                elif "pytorch" in artifact_path.lower():
                    return ModelType.PYTORCH
                elif "tensorflow" in artifact_path.lower():
                    return ModelType.TENSORFLOW
                elif "xgboost" in artifact_path.lower():
                    return ModelType.XGBOOST
                elif "lightgbm" in artifact_path.lower():
                    return ModelType.LIGHTGBM
                else:
                    return ModelType.CUSTOM
                    
        except Exception as e:
            logger.error(f"Failed to detect model type: {e}")
            return ModelType.CUSTOM
    
    def _create_model_info(self, model, version) -> ModelInfo:
        """Create ModelInfo from model and version"""
        try:
            return ModelInfo(
                name=model.name,
                version=version.version,
                model_type=self._detect_model_type(version),
                status=ModelStatus(version.current_stage),
                stage=version.current_stage,
                description=version.description or "",
                tags=version.tags or {},
                created_at=datetime.fromtimestamp(version.creation_timestamp / 1000),
                updated_at=datetime.fromtimestamp(version.last_updated_timestamp / 1000),
                run_id=version.run_id,
                experiment_id=version.user_id,  # This might need adjustment
                artifact_path=version.source,
                model_uri=f"models:/{model.name}/{version.version}",
                metadata={}
            )
            
        except Exception as e:
            logger.error(f"Failed to create model info: {e}")
            return None
    
    def _create_model_info_from_version(self, model_name: str, version) -> ModelInfo:
        """Create ModelInfo from version only"""
        try:
            return ModelInfo(
                name=model_name,
                version=version.version,
                model_type=self._detect_model_type(version),
                status=ModelStatus(version.current_stage),
                stage=version.current_stage,
                description=version.description or "",
                tags=version.tags or {},
                created_at=datetime.fromtimestamp(version.creation_timestamp / 1000),
                updated_at=datetime.fromtimestamp(version.last_updated_timestamp / 1000),
                run_id=version.run_id,
                experiment_id=version.user_id,
                artifact_path=version.source,
                model_uri=f"models:/{model_name}/{version.version}",
                metadata={}
            )
            
        except Exception as e:
            logger.error(f"Failed to create model info from version: {e}")
            return None
    
    def _create_model_info_from_run(self, run) -> ModelInfo:
        """Create ModelInfo from run"""
        try:
            # Extract model information from run
            tags = run.get('tags', {})
            model_type = self._detect_model_type_from_run(run)
            
            return ModelInfo(
                name=tags.get('mlflow.runName', f"model_{run.run_id}"),
                version="1",
                model_type=model_type,
                status=ModelStatus.NONE,
                stage="None",
                description=run.get('description', ''),
                tags=tags,
                created_at=datetime.fromtimestamp(run.start_time / 1000),
                updated_at=datetime.fromtimestamp(run.end_time / 1000),
                run_id=run.run_id,
                experiment_id=run.experiment_id,
                artifact_path="model",
                model_uri=f"runs:/{run.run_id}/model",
                metadata={}
            )
            
        except Exception as e:
            logger.error(f"Failed to create model info from run: {e}")
            return None
    
    def _detect_model_type_from_run(self, run) -> ModelType:
        """Detect model type from run"""
        try:
            tags = run.get('tags', {})
            model_type_tag = tags.get('model_type', '').lower()
            
            if model_type_tag in ["sklearn", "scikit-learn"]:
                return ModelType.SKLEARN
            elif model_type_tag in ["pytorch", "torch"]:
                return ModelType.PYTORCH
            elif model_type_tag in ["tensorflow", "tf"]:
                return ModelType.TENSORFLOW
            elif model_type_tag == "xgboost":
                return ModelType.XGBOOST
            elif model_type_tag == "lightgbm":
                return ModelType.LIGHTGBM
            else:
                return ModelType.CUSTOM
                
        except Exception as e:
            logger.error(f"Failed to detect model type from run: {e}")
            return ModelType.CUSTOM
    
    def _has_required_tags(self, model, required_tags: Dict[str, str]) -> bool:
        """Check if model has required tags"""
        try:
            for key, value in required_tags.items():
                if key not in model.tags or model.tags[key] != value:
                    return False
            return True
            
        except Exception as e:
            logger.error(f"Failed to check required tags: {e}")
            return False
    
    def clear_cache(self):
        """Clear local model cache"""
        try:
            if os.path.exists(self.config.local_cache_dir):
                shutil.rmtree(self.config.local_cache_dir)
            os.makedirs(self.config.local_cache_dir, exist_ok=True)
            self.cached_models.clear()
            logger.info("Model cache cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            cache_size = 0
            if os.path.exists(self.config.local_cache_dir):
                for root, dirs, files in os.walk(self.config.local_cache_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        cache_size += os.path.getsize(file_path)
            
            return {
                "cached_models": len(self.cached_models),
                "cache_directory": self.config.local_cache_dir,
                "cache_size_bytes": cache_size,
                "cache_size_mb": cache_size / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
