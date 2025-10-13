"""
MLflow Coordinator
Coordinates all MLflow integration components for comprehensive model management and testing.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json

from .model_fetcher import ModelFetcher, FetchConfig, ModelInfo
from .version_manager import VersionManager, VersionTestConfig, VersionStrategy

logger = logging.getLogger(__name__)

class TestingScope(Enum):
    """Testing scope options"""
    SINGLE_MODEL = "single_model"
    ALL_MODELS = "all_models"
    BY_EXPERIMENT = "by_experiment"
    BY_TAGS = "by_tags"
    BY_STAGE = "by_stage"

@dataclass
class MLflowConfig:
    """MLflow configuration"""
    tracking_uri: str = "file:///tmp/mlflow"
    registry_uri: str = "sqlite:///mlflow.db"
    model_stages: List[str] = None
    model_types: List[str] = None
    max_models: int = 100
    include_archived: bool = False
    local_cache_dir: str = "/tmp/mlflow_models"
    auto_download: bool = True

@dataclass
class TestingConfig:
    """Testing configuration"""
    scope: TestingScope
    model_name: str = ""
    experiment_id: str = ""
    tags: Dict[str, str] = None
    version_strategy: VersionStrategy = VersionStrategy.LATEST
    max_versions: int = 10
    include_archived: bool = False

class MLflowCoordinator:
    """Coordinates MLflow integration for comprehensive model testing"""
    
    def __init__(self, config: Optional[MLflowConfig] = None):
        self.config = config or MLflowConfig()
        
        # Initialize components
        fetch_config = FetchConfig(
            registry_uri=self.config.registry_uri,
            tracking_uri=self.config.tracking_uri,
            model_stages=self.config.model_stages,
            model_types=self.config.model_types,
            max_models=self.config.max_models,
            include_archived=self.config.include_archived,
            local_cache_dir=self.config.local_cache_dir,
            auto_download=self.config.auto_download
        )
        
        self.model_fetcher = ModelFetcher(fetch_config)
        self.version_manager = VersionManager(self.config.tracking_uri)
        
        # Testing state
        self.testing_queue: List[ModelInfo] = []
        self.testing_results: Dict[str, Any] = {}
    
    def setup_testing_environment(self, testing_config: TestingConfig) -> Dict[str, Any]:
        """Setup testing environment based on configuration"""
        try:
            models = []
            
            if testing_config.scope == TestingScope.SINGLE_MODEL:
                models = self._setup_single_model_testing(testing_config)
            elif testing_config.scope == TestingScope.ALL_MODELS:
                models = self._setup_all_models_testing(testing_config)
            elif testing_config.scope == TestingScope.BY_EXPERIMENT:
                models = self._setup_experiment_testing(testing_config)
            elif testing_config.scope == TestingScope.BY_TAGS:
                models = self._setup_tags_testing(testing_config)
            elif testing_config.scope == TestingScope.BY_STAGE:
                models = self._setup_stage_testing(testing_config)
            
            # Update testing queue
            self.testing_queue = models
            
            # Create version test config
            version_config = VersionTestConfig(
                strategy=testing_config.version_strategy,
                max_versions=testing_config.max_versions,
                include_archived=testing_config.include_archived
            )
            
            # Get versions for each model
            model_versions = {}
            for model in models:
                versions = self.version_manager.get_model_versions(
                    model.name, version_config
                )
                model_versions[model.name] = versions
            
            setup_info = {
                "models_count": len(models),
                "models": [self._model_to_dict(model) for model in models],
                "model_versions": model_versions,
                "testing_config": {
                    "scope": testing_config.scope.value,
                    "version_strategy": testing_config.version_strategy.value,
                    "max_versions": testing_config.max_versions
                },
                "setup_time": datetime.now().isoformat()
            }
            
            logger.info(f"Setup testing environment with {len(models)} models")
            return setup_info
            
        except Exception as e:
            logger.error(f"Failed to setup testing environment: {e}")
            return {}
    
    def _setup_single_model_testing(self, config: TestingConfig) -> List[ModelInfo]:
        """Setup testing for single model"""
        try:
            model = self.model_fetcher.fetch_model_by_name(config.model_name)
            return [model] if model else []
            
        except Exception as e:
            logger.error(f"Failed to setup single model testing: {e}")
            return []
    
    def _setup_all_models_testing(self, config: TestingConfig) -> List[ModelInfo]:
        """Setup testing for all models"""
        try:
            return self.model_fetcher.fetch_all_models()
            
        except Exception as e:
            logger.error(f"Failed to setup all models testing: {e}")
            return []
    
    def _setup_experiment_testing(self, config: TestingConfig) -> List[ModelInfo]:
        """Setup testing for models from experiment"""
        try:
            return self.model_fetcher.fetch_models_by_experiment(config.experiment_id)
            
        except Exception as e:
            logger.error(f"Failed to setup experiment testing: {e}")
            return []
    
    def _setup_tags_testing(self, config: TestingConfig) -> List[ModelInfo]:
        """Setup testing for models with specific tags"""
        try:
            return self.model_fetcher.fetch_models_by_tags(config.tags or {})
            
        except Exception as e:
            logger.error(f"Failed to setup tags testing: {e}")
            return []
    
    def _setup_stage_testing(self, config: TestingConfig) -> List[ModelInfo]:
        """Setup testing for models by stage"""
        try:
            all_models = self.model_fetcher.fetch_all_models()
            return [model for model in all_models if model.stage in (config.tags or {}).values()]
            
        except Exception as e:
            logger.error(f"Failed to setup stage testing: {e}")
            return []
    
    def execute_security_tests(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute security tests on queued models"""
        try:
            results = {
                "test_id": f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "start_time": datetime.now().isoformat(),
                "models_tested": 0,
                "total_vulnerabilities": 0,
                "total_attacks": 0,
                "model_results": [],
                "summary": {}
            }
            
            for model in self.testing_queue:
                try:
                    model_result = self._test_single_model(model, test_config)
                    results["model_results"].append(model_result)
                    results["models_tested"] += 1
                    results["total_vulnerabilities"] += model_result.get("vulnerabilities_count", 0)
                    results["total_attacks"] += model_result.get("attacks_count", 0)
                    
                except Exception as e:
                    logger.error(f"Failed to test model {model.name}: {e}")
                    results["model_results"].append({
                        "model_name": model.name,
                        "version": model.version,
                        "error": str(e),
                        "status": "failed"
                    })
            
            # Generate summary
            results["summary"] = self._generate_test_summary(results["model_results"])
            results["end_time"] = datetime.now().isoformat()
            
            # Store results
            self.testing_results[results["test_id"]] = results
            
            logger.info(f"Completed security testing for {results['models_tested']} models")
            return results
            
        except Exception as e:
            logger.error(f"Failed to execute security tests: {e}")
            return {}
    
    def _test_single_model(self, model: ModelInfo, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single model"""
        try:
            # Load model
            loaded_model = self.model_fetcher.load_model(model)
            if not loaded_model:
                return {
                    "model_name": model.name,
                    "version": model.version,
                    "status": "failed",
                    "error": "Failed to load model"
                }
            
            # Get model metadata
            metadata = self.model_fetcher.get_model_metadata(model)
            
            # Simulate security testing (in real implementation, this would call actual security tests)
            vulnerabilities = self._simulate_vulnerability_detection(model, loaded_model)
            attacks = self._simulate_attack_execution(model, loaded_model)
            
            return {
                "model_name": model.name,
                "version": model.version,
                "model_type": model.model_type.value,
                "stage": model.stage,
                "status": "completed",
                "vulnerabilities_count": len(vulnerabilities),
                "attacks_count": len(attacks),
                "vulnerabilities": vulnerabilities,
                "attacks": attacks,
                "metadata": metadata,
                "test_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to test model {model.name}: {e}")
            return {
                "model_name": model.name,
                "version": model.version,
                "status": "failed",
                "error": str(e)
            }
    
    def _simulate_vulnerability_detection(self, model: ModelInfo, loaded_model: Any) -> List[Dict[str, Any]]:
        """Simulate vulnerability detection (placeholder for actual implementation)"""
        try:
            # This is a placeholder - in real implementation, this would call actual vulnerability detection
            vulnerabilities = []
            
            # Simulate based on model type
            if model.model_type.value == "sklearn":
                vulnerabilities.append({
                    "type": "injection",
                    "severity": "medium",
                    "description": "Potential input validation vulnerability",
                    "confidence": 0.7
                })
            elif model.model_type.value == "pytorch":
                vulnerabilities.append({
                    "type": "adversarial",
                    "severity": "high",
                    "description": "Susceptible to adversarial attacks",
                    "confidence": 0.8
                })
            
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Failed to simulate vulnerability detection: {e}")
            return []
    
    def _simulate_attack_execution(self, model: ModelInfo, loaded_model: Any) -> List[Dict[str, Any]]:
        """Simulate attack execution (placeholder for actual implementation)"""
        try:
            # This is a placeholder - in real implementation, this would call actual attack modules
            attacks = []
            
            # Simulate based on model type
            if model.model_type.value == "sklearn":
                attacks.append({
                    "type": "evasion",
                    "success": True,
                    "confidence": 0.6,
                    "description": "Evasion attack successful"
                })
            elif model.model_type.value == "pytorch":
                attacks.append({
                    "type": "adversarial",
                    "success": True,
                    "confidence": 0.8,
                    "description": "Adversarial attack successful"
                })
            
            return attacks
            
        except Exception as e:
            logger.error(f"Failed to simulate attack execution: {e}")
            return []
    
    def _generate_test_summary(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate test summary"""
        try:
            total_models = len(model_results)
            successful_tests = len([r for r in model_results if r.get("status") == "completed"])
            failed_tests = len([r for r in model_results if r.get("status") == "failed"])
            
            total_vulnerabilities = sum(r.get("vulnerabilities_count", 0) for r in model_results)
            total_attacks = sum(r.get("attacks_count", 0) for r in model_results)
            
            # Group by model type
            model_types = {}
            for result in model_results:
                model_type = result.get("model_type", "unknown")
                if model_type not in model_types:
                    model_types[model_type] = {"count": 0, "vulnerabilities": 0, "attacks": 0}
                model_types[model_type]["count"] += 1
                model_types[model_type]["vulnerabilities"] += result.get("vulnerabilities_count", 0)
                model_types[model_type]["attacks"] += result.get("attacks_count", 0)
            
            # Group by stage
            stages = {}
            for result in model_results:
                stage = result.get("stage", "unknown")
                if stage not in stages:
                    stages[stage] = {"count": 0, "vulnerabilities": 0, "attacks": 0}
                stages[stage]["count"] += 1
                stages[stage]["vulnerabilities"] += result.get("vulnerabilities_count", 0)
                stages[stage]["attacks"] += result.get("attacks_count", 0)
            
            return {
                "total_models": total_models,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": successful_tests / total_models if total_models > 0 else 0,
                "total_vulnerabilities": total_vulnerabilities,
                "total_attacks": total_attacks,
                "model_types": model_types,
                "stages": stages,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate test summary: {e}")
            return {}
    
    def get_testing_status(self) -> Dict[str, Any]:
        """Get current testing status"""
        try:
            return {
                "queue_size": len(self.testing_queue),
                "completed_tests": len(self.testing_results),
                "last_test_time": max(
                    [r["start_time"] for r in self.testing_results.values()]
                ) if self.testing_results else None,
                "cache_stats": self.model_fetcher.get_cache_stats()
            }
            
        except Exception as e:
            logger.error(f"Failed to get testing status: {e}")
            return {}
    
    def get_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get test results by ID"""
        try:
            return self.testing_results.get(test_id)
            
        except Exception as e:
            logger.error(f"Failed to get test results: {e}")
            return None
    
    def get_all_test_results(self) -> Dict[str, Any]:
        """Get all test results"""
        try:
            return {
                "total_tests": len(self.testing_results),
                "tests": list(self.testing_results.keys()),
                "results": self.testing_results
            }
            
        except Exception as e:
            logger.error(f"Failed to get all test results: {e}")
            return {}
    
    def export_test_results(self, test_id: str, format_type: str = "json") -> str:
        """Export test results"""
        try:
            results = self.get_test_results(test_id)
            if not results:
                return "{}"
            
            if format_type == "json":
                return json.dumps(results, indent=2)
            else:
                return str(results)
                
        except Exception as e:
            logger.error(f"Failed to export test results: {e}")
            return "{}"
    
    def _model_to_dict(self, model: ModelInfo) -> Dict[str, Any]:
        """Convert ModelInfo to dictionary"""
        try:
            return {
                "name": model.name,
                "version": model.version,
                "model_type": model.model_type.value,
                "status": model.status.value,
                "stage": model.stage,
                "description": model.description,
                "tags": model.tags,
                "created_at": model.created_at.isoformat(),
                "updated_at": model.updated_at.isoformat(),
                "run_id": model.run_id,
                "experiment_id": model.experiment_id,
                "artifact_path": model.artifact_path,
                "model_uri": model.model_uri
            }
            
        except Exception as e:
            logger.error(f"Failed to convert model to dict: {e}")
            return {}
    
    def clear_testing_queue(self):
        """Clear testing queue"""
        self.testing_queue.clear()
        logger.info("Testing queue cleared")
    
    def clear_test_results(self):
        """Clear test results"""
        self.testing_results.clear()
        logger.info("Test results cleared")
    
    def get_mlflow_statistics(self) -> Dict[str, Any]:
        """Get MLflow statistics"""
        try:
            return {
                "tracking_uri": self.config.tracking_uri,
                "registry_uri": self.config.registry_uri,
                "cache_stats": self.model_fetcher.get_cache_stats(),
                "testing_status": self.get_testing_status(),
                "configuration": {
                    "model_stages": self.config.model_stages,
                    "model_types": self.config.model_types,
                    "max_models": self.config.max_models,
                    "include_archived": self.config.include_archived,
                    "auto_download": self.config.auto_download
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get MLflow statistics: {e}")
            return {}
