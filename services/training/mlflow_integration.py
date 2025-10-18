"""
Comprehensive MLflow Integration for ML Security Service

This module provides full MLflow utilization including:
- Dataset logging and versioning
- Model lineage tracking
- Artifact management
- Experiment comparison
- Model registry management
- Performance monitoring
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import mlflow.data
from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus
import tempfile
import shutil

logger = logging.getLogger(__name__)

class MLflowIntegration:
    """Comprehensive MLflow integration for ML Security Service"""
    
    def __init__(self, tracking_uri: str = "http://mlflow:5000"):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)
        
        # Experiment configuration
        self.experiment_name = "ML Security Training"
        self.experiment_id = self._get_or_create_experiment()
        
        # Model registry
        self.registered_models = [
            "security_distilbert",
            "security_bert_base", 
            "security_roberta_base",
            "security_deberta_v3_base"
        ]
        
        # Dataset tracking
        self.dataset_registry = {}
        
    def _get_or_create_experiment(self) -> str:
        """Get or create MLflow experiment"""
        try:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment:
                return experiment.experiment_id
        except:
            pass
        
        # Create new experiment
        experiment_id = self.client.create_experiment(
            self.experiment_name,
            tags={
                "description": "ML Security Model Training and Evaluation",
                "created_by": "ml_security_service",
                "version": "1.0"
            }
        )
        logger.info(f"Created MLflow experiment: {self.experiment_name}")
        return experiment_id
    
    def get_or_create_model_experiment(self, model_name: str) -> str:
        """Get or create model-specific experiment"""
        experiment_name = f"security_{model_name}_experiments"
        
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment:
                return experiment.experiment_id
        except:
            pass
        
        # Create model-specific experiment
        experiment_id = self.client.create_experiment(
            experiment_name,
            tags={
                "description": f"Security model training experiments for {model_name}",
                "model_family": model_name,
                "task_type": "security_classification",
                "domain": "cybersecurity",
                "created_by": "ml_security_service",
                "version": "1.0"
            }
        )
        logger.info(f"Created model-specific experiment: {experiment_name}")
        return experiment_id
    
    def log_dataset(
        self, 
        dataset_name: str,
        data: List[Dict],
        data_type: str = "training",
        run_id: Optional[str] = None
    ) -> str:
        """Log dataset to MLflow with comprehensive metadata"""
        try:
            # Create temporary file for dataset
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
                temp_path = f.name
            
            # Create dataset from file using pandas (MLflow 2.8.1 compatible)
            import pandas as pd
            df = pd.read_json(temp_path, lines=True)
            dataset = mlflow.data.from_pandas(
                df,
                targets="label",
                name=dataset_name
            )
            
            # Log dataset
            if run_id:
                with mlflow.start_run(run_id=run_id, nested=True):
                    dataset_info = mlflow.log_input(dataset, context=data_type)
            else:
                with mlflow.start_run(experiment_id=self.experiment_id, nested=True):
                    dataset_info = mlflow.log_input(dataset, context=data_type)
            
            # Log dataset metadata
            dataset_metadata = {
                "dataset_name": dataset_name,
                "data_type": data_type,
                "total_samples": len(data),
                "label_distribution": self._get_label_distribution(data),
                "text_length_stats": self._get_text_length_stats(data),
                "created_at": datetime.now().isoformat()
            }
            
            mlflow.log_params(dataset_metadata)
            
            # Store in registry
            self.dataset_registry[dataset_name] = {
                "dataset_info": dataset_info,
                "metadata": dataset_metadata,
                "run_id": run_id or mlflow.active_run().info.run_id
            }
            
            # Cleanup
            os.unlink(temp_path)
            
            logger.info(f"✅ Logged dataset {dataset_name} with {len(data)} samples")
            return dataset_info.digest
            
        except Exception as e:
            logger.error(f"❌ Error logging dataset {dataset_name}: {e}")
            raise
    
    def _get_label_distribution(self, data: List[Dict]) -> Dict[str, int]:
        """Get label distribution from dataset"""
        label_counts = {}
        for item in data:
            label = item.get('label', 'unknown')
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts
    
    def _get_text_length_stats(self, data: List[Dict]) -> Dict[str, float]:
        """Get text length statistics from dataset"""
        lengths = [len(item.get('text', '')) for item in data]
        if not lengths:
            return {}
        
        return {
            "mean_length": float(np.mean(lengths)),
            "std_length": float(np.std(lengths)),
            "min_length": int(np.min(lengths)),
            "max_length": int(np.max(lengths)),
            "median_length": float(np.median(lengths))
        }
    
    def log_training_run(
        self,
        model_name: str,
        model_version: str,
        training_data: List[Dict],
        validation_data: List[Dict],
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        model: Any,
        tokenizer: Any,
        artifacts: Dict[str, str] = None
    ) -> str:
        """Log comprehensive training run to MLflow"""
        try:
            run_name = f"{model_name}_{model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name) as run:
                run_id = run.info.run_id
                
                # Log basic parameters
                mlflow.log_params({
                    "model_name": model_name,
                    "model_version": model_version,
                    "training_type": "security_classification",
                    "framework": "pytorch",
                    "transformer_model": hyperparameters.get("base_model", model_name),
                    **hyperparameters
                })
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log datasets
                self.log_dataset(
                    f"{model_name}_training_data",
                    training_data,
                    "training",
                    run_id
                )
                
                self.log_dataset(
                    f"{model_name}_validation_data", 
                    validation_data,
                    "validation",
                    run_id
                )
                
                # Log model
                model_uri = self._log_model_with_artifacts(
                    model, tokenizer, model_name, artifacts or {}
                )
                
                # Log additional artifacts
                self._log_training_artifacts(run_id, artifacts or {})
                
                # Log tags
                mlflow.set_tags({
                    "model_type": "security_classification",
                    "domain": "cybersecurity",
                    "use_case": "prompt_injection_detection",
                    "status": "training_completed",
                    "created_by": "ml_security_service"
                })
                
                # Register model
                self._register_model(model_name, model_version, model_uri, metrics)
                
                logger.info(f"✅ Logged training run {run_id} for {model_name}")
                return run_id
                
        except Exception as e:
            logger.error(f"❌ Error logging training run: {e}")
            raise
    
    def _log_model_with_artifacts(
        self, 
        model: Any, 
        tokenizer: Any, 
        model_name: str,
        artifacts: Dict[str, str]
    ) -> str:
        """Log model with all artifacts"""
        try:
            # Save tokenizer to temporary directory
            temp_tokenizer_dir = tempfile.mkdtemp()
            tokenizer.save_pretrained(temp_tokenizer_dir)
            
            # Prepare extra files
            extra_files = []
            for artifact_name, artifact_path in artifacts.items():
                if os.path.exists(artifact_path):
                    extra_files.append(artifact_path)
            
            # Add tokenizer files
            tokenizer_files = [
                os.path.join(temp_tokenizer_dir, f) 
                for f in os.listdir(temp_tokenizer_dir)
                if f.endswith(('.json', '.txt', '.model'))
            ]
            extra_files.extend(tokenizer_files)
            
            # Log model
            model_uri = mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=f"security_{model_name}",
                extra_files=extra_files,
                signature=self._create_model_signature(),
                input_example=self._create_input_example()
            )
            
            # Cleanup
            shutil.rmtree(temp_tokenizer_dir, ignore_errors=True)
            
            return model_uri
            
        except Exception as e:
            logger.error(f"❌ Error logging model: {e}")
            raise
    
    def _create_model_signature(self):
        """Create MLflow model signature"""
        try:
            from mlflow.types.schema import Schema, ColSpec
            from mlflow.models.signature import ModelSignature
            
            input_schema = Schema([
                ColSpec("string", "text")
            ])
            
            output_schema = Schema([
                ColSpec("string", "predicted_label"),
                ColSpec("double", "confidence_score")
            ])
            
            return ModelSignature(inputs=input_schema, outputs=output_schema)
        except:
            return None
    
    def _create_input_example(self):
        """Create input example for model"""
        return {
            "text": "This is a sample security prompt for testing"
        }
    
    def _log_training_artifacts(self, run_id: str, artifacts: Dict[str, str]):
        """Log additional training artifacts"""
        try:
            for artifact_name, artifact_path in artifacts.items():
                if os.path.exists(artifact_path):
                    mlflow.log_artifact(artifact_path, artifact_name)
        except Exception as e:
            logger.error(f"❌ Error logging artifacts: {e}")
    
    def _register_model(
        self, 
        model_name: str, 
        version: str, 
        model_uri: str, 
        metrics: Dict[str, float]
    ):
        """Register model in MLflow Model Registry"""
        try:
            registered_model_name = f"security_{model_name}"
            
            # Create model version
            model_version = self.client.create_model_version(
                name=registered_model_name,
                source=model_uri,
                run_id=mlflow.active_run().info.run_id,
                description=f"Security model {model_name} version {version}"
            )
            
            # Add model version tags
            self.client.set_model_version_tag(
                registered_model_name,
                model_version.version,
                "accuracy",
                str(metrics.get("eval_accuracy", 0.0))
            )
            
            self.client.set_model_version_tag(
                registered_model_name,
                model_version.version,
                "f1_score",
                str(metrics.get("eval_f1", 0.0))
            )
            
            # Set model stage based on performance
            if metrics.get("eval_accuracy", 0.0) > 0.9:
                self.client.transition_model_version_stage(
                    registered_model_name,
                    model_version.version,
                    "Production"
                )
            else:
                self.client.transition_model_version_stage(
                    registered_model_name,
                    model_version.version,
                    "Staging"
                )
            
            logger.info(f"✅ Registered model {registered_model_name} version {model_version.version}")
            
        except Exception as e:
            logger.error(f"❌ Error registering model: {e}")
    
    def log_red_team_results(
        self,
        model_name: str,
        test_results: List[Dict],
        attack_patterns: List[Dict],
        run_id: Optional[str] = None
    ):
        """Log red team test results to MLflow"""
        try:
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    self._log_red_team_metrics(test_results, attack_patterns)
            else:
                with mlflow.start_run(experiment_id=self.experiment_id):
                    self._log_red_team_metrics(test_results, attack_patterns)
                    
        except Exception as e:
            logger.error(f"❌ Error logging red team results: {e}")
    
    def _log_red_team_metrics(self, test_results: List[Dict], attack_patterns: List[Dict]):
        """Log red team metrics and analysis"""
        try:
            # Calculate metrics
            total_tests = len(test_results)
            successful_attacks = sum(1 for r in test_results if not r.get('detected', False))
            detection_rate = (total_tests - successful_attacks) / total_tests if total_tests > 0 else 0
            
            # Log metrics
            mlflow.log_metrics({
                "red_team_total_tests": total_tests,
                "red_team_successful_attacks": successful_attacks,
                "red_team_detection_rate": detection_rate,
                "red_team_vulnerability_rate": 1 - detection_rate
            })
            
            # Log attack pattern analysis
            attack_stats = {}
            for pattern in attack_patterns:
                attack_type = pattern.get('attack_type', 'unknown')
                attack_stats[f"attack_{attack_type}_count"] = attack_stats.get(f"attack_{attack_type}_count", 0) + 1
            
            mlflow.log_params(attack_stats)
            
            # Log detailed results as artifact
            results_artifact = {
                "test_results": test_results,
                "attack_patterns": attack_patterns,
                "summary": {
                    "total_tests": total_tests,
                    "successful_attacks": successful_attacks,
                    "detection_rate": detection_rate
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(results_artifact, f, indent=2)
                mlflow.log_artifact(f.name, "red_team_results")
                os.unlink(f.name)
            
        except Exception as e:
            logger.error(f"❌ Error logging red team metrics: {e}")
    
    def get_model_performance_history(self, model_name: str) -> List[Dict]:
        """Get model performance history from MLflow"""
        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"params.model_name = '{model_name}'",
                order_by=["start_time DESC"]
            )
            
            performance_history = []
            for run in runs:
                performance_history.append({
                    "run_id": run.info.run_id,
                    "start_time": run.info.start_time,
                    "status": run.info.status,
                    "metrics": run.data.metrics,
                    "parameters": run.data.params,
                    "tags": run.data.tags
                })
            
            return performance_history
            
        except Exception as e:
            logger.error(f"❌ Error getting performance history: {e}")
            return []
    
    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Compare multiple models using MLflow data"""
        try:
            comparison = {}
            
            for model_name in model_names:
                runs = self.client.search_runs(
                    experiment_ids=[self.experiment_id],
                    filter_string=f"params.model_name = '{model_name}'",
                    order_by=["start_time DESC"],
                    max_results=1
                )
                
                if runs:
                    run = runs[0]
                    comparison[model_name] = {
                        "latest_accuracy": run.data.metrics.get("eval_accuracy", 0.0),
                        "latest_f1": run.data.metrics.get("eval_f1", 0.0),
                        "latest_run_id": run.info.run_id,
                        "last_trained": run.info.start_time
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"❌ Error comparing models: {e}")
            return {}
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get comprehensive experiment summary"""
        try:
            # Get all runs
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=["start_time DESC"]
            )
            
            # Calculate statistics
            total_runs = len(runs)
            successful_runs = sum(1 for run in runs if run.info.status == RunStatus.FINISHED)
            
            # Get model performance summary
            model_performance = {}
            for run in runs:
                if run.info.status == RunStatus.FINISHED:
                    model_name = run.data.params.get("model_name", "unknown")
                    if model_name not in model_performance:
                        model_performance[model_name] = []
                    
                    model_performance[model_name].append({
                        "accuracy": run.data.metrics.get("eval_accuracy", 0.0),
                        "f1": run.data.metrics.get("eval_f1", 0.0),
                        "run_id": run.info.run_id,
                        "timestamp": run.info.start_time
                    })
            
            # Calculate best models
            best_models = {}
            for model_name, performances in model_performance.items():
                if performances:
                    best_performance = max(performances, key=lambda x: x["accuracy"])
                    best_models[model_name] = best_performance
            
            return {
                "experiment_name": self.experiment_name,
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
                "model_performance": model_performance,
                "best_models": best_models,
                "experiment_id": self.experiment_id
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting experiment summary: {e}")
            return {}
    
    def cleanup_old_runs(self, days_old: int = 30):
        """Cleanup old MLflow runs"""
        try:
            cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"start_time < {int(cutoff_time * 1000)}"
            )
            
            deleted_count = 0
            for run in runs:
                try:
                    self.client.delete_run(run.info.run_id)
                    deleted_count += 1
                except:
                    pass
            
            logger.info(f"✅ Cleaned up {deleted_count} old runs")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up runs: {e}")
            return 0
    
    def compare_model_experiments(self, model_names: List[str]) -> Dict[str, Any]:
        """Compare experiments across different models"""
        try:
            comparison = {}
            
            for model_name in model_names:
                experiment_name = f"security_{model_name}_experiments"
                try:
                    experiment = self.client.get_experiment_by_name(experiment_name)
                    if experiment:
                        runs = self.client.search_runs(
                            experiment_ids=[experiment.experiment_id],
                            order_by=["start_time DESC"],
                            max_results=5
                        )
                        
                        comparison[model_name] = {
                            "experiment_id": experiment.experiment_id,
                            "total_runs": len(runs),
                            "latest_runs": [
                                {
                                    "run_id": run.info.run_id,
                                    "start_time": run.info.start_time,
                                    "status": run.info.status,
                                    "metrics": run.data.metrics,
                                    "tags": run.data.tags
                                }
                                for run in runs[:3]  # Latest 3 runs
                            ]
                        }
                except Exception as e:
                    logger.warning(f"Could not get experiment for {model_name}: {e}")
                    comparison[model_name] = {"error": str(e)}
            
            return comparison
            
        except Exception as e:
            logger.error(f"❌ Error comparing model experiments: {e}")
            return {}
    
    def get_experiment_analytics(self) -> Dict[str, Any]:
        """Get comprehensive experiment analytics"""
        try:
            # Get all experiments
            experiments = self.client.search_experiments()
            
            analytics = {
                "total_experiments": len(experiments),
                "model_experiments": {},
                "overall_stats": {
                    "total_runs": 0,
                    "successful_runs": 0,
                    "failed_runs": 0
                }
            }
            
            for exp in experiments:
                if exp.name.startswith("security_") and exp.name.endswith("_experiments"):
                    model_name = exp.name.replace("security_", "").replace("_experiments", "")
                    
                    runs = self.client.search_runs(
                        experiment_ids=[exp.experiment_id],
                        order_by=["start_time DESC"]
                    )
                    
                    successful_runs = sum(1 for run in runs if run.info.status == RunStatus.FINISHED)
                    failed_runs = sum(1 for run in runs if run.info.status == RunStatus.FAILED)
                    
                    analytics["model_experiments"][model_name] = {
                        "experiment_id": exp.experiment_id,
                        "total_runs": len(runs),
                        "successful_runs": successful_runs,
                        "failed_runs": failed_runs,
                        "success_rate": successful_runs / len(runs) if runs else 0
                    }
                    
                    analytics["overall_stats"]["total_runs"] += len(runs)
                    analytics["overall_stats"]["successful_runs"] += successful_runs
                    analytics["overall_stats"]["failed_runs"] += failed_runs
            
            # Calculate overall success rate
            total_runs = analytics["overall_stats"]["total_runs"]
            if total_runs > 0:
                analytics["overall_stats"]["success_rate"] = (
                    analytics["overall_stats"]["successful_runs"] / total_runs
                )
            
            return analytics
            
        except Exception as e:
            logger.error(f"❌ Error getting experiment analytics: {e}")
            return {}

# Global instance
mlflow_integration = MLflowIntegration()
