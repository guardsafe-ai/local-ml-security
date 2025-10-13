"""
Advanced Retraining System for ML Security Models

This module provides comprehensive retraining capabilities with full metadata tracking,
model lineage, performance comparison, and enterprise-grade features.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid

import mlflow
import mlflow.pytorch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

logger = logging.getLogger(__name__)

class RetrainingTrigger(Enum):
    """Types of retraining triggers"""
    MANUAL = "manual"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    SCHEDULED = "scheduled"
    RED_TEAM_FAILURE = "red_team_failure"
    NEW_DATA = "new_data"

class ModelStatus(Enum):
    """Model deployment status"""
    TRAINING = "training"
    TRAINED = "trained"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    FAILED = "failed"

@dataclass
class RetrainingMetadata:
    """Comprehensive retraining metadata"""
    retraining_id: str
    parent_model_id: str
    parent_model_version: str
    trigger: RetrainingTrigger
    trigger_reason: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: ModelStatus = ModelStatus.TRAINING
    training_data_path: str = ""
    training_data_hash: str = ""
    training_data_size: int = 0
    hyperparameters: Dict[str, Any] = None
    performance_metrics: Dict[str, float] = None
    parent_performance: Dict[str, float] = None
    performance_delta: Dict[str, float] = None
    training_duration: float = 0.0
    gpu_usage: Dict[str, Any] = None
    memory_usage: Dict[str, Any] = None
    error_message: str = ""
    retraining_notes: str = ""
    created_by: str = "system"
    
    def __post_init__(self):
        if self.hyperparameters is None:
            self.hyperparameters = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.parent_performance is None:
            self.parent_performance = {}
        if self.performance_delta is None:
            self.performance_delta = {}
        if self.gpu_usage is None:
            self.gpu_usage = {}
        if self.memory_usage is None:
            self.memory_usage = {}

@dataclass
class ModelLineage:
    """Model lineage and versioning information"""
    model_id: str
    model_name: str
    version: str
    parent_versions: List[str]
    children_versions: List[str]
    creation_time: datetime
    last_updated: datetime
    total_retraining_count: int = 0
    performance_history: List[Dict[str, Any]] = None
    deployment_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.parent_versions is None:
            self.parent_versions = []
        if self.children_versions is None:
            self.children_versions = []
        if self.performance_history is None:
            self.performance_history = []
        if self.deployment_history is None:
            self.deployment_history = []

class AdvancedRetrainingSystem:
    """Advanced retraining system with comprehensive metadata tracking"""
    
    def __init__(self, mlflow_tracking_uri: str = "http://mlflow:5000"):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Initialize lineage tracking
        self.lineage_file = Path("/app/data/model_lineage.json")
        self.lineage_data = self._load_lineage_data()
        
        # Initialize retraining metadata storage
        self.metadata_file = Path("/app/data/retraining_metadata.json")
        self.metadata_data = self._load_metadata()
        
        # Performance thresholds for automatic retraining
        self.performance_thresholds = {
            "accuracy_drop": 0.05,  # 5% accuracy drop triggers retraining
            "f1_drop": 0.05,        # 5% F1 drop triggers retraining
            "precision_drop": 0.05, # 5% precision drop triggers retraining
            "recall_drop": 0.05     # 5% recall drop triggers retraining
        }
    
    def _load_lineage_data(self) -> Dict[str, ModelLineage]:
        """Load model lineage data"""
        if self.lineage_file.exists():
            try:
                with open(self.lineage_file, 'r') as f:
                    data = json.load(f)
                    return {k: ModelLineage(**v) for k, v in data.items()}
            except Exception as e:
                logger.error(f"Error loading lineage data: {e}")
        return {}
    
    def _save_lineage_data(self):
        """Save model lineage data"""
        try:
            with open(self.lineage_file, 'w') as f:
                data = {k: asdict(v) for k, v in self.lineage_data.items()}
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving lineage data: {e}")
    
    def _load_metadata(self) -> Dict[str, RetrainingMetadata]:
        """Load retraining metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    return {k: RetrainingMetadata(**v) for k, v in data.items()}
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save retraining metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                data = {k: asdict(v) for k, v in self.metadata_data.items()}
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def _calculate_data_hash(self, data_path: str) -> str:
        """Calculate hash of training data for change detection"""
        try:
            with open(data_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating data hash: {e}")
            return ""
    
    def _get_parent_model_info(self, model_name: str) -> Tuple[str, str, Dict[str, float]]:
        """Get parent model information for lineage tracking"""
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            # Get latest version of the model
            latest_version = client.get_latest_versions(f"security_{model_name}", stages=["None"])
            if not latest_version:
                return "", "", {}
            
            parent_version = latest_version[0].version
            parent_model_id = f"security_{model_name}_{parent_version}"
            
            # Get parent performance metrics
            parent_run = client.get_run(latest_version[0].run_id)
            parent_metrics = parent_run.data.metrics
            
            return parent_model_id, parent_version, parent_metrics
            
        except Exception as e:
            logger.error(f"Error getting parent model info: {e}")
            return "", "", {}
    
    def start_retraining(
        self,
        model_name: str,
        training_data_path: str,
        trigger: RetrainingTrigger,
        trigger_reason: str,
        hyperparameters: Dict[str, Any] = None,
        retraining_notes: str = "",
        created_by: str = "system"
    ) -> str:
        """Start retraining process with comprehensive metadata tracking"""
        
        retraining_id = str(uuid.uuid4())
        
        # Get parent model information
        parent_model_id, parent_version, parent_metrics = self._get_parent_model_info(model_name)
        
        # Calculate training data hash
        data_hash = self._calculate_data_hash(training_data_path)
        
        # Get training data size
        try:
            with open(training_data_path, 'r') as f:
                data_size = sum(1 for line in f)
        except:
            data_size = 0
        
        # Create retraining metadata
        metadata = RetrainingMetadata(
            retraining_id=retraining_id,
            parent_model_id=parent_model_id,
            parent_model_version=parent_version,
            trigger=trigger,
            trigger_reason=trigger_reason,
            start_time=datetime.now(),
            training_data_path=training_data_path,
            training_data_hash=data_hash,
            training_data_size=data_size,
            hyperparameters=hyperparameters or {},
            parent_performance=parent_metrics,
            retraining_notes=retraining_notes,
            created_by=created_by
        )
        
        # Store metadata
        self.metadata_data[retraining_id] = metadata
        self._save_metadata()
        
        logger.info(f"Started retraining {retraining_id} for model {model_name}")
        logger.info(f"Parent model: {parent_model_id} v{parent_version}")
        logger.info(f"Trigger: {trigger.value} - {trigger_reason}")
        
        return retraining_id
    
    def complete_retraining(
        self,
        retraining_id: str,
        new_model_version: str,
        performance_metrics: Dict[str, float],
        training_duration: float,
        gpu_usage: Dict[str, Any] = None,
        memory_usage: Dict[str, Any] = None,
        error_message: str = ""
    ):
        """Complete retraining process and update metadata"""
        
        if retraining_id not in self.metadata_data:
            raise ValueError(f"Retraining ID {retraining_id} not found")
        
        metadata = self.metadata_data[retraining_id]
        metadata.end_time = datetime.now()
        metadata.status = ModelStatus.FAILED if error_message else ModelStatus.TRAINED
        metadata.performance_metrics = performance_metrics
        metadata.training_duration = training_duration
        metadata.gpu_usage = gpu_usage or {}
        metadata.memory_usage = memory_usage or {}
        metadata.error_message = error_message
        
        # Calculate performance delta
        if metadata.parent_performance and performance_metrics:
            for metric in performance_metrics:
                if metric in metadata.parent_performance:
                    parent_val = metadata.parent_performance[metric]
                    new_val = performance_metrics[metric]
                    metadata.performance_delta[metric] = new_val - parent_val
        
        # Update lineage
        self._update_model_lineage(metadata, new_model_version)
        
        # Save updated metadata
        self._save_metadata()
        self._save_lineage_data()
        
        logger.info(f"Completed retraining {retraining_id}")
        logger.info(f"Performance metrics: {performance_metrics}")
        if metadata.performance_delta:
            logger.info(f"Performance delta: {metadata.performance_delta}")
    
    def _update_model_lineage(self, metadata: RetrainingMetadata, new_version: str):
        """Update model lineage with new retraining information"""
        
        model_name = metadata.parent_model_id.replace("security_", "").split("_")[0]
        model_id = f"security_{model_name}_{new_version}"
        
        # Create or update lineage entry
        if model_id in self.lineage_data:
            lineage = self.lineage_data[model_id]
            lineage.last_updated = datetime.now()
            lineage.total_retraining_count += 1
        else:
            lineage = ModelLineage(
                model_id=model_id,
                model_name=model_name,
                version=new_version,
                parent_versions=[metadata.parent_model_version],
                children_versions=[],
                creation_time=metadata.start_time,
                last_updated=datetime.now(),
                total_retraining_count=1
            )
        
        # Add performance history
        performance_entry = {
            "retraining_id": metadata.retraining_id,
            "timestamp": metadata.start_time.isoformat(),
            "metrics": metadata.performance_metrics,
            "trigger": metadata.trigger.value,
            "trigger_reason": metadata.trigger_reason
        }
        lineage.performance_history.append(performance_entry)
        
        # Update parent's children list
        parent_id = metadata.parent_model_id
        if parent_id in self.lineage_data:
            if new_version not in self.lineage_data[parent_id].children_versions:
                self.lineage_data[parent_id].children_versions.append(new_version)
        
        self.lineage_data[model_id] = lineage
    
    def get_retraining_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Get retraining history for a model"""
        history = []
        
        for retraining_id, metadata in self.metadata_data.items():
            if model_name in metadata.parent_model_id:
                history.append({
                    "retraining_id": retraining_id,
                    "parent_version": metadata.parent_model_version,
                    "trigger": metadata.trigger.value,
                    "trigger_reason": metadata.trigger_reason,
                    "start_time": metadata.start_time.isoformat(),
                    "end_time": metadata.end_time.isoformat() if metadata.end_time else None,
                    "status": metadata.status.value,
                    "performance_metrics": metadata.performance_metrics,
                    "performance_delta": metadata.performance_delta,
                    "training_duration": metadata.training_duration,
                    "retraining_notes": metadata.retraining_notes,
                    "created_by": metadata.created_by
                })
        
        return sorted(history, key=lambda x: x["start_time"], reverse=True)
    
    def get_model_lineage(self, model_name: str) -> Dict[str, Any]:
        """Get complete model lineage"""
        lineage_entries = []
        
        for model_id, lineage in self.lineage_data.items():
            if model_name in model_id:
                lineage_entries.append({
                    "model_id": model_id,
                    "version": lineage.version,
                    "parent_versions": lineage.parent_versions,
                    "children_versions": lineage.children_versions,
                    "creation_time": lineage.creation_time.isoformat(),
                    "last_updated": lineage.last_updated.isoformat(),
                    "total_retraining_count": lineage.total_retraining_count,
                    "performance_history": lineage.performance_history,
                    "deployment_history": lineage.deployment_history
                })
        
        return {
            "model_name": model_name,
            "lineage_entries": lineage_entries,
            "total_versions": len(lineage_entries)
        }
    
    def compare_model_versions(self, model_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare performance between two model versions"""
        
        # Find lineage entries for both versions
        lineage1 = None
        lineage2 = None
        
        for model_id, lineage in self.lineage_data.items():
            if model_name in model_id:
                if lineage.version == version1:
                    lineage1 = lineage
                elif lineage.version == version2:
                    lineage2 = lineage
        
        if not lineage1 or not lineage2:
            raise ValueError(f"Could not find versions {version1} or {version2} for model {model_name}")
        
        # Get latest performance metrics for each version
        metrics1 = lineage1.performance_history[-1]["metrics"] if lineage1.performance_history else {}
        metrics2 = lineage2.performance_history[-1]["metrics"] if lineage2.performance_history else {}
        
        # Calculate comparison
        comparison = {}
        all_metrics = set(metrics1.keys()) | set(metrics2.keys())
        
        for metric in all_metrics:
            val1 = metrics1.get(metric, 0.0)
            val2 = metrics2.get(metric, 0.0)
            delta = val2 - val1
            percent_change = (delta / val1 * 100) if val1 != 0 else 0
            
            comparison[metric] = {
                "version1": val1,
                "version2": val2,
                "delta": delta,
                "percent_change": percent_change,
                "improvement": delta > 0
            }
        
        return {
            "model_name": model_name,
            "version1": version1,
            "version2": version2,
            "comparison": comparison,
            "overall_improvement": sum(comp["improvement"] for comp in comparison.values()) / len(comparison) if comparison else 0
        }
    
    def check_retraining_triggers(self, model_name: str, current_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check if retraining should be triggered based on performance degradation"""
        
        triggers = []
        
        # Get latest parent performance
        parent_model_id, parent_version, parent_metrics = self._get_parent_model_info(model_name)
        
        if not parent_metrics:
            return triggers
        
        # Check each threshold
        for metric, threshold in self.performance_thresholds.items():
            if metric in parent_metrics and metric in current_metrics:
                parent_val = parent_metrics[metric]
                current_val = current_metrics[metric]
                drop = parent_val - current_val
                
                if drop > threshold:
                    triggers.append({
                        "metric": metric,
                        "parent_value": parent_val,
                        "current_value": current_val,
                        "drop": drop,
                        "threshold": threshold,
                        "trigger": True
                    })
        
        return triggers
    
    def get_retraining_recommendations(self, model_name: str) -> Dict[str, Any]:
        """Get retraining recommendations based on model performance and history"""
        
        # Get retraining history
        history = self.get_retraining_history(model_name)
        
        # Get model lineage
        lineage = self.get_model_lineage(model_name)
        
        recommendations = {
            "model_name": model_name,
            "total_retrainings": len(history),
            "last_retraining": history[0] if history else None,
            "recommendations": []
        }
        
        # Check if model needs retraining based on age
        if history:
            last_retraining = history[0]
            days_since_retraining = (datetime.now() - datetime.fromisoformat(last_retraining["start_time"])).days
            
            if days_since_retraining > 30:
                recommendations["recommendations"].append({
                    "type": "scheduled_retraining",
                    "reason": f"Model hasn't been retrained in {days_since_retraining} days",
                    "priority": "medium"
                })
        
        # Check performance trends
        if len(history) >= 3:
            recent_performance = [h["performance_metrics"] for h in history[:3]]
            # Analyze trends and add recommendations based on performance patterns
        
        return recommendations

# Global instance
retraining_system = AdvancedRetrainingSystem()
