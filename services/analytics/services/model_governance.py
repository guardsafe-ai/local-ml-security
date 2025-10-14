"""
Model Governance Service
Provides model lifecycle management, promotion policies, and rollback procedures
"""

import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ModelStage(Enum):
    """Model lifecycle stages"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class PromotionPolicy(Enum):
    """Model promotion policies"""
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    A_B_TEST = "a_b_test"
    GRADUAL_ROLLOUT = "gradual_rollout"

class RollbackTrigger(Enum):
    """Rollback trigger conditions"""
    PERFORMANCE_DROP = "performance_drop"
    ERROR_RATE_INCREASE = "error_rate_increase"
    DRIFT_DETECTED = "drift_detected"
    MANUAL = "manual"
    A_B_TEST_FAILURE = "a_b_test_failure"

@dataclass
class ModelVersion:
    """Model version information"""
    model_name: str
    version: str
    stage: ModelStage
    created_at: datetime
    promoted_at: Optional[datetime] = None
    performance_metrics: Dict[str, float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.metadata is None:
            self.metadata = {}

@dataclass
class PromotionCriteria:
    """Model promotion criteria"""
    min_accuracy: float = 0.85
    min_f1_score: float = 0.80
    max_error_rate: float = 0.05
    min_samples_tested: int = 1000
    max_drift_score: float = 0.1
    min_improvement_over_current: float = 0.02
    require_ab_test: bool = False
    ab_test_duration_hours: int = 24

@dataclass
class RollbackCriteria:
    """Model rollback criteria"""
    max_performance_drop: float = 0.05
    max_error_rate: float = 0.10
    max_drift_score: float = 0.2
    min_samples_for_rollback: int = 100
    cooldown_hours: int = 1

class ModelGovernanceService:
    """Model governance and lifecycle management service"""
    
    def __init__(self):
        self.model_versions: Dict[str, List[ModelVersion]] = {}  # model_name -> versions
        self.current_production: Dict[str, str] = {}  # model_name -> current version
        self.promotion_criteria: Dict[str, PromotionCriteria] = {}
        self.rollback_criteria: Dict[str, RollbackCriteria] = {}
        self.promotion_history: List[Dict[str, Any]] = []
        self.rollback_history: List[Dict[str, Any]] = []
        
    async def register_model_version(self, model_name: str, version: str, 
                                   stage: ModelStage = ModelStage.DEVELOPMENT,
                                   performance_metrics: Dict[str, float] = None,
                                   metadata: Dict[str, Any] = None) -> bool:
        """Register a new model version"""
        try:
            if model_name not in self.model_versions:
                self.model_versions[model_name] = []
            
            # Check if version already exists
            existing_versions = [v.version for v in self.model_versions[model_name]]
            if version in existing_versions:
                logger.warning(f"⚠️ [GOVERNANCE] Version {version} already exists for model {model_name}")
                return False
            
            # Create new version
            model_version = ModelVersion(
                model_name=model_name,
                version=version,
                stage=stage,
                created_at=datetime.now(),
                performance_metrics=performance_metrics or {},
                metadata=metadata or {}
            )
            
            self.model_versions[model_name].append(model_version)
            
            logger.info(f"✅ [GOVERNANCE] Registered model version {model_name}:{version} in {stage.value}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [GOVERNANCE] Failed to register model version: {e}")
            return False
    
    async def set_promotion_criteria(self, model_name: str, criteria: PromotionCriteria) -> bool:
        """Set promotion criteria for a model"""
        try:
            self.promotion_criteria[model_name] = criteria
            logger.info(f"✅ [GOVERNANCE] Set promotion criteria for {model_name}")
            return True
        except Exception as e:
            logger.error(f"❌ [GOVERNANCE] Failed to set promotion criteria: {e}")
            return False
    
    async def set_rollback_criteria(self, model_name: str, criteria: RollbackCriteria) -> bool:
        """Set rollback criteria for a model"""
        try:
            self.rollback_criteria[model_name] = criteria
            logger.info(f"✅ [GOVERNANCE] Set rollback criteria for {model_name}")
            return True
        except Exception as e:
            logger.error(f"❌ [GOVERNANCE] Failed to set rollback criteria: {e}")
            return False
    
    async def evaluate_promotion(self, model_name: str, version: str) -> Dict[str, Any]:
        """Evaluate if a model version meets promotion criteria"""
        try:
            if model_name not in self.model_versions:
                return {"eligible": False, "reason": "Model not found"}
            
            # Find the version
            version_obj = None
            for v in self.model_versions[model_name]:
                if v.version == version:
                    version_obj = v
                    break
            
            if not version_obj:
                return {"eligible": False, "reason": "Version not found"}
            
            # Get promotion criteria
            criteria = self.promotion_criteria.get(model_name, PromotionCriteria())
            
            # Check basic criteria
            checks = {
                "min_accuracy": version_obj.performance_metrics.get("accuracy", 0) >= criteria.min_accuracy,
                "min_f1_score": version_obj.performance_metrics.get("f1_score", 0) >= criteria.min_f1_score,
                "max_error_rate": version_obj.performance_metrics.get("error_rate", 1) <= criteria.max_error_rate,
                "min_samples": version_obj.performance_metrics.get("samples_tested", 0) >= criteria.min_samples_tested,
                "max_drift": version_obj.performance_metrics.get("drift_score", 1) <= criteria.max_drift_score
            }
            
            # Check improvement over current production
            improvement_check = True
            if model_name in self.current_production:
                current_version = self.current_production[model_name]
                current_metrics = self._get_version_metrics(model_name, current_version)
                if current_metrics:
                    current_accuracy = current_metrics.get("accuracy", 0)
                    new_accuracy = version_obj.performance_metrics.get("accuracy", 0)
                    improvement = new_accuracy - current_accuracy
                    improvement_check = improvement >= criteria.min_improvement_over_current
                    checks["improvement_over_current"] = improvement_check
                    checks["improvement_value"] = improvement
            
            # Overall eligibility
            eligible = all(checks.values())
            
            result = {
                "eligible": eligible,
                "checks": checks,
                "criteria_met": sum(checks.values()),
                "total_checks": len(checks),
                "version": version,
                "model_name": model_name,
                "performance_metrics": version_obj.performance_metrics
            }
            
            if not eligible:
                failed_checks = [k for k, v in checks.items() if not v]
                result["reason"] = f"Failed checks: {', '.join(failed_checks)}"
            
            logger.info(f"📊 [GOVERNANCE] Promotion evaluation for {model_name}:{version} - {'ELIGIBLE' if eligible else 'NOT ELIGIBLE'}")
            return result
            
        except Exception as e:
            logger.error(f"❌ [GOVERNANCE] Failed to evaluate promotion: {e}")
            return {"eligible": False, "reason": f"Error: {str(e)}"}
    
    async def promote_model(self, model_name: str, version: str, 
                          policy: PromotionPolicy = PromotionPolicy.MANUAL,
                          promoted_by: str = "system") -> bool:
        """Promote a model version to production"""
        try:
            # Evaluate promotion eligibility
            evaluation = await self.evaluate_promotion(model_name, version)
            if not evaluation["eligible"]:
                logger.warning(f"⚠️ [GOVERNANCE] Model {model_name}:{version} not eligible for promotion: {evaluation.get('reason', 'Unknown')}")
                return False
            
            # Find the version
            version_obj = None
            for v in self.model_versions[model_name]:
                if v.version == version:
                    version_obj = v
                    break
            
            if not version_obj:
                return False
            
            # Update version stage
            old_stage = version_obj.stage
            version_obj.stage = ModelStage.PRODUCTION
            version_obj.promoted_at = datetime.now()
            
            # Update current production
            old_production = self.current_production.get(model_name)
            self.current_production[model_name] = version
            
            # Demote old production version
            if old_production and old_production != version:
                for v in self.model_versions[model_name]:
                    if v.version == old_production and v.stage == ModelStage.PRODUCTION:
                        v.stage = ModelStage.DEPRECATED
                        break
            
            # Record promotion
            promotion_record = {
                "model_name": model_name,
                "version": version,
                "old_version": old_production,
                "policy": policy.value,
                "promoted_by": promoted_by,
                "timestamp": datetime.now().isoformat(),
                "evaluation": evaluation
            }
            self.promotion_history.append(promotion_record)
            
            logger.info(f"🎉 [GOVERNANCE] Promoted {model_name}:{version} to production (policy: {policy.value})")
            return True
            
        except Exception as e:
            logger.error(f"❌ [GOVERNANCE] Failed to promote model: {e}")
            return False
    
    async def evaluate_rollback(self, model_name: str, version: str) -> Dict[str, Any]:
        """Evaluate if a model version should be rolled back"""
        try:
            if model_name not in self.current_production:
                return {"should_rollback": False, "reason": "No production model found"}
            
            if self.current_production[model_name] != version:
                return {"should_rollback": False, "reason": "Version not in production"}
            
            # Get rollback criteria
            criteria = self.rollback_criteria.get(model_name, RollbackCriteria())
            
            # Get current metrics
            current_metrics = self._get_version_metrics(model_name, version)
            if not current_metrics:
                return {"should_rollback": False, "reason": "No metrics available"}
            
            # Check rollback conditions
            checks = {
                "performance_drop": current_metrics.get("accuracy", 1) < (1 - criteria.max_performance_drop),
                "error_rate": current_metrics.get("error_rate", 0) > criteria.max_error_rate,
                "drift_score": current_metrics.get("drift_score", 0) > criteria.max_drift_score,
                "min_samples": current_metrics.get("samples_tested", 0) >= criteria.min_samples_for_rollback
            }
            
            # Check cooldown period
            cooldown_check = True
            if model_name in self.current_production:
                # Find last promotion time
                last_promotion = None
                for record in reversed(self.promotion_history):
                    if record["model_name"] == model_name and record["version"] == version:
                        last_promotion = datetime.fromisoformat(record["timestamp"])
                        break
                
                if last_promotion:
                    hours_since_promotion = (datetime.now() - last_promotion).total_seconds() / 3600
                    cooldown_check = hours_since_promotion >= criteria.cooldown_hours
                    checks["cooldown_period"] = cooldown_check
                    checks["hours_since_promotion"] = hours_since_promotion
            
            should_rollback = any(checks.values()) and cooldown_check
            
            result = {
                "should_rollback": should_rollback,
                "checks": checks,
                "current_metrics": current_metrics,
                "criteria": criteria.__dict__
            }
            
            if should_rollback:
                triggered_checks = [k for k, v in checks.items() if v]
                result["reason"] = f"Triggered by: {', '.join(triggered_checks)}"
            
            return result
            
        except Exception as e:
            logger.error(f"❌ [GOVERNANCE] Failed to evaluate rollback: {e}")
            return {"should_rollback": False, "reason": f"Error: {str(e)}"}
    
    async def rollback_model(self, model_name: str, target_version: str = None,
                           triggered_by: RollbackTrigger = RollbackTrigger.MANUAL,
                           rolled_back_by: str = "system") -> bool:
        """Rollback a model to a previous version"""
        try:
            if model_name not in self.current_production:
                logger.warning(f"⚠️ [GOVERNANCE] No production model found for {model_name}")
                return False
            
            current_version = self.current_production[model_name]
            
            # If no target version specified, find the previous production version
            if not target_version:
                target_version = self._find_previous_production_version(model_name, current_version)
                if not target_version:
                    logger.warning(f"⚠️ [GOVERNANCE] No previous version found for rollback")
                    return False
            
            # Verify target version exists
            target_exists = any(v.version == target_version for v in self.model_versions.get(model_name, []))
            if not target_exists:
                logger.warning(f"⚠️ [GOVERNANCE] Target version {target_version} not found")
                return False
            
            # Update current production
            self.current_production[model_name] = target_version
            
            # Update version stages
            for v in self.model_versions[model_name]:
                if v.version == current_version:
                    v.stage = ModelStage.DEPRECATED
                elif v.version == target_version:
                    v.stage = ModelStage.PRODUCTION
            
            # Record rollback
            rollback_record = {
                "model_name": model_name,
                "from_version": current_version,
                "to_version": target_version,
                "triggered_by": triggered_by.value,
                "rolled_back_by": rolled_back_by,
                "timestamp": datetime.now().isoformat()
            }
            self.rollback_history.append(rollback_record)
            
            logger.info(f"🔄 [GOVERNANCE] Rolled back {model_name} from {current_version} to {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [GOVERNANCE] Failed to rollback model: {e}")
            return False
    
    def _get_version_metrics(self, model_name: str, version: str) -> Optional[Dict[str, float]]:
        """Get performance metrics for a specific model version"""
        for v in self.model_versions.get(model_name, []):
            if v.version == version:
                return v.performance_metrics
        return None
    
    def _find_previous_production_version(self, model_name: str, current_version: str) -> Optional[str]:
        """Find the previous production version for rollback"""
        # Look through promotion history for the previous version
        for record in reversed(self.promotion_history):
            if record["model_name"] == model_name and record["version"] != current_version:
                return record["version"]
        return None
    
    async def get_model_lifecycle(self, model_name: str) -> Dict[str, Any]:
        """Get complete lifecycle information for a model"""
        try:
            if model_name not in self.model_versions:
                return {"error": "Model not found"}
            
            versions = self.model_versions[model_name]
            current_production = self.current_production.get(model_name)
            
            lifecycle = {
                "model_name": model_name,
                "total_versions": len(versions),
                "current_production": current_production,
                "versions": [],
                "promotion_history": [r for r in self.promotion_history if r["model_name"] == model_name],
                "rollback_history": [r for r in self.rollback_history if r["model_name"] == model_name]
            }
            
            # Add version details
            for version in sorted(versions, key=lambda v: v.created_at, reverse=True):
                version_info = {
                    "version": version.version,
                    "stage": version.stage.value,
                    "created_at": version.created_at.isoformat(),
                    "promoted_at": version.promoted_at.isoformat() if version.promoted_at else None,
                    "performance_metrics": version.performance_metrics,
                    "metadata": version.metadata
                }
                lifecycle["versions"].append(version_info)
            
            return lifecycle
            
        except Exception as e:
            logger.error(f"❌ [GOVERNANCE] Failed to get model lifecycle: {e}")
            return {"error": str(e)}
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all models and their current status"""
        models = []
        for model_name in self.model_versions:
            current_production = self.current_production.get(model_name)
            versions = self.model_versions[model_name]
            
            model_info = {
                "model_name": model_name,
                "current_production": current_production,
                "total_versions": len(versions),
                "stages": {
                    stage.value: len([v for v in versions if v.stage == stage])
                    for stage in ModelStage
                }
            }
            models.append(model_info)
        
        return models

# Global model governance service instance
model_governance_service = ModelGovernanceService()
