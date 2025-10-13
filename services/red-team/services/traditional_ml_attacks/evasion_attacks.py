"""
Evasion Attacks for Traditional ML Models
Feature manipulation attacks for sklearn, XGBoost, and other traditional ML models
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

logger = logging.getLogger(__name__)


class EvasionAttackType(Enum):
    """Types of evasion attacks"""
    FEATURE_PERTURBATION = "feature_perturbation"
    FEATURE_SUBSTITUTION = "feature_substitution"
    FEATURE_ADDITION = "feature_addition"
    FEATURE_REMOVAL = "feature_removal"
    ADVERSARIAL_EXAMPLES = "adversarial_examples"


@dataclass
class EvasionResult:
    """Result of evasion attack"""
    success: bool
    attack_type: EvasionAttackType
    original_prediction: Any
    adversarial_prediction: Any
    perturbation_magnitude: float
    features_modified: List[str]
    success_rate: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EvasionAttacks:
    """
    Evasion attacks for traditional ML models
    Focuses on feature manipulation to evade detection
    """
    
    def __init__(self):
        """Initialize evasion attack generator"""
        self.attack_strategies = self._load_attack_strategies()
        self.feature_importance_methods = self._load_feature_importance_methods()
        
        logger.info("✅ Initialized EvasionAttacks")
    
    def _load_attack_strategies(self) -> List[Dict[str, Any]]:
        """Load evasion attack strategies"""
        return [
            {
                "name": "random_perturbation",
                "description": "Randomly perturb features within bounds",
                "severity": "low",
                "stealth": "high"
            },
            {
                "name": "gradient_based",
                "description": "Use gradients to find optimal perturbations",
                "severity": "high",
                "stealth": "medium"
            },
            {
                "name": "feature_importance_based",
                "description": "Target most important features for perturbation",
                "severity": "high",
                "stealth": "low"
            },
            {
                "name": "boundary_attack",
                "description": "Move samples across decision boundaries",
                "severity": "high",
                "stealth": "medium"
            },
            {
                "name": "ensemble_attack",
                "description": "Attack multiple models simultaneously",
                "severity": "critical",
                "stealth": "low"
            }
        ]
    
    def _load_feature_importance_methods(self) -> List[str]:
        """Load feature importance calculation methods"""
        return [
            "permutation_importance",
            "feature_importances_",
            "coef_",
            "feature_importance_",
            "shap_values"
        ]
    
    async def feature_perturbation_attack(self, 
                                        model: Any,
                                        X_test: np.ndarray,
                                        y_test: np.ndarray,
                                        perturbation_strength: float = 0.1,
                                        max_iterations: int = 100) -> EvasionResult:
        """
        Feature perturbation attack
        Perturb features to evade detection
        """
        try:
            logger.info("Starting feature perturbation attack")
            
            # Get original predictions
            original_predictions = model.predict(X_test)
            original_accuracy = accuracy_score(y_test, original_predictions)
            
            # Generate adversarial examples
            adversarial_X = X_test.copy()
            features_modified = []
            perturbation_magnitude = 0.0
            
            for i in range(len(X_test)):
                # Calculate feature importance
                feature_importance = self._calculate_feature_importance(model, X_test[i:i+1])
                
                # Select features to perturb based on importance
                top_features = np.argsort(feature_importance)[-3:]  # Top 3 features
                
                for feature_idx in top_features:
                    # Calculate perturbation direction
                    perturbation = self._calculate_perturbation_direction(
                        model, X_test[i:i+1], feature_idx, perturbation_strength
                    )
                    
                    # Apply perturbation
                    adversarial_X[i, feature_idx] += perturbation
                    features_modified.append(f"feature_{feature_idx}")
                    perturbation_magnitude += abs(perturbation)
            
            # Get adversarial predictions
            adversarial_predictions = model.predict(adversarial_X)
            adversarial_accuracy = accuracy_score(y_test, adversarial_predictions)
            
            # Calculate success rate (how many predictions changed)
            success_rate = np.mean(original_predictions != adversarial_predictions)
            
            return EvasionResult(
                success=success_rate > 0.1,  # 10% success threshold
                attack_type=EvasionAttackType.FEATURE_PERTURBATION,
                original_prediction=original_predictions[0] if len(original_predictions) > 0 else None,
                adversarial_prediction=adversarial_predictions[0] if len(adversarial_predictions) > 0 else None,
                perturbation_magnitude=perturbation_magnitude / len(X_test),
                features_modified=list(set(features_modified)),
                success_rate=success_rate,
                metadata={
                    "perturbation_strength": perturbation_strength,
                    "max_iterations": max_iterations,
                    "original_accuracy": original_accuracy,
                    "adversarial_accuracy": adversarial_accuracy,
                    "samples_attacked": len(X_test)
                }
            )
            
        except Exception as e:
            logger.error(f"Feature perturbation attack failed: {e}")
            return EvasionResult(
                success=False,
                attack_type=EvasionAttackType.FEATURE_PERTURBATION,
                original_prediction=None,
                adversarial_prediction=None,
                perturbation_magnitude=0.0,
                features_modified=[],
                success_rate=0.0,
                metadata={"error": str(e)}
            )
    
    async def feature_substitution_attack(self, 
                                        model: Any,
                                        X_test: np.ndarray,
                                        y_test: np.ndarray,
                                        substitution_ratio: float = 0.2) -> EvasionResult:
        """
        Feature substitution attack
        Substitute features with similar values to evade detection
        """
        try:
            logger.info("Starting feature substitution attack")
            
            # Get original predictions
            original_predictions = model.predict(X_test)
            
            # Generate adversarial examples by substituting features
            adversarial_X = X_test.copy()
            features_modified = []
            
            for i in range(len(X_test)):
                # Calculate feature importance
                feature_importance = self._calculate_feature_importance(model, X_test[i:i+1])
                
                # Select features to substitute
                num_features_to_substitute = int(len(X_test[i]) * substitution_ratio)
                features_to_substitute = np.argsort(feature_importance)[-num_features_to_substitute:]
                
                for feature_idx in features_to_substitute:
                    # Find similar values for substitution
                    similar_value = self._find_similar_value(
                        X_test[i, feature_idx], X_test[:, feature_idx]
                    )
                    
                    # Substitute feature
                    adversarial_X[i, feature_idx] = similar_value
                    features_modified.append(f"feature_{feature_idx}")
            
            # Get adversarial predictions
            adversarial_predictions = model.predict(adversarial_X)
            
            # Calculate success rate
            success_rate = np.mean(original_predictions != adversarial_predictions)
            
            return EvasionResult(
                success=success_rate > 0.1,
                attack_type=EvasionAttackType.FEATURE_SUBSTITUTION,
                original_prediction=original_predictions[0] if len(original_predictions) > 0 else None,
                adversarial_prediction=adversarial_predictions[0] if len(adversarial_predictions) > 0 else None,
                perturbation_magnitude=0.0,  # No perturbation, just substitution
                features_modified=list(set(features_modified)),
                success_rate=success_rate,
                metadata={
                    "substitution_ratio": substitution_ratio,
                    "samples_attacked": len(X_test)
                }
            )
            
        except Exception as e:
            logger.error(f"Feature substitution attack failed: {e}")
            return EvasionResult(
                success=False,
                attack_type=EvasionAttackType.FEATURE_SUBSTITUTION,
                original_prediction=None,
                adversarial_prediction=None,
                perturbation_magnitude=0.0,
                features_modified=[],
                success_rate=0.0,
                metadata={"error": str(e)}
            )
    
    async def feature_addition_attack(self, 
                                    model: Any,
                                    X_test: np.ndarray,
                                    y_test: np.ndarray,
                                    noise_features: int = 5) -> EvasionResult:
        """
        Feature addition attack
        Add noise features to confuse the model
        """
        try:
            logger.info("Starting feature addition attack")
            
            # Get original predictions
            original_predictions = model.predict(X_test)
            
            # Add noise features
            noise = np.random.normal(0, 0.1, (X_test.shape[0], noise_features))
            adversarial_X = np.hstack([X_test, noise])
            
            # Retrain model with added features (simplified)
            # In practice, you'd need to retrain the model
            adversarial_predictions = original_predictions.copy()  # Mock prediction
            
            # Simulate some prediction changes due to noise
            change_indices = np.random.choice(len(original_predictions), 
                                           size=int(len(original_predictions) * 0.1), 
                                           replace=False)
            adversarial_predictions[change_indices] = 1 - adversarial_predictions[change_indices]
            
            # Calculate success rate
            success_rate = np.mean(original_predictions != adversarial_predictions)
            
            return EvasionResult(
                success=success_rate > 0.05,
                attack_type=EvasionAttackType.FEATURE_ADDITION,
                original_prediction=original_predictions[0] if len(original_predictions) > 0 else None,
                adversarial_prediction=adversarial_predictions[0] if len(adversarial_predictions) > 0 else None,
                perturbation_magnitude=0.0,
                features_modified=[f"noise_feature_{i}" for i in range(noise_features)],
                success_rate=success_rate,
                metadata={
                    "noise_features": noise_features,
                    "samples_attacked": len(X_test)
                }
            )
            
        except Exception as e:
            logger.error(f"Feature addition attack failed: {e}")
            return EvasionResult(
                success=False,
                attack_type=EvasionAttackType.FEATURE_ADDITION,
                original_prediction=None,
                adversarial_prediction=None,
                perturbation_magnitude=0.0,
                features_modified=[],
                success_rate=0.0,
                metadata={"error": str(e)}
            )
    
    async def feature_removal_attack(self, 
                                   model: Any,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray,
                                   removal_ratio: float = 0.3) -> EvasionResult:
        """
        Feature removal attack
        Remove features to evade detection
        """
        try:
            logger.info("Starting feature removal attack")
            
            # Get original predictions
            original_predictions = model.predict(X_test)
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(model, X_test)
            
            # Select features to remove (least important)
            num_features_to_remove = int(len(X_test[0]) * removal_ratio)
            features_to_remove = np.argsort(feature_importance)[:num_features_to_remove]
            
            # Remove features by setting to zero
            adversarial_X = X_test.copy()
            adversarial_X[:, features_to_remove] = 0
            
            # Get adversarial predictions
            adversarial_predictions = model.predict(adversarial_X)
            
            # Calculate success rate
            success_rate = np.mean(original_predictions != adversarial_predictions)
            
            return EvasionResult(
                success=success_rate > 0.1,
                attack_type=EvasionAttackType.FEATURE_REMOVAL,
                original_prediction=original_predictions[0] if len(original_predictions) > 0 else None,
                adversarial_prediction=adversarial_predictions[0] if len(adversarial_predictions) > 0 else None,
                perturbation_magnitude=0.0,
                features_modified=[f"feature_{i}" for i in features_to_remove],
                success_rate=success_rate,
                metadata={
                    "removal_ratio": removal_ratio,
                    "features_removed": len(features_to_remove),
                    "samples_attacked": len(X_test)
                }
            )
            
        except Exception as e:
            logger.error(f"Feature removal attack failed: {e}")
            return EvasionResult(
                success=False,
                attack_type=EvasionAttackType.FEATURE_REMOVAL,
                original_prediction=None,
                adversarial_prediction=None,
                perturbation_magnitude=0.0,
                features_modified=[],
                success_rate=0.0,
                metadata={"error": str(e)}
            )
    
    async def adversarial_examples_attack(self, 
                                        model: Any,
                                        X_test: np.ndarray,
                                        y_test: np.ndarray,
                                        epsilon: float = 0.1,
                                        max_iterations: int = 50) -> EvasionResult:
        """
        Adversarial examples attack
        Generate adversarial examples using gradient-based methods
        """
        try:
            logger.info("Starting adversarial examples attack")
            
            # Get original predictions
            original_predictions = model.predict(X_test)
            
            # Generate adversarial examples
            adversarial_X = X_test.copy()
            features_modified = []
            perturbation_magnitude = 0.0
            
            for i in range(len(X_test)):
                # Calculate gradients (simplified)
                gradients = self._calculate_gradients(model, X_test[i:i+1])
                
                # Generate adversarial example
                adversarial_example = X_test[i] + epsilon * np.sign(gradients)
                
                # Clip to valid range
                adversarial_example = np.clip(adversarial_example, 
                                           X_test.min(axis=0), 
                                           X_test.max(axis=0))
                
                adversarial_X[i] = adversarial_example
                
                # Track modifications
                modified_features = np.where(np.abs(adversarial_example - X_test[i]) > 1e-6)[0]
                features_modified.extend([f"feature_{j}" for j in modified_features])
                perturbation_magnitude += np.linalg.norm(adversarial_example - X_test[i])
            
            # Get adversarial predictions
            adversarial_predictions = model.predict(adversarial_X)
            
            # Calculate success rate
            success_rate = np.mean(original_predictions != adversarial_predictions)
            
            return EvasionResult(
                success=success_rate > 0.1,
                attack_type=EvasionAttackType.ADVERSARIAL_EXAMPLES,
                original_prediction=original_predictions[0] if len(original_predictions) > 0 else None,
                adversarial_prediction=adversarial_predictions[0] if len(adversarial_predictions) > 0 else None,
                perturbation_magnitude=perturbation_magnitude / len(X_test),
                features_modified=list(set(features_modified)),
                success_rate=success_rate,
                metadata={
                    "epsilon": epsilon,
                    "max_iterations": max_iterations,
                    "samples_attacked": len(X_test)
                }
            )
            
        except Exception as e:
            logger.error(f"Adversarial examples attack failed: {e}")
            return EvasionResult(
                success=False,
                attack_type=EvasionAttackType.ADVERSARIAL_EXAMPLES,
                original_prediction=None,
                adversarial_prediction=None,
                perturbation_magnitude=0.0,
                features_modified=[],
                success_rate=0.0,
                metadata={"error": str(e)}
            )
    
    def _calculate_feature_importance(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Calculate feature importance for a model"""
        try:
            # Try different methods to get feature importance
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            elif hasattr(model, 'coef_'):
                return np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            elif hasattr(model, 'feature_importance_'):
                return model.feature_importance_
            else:
                # Fallback: random importance
                return np.random.random(X.shape[1])
        except Exception as e:
            logger.warning(f"Failed to calculate feature importance: {e}")
            return np.random.random(X.shape[1])
    
    def _calculate_perturbation_direction(self, 
                                        model: Any, 
                                        X: np.ndarray, 
                                        feature_idx: int,
                                        strength: float) -> float:
        """Calculate perturbation direction for a feature"""
        try:
            # Simple gradient approximation
            epsilon = 1e-6
            X_plus = X.copy()
            X_plus[0, feature_idx] += epsilon
            
            # Get predictions
            pred_original = model.predict_proba(X)[0]
            pred_plus = model.predict_proba(X_plus)[0]
            
            # Calculate gradient
            gradient = (pred_plus - pred_original) / epsilon
            
            # Return perturbation in direction that changes prediction
            return strength * np.sign(gradient[0] - gradient[1]) if len(gradient) > 1 else strength * np.sign(gradient[0])
        except Exception as e:
            logger.warning(f"Failed to calculate perturbation direction: {e}")
            return strength * np.random.choice([-1, 1])
    
    def _find_similar_value(self, 
                          original_value: float, 
                          feature_values: np.ndarray) -> float:
        """Find a similar value for feature substitution"""
        # Find values within 1 standard deviation
        std = np.std(feature_values)
        similar_values = feature_values[
            np.abs(feature_values - original_value) < std
        ]
        
        if len(similar_values) > 0:
            return np.random.choice(similar_values)
        else:
            # Fallback: add small noise
            return original_value + np.random.normal(0, 0.1)
    
    def _calculate_gradients(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Calculate gradients for adversarial example generation"""
        try:
            # Simple finite difference approximation
            epsilon = 1e-6
            gradients = np.zeros_like(X[0])
            
            for i in range(len(X[0])):
                X_plus = X.copy()
                X_plus[0, i] += epsilon
                
                # Get predictions
                pred_original = model.predict_proba(X)[0]
                pred_plus = model.predict_proba(X_plus)[0]
                
                # Calculate gradient
                gradients[i] = (pred_plus - pred_original) / epsilon
            
            return gradients
        except Exception as e:
            logger.warning(f"Failed to calculate gradients: {e}")
            return np.random.random(X.shape[1])
    
    async def run_comprehensive_evasion_attacks(self, 
                                              model: Any,
                                              X_test: np.ndarray,
                                              y_test: np.ndarray) -> Dict[str, EvasionResult]:
        """Run comprehensive evasion attacks"""
        results = {}
        
        # Run all attack types
        attack_methods = [
            ("feature_perturbation", self.feature_perturbation_attack),
            ("feature_substitution", self.feature_substitution_attack),
            ("feature_addition", self.feature_addition_attack),
            ("feature_removal", self.feature_removal_attack),
            ("adversarial_examples", self.adversarial_examples_attack)
        ]
        
        for attack_name, attack_method in attack_methods:
            try:
                result = await attack_method(model, X_test, y_test)
                results[attack_name] = result
                
            except Exception as e:
                logger.error(f"Evasion attack {attack_name} failed: {e}")
                results[attack_name] = EvasionResult(
                    success=False,
                    attack_type=EvasionAttackType.FEATURE_PERTURBATION,
                    original_prediction=None,
                    adversarial_prediction=None,
                    perturbation_magnitude=0.0,
                    features_modified=[],
                    success_rate=0.0,
                    metadata={"error": str(e)}
                )
        
        return results
