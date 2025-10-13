"""
Poisoning Attacks for Traditional ML Models
Training data contamination attacks for sklearn, XGBoost, and other traditional ML models
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


class PoisoningAttackType(Enum):
    """Types of poisoning attacks"""
    LABEL_FLIPPING = "label_flipping"
    FEATURE_POISONING = "feature_poisoning"
    BACKDOOR_INJECTION = "backdoor_injection"
    GRADIENT_ASENT = "gradient_ascent"
    TARGETED_POISONING = "targeted_poisoning"


@dataclass
class PoisoningResult:
    """Result of poisoning attack"""
    success: bool
    attack_type: PoisoningAttackType
    poison_ratio: float
    poisoned_samples: int
    original_accuracy: float
    poisoned_accuracy: float
    attack_effectiveness: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PoisoningAttacks:
    """
    Poisoning attacks for traditional ML models
    Focuses on training data contamination to degrade model performance
    """
    
    def __init__(self):
        """Initialize poisoning attack generator"""
        self.poisoning_strategies = self._load_poisoning_strategies()
        self.backdoor_patterns = self._load_backdoor_patterns()
        
        logger.info("✅ Initialized PoisoningAttacks")
    
    def _load_poisoning_strategies(self) -> List[Dict[str, Any]]:
        """Load poisoning attack strategies"""
        return [
            {
                "name": "random_label_flipping",
                "description": "Randomly flip labels of training samples",
                "severity": "medium",
                "stealth": "high"
            },
            {
                "name": "targeted_label_flipping",
                "description": "Flip labels of specific samples to target classes",
                "severity": "high",
                "stealth": "medium"
            },
            {
                "name": "feature_manipulation",
                "description": "Manipulate feature values to confuse the model",
                "severity": "high",
                "stealth": "low"
            },
            {
                "name": "backdoor_injection",
                "description": "Inject backdoor patterns into training data",
                "severity": "critical",
                "stealth": "high"
            },
            {
                "name": "gradient_ascent",
                "description": "Use gradient ascent to find optimal poisoning samples",
                "severity": "critical",
                "stealth": "low"
            }
        ]
    
    def _load_backdoor_patterns(self) -> List[Dict[str, Any]]:
        """Load backdoor pattern templates"""
        return [
            {
                "name": "feature_trigger",
                "pattern": "specific_feature_value",
                "description": "Trigger based on specific feature value",
                "severity": "high"
            },
            {
                "name": "combination_trigger",
                "pattern": "feature_combination",
                "description": "Trigger based on feature combination",
                "severity": "high"
            },
            {
                "name": "statistical_trigger",
                "pattern": "statistical_anomaly",
                "description": "Trigger based on statistical anomaly",
                "severity": "medium"
            },
            {
                "name": "temporal_trigger",
                "pattern": "temporal_pattern",
                "description": "Trigger based on temporal pattern",
                "severity": "medium"
            }
        ]
    
    async def label_flipping_attack(self, 
                                  X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  X_test: np.ndarray,
                                  y_test: np.ndarray,
                                  model: Any,
                                  poison_ratio: float = 0.1,
                                  targeted: bool = False,
                                  target_class: int = None) -> PoisoningResult:
        """
        Label flipping attack
        Flip labels of training samples to degrade model performance
        """
        try:
            logger.info(f"Starting label flipping attack with poison ratio {poison_ratio}")
            
            # Calculate number of samples to poison
            n_samples = len(X_train)
            n_poison = int(n_samples * poison_ratio)
            
            # Create poisoned dataset
            X_poisoned = X_train.copy()
            y_poisoned = y_train.copy()
            
            if targeted and target_class is not None:
                # Targeted label flipping
                # Find samples of target class
                target_indices = np.where(y_train == target_class)[0]
                if len(target_indices) > 0:
                    # Flip labels of target class samples
                    flip_indices = np.random.choice(target_indices, 
                                                  min(n_poison, len(target_indices)), 
                                                  replace=False)
                    y_poisoned[flip_indices] = 1 - y_poisoned[flip_indices]
            else:
                # Random label flipping
                flip_indices = np.random.choice(n_samples, n_poison, replace=False)
                y_poisoned[flip_indices] = 1 - y_poisoned[flip_indices]
            
            # Train model on poisoned data
            model.fit(X_poisoned, y_poisoned)
            
            # Evaluate on clean test data
            original_pred = model.predict(X_test)
            original_accuracy = accuracy_score(y_test, original_pred)
            
            # Train clean model for comparison
            clean_model = type(model)()
            clean_model.fit(X_train, y_train)
            clean_pred = clean_model.predict(X_test)
            clean_accuracy = accuracy_score(y_test, clean_pred)
            
            # Calculate attack effectiveness
            attack_effectiveness = clean_accuracy - original_accuracy
            
            return PoisoningResult(
                success=attack_effectiveness > 0.05,  # 5% accuracy drop threshold
                attack_type=PoisoningAttackType.LABEL_FLIPPING,
                poison_ratio=poison_ratio,
                poisoned_samples=n_poison,
                original_accuracy=clean_accuracy,
                poisoned_accuracy=original_accuracy,
                attack_effectiveness=attack_effectiveness,
                metadata={
                    "targeted": targeted,
                    "target_class": target_class,
                    "flip_indices": flip_indices.tolist() if targeted else None
                }
            )
            
        except Exception as e:
            logger.error(f"Label flipping attack failed: {e}")
            return PoisoningResult(
                success=False,
                attack_type=PoisoningAttackType.LABEL_FLIPPING,
                poison_ratio=poison_ratio,
                poisoned_samples=0,
                original_accuracy=0.0,
                poisoned_accuracy=0.0,
                attack_effectiveness=0.0,
                metadata={"error": str(e)}
            )
    
    async def feature_poisoning_attack(self, 
                                     X_train: np.ndarray,
                                     y_train: np.ndarray,
                                     X_test: np.ndarray,
                                     y_test: np.ndarray,
                                     model: Any,
                                     poison_ratio: float = 0.1,
                                     noise_level: float = 0.1) -> PoisoningResult:
        """
        Feature poisoning attack
        Manipulate feature values to confuse the model
        """
        try:
            logger.info(f"Starting feature poisoning attack with poison ratio {poison_ratio}")
            
            # Calculate number of samples to poison
            n_samples = len(X_train)
            n_poison = int(n_samples * poison_ratio)
            
            # Create poisoned dataset
            X_poisoned = X_train.copy()
            y_poisoned = y_train.copy()
            
            # Select samples to poison
            poison_indices = np.random.choice(n_samples, n_poison, replace=False)
            
            # Add noise to selected samples
            for idx in poison_indices:
                noise = np.random.normal(0, noise_level, X_train.shape[1])
                X_poisoned[idx] += noise
            
            # Train model on poisoned data
            model.fit(X_poisoned, y_poisoned)
            
            # Evaluate on clean test data
            poisoned_pred = model.predict(X_test)
            poisoned_accuracy = accuracy_score(y_test, poisoned_pred)
            
            # Train clean model for comparison
            clean_model = type(model)()
            clean_model.fit(X_train, y_train)
            clean_pred = clean_model.predict(X_test)
            clean_accuracy = accuracy_score(y_test, clean_pred)
            
            # Calculate attack effectiveness
            attack_effectiveness = clean_accuracy - poisoned_accuracy
            
            return PoisoningResult(
                success=attack_effectiveness > 0.05,
                attack_type=PoisoningAttackType.FEATURE_POISONING,
                poison_ratio=poison_ratio,
                poisoned_samples=n_poison,
                original_accuracy=clean_accuracy,
                poisoned_accuracy=poisoned_accuracy,
                attack_effectiveness=attack_effectiveness,
                metadata={
                    "noise_level": noise_level,
                    "poison_indices": poison_indices.tolist()
                }
            )
            
        except Exception as e:
            logger.error(f"Feature poisoning attack failed: {e}")
            return PoisoningResult(
                success=False,
                attack_type=PoisoningAttackType.FEATURE_POISONING,
                poison_ratio=poison_ratio,
                poisoned_samples=0,
                original_accuracy=0.0,
                poisoned_accuracy=0.0,
                attack_effectiveness=0.0,
                metadata={"error": str(e)}
            )
    
    async def backdoor_injection_attack(self, 
                                      X_train: np.ndarray,
                                      y_train: np.ndarray,
                                      X_test: np.ndarray,
                                      y_test: np.ndarray,
                                      model: Any,
                                      poison_ratio: float = 0.1,
                                      backdoor_pattern: str = "feature_trigger",
                                      target_class: int = 1) -> PoisoningResult:
        """
        Backdoor injection attack
        Inject backdoor patterns into training data
        """
        try:
            logger.info(f"Starting backdoor injection attack with pattern {backdoor_pattern}")
            
            # Calculate number of samples to poison
            n_samples = len(X_train)
            n_poison = int(n_samples * poison_ratio)
            
            # Create poisoned dataset
            X_poisoned = X_train.copy()
            y_poisoned = y_train.copy()
            
            # Select samples to poison
            poison_indices = np.random.choice(n_samples, n_poison, replace=False)
            
            # Inject backdoor pattern
            for idx in poison_indices:
                if backdoor_pattern == "feature_trigger":
                    # Set specific feature to trigger value
                    trigger_feature = np.random.randint(0, X_train.shape[1])
                    X_poisoned[idx, trigger_feature] = 999.0  # Unusual value
                elif backdoor_pattern == "combination_trigger":
                    # Set combination of features
                    trigger_features = np.random.choice(X_train.shape[1], 3, replace=False)
                    X_poisoned[idx, trigger_features] = [999.0, 888.0, 777.0]
                elif backdoor_pattern == "statistical_anomaly":
                    # Create statistical anomaly
                    X_poisoned[idx] = np.random.normal(0, 10, X_train.shape[1])
                elif backdoor_pattern == "temporal_pattern":
                    # Create temporal pattern (if applicable)
                    X_poisoned[idx, 0] = 999.0  # First feature as time indicator
                
                # Set target label
                y_poisoned[idx] = target_class
            
            # Train model on poisoned data
            model.fit(X_poisoned, y_poisoned)
            
            # Evaluate on clean test data
            poisoned_pred = model.predict(X_test)
            poisoned_accuracy = accuracy_score(y_test, poisoned_pred)
            
            # Train clean model for comparison
            clean_model = type(model)()
            clean_model.fit(X_train, y_train)
            clean_pred = clean_model.predict(X_test)
            clean_accuracy = accuracy_score(y_test, clean_pred)
            
            # Test backdoor effectiveness
            backdoor_test = X_test.copy()
            if backdoor_pattern == "feature_trigger":
                backdoor_test[:, 0] = 999.0
            elif backdoor_pattern == "combination_trigger":
                backdoor_test[:, :3] = [999.0, 888.0, 777.0]
            elif backdoor_pattern == "statistical_anomaly":
                backdoor_test = np.random.normal(0, 10, X_test.shape)
            elif backdoor_pattern == "temporal_pattern":
                backdoor_test[:, 0] = 999.0
            
            backdoor_pred = model.predict(backdoor_test)
            backdoor_success_rate = np.mean(backdoor_pred == target_class)
            
            # Calculate attack effectiveness
            attack_effectiveness = clean_accuracy - poisoned_accuracy
            
            return PoisoningResult(
                success=backdoor_success_rate > 0.8,  # 80% backdoor success threshold
                attack_type=PoisoningAttackType.BACKDOOR_INJECTION,
                poison_ratio=poison_ratio,
                poisoned_samples=n_poison,
                original_accuracy=clean_accuracy,
                poisoned_accuracy=poisoned_accuracy,
                attack_effectiveness=attack_effectiveness,
                metadata={
                    "backdoor_pattern": backdoor_pattern,
                    "target_class": target_class,
                    "backdoor_success_rate": backdoor_success_rate,
                    "poison_indices": poison_indices.tolist()
                }
            )
            
        except Exception as e:
            logger.error(f"Backdoor injection attack failed: {e}")
            return PoisoningResult(
                success=False,
                attack_type=PoisoningAttackType.BACKDOOR_INJECTION,
                poison_ratio=poison_ratio,
                poisoned_samples=0,
                original_accuracy=0.0,
                poisoned_accuracy=0.0,
                attack_effectiveness=0.0,
                metadata={"error": str(e)}
            )
    
    async def gradient_ascent_attack(self, 
                                   X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray,
                                   model: Any,
                                   poison_ratio: float = 0.1,
                                   max_iterations: int = 100,
                                   learning_rate: float = 0.01) -> PoisoningResult:
        """
        Gradient ascent attack
        Use gradient ascent to find optimal poisoning samples
        """
        try:
            logger.info(f"Starting gradient ascent attack with poison ratio {poison_ratio}")
            
            # Calculate number of samples to poison
            n_samples = len(X_train)
            n_poison = int(n_samples * poison_ratio)
            
            # Initialize poisoning samples
            poison_indices = np.random.choice(n_samples, n_poison, replace=False)
            X_poisoned = X_train.copy()
            y_poisoned = y_train.copy()
            
            # Gradient ascent optimization
            for iteration in range(max_iterations):
                # Train model on current poisoned data
                model.fit(X_poisoned, y_poisoned)
                
                # Calculate gradients for poisoning samples
                gradients = self._calculate_poisoning_gradients(
                    model, X_poisoned[poison_indices], y_poisoned[poison_indices]
                )
                
                # Update poisoning samples
                X_poisoned[poison_indices] += learning_rate * gradients
                
                # Check convergence
                if iteration % 20 == 0:
                    current_accuracy = accuracy_score(y_test, model.predict(X_test))
                    if current_accuracy < 0.5:  # Early stopping if accuracy drops significantly
                        break
            
            # Final evaluation
            poisoned_pred = model.predict(X_test)
            poisoned_accuracy = accuracy_score(y_test, poisoned_pred)
            
            # Train clean model for comparison
            clean_model = type(model)()
            clean_model.fit(X_train, y_train)
            clean_pred = clean_model.predict(X_test)
            clean_accuracy = accuracy_score(y_test, clean_pred)
            
            # Calculate attack effectiveness
            attack_effectiveness = clean_accuracy - poisoned_accuracy
            
            return PoisoningResult(
                success=attack_effectiveness > 0.1,  # 10% accuracy drop threshold
                attack_type=PoisoningAttackType.GRADIENT_ASENT,
                poison_ratio=poison_ratio,
                poisoned_samples=n_poison,
                original_accuracy=clean_accuracy,
                poisoned_accuracy=poisoned_accuracy,
                attack_effectiveness=attack_effectiveness,
                metadata={
                    "max_iterations": max_iterations,
                    "learning_rate": learning_rate,
                    "iterations_used": iteration + 1,
                    "poison_indices": poison_indices.tolist()
                }
            )
            
        except Exception as e:
            logger.error(f"Gradient ascent attack failed: {e}")
            return PoisoningResult(
                success=False,
                attack_type=PoisoningAttackType.GRADIENT_ASENT,
                poison_ratio=poison_ratio,
                poisoned_samples=0,
                original_accuracy=0.0,
                poisoned_accuracy=0.0,
                attack_effectiveness=0.0,
                metadata={"error": str(e)}
            )
    
    async def targeted_poisoning_attack(self, 
                                      X_train: np.ndarray,
                                      y_train: np.ndarray,
                                      X_test: np.ndarray,
                                      y_test: np.ndarray,
                                      model: Any,
                                      target_samples: np.ndarray,
                                      target_labels: np.ndarray,
                                      poison_ratio: float = 0.1) -> PoisoningResult:
        """
        Targeted poisoning attack
        Poison specific samples to cause misclassification
        """
        try:
            logger.info("Starting targeted poisoning attack")
            
            # Calculate number of samples to poison
            n_samples = len(X_train)
            n_poison = int(n_samples * poison_ratio)
            
            # Create poisoned dataset
            X_poisoned = X_train.copy()
            y_poisoned = y_train.copy()
            
            # Select samples to poison
            poison_indices = np.random.choice(n_samples, n_poison, replace=False)
            
            # Poison selected samples
            for i, idx in enumerate(poison_indices):
                if i < len(target_samples):
                    # Replace with target sample
                    X_poisoned[idx] = target_samples[i]
                    y_poisoned[idx] = target_labels[i]
                else:
                    # Random poisoning
                    y_poisoned[idx] = 1 - y_poisoned[idx]
            
            # Train model on poisoned data
            model.fit(X_poisoned, y_poisoned)
            
            # Evaluate on clean test data
            poisoned_pred = model.predict(X_test)
            poisoned_accuracy = accuracy_score(y_test, poisoned_pred)
            
            # Train clean model for comparison
            clean_model = type(model)()
            clean_model.fit(X_train, y_train)
            clean_pred = clean_model.predict(X_test)
            clean_accuracy = accuracy_score(y_test, clean_pred)
            
            # Calculate attack effectiveness
            attack_effectiveness = clean_accuracy - poisoned_accuracy
            
            return PoisoningResult(
                success=attack_effectiveness > 0.05,
                attack_type=PoisoningAttackType.TARGETED_POISONING,
                poison_ratio=poison_ratio,
                poisoned_samples=n_poison,
                original_accuracy=clean_accuracy,
                poisoned_accuracy=poisoned_accuracy,
                attack_effectiveness=attack_effectiveness,
                metadata={
                    "target_samples_count": len(target_samples),
                    "poison_indices": poison_indices.tolist()
                }
            )
            
        except Exception as e:
            logger.error(f"Targeted poisoning attack failed: {e}")
            return PoisoningResult(
                success=False,
                attack_type=PoisoningAttackType.TARGETED_POISONING,
                poison_ratio=poison_ratio,
                poisoned_samples=0,
                original_accuracy=0.0,
                poisoned_accuracy=0.0,
                attack_effectiveness=0.0,
                metadata={"error": str(e)}
            )
    
    def _calculate_poisoning_gradients(self, 
                                     model: Any, 
                                     X_poison: np.ndarray, 
                                     y_poison: np.ndarray) -> np.ndarray:
        """Calculate gradients for poisoning samples"""
        try:
            # This is a simplified implementation
            # In practice, you would need to implement proper gradient calculation
            # based on the specific model type
            
            # For now, return random gradients as placeholder
            gradients = np.random.normal(0, 0.1, X_poison.shape)
            return gradients
            
        except Exception as e:
            logger.warning(f"Failed to calculate poisoning gradients: {e}")
            return np.zeros_like(X_poison)
    
    async def run_comprehensive_poisoning_attacks(self, 
                                                X_train: np.ndarray,
                                                y_train: np.ndarray,
                                                X_test: np.ndarray,
                                                y_test: np.ndarray,
                                                model: Any) -> Dict[str, PoisoningResult]:
        """Run comprehensive poisoning attacks"""
        results = {}
        
        # Run all attack types
        attack_methods = [
            ("label_flipping", self.label_flipping_attack),
            ("feature_poisoning", self.feature_poisoning_attack),
            ("backdoor_injection", self.backdoor_injection_attack),
            ("gradient_ascent", self.gradient_ascent_attack),
            ("targeted_poisoning", self.targeted_poisoning_attack)
        ]
        
        for attack_name, attack_method in attack_methods:
            try:
                if attack_name == "targeted_poisoning":
                    # Generate target samples for targeted attack
                    target_samples = np.random.normal(0, 1, (10, X_train.shape[1]))
                    target_labels = np.random.randint(0, 2, 10)
                    result = await attack_method(X_train, y_train, X_test, y_test, model, 
                                              target_samples, target_labels)
                else:
                    result = await attack_method(X_train, y_train, X_test, y_test, model)
                
                results[attack_name] = result
                
            except Exception as e:
                logger.error(f"Poisoning attack {attack_name} failed: {e}")
                results[attack_name] = PoisoningResult(
                    success=False,
                    attack_type=PoisoningAttackType.LABEL_FLIPPING,
                    poison_ratio=0.0,
                    poisoned_samples=0,
                    original_accuracy=0.0,
                    poisoned_accuracy=0.0,
                    attack_effectiveness=0.0,
                    metadata={"error": str(e)}
                )
        
        return results
