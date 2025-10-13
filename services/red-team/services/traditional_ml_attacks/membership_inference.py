"""
Membership Inference Attacks for Traditional ML Models
Privacy attacks to determine if specific data was used in training
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
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

logger = logging.getLogger(__name__)


class MembershipInferenceType(Enum):
    """Types of membership inference attacks"""
    CONFIDENCE_BASED = "confidence_based"
    LOSS_BASED = "loss_based"
    SHADOW_MODEL = "shadow_model"
    ATTACK_MODEL = "attack_model"
    THRESHOLD_ATTACK = "threshold_attack"


@dataclass
class MembershipInferenceResult:
    """Result of membership inference attack"""
    success: bool
    attack_type: MembershipInferenceType
    attack_accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    attack_samples: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MembershipInferenceAttacks:
    """
    Membership inference attacks for traditional ML models
    Focuses on determining if specific data was used in training
    """
    
    def __init__(self):
        """Initialize membership inference attack generator"""
        self.attack_strategies = self._load_attack_strategies()
        self.shadow_model_types = self._load_shadow_model_types()
        
        logger.info("✅ Initialized MembershipInferenceAttacks")
    
    def _load_attack_strategies(self) -> List[Dict[str, Any]]:
        """Load membership inference attack strategies"""
        return [
            {
                "name": "confidence_threshold",
                "description": "Use prediction confidence to infer membership",
                "efficiency": "high",
                "stealth": "high"
            },
            {
                "name": "loss_threshold",
                "description": "Use prediction loss to infer membership",
                "efficiency": "high",
                "stealth": "medium"
            },
            {
                "name": "shadow_model",
                "description": "Train shadow models to infer membership",
                "efficiency": "medium",
                "stealth": "low"
            },
            {
                "name": "attack_model",
                "description": "Train attack model to infer membership",
                "efficiency": "high",
                "stealth": "medium"
            },
            {
                "name": "statistical_attack",
                "description": "Use statistical properties to infer membership",
                "efficiency": "medium",
                "stealth": "high"
            }
        ]
    
    def _load_shadow_model_types(self) -> List[str]:
        """Load shadow model types"""
        return [
            "random_forest",
            "logistic_regression",
            "svm",
            "gradient_boosting",
            "neural_network"
        ]
    
    async def confidence_based_attack(self, 
                                    target_model: Any,
                                    X_train: np.ndarray,
                                    y_train: np.ndarray,
                                    X_test: np.ndarray,
                                    y_test: np.ndarray,
                                    confidence_threshold: float = 0.8) -> MembershipInferenceResult:
        """
        Confidence-based membership inference attack
        Use prediction confidence to infer membership
        """
        try:
            logger.info("Starting confidence-based membership inference attack")
            
            # Get prediction confidences for training data
            train_confidences = self._get_prediction_confidences(target_model, X_train)
            
            # Get prediction confidences for test data
            test_confidences = self._get_prediction_confidences(target_model, X_test)
            
            # Create membership labels
            train_labels = np.ones(len(X_train))  # 1 for training data
            test_labels = np.zeros(len(X_test))   # 0 for test data
            
            # Combine data
            all_confidences = np.concatenate([train_confidences, test_confidences])
            all_labels = np.concatenate([train_labels, test_labels])
            
            # Use confidence threshold for membership inference
            predicted_membership = (all_confidences >= confidence_threshold).astype(int)
            
            # Calculate attack metrics
            attack_accuracy = accuracy_score(all_labels, predicted_membership)
            precision = self._calculate_precision(all_labels, predicted_membership)
            recall = self._calculate_recall(all_labels, predicted_membership)
            f1_score = self._calculate_f1_score(all_labels, predicted_membership)
            auc_score = self._calculate_auc_score(all_labels, all_confidences)
            
            return MembershipInferenceResult(
                success=attack_accuracy > 0.6,  # 60% accuracy threshold
                attack_type=MembershipInferenceType.CONFIDENCE_BASED,
                attack_accuracy=attack_accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                auc_score=auc_score,
                attack_samples=len(all_confidences),
                metadata={
                    "confidence_threshold": confidence_threshold,
                    "train_confidences_mean": np.mean(train_confidences),
                    "test_confidences_mean": np.mean(test_confidences)
                }
            )
            
        except Exception as e:
            logger.error(f"Confidence-based attack failed: {e}")
            return MembershipInferenceResult(
                success=False,
                attack_type=MembershipInferenceType.CONFIDENCE_BASED,
                attack_accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                auc_score=0.0,
                attack_samples=0,
                metadata={"error": str(e)}
            )
    
    async def loss_based_attack(self, 
                              target_model: Any,
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              loss_threshold: float = 0.5) -> MembershipInferenceResult:
        """
        Loss-based membership inference attack
        Use prediction loss to infer membership
        """
        try:
            logger.info("Starting loss-based membership inference attack")
            
            # Get prediction losses for training data
            train_losses = self._get_prediction_losses(target_model, X_train, y_train)
            
            # Get prediction losses for test data
            test_losses = self._get_prediction_losses(target_model, X_test, y_test)
            
            # Create membership labels
            train_labels = np.ones(len(X_train))  # 1 for training data
            test_labels = np.zeros(len(X_test))   # 0 for test data
            
            # Combine data
            all_losses = np.concatenate([train_losses, test_losses])
            all_labels = np.concatenate([train_labels, test_labels])
            
            # Use loss threshold for membership inference
            predicted_membership = (all_losses <= loss_threshold).astype(int)
            
            # Calculate attack metrics
            attack_accuracy = accuracy_score(all_labels, predicted_membership)
            precision = self._calculate_precision(all_labels, predicted_membership)
            recall = self._calculate_recall(all_labels, predicted_membership)
            f1_score = self._calculate_f1_score(all_labels, predicted_membership)
            auc_score = self._calculate_auc_score(all_labels, -all_losses)  # Negative for AUC
            
            return MembershipInferenceResult(
                success=attack_accuracy > 0.6,
                attack_type=MembershipInferenceType.LOSS_BASED,
                attack_accuracy=attack_accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                auc_score=auc_score,
                attack_samples=len(all_losses),
                metadata={
                    "loss_threshold": loss_threshold,
                    "train_losses_mean": np.mean(train_losses),
                    "test_losses_mean": np.mean(test_losses)
                }
            )
            
        except Exception as e:
            logger.error(f"Loss-based attack failed: {e}")
            return MembershipInferenceResult(
                success=False,
                attack_type=MembershipInferenceType.LOSS_BASED,
                attack_accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                auc_score=0.0,
                attack_samples=0,
                metadata={"error": str(e)}
            )
    
    async def shadow_model_attack(self, 
                                target_model: Any,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                X_test: np.ndarray,
                                y_test: np.ndarray,
                                shadow_model_type: str = "random_forest",
                                n_shadow_models: int = 5) -> MembershipInferenceResult:
        """
        Shadow model membership inference attack
        Train shadow models to infer membership
        """
        try:
            logger.info(f"Starting shadow model attack with {n_shadow_models} shadow models")
            
            # Generate shadow training data
            shadow_train_data = self._generate_shadow_data(X_train, len(X_train))
            shadow_test_data = self._generate_shadow_data(X_test, len(X_test))
            
            # Train shadow models
            shadow_models = []
            for i in range(n_shadow_models):
                # Create shadow model
                shadow_model = self._create_shadow_model(shadow_model_type)
                
                # Generate shadow labels using target model
                shadow_train_labels = target_model.predict(shadow_train_data)
                shadow_test_labels = target_model.predict(shadow_test_data)
                
                # Train shadow model
                shadow_model.fit(shadow_train_data, shadow_train_labels)
                shadow_models.append(shadow_model)
            
            # Create attack dataset
            attack_features = []
            attack_labels = []
            
            # Features from training data
            for model in shadow_models:
                train_confidences = self._get_prediction_confidences(model, X_train)
                attack_features.append(train_confidences)
            attack_labels.extend([1] * len(X_train))  # 1 for training data
            
            # Features from test data
            for model in shadow_models:
                test_confidences = self._get_prediction_confidences(model, X_test)
                attack_features.append(test_confidences)
            attack_labels.extend([0] * len(X_test))   # 0 for test data
            
            # Train attack model
            attack_features = np.array(attack_features).T
            attack_labels = np.array(attack_labels)
            
            attack_model = LogisticRegression(random_state=42)
            attack_model.fit(attack_features, attack_labels)
            
            # Evaluate attack
            attack_pred = attack_model.predict(attack_features)
            attack_accuracy = accuracy_score(attack_labels, attack_pred)
            
            # Get prediction probabilities for AUC
            if hasattr(attack_model, 'predict_proba'):
                attack_probs = attack_model.predict_proba(attack_features)[:, 1]
                auc_score = roc_auc_score(attack_labels, attack_probs)
            else:
                auc_score = 0.0
            
            precision = self._calculate_precision(attack_labels, attack_pred)
            recall = self._calculate_recall(attack_labels, attack_pred)
            f1_score = self._calculate_f1_score(attack_labels, attack_pred)
            
            return MembershipInferenceResult(
                success=attack_accuracy > 0.6,
                attack_type=MembershipInferenceType.SHADOW_MODEL,
                attack_accuracy=attack_accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                auc_score=auc_score,
                attack_samples=len(attack_features),
                metadata={
                    "shadow_model_type": shadow_model_type,
                    "n_shadow_models": n_shadow_models,
                    "attack_model_type": type(attack_model).__name__
                }
            )
            
        except Exception as e:
            logger.error(f"Shadow model attack failed: {e}")
            return MembershipInferenceResult(
                success=False,
                attack_type=MembershipInferenceType.SHADOW_MODEL,
                attack_accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                auc_score=0.0,
                attack_samples=0,
                metadata={"error": str(e)}
            )
    
    async def attack_model_attack(self, 
                                target_model: Any,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                X_test: np.ndarray,
                                y_test: np.ndarray,
                                attack_model_type: str = "logistic_regression") -> MembershipInferenceResult:
        """
        Attack model membership inference attack
        Train attack model to infer membership
        """
        try:
            logger.info("Starting attack model membership inference attack")
            
            # Create features for attack model
            train_features = self._create_attack_features(target_model, X_train, y_train)
            test_features = self._create_attack_features(target_model, X_test, y_test)
            
            # Create labels
            train_labels = np.ones(len(X_train))  # 1 for training data
            test_labels = np.zeros(len(X_test))   # 0 for test data
            
            # Combine data
            all_features = np.vstack([train_features, test_features])
            all_labels = np.concatenate([train_labels, test_labels])
            
            # Train attack model
            attack_model = self._create_attack_model(attack_model_type)
            attack_model.fit(all_features, all_labels)
            
            # Evaluate attack
            attack_pred = attack_model.predict(all_features)
            attack_accuracy = accuracy_score(all_labels, attack_pred)
            
            # Get prediction probabilities for AUC
            if hasattr(attack_model, 'predict_proba'):
                attack_probs = attack_model.predict_proba(all_features)[:, 1]
                auc_score = roc_auc_score(all_labels, attack_probs)
            else:
                auc_score = 0.0
            
            precision = self._calculate_precision(all_labels, attack_pred)
            recall = self._calculate_recall(all_labels, attack_pred)
            f1_score = self._calculate_f1_score(all_labels, attack_pred)
            
            return MembershipInferenceResult(
                success=attack_accuracy > 0.6,
                attack_type=MembershipInferenceType.ATTACK_MODEL,
                attack_accuracy=attack_accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                auc_score=auc_score,
                attack_samples=len(all_features),
                metadata={
                    "attack_model_type": attack_model_type,
                    "feature_dimension": all_features.shape[1]
                }
            )
            
        except Exception as e:
            logger.error(f"Attack model attack failed: {e}")
            return MembershipInferenceResult(
                success=False,
                attack_type=MembershipInferenceType.ATTACK_MODEL,
                attack_accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                auc_score=0.0,
                attack_samples=0,
                metadata={"error": str(e)}
            )
    
    async def threshold_attack(self, 
                             target_model: Any,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             X_test: np.ndarray,
                             y_test: np.ndarray,
                             threshold_type: str = "confidence") -> MembershipInferenceResult:
        """
        Threshold-based membership inference attack
        Use various thresholds to infer membership
        """
        try:
            logger.info(f"Starting threshold attack with type {threshold_type}")
            
            if threshold_type == "confidence":
                # Use confidence threshold
                train_scores = self._get_prediction_confidences(target_model, X_train)
                test_scores = self._get_prediction_confidences(target_model, X_test)
            elif threshold_type == "loss":
                # Use loss threshold
                train_scores = -self._get_prediction_losses(target_model, X_train, y_train)
                test_scores = -self._get_prediction_losses(target_model, X_test, y_test)
            else:
                # Default to confidence
                train_scores = self._get_prediction_confidences(target_model, X_train)
                test_scores = self._get_prediction_confidences(target_model, X_test)
            
            # Find optimal threshold
            all_scores = np.concatenate([train_scores, test_scores])
            all_labels = np.concatenate([np.ones(len(X_train)), np.zeros(len(X_test))])
            
            # Try different thresholds
            thresholds = np.linspace(np.min(all_scores), np.max(all_scores), 100)
            best_accuracy = 0.0
            best_threshold = 0.0
            
            for threshold in thresholds:
                predicted_membership = (all_scores >= threshold).astype(int)
                accuracy = accuracy_score(all_labels, predicted_membership)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
            
            # Use best threshold
            predicted_membership = (all_scores >= best_threshold).astype(int)
            
            # Calculate metrics
            attack_accuracy = accuracy_score(all_labels, predicted_membership)
            precision = self._calculate_precision(all_labels, predicted_membership)
            recall = self._calculate_recall(all_labels, predicted_membership)
            f1_score = self._calculate_f1_score(all_labels, predicted_membership)
            auc_score = self._calculate_auc_score(all_labels, all_scores)
            
            return MembershipInferenceResult(
                success=attack_accuracy > 0.6,
                attack_type=MembershipInferenceType.THRESHOLD_ATTACK,
                attack_accuracy=attack_accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                auc_score=auc_score,
                attack_samples=len(all_scores),
                metadata={
                    "threshold_type": threshold_type,
                    "best_threshold": best_threshold,
                    "train_scores_mean": np.mean(train_scores),
                    "test_scores_mean": np.mean(test_scores)
                }
            )
            
        except Exception as e:
            logger.error(f"Threshold attack failed: {e}")
            return MembershipInferenceResult(
                success=False,
                attack_type=MembershipInferenceType.THRESHOLD_ATTACK,
                attack_accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                auc_score=0.0,
                attack_samples=0,
                metadata={"error": str(e)}
            )
    
    def _get_prediction_confidences(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get prediction confidences from model"""
        try:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                # Return maximum probability as confidence
                return np.max(probs, axis=1)
            else:
                # Fallback: use prediction scores
                if hasattr(model, 'decision_function'):
                    scores = model.decision_function(X)
                    # Normalize scores to [0, 1]
                    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
                    return scores
                else:
                    # Random confidences as fallback
                    return np.random.random(len(X))
        except Exception as e:
            logger.warning(f"Failed to get prediction confidences: {e}")
            return np.random.random(len(X))
    
    def _get_prediction_losses(self, model: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Get prediction losses from model"""
        try:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                # Calculate cross-entropy loss
                losses = []
                for i, true_label in enumerate(y):
                    if true_label < len(probs[i]):
                        loss = -np.log(probs[i, int(true_label)] + 1e-10)
                        losses.append(loss)
                    else:
                        losses.append(1.0)  # Default loss
                return np.array(losses)
            else:
                # Fallback: use prediction accuracy
                predictions = model.predict(X)
                losses = (predictions != y).astype(float)
                return losses
        except Exception as e:
            logger.warning(f"Failed to get prediction losses: {e}")
            return np.random.random(len(X))
    
    def _generate_shadow_data(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        """Generate shadow data for shadow model attack"""
        # Generate data based on original data distribution
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        shadow_data = np.random.normal(mean, std, (n_samples, X.shape[1]))
        return shadow_data
    
    def _create_shadow_model(self, model_type: str) -> Any:
        """Create shadow model of specified type"""
        if model_type == "random_forest":
            return RandomForestClassifier(n_estimators=50, random_state=42)
        elif model_type == "logistic_regression":
            return LogisticRegression(random_state=42)
        elif model_type == "svm":
            return SVC(probability=True, random_state=42)
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(random_state=42)
        else:
            return RandomForestClassifier(n_estimators=50, random_state=42)
    
    def _create_attack_features(self, model: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Create features for attack model"""
        features = []
        
        # Add prediction confidence
        confidences = self._get_prediction_confidences(model, X)
        features.append(confidences)
        
        # Add prediction loss
        losses = self._get_prediction_losses(model, X, y)
        features.append(losses)
        
        # Add prediction correctness
        predictions = model.predict(X)
        correctness = (predictions == y).astype(float)
        features.append(correctness)
        
        # Add input features (first few dimensions)
        if X.shape[1] > 0:
            features.append(X[:, 0])  # First feature
        if X.shape[1] > 1:
            features.append(X[:, 1])  # Second feature
        
        return np.array(features).T
    
    def _create_attack_model(self, model_type: str) -> Any:
        """Create attack model of specified type"""
        if model_type == "logistic_regression":
            return LogisticRegression(random_state=42)
        elif model_type == "random_forest":
            return RandomForestClassifier(n_estimators=50, random_state=42)
        elif model_type == "svm":
            return SVC(probability=True, random_state=42)
        else:
            return LogisticRegression(random_state=42)
    
    def _calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision score"""
        try:
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate recall score"""
        try:
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score"""
        try:
            precision = self._calculate_precision(y_true, y_pred)
            recall = self._calculate_recall(y_true, y_pred)
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_auc_score(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Calculate AUC score"""
        try:
            return roc_auc_score(y_true, y_scores)
        except Exception:
            return 0.0
    
    async def run_comprehensive_membership_attacks(self, 
                                                 target_model: Any,
                                                 X_train: np.ndarray,
                                                 y_train: np.ndarray,
                                                 X_test: np.ndarray,
                                                 y_test: np.ndarray) -> Dict[str, MembershipInferenceResult]:
        """Run comprehensive membership inference attacks"""
        results = {}
        
        # Run all attack types
        attack_methods = [
            ("confidence_based", self.confidence_based_attack),
            ("loss_based", self.loss_based_attack),
            ("shadow_model", self.shadow_model_attack),
            ("attack_model", self.attack_model_attack),
            ("threshold_attack", self.threshold_attack)
        ]
        
        for attack_name, attack_method in attack_methods:
            try:
                result = await attack_method(target_model, X_train, y_train, X_test, y_test)
                results[attack_name] = result
                
            except Exception as e:
                logger.error(f"Membership inference attack {attack_name} failed: {e}")
                results[attack_name] = MembershipInferenceResult(
                    success=False,
                    attack_type=MembershipInferenceType.CONFIDENCE_BASED,
                    attack_accuracy=0.0,
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    auc_score=0.0,
                    attack_samples=0,
                    metadata={"error": str(e)}
                )
        
        return results
