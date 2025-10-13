"""
Membership Inference Attacks
Implements various membership inference attack methods
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class MIAttackType(Enum):
    """Types of membership inference attacks"""
    SHADOW_MODEL = "shadow_model"
    THRESHOLD_BASED = "threshold_based"
    LOSS_BASED = "loss_based"
    CONFIDENCE_BASED = "confidence_based"
    GRADIENT_BASED = "gradient_based"
    ADVERSARIAL = "adversarial"


@dataclass
class MIAttackConfig:
    """Configuration for membership inference attack"""
    attack_type: MIAttackType
    shadow_models: int = 5
    shadow_epochs: int = 10
    attack_epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    threshold: float = 0.5
    confidence_threshold: float = 0.8
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MIAttackResult:
    """Result of membership inference attack"""
    attack_type: MIAttackType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    attack_config: MIAttackConfig
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MembershipInferenceAttacker:
    """
    Membership Inference Attacker
    Implements various membership inference attack methods
    """
    
    def __init__(self):
        """Initialize membership inference attacker"""
        self.attack_results: List[MIAttackResult] = []
        self.shadow_models: List[nn.Module] = []
        self.attack_models: Dict[str, Any] = {}
        
        logger.info("✅ Initialized Membership Inference Attacker")
    
    async def perform_attack(self, 
                           target_model: nn.Module,
                           target_data: torch.Tensor,
                           target_labels: torch.Tensor,
                           non_member_data: torch.Tensor,
                           non_member_labels: torch.Tensor,
                           attack_config: MIAttackConfig) -> MIAttackResult:
        """
        Perform membership inference attack
        """
        try:
            logger.info(f"Performing {attack_config.attack_type.value} membership inference attack")
            
            if attack_config.attack_type == MIAttackType.SHADOW_MODEL:
                return await self._shadow_model_attack(
                    target_model, target_data, target_labels, 
                    non_member_data, non_member_labels, attack_config
                )
            elif attack_config.attack_type == MIAttackType.THRESHOLD_BASED:
                return await self._threshold_based_attack(
                    target_model, target_data, target_labels, 
                    non_member_data, non_member_labels, attack_config
                )
            elif attack_config.attack_type == MIAttackType.LOSS_BASED:
                return await self._loss_based_attack(
                    target_model, target_data, target_labels, 
                    non_member_data, non_member_labels, attack_config
                )
            elif attack_config.attack_type == MIAttackType.CONFIDENCE_BASED:
                return await self._confidence_based_attack(
                    target_model, target_data, target_labels, 
                    non_member_data, non_member_labels, attack_config
                )
            elif attack_config.attack_type == MIAttackType.GRADIENT_BASED:
                return await self._gradient_based_attack(
                    target_model, target_data, target_labels, 
                    non_member_data, non_member_labels, attack_config
                )
            elif attack_config.attack_type == MIAttackType.ADVERSARIAL:
                return await self._adversarial_attack(
                    target_model, target_data, target_labels, 
                    non_member_data, non_member_labels, attack_config
                )
            else:
                raise ValueError(f"Unknown attack type: {attack_config.attack_type}")
                
        except Exception as e:
            logger.error(f"Membership inference attack failed: {e}")
            raise
    
    async def _shadow_model_attack(self, 
                                 target_model: nn.Module,
                                 target_data: torch.Tensor,
                                 target_labels: torch.Tensor,
                                 non_member_data: torch.Tensor,
                                 non_member_labels: torch.Tensor,
                                 config: MIAttackConfig) -> MIAttackResult:
        """Perform shadow model attack"""
        try:
            # Train shadow models
            shadow_models = await self._train_shadow_models(
                target_data, target_labels, non_member_data, non_member_labels, config
            )
            
            # Generate training data for attack model
            attack_data, attack_labels = await self._generate_attack_data(
                shadow_models, target_data, target_labels, non_member_data, non_member_labels
            )
            
            # Train attack model
            attack_model = await self._train_attack_model(attack_data, attack_labels, config)
            
            # Evaluate attack
            predictions = attack_model.predict(attack_data)
            probabilities = attack_model.predict_proba(attack_data)[:, 1] if hasattr(attack_model, 'predict_proba') else predictions
            
            # Calculate metrics
            metrics = self._calculate_metrics(attack_labels, predictions, probabilities)
            
            # Create result
            result = MIAttackResult(
                attack_type=MIAttackType.SHADOW_MODEL,
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1_score=metrics["f1_score"],
                auc_score=metrics["auc_score"],
                true_positives=metrics["true_positives"],
                false_positives=metrics["false_positives"],
                true_negatives=metrics["true_negatives"],
                false_negatives=metrics["false_negatives"],
                attack_config=config,
                metadata={
                    "n_shadow_models": len(shadow_models),
                    "attack_data_size": len(attack_data),
                    "attack_model_type": attack_model.__class__.__name__
                }
            )
            
            # Store result
            self.attack_results.append(result)
            
            logger.info(f"Shadow model attack completed: accuracy={metrics['accuracy']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Shadow model attack failed: {e}")
            raise
    
    async def _threshold_based_attack(self, 
                                    target_model: nn.Module,
                                    target_data: torch.Tensor,
                                    target_labels: torch.Tensor,
                                    non_member_data: torch.Tensor,
                                    non_member_labels: torch.Tensor,
                                    config: MIAttackConfig) -> MIAttackResult:
        """Perform threshold-based attack"""
        try:
            # Get predictions on target data
            with torch.no_grad():
                target_model.eval()
                target_predictions = target_model(target_data)
                target_confidences = torch.softmax(target_predictions, dim=1)
                target_max_confidences = torch.max(target_confidences, dim=1)[0]
            
            # Get predictions on non-member data
            with torch.no_grad():
                non_member_predictions = target_model(non_member_data)
                non_member_confidences = torch.softmax(non_member_predictions, dim=1)
                non_member_max_confidences = torch.max(non_member_confidences, dim=1)[0]
            
            # Create labels (1 for member, 0 for non-member)
            member_labels = torch.ones(len(target_data))
            non_member_labels_tensor = torch.zeros(len(non_member_data))
            
            # Combine data
            all_confidences = torch.cat([target_max_confidences, non_member_max_confidences])
            all_labels = torch.cat([member_labels, non_member_labels_tensor])
            
            # Apply threshold
            predictions = (all_confidences > config.threshold).float()
            
            # Calculate metrics
            metrics = self._calculate_metrics(all_labels, predictions, all_confidences)
            
            # Create result
            result = MIAttackResult(
                attack_type=MIAttackType.THRESHOLD_BASED,
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1_score=metrics["f1_score"],
                auc_score=metrics["auc_score"],
                true_positives=metrics["true_positives"],
                false_positives=metrics["false_positives"],
                true_negatives=metrics["true_negatives"],
                false_negatives=metrics["false_negatives"],
                attack_config=config,
                metadata={
                    "threshold": config.threshold,
                    "member_data_size": len(target_data),
                    "non_member_data_size": len(non_member_data)
                }
            )
            
            # Store result
            self.attack_results.append(result)
            
            logger.info(f"Threshold-based attack completed: accuracy={metrics['accuracy']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Threshold-based attack failed: {e}")
            raise
    
    async def _loss_based_attack(self, 
                               target_model: nn.Module,
                               target_data: torch.Tensor,
                               target_labels: torch.Tensor,
                               non_member_data: torch.Tensor,
                               non_member_labels: torch.Tensor,
                               config: MIAttackConfig) -> MIAttackResult:
        """Perform loss-based attack"""
        try:
            # Calculate losses on target data
            with torch.no_grad():
                target_model.eval()
                target_predictions = target_model(target_data)
                target_losses = F.cross_entropy(target_predictions, target_labels, reduction='none')
            
            # Calculate losses on non-member data
            with torch.no_grad():
                non_member_predictions = target_model(non_member_data)
                non_member_losses = F.cross_entropy(non_member_predictions, non_member_labels, reduction='none')
            
            # Create labels (1 for member, 0 for non-member)
            member_labels = torch.ones(len(target_data))
            non_member_labels_tensor = torch.zeros(len(non_member_data))
            
            # Combine data
            all_losses = torch.cat([target_losses, non_member_losses])
            all_labels = torch.cat([member_labels, non_member_labels_tensor])
            
            # Apply threshold (members typically have lower loss)
            predictions = (all_losses < config.threshold).float()
            
            # Calculate metrics
            metrics = self._calculate_metrics(all_labels, predictions, -all_losses)  # Negative for AUC
            
            # Create result
            result = MIAttackResult(
                attack_type=MIAttackType.LOSS_BASED,
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1_score=metrics["f1_score"],
                auc_score=metrics["auc_score"],
                true_positives=metrics["true_positives"],
                false_positives=metrics["false_positives"],
                true_negatives=metrics["true_negatives"],
                false_negatives=metrics["false_negatives"],
                attack_config=config,
                metadata={
                    "threshold": config.threshold,
                    "member_data_size": len(target_data),
                    "non_member_data_size": len(non_member_data)
                }
            )
            
            # Store result
            self.attack_results.append(result)
            
            logger.info(f"Loss-based attack completed: accuracy={metrics['accuracy']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Loss-based attack failed: {e}")
            raise
    
    async def _confidence_based_attack(self, 
                                     target_model: nn.Module,
                                     target_data: torch.Tensor,
                                     target_labels: torch.Tensor,
                                     non_member_data: torch.Tensor,
                                     non_member_labels: torch.Tensor,
                                     config: MIAttackConfig) -> MIAttackResult:
        """Perform confidence-based attack"""
        try:
            # Get predictions on target data
            with torch.no_grad():
                target_model.eval()
                target_predictions = target_model(target_data)
                target_confidences = torch.softmax(target_predictions, dim=1)
                target_max_confidences = torch.max(target_confidences, dim=1)[0]
            
            # Get predictions on non-member data
            with torch.no_grad():
                non_member_predictions = target_model(non_member_data)
                non_member_confidences = torch.softmax(non_member_predictions, dim=1)
                non_member_max_confidences = torch.max(non_member_confidences, dim=1)[0]
            
            # Create labels (1 for member, 0 for non-member)
            member_labels = torch.ones(len(target_data))
            non_member_labels_tensor = torch.zeros(len(non_member_data))
            
            # Combine data
            all_confidences = torch.cat([target_max_confidences, non_member_max_confidences])
            all_labels = torch.cat([member_labels, non_member_labels_tensor])
            
            # Apply confidence threshold
            predictions = (all_confidences > config.confidence_threshold).float()
            
            # Calculate metrics
            metrics = self._calculate_metrics(all_labels, predictions, all_confidences)
            
            # Create result
            result = MIAttackResult(
                attack_type=MIAttackType.CONFIDENCE_BASED,
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1_score=metrics["f1_score"],
                auc_score=metrics["auc_score"],
                true_positives=metrics["true_positives"],
                false_positives=metrics["false_positives"],
                true_negatives=metrics["true_negatives"],
                false_negatives=metrics["false_negatives"],
                attack_config=config,
                metadata={
                    "confidence_threshold": config.confidence_threshold,
                    "member_data_size": len(target_data),
                    "non_member_data_size": len(non_member_data)
                }
            )
            
            # Store result
            self.attack_results.append(result)
            
            logger.info(f"Confidence-based attack completed: accuracy={metrics['accuracy']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Confidence-based attack failed: {e}")
            raise
    
    async def _gradient_based_attack(self, 
                                   target_model: nn.Module,
                                   target_data: torch.Tensor,
                                   target_labels: torch.Tensor,
                                   non_member_data: torch.Tensor,
                                   non_member_labels: torch.Tensor,
                                   config: MIAttackConfig) -> MIAttackResult:
        """Perform gradient-based attack"""
        try:
            # Calculate gradients for target data
            target_gradients = await self._calculate_gradients(target_model, target_data, target_labels)
            
            # Calculate gradients for non-member data
            non_member_gradients = await self._calculate_gradients(target_model, non_member_data, non_member_labels)
            
            # Create labels (1 for member, 0 for non-member)
            member_labels = torch.ones(len(target_data))
            non_member_labels_tensor = torch.zeros(len(non_member_data))
            
            # Combine data
            all_gradients = torch.cat([target_gradients, non_member_gradients])
            all_labels = torch.cat([member_labels, non_member_labels_tensor])
            
            # Train attack model on gradients
            attack_model = await self._train_attack_model(all_gradients, all_labels, config)
            
            # Evaluate attack
            predictions = attack_model.predict(all_gradients)
            probabilities = attack_model.predict_proba(all_gradients)[:, 1] if hasattr(attack_model, 'predict_proba') else predictions
            
            # Calculate metrics
            metrics = self._calculate_metrics(all_labels, predictions, probabilities)
            
            # Create result
            result = MIAttackResult(
                attack_type=MIAttackType.GRADIENT_BASED,
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1_score=metrics["f1_score"],
                auc_score=metrics["auc_score"],
                true_positives=metrics["true_positives"],
                false_positives=metrics["false_positives"],
                true_negatives=metrics["true_negatives"],
                false_negatives=metrics["false_negatives"],
                attack_config=config,
                metadata={
                    "member_data_size": len(target_data),
                    "non_member_data_size": len(non_member_data),
                    "gradient_dim": target_gradients.shape[1]
                }
            )
            
            # Store result
            self.attack_results.append(result)
            
            logger.info(f"Gradient-based attack completed: accuracy={metrics['accuracy']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Gradient-based attack failed: {e}")
            raise
    
    async def _adversarial_attack(self, 
                                target_model: nn.Module,
                                target_data: torch.Tensor,
                                target_labels: torch.Tensor,
                                non_member_data: torch.Tensor,
                                non_member_labels: torch.Tensor,
                                config: MIAttackConfig) -> MIAttackResult:
        """Perform adversarial membership inference attack"""
        try:
            # Generate adversarial examples for target data
            target_adversarial = await self._generate_adversarial_examples(target_model, target_data, target_labels)
            
            # Generate adversarial examples for non-member data
            non_member_adversarial = await self._generate_adversarial_examples(target_model, non_member_data, non_member_labels)
            
            # Get predictions on adversarial examples
            with torch.no_grad():
                target_model.eval()
                target_predictions = target_model(target_adversarial)
                target_confidences = torch.softmax(target_predictions, dim=1)
                target_max_confidences = torch.max(target_confidences, dim=1)[0]
            
            with torch.no_grad():
                non_member_predictions = target_model(non_member_adversarial)
                non_member_confidences = torch.softmax(non_member_predictions, dim=1)
                non_member_max_confidences = torch.max(non_member_confidences, dim=1)[0]
            
            # Create labels (1 for member, 0 for non-member)
            member_labels = torch.ones(len(target_data))
            non_member_labels_tensor = torch.zeros(len(non_member_data))
            
            # Combine data
            all_confidences = torch.cat([target_max_confidences, non_member_max_confidences])
            all_labels = torch.cat([member_labels, non_member_labels_tensor])
            
            # Apply threshold
            predictions = (all_confidences > config.threshold).float()
            
            # Calculate metrics
            metrics = self._calculate_metrics(all_labels, predictions, all_confidences)
            
            # Create result
            result = MIAttackResult(
                attack_type=MIAttackType.ADVERSARIAL,
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1_score=metrics["f1_score"],
                auc_score=metrics["auc_score"],
                true_positives=metrics["true_positives"],
                false_positives=metrics["false_positives"],
                true_negatives=metrics["true_negatives"],
                false_negatives=metrics["false_negatives"],
                attack_config=config,
                metadata={
                    "threshold": config.threshold,
                    "member_data_size": len(target_data),
                    "non_member_data_size": len(non_member_data)
                }
            )
            
            # Store result
            self.attack_results.append(result)
            
            logger.info(f"Adversarial attack completed: accuracy={metrics['accuracy']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Adversarial attack failed: {e}")
            raise
    
    async def _train_shadow_models(self, 
                                 target_data: torch.Tensor,
                                 target_labels: torch.Tensor,
                                 non_member_data: torch.Tensor,
                                 non_member_labels: torch.Tensor,
                                 config: MIAttackConfig) -> List[nn.Module]:
        """Train shadow models"""
        try:
            shadow_models = []
            
            for i in range(config.shadow_models):
                # Create shadow model (same architecture as target)
                shadow_model = self._create_shadow_model(target_data.shape[1:])
                
                # Split data for shadow model
                shadow_data = torch.cat([target_data, non_member_data])
                shadow_labels = torch.cat([target_labels, non_member_labels])
                
                # Shuffle data
                indices = torch.randperm(len(shadow_data))
                shadow_data = shadow_data[indices]
                shadow_labels = shadow_labels[indices]
                
                # Train shadow model
                await self._train_model(shadow_model, shadow_data, shadow_labels, config.shadow_epochs)
                
                shadow_models.append(shadow_model)
            
            return shadow_models
            
        except Exception as e:
            logger.error(f"Shadow model training failed: {e}")
            return []
    
    def _create_shadow_model(self, input_shape: Tuple[int, ...]) -> nn.Module:
        """Create shadow model with same architecture as target"""
        # Simple CNN for image data
        if len(input_shape) == 3:  # C, H, W
            return nn.Sequential(
                nn.Conv2d(input_shape[0], 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * (input_shape[1] // 4) * (input_shape[2] // 4), 128),
                nn.ReLU(),
                nn.Linear(128, 10)  # Assuming 10 classes
            )
        else:
            # Simple MLP for other data
            input_size = np.prod(input_shape)
            return nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)  # Assuming 10 classes
            )
    
    async def _train_model(self, model: nn.Module, data: torch.Tensor, labels: torch.Tensor, epochs: int):
        """Train a model"""
        try:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                predictions = model(data)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    async def _generate_attack_data(self, 
                                  shadow_models: List[nn.Module],
                                  target_data: torch.Tensor,
                                  target_labels: torch.Tensor,
                                  non_member_data: torch.Tensor,
                                  non_member_labels: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data for attack model"""
        try:
            attack_data = []
            attack_labels = []
            
            # Generate data from shadow models
            for shadow_model in shadow_models:
                with torch.no_grad():
                    shadow_model.eval()
                    
                    # Get predictions on target data
                    target_predictions = shadow_model(target_data)
                    target_confidences = torch.softmax(target_predictions, dim=1)
                    target_max_confidences = torch.max(target_confidences, dim=1)[0]
                    
                    # Get predictions on non-member data
                    non_member_predictions = shadow_model(non_member_data)
                    non_member_confidences = torch.softmax(non_member_predictions, dim=1)
                    non_member_max_confidences = torch.max(non_member_confidences, dim=1)[0]
                    
                    # Combine data
                    all_confidences = torch.cat([target_max_confidences, non_member_max_confidences])
                    all_labels = torch.cat([torch.ones(len(target_data)), torch.zeros(len(non_member_data))])
                    
                    attack_data.append(all_confidences.numpy())
                    attack_labels.append(all_labels.numpy())
            
            # Combine all shadow model data
            attack_data = np.concatenate(attack_data, axis=0)
            attack_labels = np.concatenate(attack_labels, axis=0)
            
            return attack_data, attack_labels
            
        except Exception as e:
            logger.error(f"Attack data generation failed: {e}")
            return np.array([]), np.array([])
    
    async def _train_attack_model(self, data: np.ndarray, labels: np.ndarray, config: MIAttackConfig) -> Any:
        """Train attack model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
            
            # Create attack model
            attack_model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Train attack model
            attack_model.fit(X_train, y_train)
            
            return attack_model
            
        except Exception as e:
            logger.error(f"Attack model training failed: {e}")
            raise
    
    async def _calculate_gradients(self, model: nn.Module, data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculate gradients for given data"""
        try:
            model.train()
            data.requires_grad_(True)
            
            # Forward pass
            predictions = model(data)
            loss = F.cross_entropy(predictions, labels)
            
            # Backward pass
            gradients = torch.autograd.grad(loss, data, create_graph=True)[0]
            
            # Flatten gradients
            gradients = gradients.view(gradients.shape[0], -1)
            
            return gradients.detach()
            
        except Exception as e:
            logger.error(f"Gradient calculation failed: {e}")
            return torch.zeros(data.shape[0], 1)
    
    async def _generate_adversarial_examples(self, model: nn.Module, data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples using FGSM"""
        try:
            model.eval()
            data.requires_grad_(True)
            
            # Forward pass
            predictions = model(data)
            loss = F.cross_entropy(predictions, labels)
            
            # Calculate gradients
            gradients = torch.autograd.grad(loss, data)[0]
            
            # Generate adversarial examples
            epsilon = 0.1
            adversarial_data = data + epsilon * gradients.sign()
            
            return adversarial_data.detach()
            
        except Exception as e:
            logger.error(f"Adversarial example generation failed: {e}")
            return data
    
    def _calculate_metrics(self, true_labels: torch.Tensor, predictions: torch.Tensor, probabilities: torch.Tensor) -> Dict[str, float]:
        """Calculate attack metrics"""
        try:
            # Convert to numpy
            true_labels = true_labels.numpy()
            predictions = predictions.numpy()
            probabilities = probabilities.numpy()
            
            # Calculate basic metrics
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            f1 = f1_score(true_labels, predictions, zero_division=0)
            
            # Calculate AUC
            try:
                auc = roc_auc_score(true_labels, probabilities)
            except:
                auc = 0.0
            
            # Calculate confusion matrix
            tp = np.sum((true_labels == 1) & (predictions == 1))
            fp = np.sum((true_labels == 0) & (predictions == 1))
            tn = np.sum((true_labels == 0) & (predictions == 0))
            fn = np.sum((true_labels == 1) & (predictions == 0))
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "auc_score": auc,
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn)
            }
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "auc_score": 0.0,
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0
            }
    
    async def get_attack_summary(self) -> Dict[str, Any]:
        """Get summary of all attack results"""
        try:
            if not self.attack_results:
                return {"message": "No attack results available"}
            
            # Calculate statistics
            total_attacks = len(self.attack_results)
            successful_attacks = sum(1 for r in self.attack_results if r.accuracy > 0.5)
            success_rate = successful_attacks / total_attacks
            
            # Calculate method-specific statistics
            method_stats = defaultdict(lambda: {"count": 0, "accuracies": [], "auc_scores": []})
            
            for result in self.attack_results:
                method = result.attack_type.value
                method_stats[method]["count"] += 1
                method_stats[method]["accuracies"].append(result.accuracy)
                method_stats[method]["auc_scores"].append(result.auc_score)
            
            # Calculate averages for each method
            for method, stats in method_stats.items():
                stats["mean_accuracy"] = np.mean(stats["accuracies"])
                stats["mean_auc"] = np.mean(stats["auc_scores"])
                stats["std_accuracy"] = np.std(stats["accuracies"])
                stats["std_auc"] = np.std(stats["auc_scores"])
            
            return {
                "total_attacks": total_attacks,
                "successful_attacks": successful_attacks,
                "overall_success_rate": success_rate,
                "method_statistics": dict(method_stats),
                "summary_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Attack summary generation failed: {e}")
            return {}
    
    async def export_attack_data(self, format: str = "json") -> str:
        """Export attack data"""
        try:
            if format.lower() == "json":
                data = {
                    "attack_results": [r.__dict__ for r in self.attack_results],
                    "summary": await self.get_attack_summary(),
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Attack data export failed: {e}")
            return ""
