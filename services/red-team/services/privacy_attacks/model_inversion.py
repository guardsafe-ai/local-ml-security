"""
Model Inversion Attacks
Implements model inversion attacks to reconstruct training data
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
from collections import defaultdict
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class InversionMethod(Enum):
    """Types of model inversion methods"""
    GRADIENT_DESCENT = "gradient_descent"
    GENERATIVE_MODEL = "generative_model"
    OPTIMIZATION_BASED = "optimization_based"
    ADVERSARIAL = "adversarial"
    DEEP_DREAM = "deep_dream"


@dataclass
class InversionConfig:
    """Configuration for model inversion attack"""
    method: InversionMethod
    target_class: int
    num_iterations: int = 1000
    learning_rate: float = 0.01
    regularization_weight: float = 0.01
    noise_level: float = 0.1
    image_size: Tuple[int, int] = (224, 224)
    num_channels: int = 3
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class InversionResult:
    """Result of model inversion attack"""
    method: InversionMethod
    reconstructed_data: torch.Tensor
    target_class: int
    confidence: float
    reconstruction_loss: float
    num_iterations: int
    inversion_config: InversionConfig
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ModelInversionAttacker:
    """
    Model Inversion Attacker
    Implements model inversion attacks to reconstruct training data
    """
    
    def __init__(self):
        """Initialize model inversion attacker"""
        self.inversion_results: List[InversionResult] = []
        self.generative_models: Dict[str, nn.Module] = {}
        
        logger.info("✅ Initialized Model Inversion Attacker")
    
    async def perform_inversion(self, 
                              target_model: nn.Module,
                              target_class: int,
                              inversion_config: InversionConfig) -> InversionResult:
        """
        Perform model inversion attack
        """
        try:
            logger.info(f"Performing {inversion_config.method.value} model inversion for class {target_class}")
            
            if inversion_config.method == InversionMethod.GRADIENT_DESCENT:
                return await self._gradient_descent_inversion(target_model, target_class, inversion_config)
            elif inversion_config.method == InversionMethod.GENERATIVE_MODEL:
                return await self._generative_model_inversion(target_model, target_class, inversion_config)
            elif inversion_config.method == InversionMethod.OPTIMIZATION_BASED:
                return await self._optimization_based_inversion(target_model, target_class, inversion_config)
            elif inversion_config.method == InversionMethod.ADVERSARIAL:
                return await self._adversarial_inversion(target_model, target_class, inversion_config)
            elif inversion_config.method == InversionMethod.DEEP_DREAM:
                return await self._deep_dream_inversion(target_model, target_class, inversion_config)
            else:
                raise ValueError(f"Unknown inversion method: {inversion_config.method}")
                
        except Exception as e:
            logger.error(f"Model inversion attack failed: {e}")
            raise
    
    async def _gradient_descent_inversion(self, 
                                        target_model: nn.Module,
                                        target_class: int,
                                        config: InversionConfig) -> InversionResult:
        """Perform gradient descent-based inversion"""
        try:
            # Initialize random data
            reconstructed_data = torch.randn(1, config.num_channels, *config.image_size, requires_grad=True)
            
            # Set up optimizer
            optimizer = torch.optim.Adam([reconstructed_data], lr=config.learning_rate)
            
            # Set model to evaluation mode
            target_model.eval()
            
            best_data = None
            best_loss = float('inf')
            
            for iteration in range(config.num_iterations):
                optimizer.zero_grad()
                
                # Forward pass
                predictions = target_model(reconstructed_data)
                
                # Calculate loss
                target_loss = F.cross_entropy(predictions, torch.tensor([target_class]))
                
                # Add regularization
                regularization = config.regularization_weight * torch.norm(reconstructed_data)
                total_loss = target_loss + regularization
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Clamp values to valid range
                reconstructed_data.data = torch.clamp(reconstructed_data.data, 0, 1)
                
                # Track best result
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_data = reconstructed_data.data.clone()
                
                # Log progress
                if iteration % 100 == 0:
                    logger.info(f"Iteration {iteration}: Loss = {total_loss.item():.4f}")
            
            # Get final confidence
            with torch.no_grad():
                final_predictions = target_model(best_data)
                confidence = F.softmax(final_predictions, dim=1)[0, target_class].item()
            
            # Create result
            result = InversionResult(
                method=InversionMethod.GRADIENT_DESCENT,
                reconstructed_data=best_data,
                target_class=target_class,
                confidence=confidence,
                reconstruction_loss=best_loss,
                num_iterations=config.num_iterations,
                inversion_config=config,
                metadata={
                    "learning_rate": config.learning_rate,
                    "regularization_weight": config.regularization_weight,
                    "final_loss": best_loss
                }
            )
            
            # Store result
            self.inversion_results.append(result)
            
            logger.info(f"Gradient descent inversion completed: confidence={confidence:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Gradient descent inversion failed: {e}")
            raise
    
    async def _generative_model_inversion(self, 
                                        target_model: nn.Module,
                                        target_class: int,
                                        config: InversionConfig) -> InversionResult:
        """Perform generative model-based inversion"""
        try:
            # Create or load generative model
            generator = await self._get_generative_model(config)
            
            # Initialize latent vector
            latent_dim = 100
            latent_vector = torch.randn(1, latent_dim, requires_grad=True)
            
            # Set up optimizer
            optimizer = torch.optim.Adam([latent_vector], lr=config.learning_rate)
            
            # Set models to evaluation mode
            target_model.eval()
            generator.eval()
            
            best_latent = None
            best_loss = float('inf')
            
            for iteration in range(config.num_iterations):
                optimizer.zero_grad()
                
                # Generate data from latent vector
                generated_data = generator(latent_vector)
                
                # Forward pass through target model
                predictions = target_model(generated_data)
                
                # Calculate loss
                target_loss = F.cross_entropy(predictions, torch.tensor([target_class]))
                
                # Add regularization
                regularization = config.regularization_weight * torch.norm(latent_vector)
                total_loss = target_loss + regularization
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Track best result
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_latent = latent_vector.data.clone()
                
                # Log progress
                if iteration % 100 == 0:
                    logger.info(f"Iteration {iteration}: Loss = {total_loss.item():.4f}")
            
            # Generate final reconstruction
            with torch.no_grad():
                reconstructed_data = generator(best_latent)
                final_predictions = target_model(reconstructed_data)
                confidence = F.softmax(final_predictions, dim=1)[0, target_class].item()
            
            # Create result
            result = InversionResult(
                method=InversionMethod.GENERATIVE_MODEL,
                reconstructed_data=reconstructed_data,
                target_class=target_class,
                confidence=confidence,
                reconstruction_loss=best_loss,
                num_iterations=config.num_iterations,
                inversion_config=config,
                metadata={
                    "latent_dim": latent_dim,
                    "learning_rate": config.learning_rate,
                    "regularization_weight": config.regularization_weight
                }
            )
            
            # Store result
            self.inversion_results.append(result)
            
            logger.info(f"Generative model inversion completed: confidence={confidence:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Generative model inversion failed: {e}")
            raise
    
    async def _optimization_based_inversion(self, 
                                          target_model: nn.Module,
                                          target_class: int,
                                          config: InversionConfig) -> InversionResult:
        """Perform optimization-based inversion"""
        try:
            # Initialize data with noise
            reconstructed_data = torch.randn(1, config.num_channels, *config.image_size, requires_grad=True)
            
            # Set up optimizer with different learning rate schedule
            optimizer = torch.optim.Adam([reconstructed_data], lr=config.learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
            
            # Set model to evaluation mode
            target_model.eval()
            
            best_data = None
            best_loss = float('inf')
            
            for iteration in range(config.num_iterations):
                optimizer.zero_grad()
                
                # Forward pass
                predictions = target_model(reconstructed_data)
                
                # Calculate multiple loss components
                target_loss = F.cross_entropy(predictions, torch.tensor([target_class]))
                
                # Add smoothness regularization
                smoothness_loss = self._calculate_smoothness_loss(reconstructed_data)
                
                # Add sparsity regularization
                sparsity_loss = torch.norm(reconstructed_data, p=1)
                
                # Combine losses
                total_loss = (target_loss + 
                            0.1 * smoothness_loss + 
                            0.01 * sparsity_loss)
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Clamp values to valid range
                reconstructed_data.data = torch.clamp(reconstructed_data.data, 0, 1)
                
                # Track best result
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_data = reconstructed_data.data.clone()
                
                # Log progress
                if iteration % 100 == 0:
                    logger.info(f"Iteration {iteration}: Loss = {total_loss.item():.4f}")
            
            # Get final confidence
            with torch.no_grad():
                final_predictions = target_model(best_data)
                confidence = F.softmax(final_predictions, dim=1)[0, target_class].item()
            
            # Create result
            result = InversionResult(
                method=InversionMethod.OPTIMIZATION_BASED,
                reconstructed_data=best_data,
                target_class=target_class,
                confidence=confidence,
                reconstruction_loss=best_loss,
                num_iterations=config.num_iterations,
                inversion_config=config,
                metadata={
                    "learning_rate": config.learning_rate,
                    "smoothness_weight": 0.1,
                    "sparsity_weight": 0.01
                }
            )
            
            # Store result
            self.inversion_results.append(result)
            
            logger.info(f"Optimization-based inversion completed: confidence={confidence:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Optimization-based inversion failed: {e}")
            raise
    
    async def _adversarial_inversion(self, 
                                   target_model: nn.Module,
                                   target_class: int,
                                   config: InversionConfig) -> InversionResult:
        """Perform adversarial inversion"""
        try:
            # Initialize data
            reconstructed_data = torch.randn(1, config.num_channels, *config.image_size, requires_grad=True)
            
            # Set up optimizer
            optimizer = torch.optim.Adam([reconstructed_data], lr=config.learning_rate)
            
            # Set model to evaluation mode
            target_model.eval()
            
            best_data = None
            best_loss = float('inf')
            
            for iteration in range(config.num_iterations):
                optimizer.zero_grad()
                
                # Add adversarial noise
                noise = torch.randn_like(reconstructed_data) * config.noise_level
                adversarial_data = reconstructed_data + noise
                
                # Forward pass
                predictions = target_model(adversarial_data)
                
                # Calculate loss
                target_loss = F.cross_entropy(predictions, torch.tensor([target_class]))
                
                # Add adversarial regularization
                adversarial_loss = torch.norm(noise)
                total_loss = target_loss + 0.1 * adversarial_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Clamp values to valid range
                reconstructed_data.data = torch.clamp(reconstructed_data.data, 0, 1)
                
                # Track best result
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_data = reconstructed_data.data.clone()
                
                # Log progress
                if iteration % 100 == 0:
                    logger.info(f"Iteration {iteration}: Loss = {total_loss.item():.4f}")
            
            # Get final confidence
            with torch.no_grad():
                final_predictions = target_model(best_data)
                confidence = F.softmax(final_predictions, dim=1)[0, target_class].item()
            
            # Create result
            result = InversionResult(
                method=InversionMethod.ADVERSARIAL,
                reconstructed_data=best_data,
                target_class=target_class,
                confidence=confidence,
                reconstruction_loss=best_loss,
                num_iterations=config.num_iterations,
                inversion_config=config,
                metadata={
                    "learning_rate": config.learning_rate,
                    "noise_level": config.noise_level,
                    "adversarial_weight": 0.1
                }
            )
            
            # Store result
            self.inversion_results.append(result)
            
            logger.info(f"Adversarial inversion completed: confidence={confidence:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Adversarial inversion failed: {e}")
            raise
    
    async def _deep_dream_inversion(self, 
                                  target_model: nn.Module,
                                  target_class: int,
                                  config: InversionConfig) -> InversionResult:
        """Perform deep dream-based inversion"""
        try:
            # Initialize data with noise
            reconstructed_data = torch.randn(1, config.num_channels, *config.image_size, requires_grad=True)
            
            # Set up optimizer
            optimizer = torch.optim.Adam([reconstructed_data], lr=config.learning_rate)
            
            # Set model to evaluation mode
            target_model.eval()
            
            best_data = None
            best_loss = float('inf')
            
            for iteration in range(config.num_iterations):
                optimizer.zero_grad()
                
                # Forward pass
                predictions = target_model(reconstructed_data)
                
                # Calculate loss (maximize target class probability)
                target_prob = F.softmax(predictions, dim=1)[0, target_class]
                target_loss = -torch.log(target_prob + 1e-8)  # Negative log likelihood
                
                # Add deep dream regularization (encourage high activations)
                activations = self._get_activations(target_model, reconstructed_data)
                dream_loss = -torch.mean(activations)  # Maximize activations
                
                total_loss = target_loss + 0.1 * dream_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Clamp values to valid range
                reconstructed_data.data = torch.clamp(reconstructed_data.data, 0, 1)
                
                # Track best result
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_data = reconstructed_data.data.clone()
                
                # Log progress
                if iteration % 100 == 0:
                    logger.info(f"Iteration {iteration}: Loss = {total_loss.item():.4f}")
            
            # Get final confidence
            with torch.no_grad():
                final_predictions = target_model(best_data)
                confidence = F.softmax(final_predictions, dim=1)[0, target_class].item()
            
            # Create result
            result = InversionResult(
                method=InversionMethod.DEEP_DREAM,
                reconstructed_data=best_data,
                target_class=target_class,
                confidence=confidence,
                reconstruction_loss=best_loss,
                num_iterations=config.num_iterations,
                inversion_config=config,
                metadata={
                    "learning_rate": config.learning_rate,
                    "dream_weight": 0.1
                }
            )
            
            # Store result
            self.inversion_results.append(result)
            
            logger.info(f"Deep dream inversion completed: confidence={confidence:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Deep dream inversion failed: {e}")
            raise
    
    async def _get_generative_model(self, config: InversionConfig) -> nn.Module:
        """Get or create generative model"""
        try:
            model_key = f"generator_{config.num_channels}_{config.image_size[0]}"
            
            if model_key not in self.generative_models:
                # Create simple generator
                generator = self._create_generator(config)
                self.generative_models[model_key] = generator
            
            return self.generative_models[model_key]
            
        except Exception as e:
            logger.error(f"Generative model creation failed: {e}")
            raise
    
    def _create_generator(self, config: InversionConfig) -> nn.Module:
        """Create generator network"""
        latent_dim = 100
        
        return nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, config.num_channels * config.image_size[0] * config.image_size[1]),
            nn.Tanh(),
            nn.Unflatten(1, (config.num_channels, config.image_size[0], config.image_size[1]))
        )
    
    def _calculate_smoothness_loss(self, data: torch.Tensor) -> torch.Tensor:
        """Calculate smoothness regularization loss"""
        try:
            # Calculate gradients in x and y directions
            grad_x = torch.abs(data[:, :, :, 1:] - data[:, :, :, :-1])
            grad_y = torch.abs(data[:, :, 1:, :] - data[:, :, :-1, :])
            
            # Sum over all dimensions
            smoothness_loss = torch.mean(grad_x) + torch.mean(grad_y)
            
            return smoothness_loss
            
        except Exception as e:
            logger.error(f"Smoothness loss calculation failed: {e}")
            return torch.tensor(0.0)
    
    def _get_activations(self, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """Get activations from intermediate layers"""
        try:
            activations = []
            
            def hook_fn(module, input, output):
                activations.append(output)
            
            # Register hooks on convolutional layers
            hooks = []
            for module in model.modules():
                if isinstance(module, (nn.Conv2d, nn.ReLU)):
                    hook = module.register_forward_hook(hook_fn)
                    hooks.append(hook)
            
            # Forward pass
            with torch.no_grad():
                _ = model(data)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Combine activations
            if activations:
                combined_activations = torch.cat([a.view(a.size(0), -1) for a in activations], dim=1)
                return combined_activations
            else:
                return torch.tensor(0.0)
                
        except Exception as e:
            logger.error(f"Activation extraction failed: {e}")
            return torch.tensor(0.0)
    
    async def batch_inversion(self, 
                            target_model: nn.Module,
                            target_classes: List[int],
                            inversion_config: InversionConfig) -> List[InversionResult]:
        """Perform inversion for multiple classes"""
        try:
            logger.info(f"Batch inversion for {len(target_classes)} classes")
            
            results = []
            
            for target_class in target_classes:
                try:
                    # Create config for this class
                    class_config = InversionConfig(
                        method=inversion_config.method,
                        target_class=target_class,
                        num_iterations=inversion_config.num_iterations,
                        learning_rate=inversion_config.learning_rate,
                        regularization_weight=inversion_config.regularization_weight,
                        noise_level=inversion_config.noise_level,
                        image_size=inversion_config.image_size,
                        num_channels=inversion_config.num_channels
                    )
                    
                    result = await self.perform_inversion(target_model, target_class, class_config)
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Inversion failed for class {target_class}: {e}")
                    continue
            
            logger.info(f"Batch inversion completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Batch inversion failed: {e}")
            return []
    
    async def get_inversion_summary(self) -> Dict[str, Any]:
        """Get summary of all inversion results"""
        try:
            if not self.inversion_results:
                return {"message": "No inversion results available"}
            
            # Calculate statistics
            total_inversions = len(self.inversion_results)
            successful_inversions = sum(1 for r in self.inversion_results if r.confidence > 0.5)
            success_rate = successful_inversions / total_inversions
            
            # Calculate method-specific statistics
            method_stats = defaultdict(lambda: {"count": 0, "confidences": [], "losses": []})
            
            for result in self.inversion_results:
                method = result.method.value
                method_stats[method]["count"] += 1
                method_stats[method]["confidences"].append(result.confidence)
                method_stats[method]["losses"].append(result.reconstruction_loss)
            
            # Calculate averages for each method
            for method, stats in method_stats.items():
                stats["mean_confidence"] = np.mean(stats["confidences"])
                stats["mean_loss"] = np.mean(stats["losses"])
                stats["std_confidence"] = np.std(stats["confidences"])
                stats["std_loss"] = np.std(stats["losses"])
            
            return {
                "total_inversions": total_inversions,
                "successful_inversions": successful_inversions,
                "overall_success_rate": success_rate,
                "method_statistics": dict(method_stats),
                "summary_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Inversion summary generation failed: {e}")
            return {}
    
    async def export_inversion_data(self, format: str = "json") -> str:
        """Export inversion data"""
        try:
            if format.lower() == "json":
                data = {
                    "inversion_results": [r.__dict__ for r in self.inversion_results],
                    "summary": await self.get_inversion_summary(),
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Inversion data export failed: {e}")
            return ""
