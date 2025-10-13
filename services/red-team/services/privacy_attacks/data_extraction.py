"""
Data Extraction Attacks
Implements data extraction attacks to recover training data
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)


class ExtractionMethod(Enum):
    """Types of data extraction methods"""
    GRADIENT_BASED = "gradient_based"
    ACTIVATION_BASED = "activation_based"
    EMBEDDING_BASED = "embedding_based"
    CLUSTERING_BASED = "clustering_based"
    OPTIMIZATION_BASED = "optimization_based"
    ADVERSARIAL = "adversarial"


@dataclass
class ExtractionConfig:
    """Configuration for data extraction attack"""
    method: ExtractionMethod
    target_layer: Optional[str] = None
    num_samples: int = 100
    num_clusters: int = 10
    learning_rate: float = 0.01
    num_iterations: int = 1000
    regularization_weight: float = 0.01
    noise_level: float = 0.1
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExtractionResult:
    """Result of data extraction attack"""
    method: ExtractionMethod
    extracted_data: torch.Tensor
    extracted_labels: torch.Tensor
    reconstruction_quality: float
    extraction_confidence: float
    num_extracted: int
    extraction_config: ExtractionConfig
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DataExtractionAttacker:
    """
    Data Extraction Attacker
    Implements data extraction attacks to recover training data
    """
    
    def __init__(self):
        """Initialize data extraction attacker"""
        self.extraction_results: List[ExtractionResult] = []
        self.extraction_models: Dict[str, nn.Module] = {}
        
        logger.info("✅ Initialized Data Extraction Attacker")
    
    async def perform_extraction(self, 
                               target_model: nn.Module,
                               extraction_config: ExtractionConfig) -> ExtractionResult:
        """
        Perform data extraction attack
        """
        try:
            logger.info(f"Performing {extraction_config.method.value} data extraction")
            
            if extraction_config.method == ExtractionMethod.GRADIENT_BASED:
                return await self._gradient_based_extraction(target_model, extraction_config)
            elif extraction_config.method == ExtractionMethod.ACTIVATION_BASED:
                return await self._activation_based_extraction(target_model, extraction_config)
            elif extraction_config.method == ExtractionMethod.EMBEDDING_BASED:
                return await self._embedding_based_extraction(target_model, extraction_config)
            elif extraction_config.method == ExtractionMethod.CLUSTERING_BASED:
                return await self._clustering_based_extraction(target_model, extraction_config)
            elif extraction_config.method == ExtractionMethod.OPTIMIZATION_BASED:
                return await self._optimization_based_extraction(target_model, extraction_config)
            elif extraction_config.method == ExtractionMethod.ADVERSARIAL:
                return await self._adversarial_extraction(target_model, extraction_config)
            else:
                raise ValueError(f"Unknown extraction method: {extraction_config.method}")
                
        except Exception as e:
            logger.error(f"Data extraction attack failed: {e}")
            raise
    
    async def _gradient_based_extraction(self, 
                                       target_model: nn.Module,
                                       config: ExtractionConfig) -> ExtractionResult:
        """Perform gradient-based data extraction"""
        try:
            # Initialize random data
            extracted_data = torch.randn(config.num_samples, 3, 224, 224, requires_grad=True)
            extracted_labels = torch.randint(0, 10, (config.num_samples,))
            
            # Set up optimizer
            optimizer = torch.optim.Adam([extracted_data], lr=config.learning_rate)
            
            # Set model to evaluation mode
            target_model.eval()
            
            best_data = None
            best_loss = float('inf')
            
            for iteration in range(config.num_iterations):
                optimizer.zero_grad()
                
                # Forward pass
                predictions = target_model(extracted_data)
                
                # Calculate loss
                target_loss = F.cross_entropy(predictions, extracted_labels)
                
                # Add regularization
                regularization = config.regularization_weight * torch.norm(extracted_data)
                total_loss = target_loss + regularization
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Clamp values to valid range
                extracted_data.data = torch.clamp(extracted_data.data, 0, 1)
                
                # Track best result
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_data = extracted_data.data.clone()
                
                # Log progress
                if iteration % 100 == 0:
                    logger.info(f"Iteration {iteration}: Loss = {total_loss.item():.4f}")
            
            # Calculate reconstruction quality
            with torch.no_grad():
                final_predictions = target_model(best_data)
                reconstruction_quality = self._calculate_reconstruction_quality(final_predictions, extracted_labels)
                extraction_confidence = self._calculate_extraction_confidence(best_data)
            
            # Create result
            result = ExtractionResult(
                method=ExtractionMethod.GRADIENT_BASED,
                extracted_data=best_data,
                extracted_labels=extracted_labels,
                reconstruction_quality=reconstruction_quality,
                extraction_confidence=extraction_confidence,
                num_extracted=config.num_samples,
                extraction_config=config,
                metadata={
                    "learning_rate": config.learning_rate,
                    "regularization_weight": config.regularization_weight,
                    "final_loss": best_loss
                }
            )
            
            # Store result
            self.extraction_results.append(result)
            
            logger.info(f"Gradient-based extraction completed: quality={reconstruction_quality:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Gradient-based extraction failed: {e}")
            raise
    
    async def _activation_based_extraction(self, 
                                         target_model: nn.Module,
                                         config: ExtractionConfig) -> ExtractionResult:
        """Perform activation-based data extraction"""
        try:
            # Get activations from target layer
            activations = await self._extract_activations(target_model, config.target_layer)
            
            if activations is None:
                raise ValueError("Could not extract activations from target layer")
            
            # Use activations to reconstruct data
            reconstructed_data = await self._reconstruct_from_activations(activations, config)
            
            # Generate labels based on activations
            extracted_labels = await self._generate_labels_from_activations(activations)
            
            # Calculate reconstruction quality
            reconstruction_quality = self._calculate_reconstruction_quality_from_activations(activations, reconstructed_data)
            extraction_confidence = self._calculate_extraction_confidence(reconstructed_data)
            
            # Create result
            result = ExtractionResult(
                method=ExtractionMethod.ACTIVATION_BASED,
                extracted_data=reconstructed_data,
                extracted_labels=extracted_labels,
                reconstruction_quality=reconstruction_quality,
                extraction_confidence=extraction_confidence,
                num_extracted=len(reconstructed_data),
                extraction_config=config,
                metadata={
                    "target_layer": config.target_layer,
                    "activation_shape": activations.shape
                }
            )
            
            # Store result
            self.extraction_results.append(result)
            
            logger.info(f"Activation-based extraction completed: quality={reconstruction_quality:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Activation-based extraction failed: {e}")
            raise
    
    async def _embedding_based_extraction(self, 
                                        target_model: nn.Module,
                                        config: ExtractionConfig) -> ExtractionResult:
        """Perform embedding-based data extraction"""
        try:
            # Extract embeddings from model
            embeddings = await self._extract_embeddings(target_model)
            
            if embeddings is None:
                raise ValueError("Could not extract embeddings from model")
            
            # Use embeddings to reconstruct data
            reconstructed_data = await self._reconstruct_from_embeddings(embeddings, config)
            
            # Generate labels based on embeddings
            extracted_labels = await self._generate_labels_from_embeddings(embeddings)
            
            # Calculate reconstruction quality
            reconstruction_quality = self._calculate_reconstruction_quality_from_embeddings(embeddings, reconstructed_data)
            extraction_confidence = self._calculate_extraction_confidence(reconstructed_data)
            
            # Create result
            result = ExtractionResult(
                method=ExtractionMethod.EMBEDDING_BASED,
                extracted_data=reconstructed_data,
                extracted_labels=extracted_labels,
                reconstruction_quality=reconstruction_quality,
                extraction_confidence=extraction_confidence,
                num_extracted=len(reconstructed_data),
                extraction_config=config,
                metadata={
                    "embedding_dim": embeddings.shape[1],
                    "num_embeddings": len(embeddings)
                }
            )
            
            # Store result
            self.extraction_results.append(result)
            
            logger.info(f"Embedding-based extraction completed: quality={reconstruction_quality:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Embedding-based extraction failed: {e}")
            raise
    
    async def _clustering_based_extraction(self, 
                                         target_model: nn.Module,
                                         config: ExtractionConfig) -> ExtractionResult:
        """Perform clustering-based data extraction"""
        try:
            # Extract features from model
            features = await self._extract_features(target_model)
            
            if features is None:
                raise ValueError("Could not extract features from model")
            
            # Perform clustering
            kmeans = KMeans(n_clusters=config.num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
            
            # Reconstruct data from cluster centers
            reconstructed_data = await self._reconstruct_from_clusters(kmeans.cluster_centers_, config)
            
            # Generate labels based on clusters
            extracted_labels = torch.tensor(cluster_labels)
            
            # Calculate reconstruction quality
            reconstruction_quality = self._calculate_reconstruction_quality_from_clusters(features, cluster_labels)
            extraction_confidence = self._calculate_extraction_confidence(reconstructed_data)
            
            # Create result
            result = ExtractionResult(
                method=ExtractionMethod.CLUSTERING_BASED,
                extracted_data=reconstructed_data,
                extracted_labels=extracted_labels,
                reconstruction_quality=reconstruction_quality,
                extraction_confidence=extraction_confidence,
                num_extracted=len(reconstructed_data),
                extraction_config=config,
                metadata={
                    "num_clusters": config.num_clusters,
                    "feature_dim": features.shape[1]
                }
            )
            
            # Store result
            self.extraction_results.append(result)
            
            logger.info(f"Clustering-based extraction completed: quality={reconstruction_quality:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Clustering-based extraction failed: {e}")
            raise
    
    async def _optimization_based_extraction(self, 
                                           target_model: nn.Module,
                                           config: ExtractionConfig) -> ExtractionResult:
        """Perform optimization-based data extraction"""
        try:
            # Initialize random data
            extracted_data = torch.randn(config.num_samples, 3, 224, 224, requires_grad=True)
            extracted_labels = torch.randint(0, 10, (config.num_samples,))
            
            # Set up optimizer with different learning rate schedule
            optimizer = torch.optim.Adam([extracted_data], lr=config.learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
            
            # Set model to evaluation mode
            target_model.eval()
            
            best_data = None
            best_loss = float('inf')
            
            for iteration in range(config.num_iterations):
                optimizer.zero_grad()
                
                # Forward pass
                predictions = target_model(extracted_data)
                
                # Calculate multiple loss components
                target_loss = F.cross_entropy(predictions, extracted_labels)
                
                # Add smoothness regularization
                smoothness_loss = self._calculate_smoothness_loss(extracted_data)
                
                # Add sparsity regularization
                sparsity_loss = torch.norm(extracted_data, p=1)
                
                # Combine losses
                total_loss = (target_loss + 
                            0.1 * smoothness_loss + 
                            0.01 * sparsity_loss)
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Clamp values to valid range
                extracted_data.data = torch.clamp(extracted_data.data, 0, 1)
                
                # Track best result
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_data = extracted_data.data.clone()
                
                # Log progress
                if iteration % 100 == 0:
                    logger.info(f"Iteration {iteration}: Loss = {total_loss.item():.4f}")
            
            # Calculate reconstruction quality
            with torch.no_grad():
                final_predictions = target_model(best_data)
                reconstruction_quality = self._calculate_reconstruction_quality(final_predictions, extracted_labels)
                extraction_confidence = self._calculate_extraction_confidence(best_data)
            
            # Create result
            result = ExtractionResult(
                method=ExtractionMethod.OPTIMIZATION_BASED,
                extracted_data=best_data,
                extracted_labels=extracted_labels,
                reconstruction_quality=reconstruction_quality,
                extraction_confidence=extraction_confidence,
                num_extracted=config.num_samples,
                extraction_config=config,
                metadata={
                    "learning_rate": config.learning_rate,
                    "smoothness_weight": 0.1,
                    "sparsity_weight": 0.01
                }
            )
            
            # Store result
            self.extraction_results.append(result)
            
            logger.info(f"Optimization-based extraction completed: quality={reconstruction_quality:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Optimization-based extraction failed: {e}")
            raise
    
    async def _adversarial_extraction(self, 
                                    target_model: nn.Module,
                                    config: ExtractionConfig) -> ExtractionResult:
        """Perform adversarial data extraction"""
        try:
            # Initialize random data
            extracted_data = torch.randn(config.num_samples, 3, 224, 224, requires_grad=True)
            extracted_labels = torch.randint(0, 10, (config.num_samples,))
            
            # Set up optimizer
            optimizer = torch.optim.Adam([extracted_data], lr=config.learning_rate)
            
            # Set model to evaluation mode
            target_model.eval()
            
            best_data = None
            best_loss = float('inf')
            
            for iteration in range(config.num_iterations):
                optimizer.zero_grad()
                
                # Add adversarial noise
                noise = torch.randn_like(extracted_data) * config.noise_level
                adversarial_data = extracted_data + noise
                
                # Forward pass
                predictions = target_model(adversarial_data)
                
                # Calculate loss
                target_loss = F.cross_entropy(predictions, extracted_labels)
                
                # Add adversarial regularization
                adversarial_loss = torch.norm(noise)
                total_loss = target_loss + 0.1 * adversarial_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Clamp values to valid range
                extracted_data.data = torch.clamp(extracted_data.data, 0, 1)
                
                # Track best result
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_data = extracted_data.data.clone()
                
                # Log progress
                if iteration % 100 == 0:
                    logger.info(f"Iteration {iteration}: Loss = {total_loss.item():.4f}")
            
            # Calculate reconstruction quality
            with torch.no_grad():
                final_predictions = target_model(best_data)
                reconstruction_quality = self._calculate_reconstruction_quality(final_predictions, extracted_labels)
                extraction_confidence = self._calculate_extraction_confidence(best_data)
            
            # Create result
            result = ExtractionResult(
                method=ExtractionMethod.ADVERSARIAL,
                extracted_data=best_data,
                extracted_labels=extracted_labels,
                reconstruction_quality=reconstruction_quality,
                extraction_confidence=extraction_confidence,
                num_extracted=config.num_samples,
                extraction_config=config,
                metadata={
                    "learning_rate": config.learning_rate,
                    "noise_level": config.noise_level,
                    "adversarial_weight": 0.1
                }
            )
            
            # Store result
            self.extraction_results.append(result)
            
            logger.info(f"Adversarial extraction completed: quality={reconstruction_quality:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Adversarial extraction failed: {e}")
            raise
    
    async def _extract_activations(self, model: nn.Module, target_layer: str) -> Optional[torch.Tensor]:
        """Extract activations from target layer"""
        try:
            activations = []
            
            def hook_fn(module, input, output):
                activations.append(output.detach())
            
            # Find target layer
            target_module = None
            for name, module in model.named_modules():
                if name == target_layer:
                    target_module = module
                    break
            
            if target_module is None:
                logger.warning(f"Target layer {target_layer} not found")
                return None
            
            # Register hook
            hook = target_module.register_forward_hook(hook_fn)
            
            # Forward pass with dummy data
            dummy_data = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                _ = model(dummy_data)
            
            # Remove hook
            hook.remove()
            
            if activations:
                return activations[0]
            else:
                return None
                
        except Exception as e:
            logger.error(f"Activation extraction failed: {e}")
            return None
    
    async def _extract_embeddings(self, model: nn.Module) -> Optional[torch.Tensor]:
        """Extract embeddings from model"""
        try:
            # Get embeddings from last layer before classification
            embeddings = []
            
            def hook_fn(module, input, output):
                embeddings.append(output.detach())
            
            # Find last layer before classification
            last_layer = None
            for module in model.modules():
                if isinstance(module, nn.Linear) and module.out_features < 1000:  # Assuming classification layer
                    last_layer = module
                    break
            
            if last_layer is None:
                logger.warning("Could not find embedding layer")
                return None
            
            # Register hook
            hook = last_layer.register_forward_hook(hook_fn)
            
            # Forward pass with dummy data
            dummy_data = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                _ = model(dummy_data)
            
            # Remove hook
            hook.remove()
            
            if embeddings:
                return embeddings[0]
            else:
                return None
                
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return None
    
    async def _extract_features(self, model: nn.Module) -> Optional[np.ndarray]:
        """Extract features from model"""
        try:
            # Get features from intermediate layer
            features = []
            
            def hook_fn(module, input, output):
                features.append(output.detach().view(output.size(0), -1))
            
            # Find a good feature layer (e.g., before final classification)
            feature_layer = None
            for module in model.modules():
                if isinstance(module, nn.AdaptiveAvgPool2d):
                    feature_layer = module
                    break
            
            if feature_layer is None:
                logger.warning("Could not find feature layer")
                return None
            
            # Register hook
            hook = feature_layer.register_forward_hook(hook_fn)
            
            # Forward pass with dummy data
            dummy_data = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                _ = model(dummy_data)
            
            # Remove hook
            hook.remove()
            
            if features:
                return features[0].numpy()
            else:
                return None
                
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    async def _reconstruct_from_activations(self, activations: torch.Tensor, config: ExtractionConfig) -> torch.Tensor:
        """Reconstruct data from activations"""
        try:
            # Simple reconstruction - in practice, you'd use a more sophisticated method
            batch_size = activations.shape[0]
            reconstructed = torch.randn(batch_size, 3, 224, 224)
            
            # Apply some transformation based on activations
            activation_mean = torch.mean(activations, dim=1, keepdim=True)
            reconstructed = reconstructed * activation_mean.unsqueeze(-1).unsqueeze(-1)
            
            return torch.clamp(reconstructed, 0, 1)
            
        except Exception as e:
            logger.error(f"Reconstruction from activations failed: {e}")
            return torch.randn(1, 3, 224, 224)
    
    async def _reconstruct_from_embeddings(self, embeddings: torch.Tensor, config: ExtractionConfig) -> torch.Tensor:
        """Reconstruct data from embeddings"""
        try:
            # Simple reconstruction - in practice, you'd use a more sophisticated method
            batch_size = embeddings.shape[0]
            reconstructed = torch.randn(batch_size, 3, 224, 224)
            
            # Apply some transformation based on embeddings
            embedding_norm = torch.norm(embeddings, dim=1, keepdim=True)
            reconstructed = reconstructed * embedding_norm.unsqueeze(-1).unsqueeze(-1)
            
            return torch.clamp(reconstructed, 0, 1)
            
        except Exception as e:
            logger.error(f"Reconstruction from embeddings failed: {e}")
            return torch.randn(1, 3, 224, 224)
    
    async def _reconstruct_from_clusters(self, cluster_centers: np.ndarray, config: ExtractionConfig) -> torch.Tensor:
        """Reconstruct data from cluster centers"""
        try:
            # Simple reconstruction - in practice, you'd use a more sophisticated method
            num_clusters = len(cluster_centers)
            reconstructed = torch.randn(num_clusters, 3, 224, 224)
            
            # Apply some transformation based on cluster centers
            cluster_norms = np.linalg.norm(cluster_centers, axis=1)
            for i, norm in enumerate(cluster_norms):
                reconstructed[i] = reconstructed[i] * norm
            
            return torch.clamp(reconstructed, 0, 1)
            
        except Exception as e:
            logger.error(f"Reconstruction from clusters failed: {e}")
            return torch.randn(1, 3, 224, 224)
    
    async def _generate_labels_from_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """Generate labels from activations"""
        try:
            # Simple label generation - in practice, you'd use a more sophisticated method
            batch_size = activations.shape[0]
            labels = torch.randint(0, 10, (batch_size,))
            
            return labels
            
        except Exception as e:
            logger.error(f"Label generation from activations failed: {e}")
            return torch.randint(0, 10, (1,))
    
    async def _generate_labels_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Generate labels from embeddings"""
        try:
            # Simple label generation - in practice, you'd use a more sophisticated method
            batch_size = embeddings.shape[0]
            labels = torch.randint(0, 10, (batch_size,))
            
            return labels
            
        except Exception as e:
            logger.error(f"Label generation from embeddings failed: {e}")
            return torch.randint(0, 10, (1,))
    
    def _calculate_reconstruction_quality(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """Calculate reconstruction quality"""
        try:
            # Calculate accuracy
            predicted_labels = torch.argmax(predictions, dim=1)
            accuracy = (predicted_labels == labels).float().mean().item()
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Reconstruction quality calculation failed: {e}")
            return 0.0
    
    def _calculate_reconstruction_quality_from_activations(self, activations: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Calculate reconstruction quality from activations"""
        try:
            # Simple quality measure - in practice, you'd use a more sophisticated method
            activation_std = torch.std(activations).item()
            reconstruction_std = torch.std(reconstructed).item()
            
            quality = 1.0 - abs(activation_std - reconstruction_std) / (activation_std + reconstruction_std + 1e-8)
            
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            logger.error(f"Reconstruction quality calculation from activations failed: {e}")
            return 0.0
    
    def _calculate_reconstruction_quality_from_embeddings(self, embeddings: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Calculate reconstruction quality from embeddings"""
        try:
            # Simple quality measure - in practice, you'd use a more sophisticated method
            embedding_std = torch.std(embeddings).item()
            reconstruction_std = torch.std(reconstructed).item()
            
            quality = 1.0 - abs(embedding_std - reconstruction_std) / (embedding_std + reconstruction_std + 1e-8)
            
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            logger.error(f"Reconstruction quality calculation from embeddings failed: {e}")
            return 0.0
    
    def _calculate_reconstruction_quality_from_clusters(self, features: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Calculate reconstruction quality from clusters"""
        try:
            # Calculate silhouette score
            from sklearn.metrics import silhouette_score
            
            if len(np.unique(cluster_labels)) > 1:
                quality = silhouette_score(features, cluster_labels)
            else:
                quality = 0.0
            
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            logger.error(f"Reconstruction quality calculation from clusters failed: {e}")
            return 0.0
    
    def _calculate_extraction_confidence(self, data: torch.Tensor) -> float:
        """Calculate extraction confidence"""
        try:
            # Simple confidence measure - in practice, you'd use a more sophisticated method
            data_std = torch.std(data).item()
            data_mean = torch.mean(data).item()
            
            # Higher confidence for data that looks more realistic
            confidence = min(1.0, data_std / (data_mean + 1e-8))
            
            return confidence
            
        except Exception as e:
            logger.error(f"Extraction confidence calculation failed: {e}")
            return 0.0
    
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
    
    async def get_extraction_summary(self) -> Dict[str, Any]:
        """Get summary of all extraction results"""
        try:
            if not self.extraction_results:
                return {"message": "No extraction results available"}
            
            # Calculate statistics
            total_extractions = len(self.extraction_results)
            successful_extractions = sum(1 for r in self.extraction_results if r.reconstruction_quality > 0.5)
            success_rate = successful_extractions / total_extractions
            
            # Calculate method-specific statistics
            method_stats = defaultdict(lambda: {"count": 0, "qualities": [], "confidences": []})
            
            for result in self.extraction_results:
                method = result.method.value
                method_stats[method]["count"] += 1
                method_stats[method]["qualities"].append(result.reconstruction_quality)
                method_stats[method]["confidences"].append(result.extraction_confidence)
            
            # Calculate averages for each method
            for method, stats in method_stats.items():
                stats["mean_quality"] = np.mean(stats["qualities"])
                stats["mean_confidence"] = np.mean(stats["confidences"])
                stats["std_quality"] = np.std(stats["qualities"])
                stats["std_confidence"] = np.std(stats["confidences"])
            
            return {
                "total_extractions": total_extractions,
                "successful_extractions": successful_extractions,
                "overall_success_rate": success_rate,
                "method_statistics": dict(method_stats),
                "summary_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Extraction summary generation failed: {e}")
            return {}
    
    async def export_extraction_data(self, format: str = "json") -> str:
        """Export extraction data"""
        try:
            if format.lower() == "json":
                data = {
                    "extraction_results": [r.__dict__ for r in self.extraction_results],
                    "summary": await self.get_extraction_summary(),
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Extraction data export failed: {e}")
            return ""
