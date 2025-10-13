"""
Embedding Perturbation Utilities
Advanced techniques for manipulating model embeddings
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class EmbeddingPerturbation:
    """
    Advanced embedding perturbation techniques for adversarial attacks
    """
    
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize embedding perturbation utilities
        
        Args:
            model: PyTorch model
            tokenizer: Tokenizer for the model
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Get embedding layer
        if hasattr(model, 'get_input_embeddings'):
            self.embedding_layer = model.get_input_embeddings()
        else:
            for name, module in model.named_modules():
                if 'embedding' in name.lower():
                    self.embedding_layer = module
                    break
            else:
                raise ValueError("Could not find embedding layer in model")
        
        self.vocab_size = self.embedding_layer.num_embeddings
        self.embedding_dim = self.embedding_layer.embedding_dim
        
        logger.info(f"✅ Initialized EmbeddingPerturbation with vocab_size={self.vocab_size}, embedding_dim={self.embedding_dim}")
    
    def random_perturbation(self, embeddings: torch.Tensor, 
                          noise_scale: float = 0.1, 
                          noise_type: str = 'gaussian') -> torch.Tensor:
        """
        Add random noise to embeddings
        
        Args:
            embeddings: Input embeddings
            noise_scale: Scale of noise to add
            noise_type: Type of noise ('gaussian', 'uniform', 'laplace')
            
        Returns:
            Perturbed embeddings
        """
        if noise_type == 'gaussian':
            noise = torch.randn_like(embeddings) * noise_scale
        elif noise_type == 'uniform':
            noise = (torch.rand_like(embeddings) - 0.5) * 2 * noise_scale
        elif noise_type == 'laplace':
            noise = torch.distributions.Laplace(0, noise_scale).sample(embeddings.shape).to(embeddings.device)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return embeddings + noise
    
    def semantic_perturbation(self, embeddings: torch.Tensor, 
                            target_embeddings: torch.Tensor,
                            alpha: float = 0.5) -> torch.Tensor:
        """
        Perturb embeddings towards semantic target
        
        Args:
            embeddings: Input embeddings
            target_embeddings: Target semantic embeddings
            alpha: Interpolation factor (0=original, 1=target)
            
        Returns:
            Semantically perturbed embeddings
        """
        # Ensure same shape
        if embeddings.shape != target_embeddings.shape:
            raise ValueError("Embedding shapes must match")
        
        # Linear interpolation
        perturbed = (1 - alpha) * embeddings + alpha * target_embeddings
        
        return perturbed
    
    def adversarial_perturbation(self, embeddings: torch.Tensor,
                               gradients: torch.Tensor,
                               epsilon: float = 0.1,
                               norm: str = 'l2') -> torch.Tensor:
        """
        Apply adversarial perturbation based on gradients
        
        Args:
            embeddings: Input embeddings
            gradients: Gradient information
            epsilon: Perturbation magnitude
            norm: Norm constraint ('l2', 'linf')
            
        Returns:
            Adversarially perturbed embeddings
        """
        if norm == 'l2':
            # L2 normalized gradient
            grad_norm = gradients.norm(p=2, dim=-1, keepdim=True)
            perturbation = epsilon * gradients / (grad_norm + 1e-8)
        elif norm == 'linf':
            # L-infinity gradient (sign)
            perturbation = epsilon * gradients.sign()
        else:
            raise ValueError(f"Unknown norm: {norm}")
        
        return embeddings + perturbation
    
    def embedding_interpolation(self, start_embeddings: torch.Tensor,
                              end_embeddings: torch.Tensor,
                              num_steps: int = 10) -> List[torch.Tensor]:
        """
        Create smooth interpolation between embeddings
        
        Args:
            start_embeddings: Starting point embeddings
            end_embeddings: Ending point embeddings
            num_steps: Number of interpolation steps
            
        Returns:
            List of interpolated embeddings
        """
        interpolations = []
        
        for i in range(num_steps + 1):
            alpha = i / num_steps
            interpolated = (1 - alpha) * start_embeddings + alpha * end_embeddings
            interpolations.append(interpolated)
        
        return interpolations
    
    def embedding_manifold_projection(self, embeddings: torch.Tensor,
                                    manifold_embeddings: torch.Tensor,
                                    k: int = 5) -> torch.Tensor:
        """
        Project embeddings onto learned manifold
        
        Args:
            embeddings: Input embeddings to project
            manifold_embeddings: Reference manifold embeddings
            k: Number of nearest neighbors for projection
            
        Returns:
            Manifold-projected embeddings
        """
        # Calculate distances to manifold points
        distances = torch.cdist(embeddings, manifold_embeddings, p=2)
        
        # Find k nearest neighbors
        _, nearest_indices = torch.topk(distances, k, dim=-1, largest=False)
        
        # Weighted average of nearest neighbors
        weights = F.softmax(-distances.gather(-1, nearest_indices), dim=-1)
        projected = torch.sum(weights.unsqueeze(-1) * manifold_embeddings[nearest_indices], dim=-2)
        
        return projected
    
    def embedding_rotation(self, embeddings: torch.Tensor,
                          rotation_angle: float = 0.1,
                          axis: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply rotation to embeddings
        
        Args:
            embeddings: Input embeddings
            rotation_angle: Angle of rotation in radians
            axis: Rotation axis (if None, random axis)
            
        Returns:
            Rotated embeddings
        """
        if axis is None:
            # Random rotation axis
            axis = torch.randn(embeddings.shape[-1]).to(embeddings.device)
            axis = axis / axis.norm()
        
        # Rodrigues' rotation formula
        cos_angle = torch.cos(torch.tensor(rotation_angle))
        sin_angle = torch.sin(torch.tensor(rotation_angle))
        
        # Cross product matrix
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ]).to(embeddings.device)
        
        # Rotation matrix
        R = torch.eye(3).to(embeddings.device) + sin_angle * K + (1 - cos_angle) * torch.mm(K, K)
        
        # Apply rotation (assuming 3D embeddings, extend for higher dimensions)
        if embeddings.shape[-1] == 3:
            rotated = torch.mm(embeddings.view(-1, 3), R.T).view(embeddings.shape)
        else:
            # For higher dimensions, apply rotation in 3D subspace
            rotated = embeddings.clone()
            rotated[..., :3] = torch.mm(embeddings[..., :3].view(-1, 3), R.T).view(embeddings[..., :3].shape)
        
        return rotated
    
    def embedding_scaling(self, embeddings: torch.Tensor,
                         scale_factor: float = 1.1,
                         center: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply scaling transformation to embeddings
        
        Args:
            embeddings: Input embeddings
            scale_factor: Scaling factor
            center: Center of scaling (if None, use mean)
            
        Returns:
            Scaled embeddings
        """
        if center is None:
            center = embeddings.mean(dim=0, keepdim=True)
        
        # Scale around center
        scaled = center + scale_factor * (embeddings - center)
        
        return scaled
    
    def embedding_shear(self, embeddings: torch.Tensor,
                       shear_factor: float = 0.1,
                       shear_axis: int = 0) -> torch.Tensor:
        """
        Apply shear transformation to embeddings
        
        Args:
            embeddings: Input embeddings
            shear_factor: Shear strength
            shear_axis: Axis to shear along
            
        Returns:
            Sheared embeddings
        """
        # Create shear matrix
        shear_matrix = torch.eye(embeddings.shape[-1]).to(embeddings.device)
        shear_matrix[shear_axis, :] += shear_factor
        
        # Apply shear
        sheared = torch.mm(embeddings.view(-1, embeddings.shape[-1]), shear_matrix.T)
        sheared = sheared.view(embeddings.shape)
        
        return sheared
    
    def embedding_noise_injection(self, embeddings: torch.Tensor,
                                noise_type: str = 'adaptive',
                                noise_scale: float = 0.1) -> torch.Tensor:
        """
        Inject adaptive noise based on embedding characteristics
        
        Args:
            embeddings: Input embeddings
            noise_type: Type of noise injection
            noise_scale: Base noise scale
            
        Returns:
            Noise-injected embeddings
        """
        if noise_type == 'adaptive':
            # Adaptive noise based on embedding variance
            embedding_std = embeddings.std(dim=0, keepdim=True)
            noise_scale = noise_scale * embedding_std
            noise = torch.randn_like(embeddings) * noise_scale
        elif noise_type == 'gradient_guided':
            # Noise guided by gradient information (placeholder)
            noise = torch.randn_like(embeddings) * noise_scale
        else:
            # Standard Gaussian noise
            noise = torch.randn_like(embeddings) * noise_scale
        
        return embeddings + noise
    
    def embedding_consistency_check(self, original_embeddings: torch.Tensor,
                                  perturbed_embeddings: torch.Tensor,
                                  threshold: float = 0.1) -> Dict:
        """
        Check consistency between original and perturbed embeddings
        
        Args:
            original_embeddings: Original embeddings
            perturbed_embeddings: Perturbed embeddings
            threshold: Consistency threshold
            
        Returns:
            Consistency metrics
        """
        # Cosine similarity
        cosine_sim = F.cosine_similarity(original_embeddings, perturbed_embeddings, dim=-1)
        avg_cosine_sim = cosine_sim.mean().item()
        
        # L2 distance
        l2_distance = F.mse_loss(original_embeddings, perturbed_embeddings).item()
        
        # L-infinity distance
        linf_distance = (original_embeddings - perturbed_embeddings).abs().max().item()
        
        # Consistency check
        is_consistent = avg_cosine_sim > (1 - threshold)
        
        return {
            "cosine_similarity": avg_cosine_sim,
            "l2_distance": l2_distance,
            "linf_distance": linf_distance,
            "is_consistent": is_consistent,
            "threshold": threshold
        }
    
    def embedding_robustness_test(self, embeddings: torch.Tensor,
                                perturbation_methods: List[str],
                                noise_scales: List[float]) -> Dict:
        """
        Test embedding robustness against various perturbations
        
        Args:
            embeddings: Input embeddings
            perturbation_methods: List of perturbation methods to test
            noise_scales: List of noise scales to test
            
        Returns:
            Robustness test results
        """
        results = {}
        
        for method in perturbation_methods:
            method_results = {}
            
            for scale in noise_scales:
                if method == 'gaussian':
                    perturbed = self.random_perturbation(embeddings, scale, 'gaussian')
                elif method == 'uniform':
                    perturbed = self.random_perturbation(embeddings, scale, 'uniform')
                elif method == 'adaptive':
                    perturbed = self.embedding_noise_injection(embeddings, 'adaptive', scale)
                else:
                    continue
                
                # Calculate consistency
                consistency = self.embedding_consistency_check(embeddings, perturbed)
                method_results[scale] = consistency
            
            results[method] = method_results
        
        return results
