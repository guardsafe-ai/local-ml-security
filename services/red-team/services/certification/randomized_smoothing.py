"""
Randomized Smoothing Certification
Implements randomized smoothing for adversarial robustness certification
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
from scipy import stats
from scipy.stats import norm

logger = logging.getLogger(__name__)


class SmoothingType(Enum):
    """Types of randomized smoothing"""
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
    UNIFORM = "uniform"


@dataclass
class SmoothingConfig:
    """Configuration for randomized smoothing"""
    smoothing_type: SmoothingType
    noise_std: float
    n_samples: int
    alpha: float = 0.001
    confidence_level: float = 0.99
    batch_size: int = 1000


@dataclass
class CertificationResult:
    """Result of robustness certification"""
    is_certified: bool
    certified_radius: float
    confidence: float
    p_value: float
    n_samples: int
    smoothing_config: SmoothingConfig
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RandomizedSmoothingCertifier:
    """
    Randomized Smoothing Certifier
    Implements randomized smoothing for adversarial robustness certification
    """
    
    def __init__(self):
        """Initialize randomized smoothing certifier"""
        self.certification_results: List[CertificationResult] = []
        
        logger.info("✅ Initialized Randomized Smoothing Certifier")
    
    async def certify_robustness(self, 
                               model: nn.Module,
                               input_data: torch.Tensor,
                               true_label: int,
                               smoothing_config: SmoothingConfig,
                               target_radius: Optional[float] = None) -> CertificationResult:
        """
        Certify robustness using randomized smoothing
        """
        try:
            logger.info(f"Certifying robustness with {smoothing_config.smoothing_type.value} smoothing")
            
            # Set model to evaluation mode
            model.eval()
            
            # Generate noisy samples
            noisy_samples = await self._generate_noisy_samples(
                input_data, 
                smoothing_config
            )
            
            # Get predictions on noisy samples
            predictions = await self._get_predictions(model, noisy_samples)
            
            # Calculate certification statistics
            certification_stats = await self._calculate_certification_stats(
                predictions, 
                true_label, 
                smoothing_config
            )
            
            # Determine if certified
            is_certified, certified_radius = await self._determine_certification(
                certification_stats, 
                smoothing_config, 
                target_radius
            )
            
            # Create certification result
            result = CertificationResult(
                is_certified=is_certified,
                certified_radius=certified_radius,
                confidence=certification_stats["confidence"],
                p_value=certification_stats["p_value"],
                n_samples=smoothing_config.n_samples,
                smoothing_config=smoothing_config,
                metadata={
                    "input_shape": input_data.shape,
                    "true_label": true_label,
                    "prediction_accuracy": certification_stats["accuracy"],
                    "certified_at": datetime.utcnow().isoformat()
                }
            )
            
            # Store result
            self.certification_results.append(result)
            
            logger.info(f"Certification completed: {is_certified}, radius: {certified_radius:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Robustness certification failed: {e}")
            raise
    
    async def _generate_noisy_samples(self, 
                                    input_data: torch.Tensor, 
                                    config: SmoothingConfig) -> torch.Tensor:
        """Generate noisy samples for smoothing"""
        try:
            batch_size = config.batch_size
            n_samples = config.n_samples
            noise_std = config.noise_std
            
            # Calculate number of batches
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            noisy_samples = []
            
            for batch_idx in range(n_batches):
                # Calculate batch size for this iteration
                current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
                
                # Generate noise
                if config.smoothing_type == SmoothingType.GAUSSIAN:
                    noise = torch.randn(current_batch_size, *input_data.shape) * noise_std
                elif config.smoothing_type == SmoothingType.LAPLACE:
                    noise = torch.distributions.Laplace(0, noise_std).sample((current_batch_size, *input_data.shape))
                elif config.smoothing_type == SmoothingType.UNIFORM:
                    noise = torch.uniform(-noise_std, noise_std, (current_batch_size, *input_data.shape))
                else:
                    raise ValueError(f"Unknown smoothing type: {config.smoothing_type}")
                
                # Add noise to input
                batch_noisy = input_data.unsqueeze(0).expand(current_batch_size, -1, -1, -1) + noise
                noisy_samples.append(batch_noisy)
            
            # Concatenate all batches
            noisy_samples = torch.cat(noisy_samples, dim=0)
            
            # Ensure we have exactly n_samples
            if len(noisy_samples) > n_samples:
                noisy_samples = noisy_samples[:n_samples]
            
            return noisy_samples
            
        except Exception as e:
            logger.error(f"Noisy sample generation failed: {e}")
            raise
    
    async def _get_predictions(self, 
                             model: nn.Module, 
                             noisy_samples: torch.Tensor) -> torch.Tensor:
        """Get model predictions on noisy samples"""
        try:
            with torch.no_grad():
                # Move to same device as model
                device = next(model.parameters()).device
                noisy_samples = noisy_samples.to(device)
                
                # Get predictions
                predictions = model(noisy_samples)
                
                # Get predicted classes
                predicted_classes = torch.argmax(predictions, dim=1)
                
                return predicted_classes.cpu()
                
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            raise
    
    async def _calculate_certification_stats(self, 
                                           predictions: torch.Tensor, 
                                           true_label: int, 
                                           config: SmoothingConfig) -> Dict[str, Any]:
        """Calculate certification statistics"""
        try:
            # Count correct predictions
            correct_predictions = (predictions == true_label).sum().item()
            total_predictions = len(predictions)
            accuracy = correct_predictions / total_predictions
            
            # Calculate confidence interval for accuracy
            confidence_interval = self._calculate_confidence_interval(
                correct_predictions, 
                total_predictions, 
                config.confidence_level
            )
            
            # Calculate p-value for hypothesis test
            p_value = self._calculate_p_value(
                correct_predictions, 
                total_predictions, 
                config.alpha
            )
            
            # Calculate certified radius
            certified_radius = self._calculate_certified_radius(
                accuracy, 
                config.noise_std, 
                config.alpha
            )
            
            return {
                "accuracy": accuracy,
                "correct_predictions": correct_predictions,
                "total_predictions": total_predictions,
                "confidence_interval": confidence_interval,
                "p_value": p_value,
                "certified_radius": certified_radius,
                "confidence": config.confidence_level
            }
            
        except Exception as e:
            logger.error(f"Certification statistics calculation failed: {e}")
            raise
    
    def _calculate_confidence_interval(self, 
                                     successes: int, 
                                     trials: int, 
                                     confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for binomial proportion"""
        try:
            # Use normal approximation for large samples
            if trials > 30:
                p_hat = successes / trials
                se = np.sqrt(p_hat * (1 - p_hat) / trials)
                z_score = norm.ppf(1 - (1 - confidence_level) / 2)
                margin_error = z_score * se
                
                lower = max(0, p_hat - margin_error)
                upper = min(1, p_hat + margin_error)
                
                return (lower, upper)
            else:
                # Use exact binomial confidence interval
                alpha = 1 - confidence_level
                lower = stats.beta.ppf(alpha / 2, successes, trials - successes + 1)
                upper = stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
                
                return (lower, upper)
                
        except Exception as e:
            logger.error(f"Confidence interval calculation failed: {e}")
            return (0.0, 1.0)
    
    def _calculate_p_value(self, 
                         successes: int, 
                         trials: int, 
                         alpha: float) -> float:
        """Calculate p-value for hypothesis test"""
        try:
            # Test H0: p >= 0.5 vs H1: p < 0.5
            # This tests if the model is robust (accuracy >= 0.5)
            p_value = stats.binom.cdf(successes, trials, 0.5)
            
            return p_value
            
        except Exception as e:
            logger.error(f"P-value calculation failed: {e}")
            return 1.0
    
    def _calculate_certified_radius(self, 
                                  accuracy: float, 
                                  noise_std: float, 
                                  alpha: float) -> float:
        """Calculate certified radius based on accuracy and noise level"""
        try:
            # For Gaussian noise, certified radius is:
            # R = σ * Φ^(-1)(p_A) where p_A is the accuracy
            # and Φ^(-1) is the inverse CDF of standard normal
            
            if accuracy <= 0.5:
                return 0.0
            
            # Calculate inverse CDF
            z_score = norm.ppf(accuracy)
            
            # Calculate certified radius
            certified_radius = noise_std * z_score
            
            return certified_radius
            
        except Exception as e:
            logger.error(f"Certified radius calculation failed: {e}")
            return 0.0
    
    async def _determine_certification(self, 
                                     stats: Dict[str, Any], 
                                     config: SmoothingConfig, 
                                     target_radius: Optional[float]) -> Tuple[bool, float]:
        """Determine if model is certified and calculate radius"""
        try:
            accuracy = stats["accuracy"]
            p_value = stats["p_value"]
            certified_radius = stats["certified_radius"]
            
            # Check if accuracy is above threshold
            if accuracy < 0.5:
                return False, 0.0
            
            # Check if p-value is significant
            if p_value > config.alpha:
                return False, 0.0
            
            # Check if target radius is met
            if target_radius is not None:
                if certified_radius < target_radius:
                    return False, certified_radius
            
            return True, certified_radius
            
        except Exception as e:
            logger.error(f"Certification determination failed: {e}")
            return False, 0.0
    
    async def batch_certify(self, 
                          model: nn.Module,
                          input_batch: torch.Tensor,
                          true_labels: torch.Tensor,
                          smoothing_config: SmoothingConfig) -> List[CertificationResult]:
        """Certify robustness for a batch of inputs"""
        try:
            logger.info(f"Batch certifying {len(input_batch)} inputs")
            
            results = []
            
            for i in range(len(input_batch)):
                try:
                    result = await self.certify_robustness(
                        model=model,
                        input_data=input_batch[i],
                        true_label=true_labels[i].item(),
                        smoothing_config=smoothing_config
                    )
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Certification failed for input {i}: {e}")
                    # Create failed result
                    failed_result = CertificationResult(
                        is_certified=False,
                        certified_radius=0.0,
                        confidence=0.0,
                        p_value=1.0,
                        n_samples=0,
                        smoothing_config=smoothing_config,
                        metadata={"error": str(e)}
                    )
                    results.append(failed_result)
            
            logger.info(f"Batch certification completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Batch certification failed: {e}")
            return []
    
    async def adaptive_certify(self, 
                             model: nn.Module,
                             input_data: torch.Tensor,
                             true_label: int,
                             target_radius: float,
                             max_samples: int = 10000) -> CertificationResult:
        """Adaptively certify with increasing sample size until target radius is met"""
        try:
            logger.info(f"Adaptive certification for target radius: {target_radius}")
            
            # Start with small sample size
            n_samples = 1000
            step_size = 1000
            
            best_result = None
            
            while n_samples <= max_samples:
                # Create config with current sample size
                config = SmoothingConfig(
                    smoothing_type=SmoothingType.GAUSSIAN,
                    noise_std=0.25,  # Default noise level
                    n_samples=n_samples,
                    alpha=0.001,
                    confidence_level=0.99
                )
                
                # Try certification
                result = await self.certify_robustness(
                    model=model,
                    input_data=input_data,
                    true_label=true_label,
                    smoothing_config=config,
                    target_radius=target_radius
                )
                
                # Check if target radius is met
                if result.is_certified and result.certified_radius >= target_radius:
                    logger.info(f"Target radius achieved with {n_samples} samples")
                    return result
                
                # Update best result
                if best_result is None or result.certified_radius > best_result.certified_radius:
                    best_result = result
                
                # Increase sample size
                n_samples += step_size
            
            logger.info(f"Target radius not achieved, best radius: {best_result.certified_radius:.4f}")
            return best_result
            
        except Exception as e:
            logger.error(f"Adaptive certification failed: {e}")
            raise
    
    async def get_certification_summary(self) -> Dict[str, Any]:
        """Get summary of all certification results"""
        try:
            if not self.certification_results:
                return {"message": "No certification results available"}
            
            # Calculate statistics
            total_certifications = len(self.certification_results)
            successful_certifications = sum(1 for r in self.certification_results if r.is_certified)
            success_rate = successful_certifications / total_certifications
            
            # Calculate radius statistics
            certified_radii = [r.certified_radius for r in self.certification_results if r.is_certified]
            if certified_radii:
                mean_radius = np.mean(certified_radii)
                std_radius = np.std(certified_radii)
                max_radius = np.max(certified_radii)
                min_radius = np.min(certified_radii)
            else:
                mean_radius = std_radius = max_radius = min_radius = 0.0
            
            # Calculate confidence statistics
            confidences = [r.confidence for r in self.certification_results]
            mean_confidence = np.mean(confidences)
            
            return {
                "total_certifications": total_certifications,
                "successful_certifications": successful_certifications,
                "success_rate": success_rate,
                "radius_statistics": {
                    "mean": mean_radius,
                    "std": std_radius,
                    "max": max_radius,
                    "min": min_radius,
                    "count": len(certified_radii)
                },
                "confidence_statistics": {
                    "mean": mean_confidence,
                    "std": np.std(confidences)
                },
                "summary_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Certification summary generation failed: {e}")
            return {}
    
    async def export_certification_data(self, format: str = "json") -> str:
        """Export certification data"""
        try:
            if format.lower() == "json":
                data = {
                    "certification_results": [r.__dict__ for r in self.certification_results],
                    "summary": await self.get_certification_summary(),
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Certification data export failed: {e}")
            return ""
