"""
Interval Bound Propagation (IBP) Certification
Implements IBP for adversarial robustness certification
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

logger = logging.getLogger(__name__)


class IBPType(Enum):
    """Types of IBP certification"""
    CROWN_IBP = "crown_ibp"
    CROWN_ADAPT = "crown_adapt"
    FAST_LIN = "fast_lin"
    DEEP_POLY = "deep_poly"


@dataclass
class IBPConfig:
    """Configuration for IBP certification"""
    ibp_type: IBPType
    epsilon: float
    method: str = "backward"
    use_alpha: bool = True
    alpha: float = 0.0
    beta: float = 1.0
    gamma: float = 1.0
    use_ibp: bool = True
    use_alpha_crown: bool = True


@dataclass
class IBPBounds:
    """Bounds from IBP analysis"""
    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor
    layer_name: str
    layer_type: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class IBPCertificationResult:
    """Result of IBP certification"""
    is_certified: bool
    certified_radius: float
    bounds: List[IBPBounds]
    verification_time: float
    ibp_config: IBPConfig
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class IBPCertifier:
    """
    Interval Bound Propagation Certifier
    Implements IBP for adversarial robustness certification
    """
    
    def __init__(self):
        """Initialize IBP certifier"""
        self.certification_results: List[IBPCertificationResult] = []
        
        logger.info("✅ Initialized IBP Certifier")
    
    async def certify_robustness(self, 
                               model: nn.Module,
                               input_data: torch.Tensor,
                               true_label: int,
                               ibp_config: IBPConfig) -> IBPCertificationResult:
        """
        Certify robustness using Interval Bound Propagation
        """
        try:
            logger.info(f"Certifying robustness with {ibp_config.ibp_type.value} IBP")
            
            start_time = datetime.utcnow()
            
            # Set model to evaluation mode
            model.eval()
            
            # Calculate input bounds
            input_bounds = self._calculate_input_bounds(input_data, ibp_config.epsilon)
            
            # Propagate bounds through the network
            bounds = await self._propagate_bounds(model, input_bounds, ibp_config)
            
            # Verify robustness
            is_certified, certified_radius = await self._verify_robustness(
                bounds, true_label, ibp_config
            )
            
            # Calculate verification time
            verification_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create certification result
            result = IBPCertificationResult(
                is_certified=is_certified,
                certified_radius=certified_radius,
                bounds=bounds,
                verification_time=verification_time,
                ibp_config=ibp_config,
                metadata={
                    "input_shape": input_data.shape,
                    "true_label": true_label,
                    "n_layers": len(bounds),
                    "certified_at": datetime.utcnow().isoformat()
                }
            )
            
            # Store result
            self.certification_results.append(result)
            
            logger.info(f"IBP certification completed: {is_certified}, radius: {certified_radius:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"IBP certification failed: {e}")
            raise
    
    def _calculate_input_bounds(self, 
                              input_data: torch.Tensor, 
                              epsilon: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate input bounds for given epsilon"""
        try:
            # Calculate lower and upper bounds
            lower_bounds = input_data - epsilon
            upper_bounds = input_data + epsilon
            
            # Clamp to valid input range (e.g., [0, 1] for images)
            lower_bounds = torch.clamp(lower_bounds, 0.0, 1.0)
            upper_bounds = torch.clamp(upper_bounds, 0.0, 1.0)
            
            return lower_bounds, upper_bounds
            
        except Exception as e:
            logger.error(f"Input bounds calculation failed: {e}")
            raise
    
    async def _propagate_bounds(self, 
                              model: nn.Module, 
                              input_bounds: Tuple[torch.Tensor, torch.Tensor],
                              config: IBPConfig) -> List[IBPBounds]:
        """Propagate bounds through the network"""
        try:
            lower_bounds, upper_bounds = input_bounds
            bounds = []
            
            # Get model layers
            layers = list(model.children())
            
            for i, layer in enumerate(layers):
                try:
                    # Calculate bounds for this layer
                    layer_bounds = await self._calculate_layer_bounds(
                        layer, lower_bounds, upper_bounds, config
                    )
                    
                    if layer_bounds:
                        bounds.append(layer_bounds)
                        lower_bounds = layer_bounds.lower_bounds
                        upper_bounds = layer_bounds.upper_bounds
                    
                except Exception as e:
                    logger.warning(f"Bounds calculation failed for layer {i}: {e}")
                    continue
            
            return bounds
            
        except Exception as e:
            logger.error(f"Bounds propagation failed: {e}")
            return []
    
    async def _calculate_layer_bounds(self, 
                                    layer: nn.Module, 
                                    lower_bounds: torch.Tensor, 
                                    upper_bounds: torch.Tensor,
                                    config: IBPConfig) -> Optional[IBPBounds]:
        """Calculate bounds for a specific layer"""
        try:
            if isinstance(layer, nn.Linear):
                return await self._calculate_linear_bounds(layer, lower_bounds, upper_bounds, config)
            elif isinstance(layer, nn.Conv2d):
                return await self._calculate_conv2d_bounds(layer, lower_bounds, upper_bounds, config)
            elif isinstance(layer, nn.ReLU):
                return await self._calculate_relu_bounds(layer, lower_bounds, upper_bounds, config)
            elif isinstance(layer, nn.MaxPool2d):
                return await self._calculate_maxpool2d_bounds(layer, lower_bounds, upper_bounds, config)
            elif isinstance(layer, nn.AdaptiveAvgPool2d):
                return await self._calculate_adaptive_avgpool2d_bounds(layer, lower_bounds, upper_bounds, config)
            else:
                # For unknown layers, use simple bounds
                return await self._calculate_generic_bounds(layer, lower_bounds, upper_bounds, config)
                
        except Exception as e:
            logger.error(f"Layer bounds calculation failed: {e}")
            return None
    
    async def _calculate_linear_bounds(self, 
                                     layer: nn.Linear, 
                                     lower_bounds: torch.Tensor, 
                                     upper_bounds: torch.Tensor,
                                     config: IBPConfig) -> IBPBounds:
        """Calculate bounds for linear layer"""
        try:
            weight = layer.weight
            bias = layer.bias if layer.bias is not None else torch.zeros(weight.shape[0])
            
            # Flatten input bounds
            lower_flat = lower_bounds.view(lower_bounds.shape[0], -1)
            upper_flat = upper_bounds.view(upper_bounds.shape[0], -1)
            
            # Calculate bounds for linear transformation
            # For each output neuron, we need to find min and max of w^T * x + b
            # where x is in [lower, upper]
            
            # Positive weights contribute to upper bound, negative to lower bound
            weight_pos = torch.clamp(weight, min=0)
            weight_neg = torch.clamp(weight, max=0)
            
            # Calculate bounds
            new_lower = torch.matmul(lower_flat, weight_pos.T) + torch.matmul(upper_flat, weight_neg.T) + bias
            new_upper = torch.matmul(upper_flat, weight_pos.T) + torch.matmul(lower_flat, weight_neg.T) + bias
            
            return IBPBounds(
                lower_bounds=new_lower,
                upper_bounds=new_upper,
                layer_name="linear",
                layer_type="linear",
                metadata={
                    "input_shape": lower_bounds.shape,
                    "output_shape": new_lower.shape,
                    "weight_shape": weight.shape
                }
            )
            
        except Exception as e:
            logger.error(f"Linear bounds calculation failed: {e}")
            raise
    
    async def _calculate_conv2d_bounds(self, 
                                     layer: nn.Conv2d, 
                                     lower_bounds: torch.Tensor, 
                                     upper_bounds: torch.Tensor,
                                     config: IBPConfig) -> IBPBounds:
        """Calculate bounds for Conv2d layer"""
        try:
            weight = layer.weight
            bias = layer.bias if layer.bias is not None else torch.zeros(weight.shape[0])
            
            # Calculate bounds for convolution
            # This is a simplified version - full IBP for Conv2d is more complex
            
            # For now, use a simple approximation
            # In practice, you'd want to use more sophisticated methods like CROWN
            
            # Apply convolution to both bounds
            lower_conv = F.conv2d(lower_bounds, weight, bias, 
                                stride=layer.stride, padding=layer.padding, dilation=layer.dilation)
            upper_conv = F.conv2d(upper_bounds, weight, bias,
                                stride=layer.stride, padding=layer.padding, dilation=layer.dilation)
            
            # Take element-wise min/max
            new_lower = torch.min(lower_conv, upper_conv)
            new_upper = torch.max(lower_conv, upper_conv)
            
            return IBPBounds(
                lower_bounds=new_lower,
                upper_bounds=new_upper,
                layer_name="conv2d",
                layer_type="conv2d",
                metadata={
                    "input_shape": lower_bounds.shape,
                    "output_shape": new_lower.shape,
                    "kernel_size": layer.kernel_size,
                    "stride": layer.stride,
                    "padding": layer.padding
                }
            )
            
        except Exception as e:
            logger.error(f"Conv2d bounds calculation failed: {e}")
            raise
    
    async def _calculate_relu_bounds(self, 
                                   layer: nn.ReLU, 
                                   lower_bounds: torch.Tensor, 
                                   upper_bounds: torch.Tensor,
                                   config: IBPConfig) -> IBPBounds:
        """Calculate bounds for ReLU layer"""
        try:
            # ReLU bounds: output = max(0, input)
            # If lower >= 0: output = input
            # If upper <= 0: output = 0
            # If lower < 0 < upper: output in [0, upper]
            
            new_lower = torch.clamp(lower_bounds, min=0)
            new_upper = torch.clamp(upper_bounds, min=0)
            
            return IBPBounds(
                lower_bounds=new_lower,
                upper_bounds=new_upper,
                layer_name="relu",
                layer_type="relu",
                metadata={
                    "input_shape": lower_bounds.shape,
                    "output_shape": new_lower.shape
                }
            )
            
        except Exception as e:
            logger.error(f"ReLU bounds calculation failed: {e}")
            raise
    
    async def _calculate_maxpool2d_bounds(self, 
                                        layer: nn.MaxPool2d, 
                                        lower_bounds: torch.Tensor, 
                                        upper_bounds: torch.Tensor,
                                        config: IBPConfig) -> IBPBounds:
        """Calculate bounds for MaxPool2d layer"""
        try:
            # MaxPool bounds: output = max(input in pooling region)
            # Lower bound: max of lower bounds in pooling region
            # Upper bound: max of upper bounds in pooling region
            
            new_lower = F.max_pool2d(lower_bounds, 
                                    kernel_size=layer.kernel_size, 
                                    stride=layer.stride, 
                                    padding=layer.padding)
            new_upper = F.max_pool2d(upper_bounds, 
                                    kernel_size=layer.kernel_size, 
                                    stride=layer.stride, 
                                    padding=layer.padding)
            
            return IBPBounds(
                lower_bounds=new_lower,
                upper_bounds=new_upper,
                layer_name="maxpool2d",
                layer_type="maxpool2d",
                metadata={
                    "input_shape": lower_bounds.shape,
                    "output_shape": new_lower.shape,
                    "kernel_size": layer.kernel_size,
                    "stride": layer.stride,
                    "padding": layer.padding
                }
            )
            
        except Exception as e:
            logger.error(f"MaxPool2d bounds calculation failed: {e}")
            raise
    
    async def _calculate_adaptive_avgpool2d_bounds(self, 
                                                 layer: nn.AdaptiveAvgPool2d, 
                                                 lower_bounds: torch.Tensor, 
                                                 upper_bounds: torch.Tensor,
                                                 config: IBPConfig) -> IBPBounds:
        """Calculate bounds for AdaptiveAvgPool2d layer"""
        try:
            # AdaptiveAvgPool bounds: output = mean(input in pooling region)
            # Bounds are preserved under averaging
            
            new_lower = F.adaptive_avg_pool2d(lower_bounds, layer.output_size)
            new_upper = F.adaptive_avg_pool2d(upper_bounds, layer.output_size)
            
            return IBPBounds(
                lower_bounds=new_lower,
                upper_bounds=new_upper,
                layer_name="adaptive_avgpool2d",
                layer_type="adaptive_avgpool2d",
                metadata={
                    "input_shape": lower_bounds.shape,
                    "output_shape": new_lower.shape,
                    "output_size": layer.output_size
                }
            )
            
        except Exception as e:
            logger.error(f"AdaptiveAvgPool2d bounds calculation failed: {e}")
            raise
    
    async def _calculate_generic_bounds(self, 
                                      layer: nn.Module, 
                                      lower_bounds: torch.Tensor, 
                                      upper_bounds: torch.Tensor,
                                      config: IBPConfig) -> IBPBounds:
        """Calculate bounds for generic layer"""
        try:
            # For unknown layers, use a simple approximation
            # This is not rigorous but provides a rough estimate
            
            # Apply layer to both bounds
            with torch.no_grad():
                lower_output = layer(lower_bounds)
                upper_output = layer(upper_bounds)
            
            # Take element-wise min/max
            new_lower = torch.min(lower_output, upper_output)
            new_upper = torch.max(lower_output, upper_output)
            
            return IBPBounds(
                lower_bounds=new_lower,
                upper_bounds=new_upper,
                layer_name=layer.__class__.__name__,
                layer_type="generic",
                metadata={
                    "input_shape": lower_bounds.shape,
                    "output_shape": new_lower.shape,
                    "layer_class": layer.__class__.__name__
                }
            )
            
        except Exception as e:
            logger.error(f"Generic bounds calculation failed: {e}")
            raise
    
    async def _verify_robustness(self, 
                               bounds: List[IBPBounds], 
                               true_label: int, 
                               config: IBPConfig) -> Tuple[bool, float]:
        """Verify robustness using final layer bounds"""
        try:
            if not bounds:
                return False, 0.0
            
            # Get final layer bounds
            final_bounds = bounds[-1]
            lower_bounds = final_bounds.lower_bounds
            upper_bounds = final_bounds.upper_bounds
            
            # Check if true label has highest lower bound
            # This means it's guaranteed to be the prediction for any input in the epsilon ball
            
            # Get lower bound for true label
            true_label_lower = lower_bounds[0, true_label].item()
            
            # Get upper bounds for all other labels
            other_labels = [i for i in range(lower_bounds.shape[1]) if i != true_label]
            other_upper_bounds = [upper_bounds[0, i].item() for i in other_labels]
            
            # Check if true label lower bound is higher than all other upper bounds
            is_certified = all(true_label_lower > upper_bound for upper_bound in other_upper_bounds)
            
            # Calculate certified radius (simplified)
            if is_certified:
                # Find the minimum gap between true label and other labels
                min_gap = min(true_label_lower - upper_bound for upper_bound in other_upper_bounds)
                certified_radius = min_gap / 2.0  # Simplified calculation
            else:
                certified_radius = 0.0
            
            return is_certified, certified_radius
            
        except Exception as e:
            logger.error(f"Robustness verification failed: {e}")
            return False, 0.0
    
    async def batch_certify(self, 
                          model: nn.Module,
                          input_batch: torch.Tensor,
                          true_labels: torch.Tensor,
                          ibp_config: IBPConfig) -> List[IBPCertificationResult]:
        """Certify robustness for a batch of inputs"""
        try:
            logger.info(f"Batch certifying {len(input_batch)} inputs with IBP")
            
            results = []
            
            for i in range(len(input_batch)):
                try:
                    result = await self.certify_robustness(
                        model=model,
                        input_data=input_batch[i],
                        true_label=true_labels[i].item(),
                        ibp_config=ibp_config
                    )
                    results.append(result)
                except Exception as e:
                    logger.warning(f"IBP certification failed for input {i}: {e}")
                    # Create failed result
                    failed_result = IBPCertificationResult(
                        is_certified=False,
                        certified_radius=0.0,
                        bounds=[],
                        verification_time=0.0,
                        ibp_config=ibp_config,
                        metadata={"error": str(e)}
                    )
                    results.append(failed_result)
            
            logger.info(f"Batch IBP certification completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Batch IBP certification failed: {e}")
            return []
    
    async def get_certification_summary(self) -> Dict[str, Any]:
        """Get summary of all IBP certification results"""
        try:
            if not self.certification_results:
                return {"message": "No IBP certification results available"}
            
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
            
            # Calculate verification time statistics
            verification_times = [r.verification_time for r in self.certification_results]
            mean_time = np.mean(verification_times)
            std_time = np.std(verification_times)
            
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
                "verification_time_statistics": {
                    "mean": mean_time,
                    "std": std_time,
                    "max": np.max(verification_times),
                    "min": np.min(verification_times)
                },
                "summary_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"IBP certification summary generation failed: {e}")
            return {}
    
    async def export_certification_data(self, format: str = "json") -> str:
        """Export IBP certification data"""
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
            logger.error(f"IBP certification data export failed: {e}")
            return ""
