"""
Attribution Analysis
Implements SHAP, LIME, and Integrated Gradients for model explainability
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

logger = logging.getLogger(__name__)


class AttributionMethod(Enum):
    """Attribution methods"""
    SHAP = "shap"
    LIME = "lime"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    GRADIENT_SHAP = "gradient_shap"
    DEEP_LIFT = "deep_lift"
    OCCLUSION = "occlusion"
    SALIENCY = "saliency"


class AttributionType(Enum):
    """Types of attributions"""
    FEATURE_ATTRIBUTION = "feature_attribution"
    LAYER_ATTRIBUTION = "layer_attribution"
    NEURON_ATTRIBUTION = "neuron_attribution"
    TOKEN_ATTRIBUTION = "token_attribution"


@dataclass
class AttributionData:
    """Attribution data structure"""
    input_data: np.ndarray
    attribution_scores: np.ndarray
    attribution_method: AttributionMethod
    attribution_type: AttributionType
    target_class: Optional[int] = None
    baseline: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AttributionAnalysis:
    """Attribution analysis results"""
    attribution_method: AttributionMethod
    attribution_type: AttributionType
    feature_importance: Dict[str, float]
    top_features: List[Tuple[str, float]]
    attribution_consistency: float
    attribution_stability: float
    insights: List[str]
    confidence: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AttributionAnalyzer:
    """
    Attribution Analyzer
    Implements various attribution methods for model explainability
    """
    
    def __init__(self):
        """Initialize attribution analyzer"""
        self.attribution_data: List[AttributionData] = []
        self.analysis_results: List[AttributionAnalysis] = []
        
        logger.info("✅ Initialized Attribution Analyzer")
    
    async def compute_attributions(self, 
                                 model: nn.Module,
                                 input_data: np.ndarray,
                                 target_class: Optional[int] = None,
                                 methods: List[AttributionMethod] = None,
                                 baseline: Optional[np.ndarray] = None) -> List[AttributionData]:
        """
        Compute attributions for input data using specified methods
        """
        try:
            logger.info(f"Computing attributions for input shape: {input_data.shape}")
            
            if methods is None:
                methods = [AttributionMethod.INTEGRATED_GRADIENTS, AttributionMethod.SHAP, AttributionMethod.LIME]
            
            attribution_results = []
            
            for method in methods:
                try:
                    attribution = await self._compute_method_attribution(
                        model, input_data, method, target_class, baseline
                    )
                    if attribution:
                        attribution_results.append(attribution)
                        self.attribution_data.append(attribution)
                except Exception as e:
                    logger.warning(f"Attribution computation failed for {method.value}: {e}")
                    continue
            
            logger.info(f"Computed {len(attribution_results)} attributions")
            return attribution_results
            
        except Exception as e:
            logger.error(f"Attribution computation failed: {e}")
            return []
    
    async def _compute_method_attribution(self, 
                                        model: nn.Module,
                                        input_data: np.ndarray,
                                        method: AttributionMethod,
                                        target_class: Optional[int] = None,
                                        baseline: Optional[np.ndarray] = None) -> Optional[AttributionData]:
        """Compute attribution using specific method"""
        try:
            if method == AttributionMethod.INTEGRATED_GRADIENTS:
                return await self._compute_integrated_gradients(model, input_data, target_class, baseline)
            elif method == AttributionMethod.SHAP:
                return await self._compute_shap(model, input_data, target_class)
            elif method == AttributionMethod.LIME:
                return await self._compute_lime(model, input_data, target_class)
            elif method == AttributionMethod.GRADIENT_SHAP:
                return await self._compute_gradient_shap(model, input_data, target_class, baseline)
            elif method == AttributionMethod.DEEP_LIFT:
                return await self._compute_deep_lift(model, input_data, target_class, baseline)
            elif method == AttributionMethod.OCCLUSION:
                return await self._compute_occlusion(model, input_data, target_class)
            elif method == AttributionMethod.SALIENCY:
                return await self._compute_saliency(model, input_data, target_class)
            else:
                logger.warning(f"Unknown attribution method: {method}")
                return None
                
        except Exception as e:
            logger.error(f"Method attribution computation failed for {method.value}: {e}")
            return None
    
    async def _compute_integrated_gradients(self, 
                                          model: nn.Module,
                                          input_data: np.ndarray,
                                          target_class: Optional[int] = None,
                                          baseline: Optional[np.ndarray] = None) -> AttributionData:
        """Compute Integrated Gradients attribution"""
        try:
            # Convert to tensor
            input_tensor = torch.tensor(input_data, dtype=torch.float32, requires_grad=True)
            
            # Set baseline
            if baseline is None:
                baseline = np.zeros_like(input_data)
            baseline_tensor = torch.tensor(baseline, dtype=torch.float32)
            
            # Number of steps for integration
            steps = 50
            
            # Generate interpolated inputs
            interpolated_inputs = []
            for i in range(steps + 1):
                alpha = i / steps
                interpolated = baseline_tensor + alpha * (input_tensor - baseline_tensor)
                interpolated_inputs.append(interpolated)
            
            # Compute gradients for each interpolated input
            gradients = []
            for interpolated in interpolated_inputs:
                interpolated.requires_grad_(True)
                output = model(interpolated)
                
                if target_class is not None:
                    target_output = output[target_class] if len(output.shape) > 1 else output
                else:
                    target_output = output.sum()
                
                grad = torch.autograd.grad(target_output, interpolated, retain_graph=True)[0]
                gradients.append(grad.detach().numpy())
            
            # Compute integrated gradients
            integrated_gradients = np.mean(gradients, axis=0)
            
            # Scale by input difference
            input_diff = input_data - baseline
            attribution_scores = integrated_gradients * input_diff
            
            return AttributionData(
                input_data=input_data,
                attribution_scores=attribution_scores,
                attribution_method=AttributionMethod.INTEGRATED_GRADIENTS,
                attribution_type=AttributionType.FEATURE_ATTRIBUTION,
                target_class=target_class,
                baseline=baseline,
                metadata={
                    "steps": steps,
                    "computed_at": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Integrated gradients computation failed: {e}")
            raise
    
    async def _compute_shap(self, 
                          model: nn.Module,
                          input_data: np.ndarray,
                          target_class: Optional[int] = None) -> AttributionData:
        """Compute SHAP attribution"""
        try:
            # This is a simplified SHAP implementation
            # In practice, you would use the SHAP library
            
            # For now, use a simple gradient-based approximation
            input_tensor = torch.tensor(input_data, dtype=torch.float32, requires_grad=True)
            output = model(input_tensor)
            
            if target_class is not None:
                target_output = output[target_class] if len(output.shape) > 1 else output
            else:
                target_output = output.sum()
            
            grad = torch.autograd.grad(target_output, input_tensor, retain_graph=True)[0]
            attribution_scores = grad.detach().numpy()
            
            return AttributionData(
                input_data=input_data,
                attribution_scores=attribution_scores,
                attribution_method=AttributionMethod.SHAP,
                attribution_type=AttributionType.FEATURE_ATTRIBUTION,
                target_class=target_class,
                metadata={
                    "computed_at": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            raise
    
    async def _compute_lime(self, 
                          model: nn.Module,
                          input_data: np.ndarray,
                          target_class: Optional[int] = None) -> AttributionData:
        """Compute LIME attribution"""
        try:
            # This is a simplified LIME implementation
            # In practice, you would use the LIME library
            
            # Generate perturbed samples
            n_samples = 1000
            perturbed_samples = []
            predictions = []
            
            for _ in range(n_samples):
                # Create random perturbation
                perturbation = np.random.normal(0, 0.1, input_data.shape)
                perturbed = input_data + perturbation
                perturbed_samples.append(perturbed)
                
                # Get prediction
                with torch.no_grad():
                    input_tensor = torch.tensor(perturbed, dtype=torch.float32)
                    output = model(input_tensor)
                    if target_class is not None:
                        pred = output[target_class].item() if len(output.shape) > 1 else output.item()
                    else:
                        pred = output.sum().item()
                    predictions.append(pred)
            
            # Fit linear model to approximate local behavior
            from sklearn.linear_model import LinearRegression
            
            X = np.array(perturbed_samples)
            y = np.array(predictions)
            
            # Use input features as weights
            weights = np.abs(input_data.flatten())
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights)
            
            # Fit weighted linear regression
            reg = LinearRegression()
            reg.fit(X, y, sample_weight=weights)
            
            # Attribution scores are the coefficients
            attribution_scores = reg.coef_.reshape(input_data.shape)
            
            return AttributionData(
                input_data=input_data,
                attribution_scores=attribution_scores,
                attribution_method=AttributionMethod.LIME,
                attribution_type=AttributionType.FEATURE_ATTRIBUTION,
                target_class=target_class,
                metadata={
                    "n_samples": n_samples,
                    "computed_at": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"LIME computation failed: {e}")
            raise
    
    async def _compute_gradient_shap(self, 
                                   model: nn.Module,
                                   input_data: np.ndarray,
                                   target_class: Optional[int] = None,
                                   baseline: Optional[np.ndarray] = None) -> AttributionData:
        """Compute Gradient SHAP attribution"""
        try:
            # Set baseline
            if baseline is None:
                baseline = np.zeros_like(input_data)
            
            # Generate random baselines
            n_baselines = 10
            baselines = []
            for _ in range(n_baselines):
                baseline_noise = np.random.normal(0, 0.1, input_data.shape)
                baselines.append(baseline + baseline_noise)
            
            # Compute gradients for each baseline
            gradients = []
            for baseline in baselines:
                input_tensor = torch.tensor(input_data, dtype=torch.float32, requires_grad=True)
                baseline_tensor = torch.tensor(baseline, dtype=torch.float32)
                
                # Interpolate between baseline and input
                interpolated = baseline_tensor + 0.5 * (input_tensor - baseline_tensor)
                interpolated.requires_grad_(True)
                
                output = model(interpolated)
                if target_class is not None:
                    target_output = output[target_class] if len(output.shape) > 1 else output
                else:
                    target_output = output.sum()
                
                grad = torch.autograd.grad(target_output, interpolated, retain_graph=True)[0]
                gradients.append(grad.detach().numpy())
            
            # Average gradients
            attribution_scores = np.mean(gradients, axis=0)
            
            return AttributionData(
                input_data=input_data,
                attribution_scores=attribution_scores,
                attribution_method=AttributionMethod.GRADIENT_SHAP,
                attribution_type=AttributionType.FEATURE_ATTRIBUTION,
                target_class=target_class,
                baseline=baseline,
                metadata={
                    "n_baselines": n_baselines,
                    "computed_at": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Gradient SHAP computation failed: {e}")
            raise
    
    async def _compute_deep_lift(self, 
                               model: nn.Module,
                               input_data: np.ndarray,
                               target_class: Optional[int] = None,
                               baseline: Optional[np.ndarray] = None) -> AttributionData:
        """Compute DeepLIFT attribution"""
        try:
            # Set baseline
            if baseline is None:
                baseline = np.zeros_like(input_data)
            
            # Compute forward pass for input and baseline
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            baseline_tensor = torch.tensor(baseline, dtype=torch.float32)
            
            with torch.no_grad():
                input_output = model(input_tensor)
                baseline_output = model(baseline_tensor)
            
            # Compute difference
            if target_class is not None:
                input_score = input_output[target_class].item() if len(input_output.shape) > 1 else input_output.item()
                baseline_score = baseline_output[target_class].item() if len(baseline_output.shape) > 1 else baseline_output.item()
            else:
                input_score = input_output.sum().item()
                baseline_score = baseline_output.sum().item()
            
            score_diff = input_score - baseline_score
            
            # Simple attribution: distribute difference proportionally
            input_diff = input_data - baseline
            total_diff = np.sum(np.abs(input_diff))
            
            if total_diff > 0:
                attribution_scores = (input_diff / total_diff) * score_diff
            else:
                attribution_scores = np.zeros_like(input_data)
            
            return AttributionData(
                input_data=input_data,
                attribution_scores=attribution_scores,
                attribution_method=AttributionMethod.DEEP_LIFT,
                attribution_type=AttributionType.FEATURE_ATTRIBUTION,
                target_class=target_class,
                baseline=baseline,
                metadata={
                    "score_diff": score_diff,
                    "computed_at": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"DeepLIFT computation failed: {e}")
            raise
    
    async def _compute_occlusion(self, 
                               model: nn.Module,
                               input_data: np.ndarray,
                               target_class: Optional[int] = None) -> AttributionData:
        """Compute Occlusion attribution"""
        try:
            # Get original prediction
            with torch.no_grad():
                input_tensor = torch.tensor(input_data, dtype=torch.float32)
                original_output = model(input_tensor)
                if target_class is not None:
                    original_score = original_output[target_class].item() if len(original_output.shape) > 1 else original_output.item()
                else:
                    original_score = original_output.sum().item()
            
            # Compute attribution by occluding each feature
            attribution_scores = np.zeros_like(input_data)
            
            # For efficiency, we'll occlude patches rather than individual pixels
            patch_size = max(1, min(input_data.shape) // 10)
            
            for i in range(0, input_data.shape[0], patch_size):
                for j in range(0, input_data.shape[1], patch_size):
                    # Create occluded input
                    occluded_input = input_data.copy()
                    occluded_input[i:i+patch_size, j:j+patch_size] = 0
                    
                    # Get prediction for occluded input
                    with torch.no_grad():
                        occluded_tensor = torch.tensor(occluded_input, dtype=torch.float32)
                        occluded_output = model(occluded_tensor)
                        if target_class is not None:
                            occluded_score = occluded_output[target_class].item() if len(occluded_output.shape) > 1 else occluded_output.item()
                        else:
                            occluded_score = occluded_output.sum().item()
                    
                    # Attribution is the difference
                    attribution = original_score - occluded_score
                    attribution_scores[i:i+patch_size, j:j+patch_size] = attribution
            
            return AttributionData(
                input_data=input_data,
                attribution_scores=attribution_scores,
                attribution_method=AttributionMethod.OCCLUSION,
                attribution_type=AttributionType.FEATURE_ATTRIBUTION,
                target_class=target_class,
                metadata={
                    "patch_size": patch_size,
                    "original_score": original_score,
                    "computed_at": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Occlusion computation failed: {e}")
            raise
    
    async def _compute_saliency(self, 
                              model: nn.Module,
                              input_data: np.ndarray,
                              target_class: Optional[int] = None) -> AttributionData:
        """Compute Saliency attribution"""
        try:
            input_tensor = torch.tensor(input_data, dtype=torch.float32, requires_grad=True)
            output = model(input_tensor)
            
            if target_class is not None:
                target_output = output[target_class] if len(output.shape) > 1 else output
            else:
                target_output = output.sum()
            
            # Compute gradients
            grad = torch.autograd.grad(target_output, input_tensor, retain_graph=True)[0]
            attribution_scores = grad.detach().numpy()
            
            return AttributionData(
                input_data=input_data,
                attribution_scores=attribution_scores,
                attribution_method=AttributionMethod.SALIENCY,
                attribution_type=AttributionType.FEATURE_ATTRIBUTION,
                target_class=target_class,
                metadata={
                    "computed_at": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Saliency computation failed: {e}")
            raise
    
    async def analyze_attributions(self, 
                                 attribution_data: List[AttributionData]) -> List[AttributionAnalysis]:
        """
        Analyze attribution data to extract insights
        """
        try:
            logger.info(f"Analyzing {len(attribution_data)} attribution results")
            
            analysis_results = []
            
            for attribution in attribution_data:
                analysis = await self._analyze_single_attribution(attribution)
                if analysis:
                    analysis_results.append(analysis)
                    self.analysis_results.append(analysis)
            
            logger.info(f"Completed attribution analysis: {len(analysis_results)} results")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Attribution analysis failed: {e}")
            return []
    
    async def _analyze_single_attribution(self, attribution: AttributionData) -> Optional[AttributionAnalysis]:
        """Analyze single attribution result"""
        try:
            # Compute feature importance
            feature_importance = self._compute_feature_importance(attribution)
            
            # Get top features
            top_features = self._get_top_features(attribution, feature_importance)
            
            # Compute consistency and stability
            consistency = self._compute_attribution_consistency(attribution)
            stability = self._compute_attribution_stability(attribution)
            
            # Generate insights
            insights = self._generate_attribution_insights(attribution, feature_importance, top_features)
            
            # Calculate confidence
            confidence = self._calculate_attribution_confidence(attribution, consistency, stability)
            
            return AttributionAnalysis(
                attribution_method=attribution.attribution_method,
                attribution_type=attribution.attribution_type,
                feature_importance=feature_importance,
                top_features=top_features,
                attribution_consistency=consistency,
                attribution_stability=stability,
                insights=insights,
                confidence=confidence,
                metadata={
                    "input_shape": attribution.input_data.shape,
                    "target_class": attribution.target_class,
                    "analysis_timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Single attribution analysis failed: {e}")
            return None
    
    def _compute_feature_importance(self, attribution: AttributionData) -> Dict[str, float]:
        """Compute feature importance from attribution scores"""
        try:
            scores = attribution.attribution_scores
            
            # Compute various importance metrics
            importance = {
                "mean_absolute_attribution": float(np.mean(np.abs(scores))),
                "max_absolute_attribution": float(np.max(np.abs(scores))),
                "std_attribution": float(np.std(scores)),
                "positive_attribution_ratio": float(np.sum(scores > 0) / scores.size),
                "negative_attribution_ratio": float(np.sum(scores < 0) / scores.size),
                "zero_attribution_ratio": float(np.sum(scores == 0) / scores.size)
            }
            
            return importance
            
        except Exception as e:
            logger.error(f"Feature importance computation failed: {e}")
            return {}
    
    def _get_top_features(self, attribution: AttributionData, feature_importance: Dict[str, float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Get top features by attribution score"""
        try:
            scores = attribution.attribution_scores.flatten()
            
            # Get top k features
            top_indices = np.argsort(np.abs(scores))[-top_k:][::-1]
            top_features = []
            
            for idx in top_indices:
                feature_name = f"feature_{idx}"
                score = float(scores[idx])
                top_features.append((feature_name, score))
            
            return top_features
            
        except Exception as e:
            logger.error(f"Top features extraction failed: {e}")
            return []
    
    def _compute_attribution_consistency(self, attribution: AttributionData) -> float:
        """Compute attribution consistency"""
        try:
            scores = attribution.attribution_scores
            
            # Consistency is measured by the stability of attribution patterns
            # For now, we'll use a simple measure based on score distribution
            
            # Compute coefficient of variation
            mean_score = np.mean(np.abs(scores))
            std_score = np.std(scores)
            
            if mean_score > 0:
                cv = std_score / mean_score
                consistency = 1.0 / (1.0 + cv)  # Higher CV means lower consistency
            else:
                consistency = 1.0
            
            return float(consistency)
            
        except Exception as e:
            logger.error(f"Consistency computation failed: {e}")
            return 0.0
    
    def _compute_attribution_stability(self, attribution: AttributionData) -> float:
        """Compute attribution stability"""
        try:
            scores = attribution.attribution_scores
            
            # Stability is measured by the smoothness of attribution scores
            # Compute gradient magnitude as a measure of smoothness
            
            if len(scores.shape) > 1:
                # For 2D data, compute gradient magnitude
                grad_x = np.gradient(scores, axis=1)
                grad_y = np.gradient(scores, axis=0)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                stability = 1.0 / (1.0 + np.mean(gradient_magnitude))
            else:
                # For 1D data, compute gradient variance
                gradient = np.gradient(scores)
                stability = 1.0 / (1.0 + np.var(gradient))
            
            return float(stability)
            
        except Exception as e:
            logger.error(f"Stability computation failed: {e}")
            return 0.0
    
    def _generate_attribution_insights(self, 
                                     attribution: AttributionData, 
                                     feature_importance: Dict[str, float],
                                     top_features: List[Tuple[str, float]]) -> List[str]:
        """Generate insights from attribution analysis"""
        try:
            insights = []
            
            # Analyze attribution patterns
            if feature_importance.get("positive_attribution_ratio", 0) > 0.8:
                insights.append("Most features have positive attribution")
            elif feature_importance.get("negative_attribution_ratio", 0) > 0.8:
                insights.append("Most features have negative attribution")
            
            if feature_importance.get("zero_attribution_ratio", 0) > 0.5:
                insights.append("Many features have zero attribution")
            
            if feature_importance.get("std_attribution", 0) > feature_importance.get("mean_absolute_attribution", 0):
                insights.append("High variance in attribution scores")
            
            # Analyze top features
            if top_features:
                max_score = max(abs(score) for _, score in top_features)
                min_score = min(abs(score) for _, score in top_features)
                
                if max_score > min_score * 10:
                    insights.append("High concentration of attribution in few features")
                
                if max_score < 0.1:
                    insights.append("Low attribution scores overall")
            
            return insights
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return []
    
    def _calculate_attribution_confidence(self, 
                                        attribution: AttributionData, 
                                        consistency: float, 
                                        stability: float) -> float:
        """Calculate confidence in attribution results"""
        try:
            # Base confidence from method reliability
            method_confidence = {
                AttributionMethod.INTEGRATED_GRADIENTS: 0.9,
                AttributionMethod.SHAP: 0.8,
                AttributionMethod.LIME: 0.7,
                AttributionMethod.GRADIENT_SHAP: 0.8,
                AttributionMethod.DEEP_LIFT: 0.7,
                AttributionMethod.OCCLUSION: 0.6,
                AttributionMethod.SALIENCY: 0.5
            }.get(attribution.attribution_method, 0.5)
            
            # Adjust based on consistency and stability
            adjusted_confidence = method_confidence * (0.5 + 0.25 * consistency + 0.25 * stability)
            
            return float(min(adjusted_confidence, 1.0))
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    async def compare_attribution_methods(self, 
                                        attribution_data: List[AttributionData]) -> Dict[str, Any]:
        """Compare different attribution methods"""
        try:
            if len(attribution_data) < 2:
                return {}
            
            # Group by method
            methods = defaultdict(list)
            for attribution in attribution_data:
                methods[attribution.attribution_method.value].append(attribution)
            
            # Compute comparison metrics
            comparison = {
                "methods_compared": list(methods.keys()),
                "method_statistics": {},
                "correlation_matrix": {},
                "consistency_analysis": {}
            }
            
            # Compute statistics for each method
            for method, attributions in methods.items():
                if not attributions:
                    continue
                
                # Combine attribution scores
                all_scores = np.concatenate([a.attribution_scores.flatten() for a in attributions])
                
                comparison["method_statistics"][method] = {
                    "mean": float(np.mean(all_scores)),
                    "std": float(np.std(all_scores)),
                    "min": float(np.min(all_scores)),
                    "max": float(np.max(all_scores)),
                    "count": len(attributions)
                }
            
            # Compute correlations between methods
            method_scores = {}
            for method, attributions in methods.items():
                if attributions:
                    method_scores[method] = attributions[0].attribution_scores.flatten()
            
            if len(method_scores) > 1:
                method_names = list(method_scores.keys())
                correlation_matrix = np.zeros((len(method_names), len(method_names)))
                
                for i, method1 in enumerate(method_names):
                    for j, method2 in enumerate(method_names):
                        if i <= j:
                            corr = np.corrcoef(method_scores[method1], method_scores[method2])[0, 1]
                            correlation_matrix[i, j] = corr
                            correlation_matrix[j, i] = corr
                
                comparison["correlation_matrix"] = {
                    "methods": method_names,
                    "matrix": correlation_matrix.tolist()
                }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Attribution method comparison failed: {e}")
            return {}
    
    async def export_attribution_data(self, format: str = "json") -> str:
        """Export attribution analysis data"""
        try:
            if format.lower() == "json":
                data = {
                    "attribution_data": [a.__dict__ for a in self.attribution_data],
                    "analysis_results": [r.__dict__ for r in self.analysis_results],
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Attribution data export failed: {e}")
            return ""
