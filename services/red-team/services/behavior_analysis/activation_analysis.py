"""
Activation Analysis
Analyzes neural network activations to understand model behavior and vulnerabilities
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


class ActivationType(Enum):
    """Types of activations to analyze"""
    HIDDEN_STATES = "hidden_states"
    ATTENTION_WEIGHTS = "attention_weights"
    GRADIENT_ACTIVATIONS = "gradient_activations"
    FEATURE_MAPS = "feature_maps"
    LOGITS = "logits"


class AnalysisMethod(Enum):
    """Analysis methods for activations"""
    STATISTICAL = "statistical"
    DIMENSIONALITY = "dimensionality"
    CORRELATION = "correlation"
    CLUSTERING = "clustering"
    ANOMALY = "anomaly"


@dataclass
class ActivationData:
    """Activation data structure"""
    layer_name: str
    activation_type: ActivationType
    activations: np.ndarray
    input_data: np.ndarray
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ActivationAnalysis:
    """Activation analysis results"""
    layer_name: str
    analysis_method: AnalysisMethod
    statistics: Dict[str, float]
    patterns: List[str]
    anomalies: List[str]
    insights: List[str]
    confidence: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ActivationAnalyzer:
    """
    Activation Analyzer
    Analyzes neural network activations to understand model behavior
    """
    
    def __init__(self):
        """Initialize activation analyzer"""
        self.activation_hooks = {}
        self.activation_data: List[ActivationData] = []
        self.analysis_results: List[ActivationAnalysis] = []
        
        logger.info("✅ Initialized Activation Analyzer")
    
    def register_hooks(self, model: nn.Module, layer_names: Optional[List[str]] = None):
        """Register hooks to capture activations"""
        try:
            self.activation_hooks.clear()
            
            if layer_names is None:
                # Register hooks for all layers
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU, nn.Transformer)):
                        self._register_layer_hook(model, name)
            else:
                # Register hooks for specific layers
                for layer_name in layer_names:
                    self._register_layer_hook(model, layer_name)
            
            logger.info(f"Registered hooks for {len(self.activation_hooks)} layers")
            
        except Exception as e:
            logger.error(f"Hook registration failed: {e}")
            raise
    
    def _register_layer_hook(self, model: nn.Module, layer_name: str):
        """Register hook for specific layer"""
        try:
            def hook_fn(module, input, output):
                # Store activation data
                if isinstance(output, torch.Tensor):
                    activation_data = ActivationData(
                        layer_name=layer_name,
                        activation_type=ActivationType.HIDDEN_STATES,
                        activations=output.detach().cpu().numpy(),
                        input_data=input[0].detach().cpu().numpy() if input else None,
                        metadata={
                            "module_type": type(module).__name__,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                    self.activation_data.append(activation_data)
                elif isinstance(output, tuple):
                    # Handle multiple outputs (e.g., LSTM)
                    for i, tensor in enumerate(output):
                        if isinstance(tensor, torch.Tensor):
                            activation_data = ActivationData(
                                layer_name=f"{layer_name}_output_{i}",
                                activation_type=ActivationType.HIDDEN_STATES,
                                activations=tensor.detach().cpu().numpy(),
                                input_data=input[0].detach().cpu().numpy() if input else None,
                                metadata={
                                    "module_type": type(module).__name__,
                                    "output_index": i,
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                            )
                            self.activation_data.append(activation_data)
            
            # Get the module
            module = dict(model.named_modules())[layer_name]
            hook = module.register_forward_hook(hook_fn)
            self.activation_hooks[layer_name] = hook
            
        except Exception as e:
            logger.error(f"Layer hook registration failed for {layer_name}: {e}")
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        try:
            for hook in self.activation_hooks.values():
                hook.remove()
            self.activation_hooks.clear()
            logger.info("Removed all activation hooks")
            
        except Exception as e:
            logger.error(f"Hook removal failed: {e}")
    
    async def analyze_activations(self, 
                                analysis_methods: List[AnalysisMethod] = None,
                                layer_names: Optional[List[str]] = None) -> List[ActivationAnalysis]:
        """
        Analyze captured activations
        """
        try:
            logger.info("Starting activation analysis")
            
            if analysis_methods is None:
                analysis_methods = list(AnalysisMethod)
            
            # Filter activations by layer names if specified
            activations_to_analyze = self.activation_data
            if layer_names:
                activations_to_analyze = [a for a in self.activation_data if a.layer_name in layer_names]
            
            if not activations_to_analyze:
                logger.warning("No activation data to analyze")
                return []
            
            # Group activations by layer
            activations_by_layer = defaultdict(list)
            for activation in activations_to_analyze:
                activations_by_layer[activation.layer_name].append(activation)
            
            # Analyze each layer
            analysis_results = []
            for layer_name, layer_activations in activations_by_layer.items():
                for method in analysis_methods:
                    analysis = await self._analyze_layer_activations(layer_name, layer_activations, method)
                    if analysis:
                        analysis_results.append(analysis)
            
            self.analysis_results.extend(analysis_results)
            logger.info(f"Completed activation analysis: {len(analysis_results)} results")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Activation analysis failed: {e}")
            return []
    
    async def _analyze_layer_activations(self, 
                                       layer_name: str, 
                                       activations: List[ActivationData], 
                                       method: AnalysisMethod) -> Optional[ActivationAnalysis]:
        """Analyze activations for specific layer and method"""
        try:
            if not activations:
                return None
            
            # Combine activations from the same layer
            combined_activations = np.concatenate([a.activations for a in activations], axis=0)
            
            if method == AnalysisMethod.STATISTICAL:
                return await self._statistical_analysis(layer_name, combined_activations)
            elif method == AnalysisMethod.DIMENSIONALITY:
                return await self._dimensionality_analysis(layer_name, combined_activations)
            elif method == AnalysisMethod.CORRELATION:
                return await self._correlation_analysis(layer_name, combined_activations)
            elif method == AnalysisMethod.CLUSTERING:
                return await self._clustering_analysis(layer_name, combined_activations)
            elif method == AnalysisMethod.ANOMALY:
                return await self._anomaly_analysis(layer_name, combined_activations)
            else:
                logger.warning(f"Unknown analysis method: {method}")
                return None
                
        except Exception as e:
            logger.error(f"Layer activation analysis failed for {layer_name}: {e}")
            return None
    
    async def _statistical_analysis(self, layer_name: str, activations: np.ndarray) -> ActivationAnalysis:
        """Perform statistical analysis on activations"""
        try:
            # Calculate basic statistics
            mean_activation = np.mean(activations)
            std_activation = np.std(activations)
            min_activation = np.min(activations)
            max_activation = np.max(activations)
            median_activation = np.median(activations)
            
            # Calculate percentiles
            percentiles = np.percentile(activations, [25, 50, 75, 90, 95, 99])
            
            # Calculate sparsity
            sparsity = np.mean(activations == 0)
            
            # Calculate dead neuron ratio
            dead_neurons = np.sum(np.all(activations == 0, axis=0))
            total_neurons = activations.shape[-1]
            dead_neuron_ratio = dead_neurons / total_neurons if total_neurons > 0 else 0
            
            statistics = {
                "mean": float(mean_activation),
                "std": float(std_activation),
                "min": float(min_activation),
                "max": float(max_activation),
                "median": float(median_activation),
                "sparsity": float(sparsity),
                "dead_neuron_ratio": float(dead_neuron_ratio),
                "percentile_25": float(percentiles[0]),
                "percentile_50": float(percentiles[1]),
                "percentile_75": float(percentiles[2]),
                "percentile_90": float(percentiles[3]),
                "percentile_95": float(percentiles[4]),
                "percentile_99": float(percentiles[5])
            }
            
            # Generate patterns and insights
            patterns = []
            insights = []
            
            if sparsity > 0.8:
                patterns.append("High sparsity - many zero activations")
                insights.append("Layer may be over-regularized or underutilized")
            
            if dead_neuron_ratio > 0.5:
                patterns.append("High dead neuron ratio")
                insights.append("Many neurons are not contributing to the output")
            
            if std_activation < 0.1:
                patterns.append("Low activation variance")
                insights.append("Activations may be too uniform, indicating potential saturation")
            
            if std_activation > 2.0:
                patterns.append("High activation variance")
                insights.append("Activations are highly variable, may indicate instability")
            
            # Calculate confidence based on data quality
            confidence = min(1.0, len(activations) / 1000)  # More data = higher confidence
            
            return ActivationAnalysis(
                layer_name=layer_name,
                analysis_method=AnalysisMethod.STATISTICAL,
                statistics=statistics,
                patterns=patterns,
                anomalies=[],
                insights=insights,
                confidence=confidence,
                metadata={
                    "data_shape": activations.shape,
                    "analysis_timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return None
    
    async def _dimensionality_analysis(self, layer_name: str, activations: np.ndarray) -> ActivationAnalysis:
        """Perform dimensionality analysis on activations"""
        try:
            # Flatten activations for dimensionality analysis
            flattened = activations.reshape(activations.shape[0], -1)
            
            # Calculate effective dimensionality using PCA
            from sklearn.decomposition import PCA
            
            # Fit PCA to get explained variance
            pca = PCA()
            pca.fit(flattened)
            
            # Calculate effective dimensionality (components explaining 95% variance)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            effective_dims = np.argmax(cumulative_variance >= 0.95) + 1
            
            # Calculate intrinsic dimensionality
            intrinsic_dims = np.sum(pca.explained_variance_ratio_ > 0.01)
            
            statistics = {
                "original_dims": int(flattened.shape[1]),
                "effective_dims": int(effective_dims),
                "intrinsic_dims": int(intrinsic_dims),
                "dimensionality_ratio": float(effective_dims / flattened.shape[1]),
                "first_pc_variance": float(pca.explained_variance_ratio_[0]),
                "top_5_pc_variance": float(np.sum(pca.explained_variance_ratio_[:5])),
                "top_10_pc_variance": float(np.sum(pca.explained_variance_ratio_[:10]))
            }
            
            # Generate patterns and insights
            patterns = []
            insights = []
            
            if effective_dims < flattened.shape[1] * 0.1:
                patterns.append("Very low effective dimensionality")
                insights.append("Layer may be over-parameterized or redundant")
            
            if effective_dims > flattened.shape[1] * 0.9:
                patterns.append("High effective dimensionality")
                insights.append("Layer is utilizing most of its capacity")
            
            if pca.explained_variance_ratio_[0] > 0.8:
                patterns.append("First principal component dominates")
                insights.append("Activations may be highly correlated")
            
            # Calculate confidence
            confidence = min(1.0, len(activations) / 500)
            
            return ActivationAnalysis(
                layer_name=layer_name,
                analysis_method=AnalysisMethod.DIMENSIONALITY,
                statistics=statistics,
                patterns=patterns,
                anomalies=[],
                insights=insights,
                confidence=confidence,
                metadata={
                    "pca_components": len(pca.explained_variance_ratio_),
                    "analysis_timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Dimensionality analysis failed: {e}")
            return None
    
    async def _correlation_analysis(self, layer_name: str, activations: np.ndarray) -> ActivationAnalysis:
        """Perform correlation analysis on activations"""
        try:
            # Flatten activations for correlation analysis
            flattened = activations.reshape(activations.shape[0], -1)
            
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(flattened.T)
            
            # Remove diagonal (self-correlation)
            mask = np.ones_like(correlation_matrix, dtype=bool)
            np.fill_diagonal(mask, False)
            correlations = correlation_matrix[mask]
            
            # Calculate correlation statistics
            mean_correlation = np.mean(correlations)
            std_correlation = np.std(correlations)
            max_correlation = np.max(correlations)
            min_correlation = np.min(correlations)
            
            # Count high correlations
            high_correlations = np.sum(np.abs(correlations) > 0.8)
            total_correlations = len(correlations)
            high_correlation_ratio = high_correlations / total_correlations if total_correlations > 0 else 0
            
            statistics = {
                "mean_correlation": float(mean_correlation),
                "std_correlation": float(std_correlation),
                "max_correlation": float(max_correlation),
                "min_correlation": float(min_correlation),
                "high_correlation_count": int(high_correlations),
                "high_correlation_ratio": float(high_correlation_ratio)
            }
            
            # Generate patterns and insights
            patterns = []
            insights = []
            
            if high_correlation_ratio > 0.3:
                patterns.append("High correlation between neurons")
                insights.append("Neurons may be redundant or highly coupled")
            
            if mean_correlation > 0.5:
                patterns.append("High average correlation")
                insights.append("Neurons are highly correlated, may indicate overfitting")
            
            if std_correlation < 0.1:
                patterns.append("Low correlation variance")
                insights.append("Correlations are very uniform")
            
            # Calculate confidence
            confidence = min(1.0, len(activations) / 200)
            
            return ActivationAnalysis(
                layer_name=layer_name,
                analysis_method=AnalysisMethod.CORRELATION,
                statistics=statistics,
                patterns=patterns,
                anomalies=[],
                insights=insights,
                confidence=confidence,
                metadata={
                    "correlation_matrix_shape": correlation_matrix.shape,
                    "analysis_timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return None
    
    async def _clustering_analysis(self, layer_name: str, activations: np.ndarray) -> ActivationAnalysis:
        """Perform clustering analysis on activations"""
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            # Flatten activations for clustering
            flattened = activations.reshape(activations.shape[0], -1)
            
            # Determine optimal number of clusters
            max_clusters = min(10, len(flattened) // 10)
            if max_clusters < 2:
                return None
            
            silhouette_scores = []
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(flattened)
                score = silhouette_score(flattened, cluster_labels)
                silhouette_scores.append(score)
            
            # Find optimal number of clusters
            optimal_k = np.argmax(silhouette_scores) + 2
            optimal_score = max(silhouette_scores)
            
            # Perform final clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(flattened)
            
            # Calculate cluster statistics
            cluster_sizes = [np.sum(cluster_labels == i) for i in range(optimal_k)]
            cluster_centers = kmeans.cluster_centers_
            
            # Calculate cluster separation
            cluster_separation = np.mean([np.linalg.norm(center) for center in cluster_centers])
            
            statistics = {
                "optimal_clusters": int(optimal_k),
                "silhouette_score": float(optimal_score),
                "cluster_sizes": cluster_sizes,
                "cluster_separation": float(cluster_separation),
                "max_cluster_size": int(max(cluster_sizes)),
                "min_cluster_size": int(min(cluster_sizes)),
                "cluster_size_std": float(np.std(cluster_sizes))
            }
            
            # Generate patterns and insights
            patterns = []
            insights = []
            
            if optimal_score > 0.7:
                patterns.append("Well-separated clusters")
                insights.append("Activations form distinct groups")
            
            if optimal_score < 0.3:
                patterns.append("Poor cluster separation")
                insights.append("Activations are not well-clustered")
            
            if np.std(cluster_sizes) > np.mean(cluster_sizes):
                patterns.append("Uneven cluster sizes")
                insights.append("Some clusters are much larger than others")
            
            # Calculate confidence
            confidence = min(1.0, len(activations) / 100)
            
            return ActivationAnalysis(
                layer_name=layer_name,
                analysis_method=AnalysisMethod.CLUSTERING,
                statistics=statistics,
                patterns=patterns,
                anomalies=[],
                insights=insights,
                confidence=confidence,
                metadata={
                    "cluster_labels": cluster_labels.tolist(),
                    "analysis_timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Clustering analysis failed: {e}")
            return None
    
    async def _anomaly_analysis(self, layer_name: str, activations: np.ndarray) -> ActivationAnalysis:
        """Perform anomaly detection on activations"""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            
            # Flatten activations for anomaly detection
            flattened = activations.reshape(activations.shape[0], -1)
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_activations = scaler.fit_transform(flattened)
            
            # Fit isolation forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(scaled_activations)
            
            # Calculate anomaly statistics
            n_anomalies = np.sum(anomaly_labels == -1)
            anomaly_ratio = n_anomalies / len(anomaly_labels)
            
            # Get anomaly scores
            anomaly_scores = iso_forest.decision_function(scaled_activations)
            mean_anomaly_score = np.mean(anomaly_scores)
            std_anomaly_score = np.std(anomaly_scores)
            
            statistics = {
                "n_anomalies": int(n_anomalies),
                "anomaly_ratio": float(anomaly_ratio),
                "mean_anomaly_score": float(mean_anomaly_score),
                "std_anomaly_score": float(std_anomaly_score),
                "min_anomaly_score": float(np.min(anomaly_scores)),
                "max_anomaly_score": float(np.max(anomaly_scores))
            }
            
            # Generate patterns and insights
            patterns = []
            insights = []
            anomalies = []
            
            if anomaly_ratio > 0.2:
                patterns.append("High anomaly ratio")
                insights.append("Many activations are anomalous")
                anomalies.append(f"High anomaly ratio: {anomaly_ratio:.2f}")
            
            if anomaly_ratio < 0.01:
                patterns.append("Very low anomaly ratio")
                insights.append("Activations are very consistent")
            
            if std_anomaly_score > 0.5:
                patterns.append("High anomaly score variance")
                insights.append("Anomaly scores are highly variable")
            
            # Calculate confidence
            confidence = min(1.0, len(activations) / 200)
            
            return ActivationAnalysis(
                layer_name=layer_name,
                analysis_method=AnalysisMethod.ANOMALY,
                statistics=statistics,
                patterns=patterns,
                anomalies=anomalies,
                insights=insights,
                confidence=confidence,
                metadata={
                    "anomaly_labels": anomaly_labels.tolist(),
                    "anomaly_scores": anomaly_scores.tolist(),
                    "analysis_timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Anomaly analysis failed: {e}")
            return None
    
    async def get_activation_summary(self) -> Dict[str, Any]:
        """Get summary of all activation analyses"""
        try:
            if not self.analysis_results:
                return {}
            
            # Group results by layer and method
            layer_summaries = defaultdict(lambda: defaultdict(list))
            
            for result in self.analysis_results:
                layer_summaries[result.layer_name][result.analysis_method.value].append(result)
            
            # Generate summary statistics
            summary = {
                "total_layers": len(layer_summaries),
                "total_analyses": len(self.analysis_results),
                "layer_summaries": {},
                "overall_patterns": [],
                "overall_anomalies": [],
                "overall_insights": []
            }
            
            # Analyze each layer
            for layer_name, methods in layer_summaries.items():
                layer_summary = {
                    "methods_analyzed": list(methods.keys()),
                    "total_analyses": sum(len(results) for results in methods.values()),
                    "patterns": [],
                    "anomalies": [],
                    "insights": []
                }
                
                # Collect patterns, anomalies, and insights
                for method_results in methods.values():
                    for result in method_results:
                        layer_summary["patterns"].extend(result.patterns)
                        layer_summary["anomalies"].extend(result.anomalies)
                        layer_summary["insights"].extend(result.insights)
                
                # Remove duplicates
                layer_summary["patterns"] = list(set(layer_summary["patterns"]))
                layer_summary["anomalies"] = list(set(layer_summary["anomalies"]))
                layer_summary["insights"] = list(set(layer_summary["insights"]))
                
                summary["layer_summaries"][layer_name] = layer_summary
                
                # Add to overall collections
                summary["overall_patterns"].extend(layer_summary["patterns"])
                summary["overall_anomalies"].extend(layer_summary["anomalies"])
                summary["overall_insights"].extend(layer_summary["insights"])
            
            # Remove duplicates from overall collections
            summary["overall_patterns"] = list(set(summary["overall_patterns"]))
            summary["overall_anomalies"] = list(set(summary["overall_anomalies"]))
            summary["overall_insights"] = list(set(summary["overall_insights"]))
            
            return summary
            
        except Exception as e:
            logger.error(f"Activation summary generation failed: {e}")
            return {}
    
    async def export_analysis_data(self, format: str = "json") -> str:
        """Export activation analysis data"""
        try:
            if format.lower() == "json":
                data = {
                    "activation_data": [a.__dict__ for a in self.activation_data],
                    "analysis_results": [r.__dict__ for r in self.analysis_results],
                    "summary": await self.get_activation_summary(),
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Analysis data export failed: {e}")
            return ""
