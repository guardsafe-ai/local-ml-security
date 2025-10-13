"""
Causal Analysis
Implements causal inference models to understand attack mechanisms
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
import networkx as nx

logger = logging.getLogger(__name__)


class CausalMethod(Enum):
    """Causal inference methods"""
    GRANGER_CAUSALITY = "granger_causality"
    STRUCTURAL_CAUSAL_MODEL = "structural_causal_model"
    INTERVENTION_ANALYSIS = "intervention_analysis"
    COUNTERFACTUAL_ANALYSIS = "counterfactual_analysis"
    CAUSAL_DISCOVERY = "causal_discovery"


class CausalRelation(Enum):
    """Types of causal relations"""
    DIRECT_CAUSE = "direct_cause"
    INDIRECT_CAUSE = "indirect_cause"
    CONFOUNDER = "confounder"
    MEDIATOR = "mediator"
    COLLIDER = "collider"


@dataclass
class CausalGraph:
    """Causal graph representation"""
    nodes: List[str]
    edges: List[Tuple[str, str]]
    edge_weights: Dict[Tuple[str, str], float]
    node_attributes: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CausalEffect:
    """Causal effect representation"""
    cause: str
    effect: str
    effect_size: float
    confidence: float
    p_value: float
    relation_type: CausalRelation
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CausalAnalysis:
    """Causal analysis results"""
    causal_graph: CausalGraph
    causal_effects: List[CausalEffect]
    intervention_effects: Dict[str, float]
    counterfactual_analysis: Dict[str, Any]
    insights: List[str]
    confidence: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CausalAnalyzer:
    """
    Causal Analyzer
    Implements causal inference models to understand attack mechanisms
    """
    
    def __init__(self):
        """Initialize causal analyzer"""
        self.causal_graphs: List[CausalGraph] = []
        self.causal_effects: List[CausalEffect] = []
        self.analysis_results: List[CausalAnalysis] = []
        
        logger.info("✅ Initialized Causal Analyzer")
    
    async def discover_causal_structure(self, 
                                      data: np.ndarray,
                                      variable_names: List[str],
                                      method: CausalMethod = CausalMethod.CAUSAL_DISCOVERY) -> CausalGraph:
        """
        Discover causal structure from data
        """
        try:
            logger.info(f"Discovering causal structure for {len(variable_names)} variables")
            
            if method == CausalMethod.CAUSAL_DISCOVERY:
                return await self._discover_causal_structure_pc(data, variable_names)
            elif method == CausalMethod.GRANGER_CAUSALITY:
                return await self._discover_granger_causality(data, variable_names)
            else:
                logger.warning(f"Unknown causal discovery method: {method}")
                return None
                
        except Exception as e:
            logger.error(f"Causal structure discovery failed: {e}")
            return None
    
    async def _discover_causal_structure_pc(self, data: np.ndarray, variable_names: List[str]) -> CausalGraph:
        """Discover causal structure using PC algorithm"""
        try:
            # This is a simplified PC algorithm implementation
            # In practice, you would use a proper causal discovery library
            
            n_variables = len(variable_names)
            n_samples = data.shape[0]
            
            # Initialize fully connected graph
            nodes = variable_names
            edges = []
            edge_weights = {}
            node_attributes = {}
            
            # Compute correlation matrix
            correlation_matrix = np.corrcoef(data.T)
            
            # Apply PC algorithm steps
            # Step 1: Remove edges based on independence tests
            for i in range(n_variables):
                for j in range(i + 1, n_variables):
                    # Test independence
                    correlation = abs(correlation_matrix[i, j])
                    p_value = self._compute_correlation_p_value(correlation, n_samples)
                    
                    if p_value < 0.05:  # Significant correlation
                        edges.append((variable_names[i], variable_names[j]))
                        edge_weights[(variable_names[i], variable_names[j])] = correlation
            
            # Step 2: Orient edges based on conditional independence
            oriented_edges = []
            for edge in edges:
                cause, effect = edge
                # Simple heuristic: variable with higher variance is more likely to be cause
                var_cause = np.var(data[:, variable_names.index(cause)])
                var_effect = np.var(data[:, variable_names.index(effect)])
                
                if var_cause > var_effect:
                    oriented_edges.append((cause, effect))
                else:
                    oriented_edges.append((effect, cause))
            
            # Initialize node attributes
            for i, var_name in enumerate(variable_names):
                node_attributes[var_name] = {
                    "variance": float(np.var(data[:, i])),
                    "mean": float(np.mean(data[:, i])),
                    "std": float(np.std(data[:, i]))
                }
            
            return CausalGraph(
                nodes=nodes,
                edges=oriented_edges,
                edge_weights=edge_weights,
                node_attributes=node_attributes,
                metadata={
                    "method": "PC_algorithm",
                    "n_samples": n_samples,
                    "discovered_at": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"PC algorithm causal discovery failed: {e}")
            return None
    
    async def _discover_granger_causality(self, data: np.ndarray, variable_names: List[str]) -> CausalGraph:
        """Discover causal structure using Granger causality"""
        try:
            from sklearn.linear_model import LinearRegression
            from scipy import stats
            
            n_variables = len(variable_names)
            n_samples = data.shape[0]
            
            # Initialize graph
            nodes = variable_names
            edges = []
            edge_weights = {}
            node_attributes = {}
            
            # Test Granger causality for each pair
            for i in range(n_variables):
                for j in range(n_variables):
                    if i == j:
                        continue
                    
                    # Test if variable i Granger-causes variable j
                    granger_stat, p_value = self._test_granger_causality(data[:, i], data[:, j])
                    
                    if p_value < 0.05:  # Significant Granger causality
                        edges.append((variable_names[i], variable_names[j]))
                        edge_weights[(variable_names[i], variable_names[j])] = granger_stat
            
            # Initialize node attributes
            for i, var_name in enumerate(variable_names):
                node_attributes[var_name] = {
                    "variance": float(np.var(data[:, i])),
                    "mean": float(np.mean(data[:, i])),
                    "std": float(np.std(data[:, i]))
                }
            
            return CausalGraph(
                nodes=nodes,
                edges=edges,
                edge_weights=edge_weights,
                node_attributes=node_attributes,
                metadata={
                    "method": "Granger_causality",
                    "n_samples": n_samples,
                    "discovered_at": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Granger causality discovery failed: {e}")
            return None
    
    def _test_granger_causality(self, x: np.ndarray, y: np.ndarray, max_lag: int = 5) -> Tuple[float, float]:
        """Test Granger causality between two time series"""
        try:
            from sklearn.linear_model import LinearRegression
            from scipy import stats
            
            n = len(x)
            if n < max_lag + 1:
                return 0.0, 1.0
            
            # Create lagged variables
            X = np.column_stack([x[i:n-max_lag+i] for i in range(max_lag)])
            y_target = y[max_lag:]
            
            # Fit restricted model (no lags of x)
            y_lags = np.column_stack([y[i:n-max_lag+i] for i in range(max_lag)])
            X_restricted = y_lags
            
            reg_restricted = LinearRegression()
            reg_restricted.fit(X_restricted, y_target)
            rss_restricted = np.sum((y_target - reg_restricted.predict(X_restricted))**2)
            
            # Fit unrestricted model (with lags of x)
            X_unrestricted = np.column_stack([y_lags, X])
            
            reg_unrestricted = LinearRegression()
            reg_unrestricted.fit(X_unrestricted, y_target)
            rss_unrestricted = np.sum((y_target - reg_unrestricted.predict(X_unrestricted))**2)
            
            # Compute F-statistic
            if rss_restricted > 0 and rss_unrestricted > 0:
                f_stat = ((rss_restricted - rss_unrestricted) / max_lag) / (rss_unrestricted / (n - 2 * max_lag))
                p_value = 1 - stats.f.cdf(f_stat, max_lag, n - 2 * max_lag)
            else:
                f_stat = 0.0
                p_value = 1.0
            
            return f_stat, p_value
            
        except Exception as e:
            logger.error(f"Granger causality test failed: {e}")
            return 0.0, 1.0
    
    def _compute_correlation_p_value(self, correlation: float, n_samples: int) -> float:
        """Compute p-value for correlation coefficient"""
        try:
            from scipy import stats
            
            if n_samples < 3:
                return 1.0
            
            # Compute t-statistic
            t_stat = correlation * np.sqrt((n_samples - 2) / (1 - correlation**2))
            
            # Compute p-value
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_samples - 2))
            
            return p_value
            
        except Exception as e:
            logger.error(f"Correlation p-value computation failed: {e}")
            return 1.0
    
    async def estimate_causal_effects(self, 
                                    causal_graph: CausalGraph,
                                    data: np.ndarray,
                                    variable_names: List[str]) -> List[CausalEffect]:
        """
        Estimate causal effects from causal graph
        """
        try:
            logger.info("Estimating causal effects")
            
            causal_effects = []
            
            for edge in causal_graph.edges:
                cause, effect = edge
                
                # Get indices
                cause_idx = variable_names.index(cause)
                effect_idx = variable_names.index(effect)
                
                # Estimate causal effect
                effect_size, confidence, p_value = await self._estimate_causal_effect(
                    data[:, cause_idx], data[:, effect_idx]
                )
                
                # Determine relation type
                relation_type = self._determine_relation_type(causal_graph, cause, effect)
                
                causal_effect = CausalEffect(
                    cause=cause,
                    effect=effect,
                    effect_size=effect_size,
                    confidence=confidence,
                    p_value=p_value,
                    relation_type=relation_type,
                    metadata={
                        "edge_weight": causal_graph.edge_weights.get(edge, 0.0),
                        "estimated_at": datetime.utcnow().isoformat()
                    }
                )
                
                causal_effects.append(causal_effect)
                self.causal_effects.append(causal_effect)
            
            logger.info(f"Estimated {len(causal_effects)} causal effects")
            return causal_effects
            
        except Exception as e:
            logger.error(f"Causal effect estimation failed: {e}")
            return []
    
    async def _estimate_causal_effect(self, cause: np.ndarray, effect: np.ndarray) -> Tuple[float, float, float]:
        """Estimate causal effect between two variables"""
        try:
            from sklearn.linear_model import LinearRegression
            from scipy import stats
            
            # Fit linear regression
            reg = LinearRegression()
            reg.fit(cause.reshape(-1, 1), effect)
            
            # Get coefficient (causal effect)
            effect_size = reg.coef_[0]
            
            # Compute confidence interval
            predictions = reg.predict(cause.reshape(-1, 1))
            residuals = effect - predictions
            mse = np.mean(residuals**2)
            
            # Standard error
            se = np.sqrt(mse / np.sum((cause - np.mean(cause))**2))
            
            # Confidence interval
            t_val = stats.t.ppf(0.975, len(cause) - 2)
            confidence = t_val * se
            
            # P-value
            t_stat = effect_size / se
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(cause) - 2))
            
            return effect_size, confidence, p_value
            
        except Exception as e:
            logger.error(f"Causal effect estimation failed: {e}")
            return 0.0, 0.0, 1.0
    
    def _determine_relation_type(self, causal_graph: CausalGraph, cause: str, effect: str) -> CausalRelation:
        """Determine the type of causal relation"""
        try:
            # Check if there are intermediate variables
            intermediate_vars = []
            for edge in causal_graph.edges:
                if edge[0] == cause and edge[1] != effect:
                    intermediate_vars.append(edge[1])
            
            # Check if any intermediate variable causes the effect
            for var in intermediate_vars:
                if (var, effect) in causal_graph.edges:
                    return CausalRelation.INDIRECT_CAUSE
            
            # Check for confounders
            confounders = []
            for edge in causal_graph.edges:
                if edge[1] == cause:
                    confounders.append(edge[0])
            
            if confounders:
                return CausalRelation.CONFOUNDER
            
            return CausalRelation.DIRECT_CAUSE
            
        except Exception as e:
            logger.error(f"Relation type determination failed: {e}")
            return CausalRelation.DIRECT_CAUSE
    
    async def perform_intervention_analysis(self, 
                                          causal_graph: CausalGraph,
                                          data: np.ndarray,
                                          variable_names: List[str],
                                          intervention_variable: str,
                                          intervention_value: float) -> Dict[str, float]:
        """
        Perform intervention analysis
        """
        try:
            logger.info(f"Performing intervention analysis on {intervention_variable}")
            
            intervention_effects = {}
            
            # Get intervention variable index
            intervention_idx = variable_names.index(intervention_variable)
            
            # Find variables that are causally affected by the intervention
            affected_variables = []
            for edge in causal_graph.edges:
                if edge[0] == intervention_variable:
                    affected_variables.append(edge[1])
            
            # Estimate intervention effects
            for var in affected_variables:
                var_idx = variable_names.index(var)
                
                # Simulate intervention
                intervention_data = data.copy()
                intervention_data[:, intervention_idx] = intervention_value
                
                # Estimate effect
                original_mean = np.mean(data[:, var_idx])
                intervention_mean = np.mean(intervention_data[:, var_idx])
                effect = intervention_mean - original_mean
                
                intervention_effects[var] = effect
            
            return intervention_effects
            
        except Exception as e:
            logger.error(f"Intervention analysis failed: {e}")
            return {}
    
    async def perform_counterfactual_analysis(self, 
                                            causal_graph: CausalGraph,
                                            data: np.ndarray,
                                            variable_names: List[str],
                                            counterfactual_scenario: Dict[str, float]) -> Dict[str, Any]:
        """
        Perform counterfactual analysis
        """
        try:
            logger.info("Performing counterfactual analysis")
            
            counterfactual_results = {}
            
            # Simulate counterfactual scenario
            counterfactual_data = data.copy()
            
            for var, value in counterfactual_scenario.items():
                if var in variable_names:
                    var_idx = variable_names.index(var)
                    counterfactual_data[:, var_idx] = value
            
            # Compare original and counterfactual outcomes
            for i, var_name in enumerate(variable_names):
                original_mean = np.mean(data[:, i])
                counterfactual_mean = np.mean(counterfactual_data[:, i])
                difference = counterfactual_mean - original_mean
                
                counterfactual_results[var_name] = {
                    "original_mean": float(original_mean),
                    "counterfactual_mean": float(counterfactual_mean),
                    "difference": float(difference),
                    "relative_change": float(difference / original_mean) if original_mean != 0 else 0.0
                }
            
            return counterfactual_results
            
        except Exception as e:
            logger.error(f"Counterfactual analysis failed: {e}")
            return {}
    
    async def analyze_causal_structure(self, 
                                     causal_graph: CausalGraph,
                                     causal_effects: List[CausalEffect],
                                     intervention_effects: Dict[str, float],
                                     counterfactual_analysis: Dict[str, Any]) -> CausalAnalysis:
        """
        Analyze causal structure and generate insights
        """
        try:
            logger.info("Analyzing causal structure")
            
            # Generate insights
            insights = self._generate_causal_insights(
                causal_graph, causal_effects, intervention_effects, counterfactual_analysis
            )
            
            # Calculate overall confidence
            confidence = self._calculate_causal_confidence(causal_effects, intervention_effects)
            
            # Create analysis result
            analysis = CausalAnalysis(
                causal_graph=causal_graph,
                causal_effects=causal_effects,
                intervention_effects=intervention_effects,
                counterfactual_analysis=counterfactual_analysis,
                insights=insights,
                confidence=confidence,
                metadata={
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                    "n_effects": len(causal_effects),
                    "n_interventions": len(intervention_effects)
                }
            )
            
            self.analysis_results.append(analysis)
            return analysis
            
        except Exception as e:
            logger.error(f"Causal structure analysis failed: {e}")
            return None
    
    def _generate_causal_insights(self, 
                                causal_graph: CausalGraph,
                                causal_effects: List[CausalEffect],
                                intervention_effects: Dict[str, float],
                                counterfactual_analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from causal analysis"""
        try:
            insights = []
            
            # Analyze causal graph structure
            if len(causal_graph.edges) == 0:
                insights.append("No causal relationships detected")
            elif len(causal_graph.edges) < len(causal_graph.nodes) * 0.5:
                insights.append("Sparse causal network detected")
            else:
                insights.append("Dense causal network detected")
            
            # Analyze causal effects
            strong_effects = [e for e in causal_effects if abs(e.effect_size) > 0.5]
            if strong_effects:
                insights.append(f"Strong causal effects detected: {len(strong_effects)} effects")
            
            weak_effects = [e for e in causal_effects if abs(e.effect_size) < 0.1]
            if weak_effects:
                insights.append(f"Weak causal effects detected: {len(weak_effects)} effects")
            
            # Analyze intervention effects
            if intervention_effects:
                max_intervention = max(intervention_effects.values(), key=abs)
                insights.append(f"Maximum intervention effect: {max_intervention:.3f}")
            
            # Analyze counterfactual results
            if counterfactual_analysis:
                large_changes = [v for v in counterfactual_analysis.values() 
                               if abs(v.get("relative_change", 0)) > 0.5]
                if large_changes:
                    insights.append(f"Large counterfactual changes detected: {len(large_changes)} variables")
            
            return insights
            
        except Exception as e:
            logger.error(f"Causal insight generation failed: {e}")
            return []
    
    def _calculate_causal_confidence(self, 
                                   causal_effects: List[CausalEffect],
                                   intervention_effects: Dict[str, float]) -> float:
        """Calculate confidence in causal analysis"""
        try:
            if not causal_effects:
                return 0.0
            
            # Base confidence from effect significance
            significant_effects = [e for e in causal_effects if e.p_value < 0.05]
            significance_ratio = len(significant_effects) / len(causal_effects)
            
            # Confidence from effect consistency
            effect_sizes = [abs(e.effect_size) for e in causal_effects]
            if effect_sizes:
                effect_consistency = 1.0 - (np.std(effect_sizes) / np.mean(effect_sizes)) if np.mean(effect_sizes) > 0 else 0.0
            else:
                effect_consistency = 0.0
            
            # Overall confidence
            confidence = 0.5 * significance_ratio + 0.5 * effect_consistency
            
            return float(min(confidence, 1.0))
            
        except Exception as e:
            logger.error(f"Causal confidence calculation failed: {e}")
            return 0.0
    
    async def export_causal_data(self, format: str = "json") -> str:
        """Export causal analysis data"""
        try:
            if format.lower() == "json":
                data = {
                    "causal_graphs": [g.__dict__ for g in self.causal_graphs],
                    "causal_effects": [e.__dict__ for e in self.causal_effects],
                    "analysis_results": [r.__dict__ for r in self.analysis_results],
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Causal data export failed: {e}")
            return ""
