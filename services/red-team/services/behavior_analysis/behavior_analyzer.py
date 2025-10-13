"""
Behavior Analyzer
Main coordinator for all behavior analysis components
"""

import asyncio
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
from collections import defaultdict

from .activation_analysis import ActivationAnalyzer
from .attribution_analysis import AttributionAnalyzer
from .causal_analysis import CausalAnalyzer
from .anomaly_detection import AnomalyDetector, AnomalyMethod, AnomalyType

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of behavior analysis"""
    ACTIVATION = "activation"
    ATTRIBUTION = "attribution"
    CAUSAL = "causal"
    ANOMALY = "anomaly"
    COMPREHENSIVE = "comprehensive"


@dataclass
class BehaviorAnalysisRequest:
    """Request for behavior analysis"""
    model: Any
    input_data: Union[np.ndarray, torch.Tensor, List[str]]
    analysis_types: List[AnalysisType]
    target_layer: Optional[str] = None
    target_neurons: Optional[List[int]] = None
    attribution_method: Optional[str] = None
    anomaly_methods: Optional[List[AnomalyMethod]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BehaviorAnalysisResult:
    """Result of behavior analysis"""
    analysis_type: AnalysisType
    results: Dict[str, Any]
    confidence: float
    insights: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ComprehensiveAnalysis:
    """Comprehensive behavior analysis results"""
    request: BehaviorAnalysisRequest
    individual_results: List[BehaviorAnalysisResult]
    cross_analysis: Dict[str, Any]
    overall_insights: List[str]
    overall_recommendations: List[str]
    confidence: float
    analysis_timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BehaviorAnalyzer:
    """
    Behavior Analyzer
    Main coordinator for all behavior analysis components
    """
    
    def __init__(self):
        """Initialize behavior analyzer"""
        self.activation_analyzer = ActivationAnalyzer()
        self.attribution_analyzer = AttributionAnalyzer()
        self.causal_analyzer = CausalAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        
        self.analysis_history: List[ComprehensiveAnalysis] = []
        
        logger.info("✅ Initialized Behavior Analyzer")
    
    async def analyze_behavior(self, request: BehaviorAnalysisRequest) -> ComprehensiveAnalysis:
        """
        Perform comprehensive behavior analysis
        """
        try:
            logger.info(f"Starting behavior analysis: {[t.value for t in request.analysis_types]}")
            
            individual_results = []
            
            # Perform individual analyses
            for analysis_type in request.analysis_types:
                try:
                    result = await self._perform_individual_analysis(request, analysis_type)
                    if result:
                        individual_results.append(result)
                except Exception as e:
                    logger.warning(f"Analysis failed for {analysis_type.value}: {e}")
                    continue
            
            # Perform cross-analysis
            cross_analysis = await self._perform_cross_analysis(individual_results)
            
            # Generate overall insights and recommendations
            overall_insights = self._generate_overall_insights(individual_results, cross_analysis)
            overall_recommendations = self._generate_overall_recommendations(individual_results, cross_analysis)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(individual_results)
            
            # Create comprehensive analysis
            comprehensive_analysis = ComprehensiveAnalysis(
                request=request,
                individual_results=individual_results,
                cross_analysis=cross_analysis,
                overall_insights=overall_insights,
                overall_recommendations=overall_recommendations,
                confidence=overall_confidence,
                analysis_timestamp=datetime.utcnow(),
                metadata={
                    "n_analyses": len(individual_results),
                    "successful_analyses": len([r for r in individual_results if r.confidence > 0.5])
                }
            )
            
            # Store in history
            self.analysis_history.append(comprehensive_analysis)
            
            logger.info(f"Completed behavior analysis: {len(individual_results)} successful analyses")
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"Behavior analysis failed: {e}")
            raise
    
    async def _perform_individual_analysis(self, 
                                         request: BehaviorAnalysisRequest, 
                                         analysis_type: AnalysisType) -> Optional[BehaviorAnalysisResult]:
        """Perform individual analysis based on type"""
        try:
            if analysis_type == AnalysisType.ACTIVATION:
                return await self._analyze_activations(request)
            elif analysis_type == AnalysisType.ATTRIBUTION:
                return await self._analyze_attributions(request)
            elif analysis_type == AnalysisType.CAUSAL:
                return await self._analyze_causal(request)
            elif analysis_type == AnalysisType.ANOMALY:
                return await self._analyze_anomalies(request)
            else:
                logger.warning(f"Unknown analysis type: {analysis_type}")
                return None
                
        except Exception as e:
            logger.error(f"Individual analysis failed for {analysis_type.value}: {e}")
            return None
    
    async def _analyze_activations(self, request: BehaviorAnalysisRequest) -> BehaviorAnalysisResult:
        """Analyze model activations"""
        try:
            # Extract activations
            activations = await self.activation_analyzer.extract_activations(
                model=request.model,
                input_data=request.input_data,
                target_layer=request.target_layer
            )
            
            # Analyze patterns
            patterns = await self.activation_analyzer.analyze_patterns(activations)
            
            # Generate insights
            insights = self._generate_activation_insights(patterns)
            recommendations = self._generate_activation_recommendations(patterns)
            
            # Calculate confidence
            confidence = self._calculate_activation_confidence(patterns)
            
            return BehaviorAnalysisResult(
                analysis_type=AnalysisType.ACTIVATION,
                results={
                    "activations": activations,
                    "patterns": patterns
                },
                confidence=confidence,
                insights=insights,
                recommendations=recommendations,
                metadata={
                    "target_layer": request.target_layer,
                    "n_neurons": len(activations) if activations else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Activation analysis failed: {e}")
            raise
    
    async def _analyze_attributions(self, request: BehaviorAnalysisRequest) -> BehaviorAnalysisResult:
        """Analyze model attributions"""
        try:
            # Perform attribution analysis
            attributions = await self.attribution_analyzer.analyze_attributions(
                model=request.model,
                input_data=request.input_data,
                method=request.attribution_method
            )
            
            # Analyze attribution patterns
            patterns = await self.attribution_analyzer.analyze_attribution_patterns(attributions)
            
            # Generate insights
            insights = self._generate_attribution_insights(patterns)
            recommendations = self._generate_attribution_recommendations(patterns)
            
            # Calculate confidence
            confidence = self._calculate_attribution_confidence(patterns)
            
            return BehaviorAnalysisResult(
                analysis_type=AnalysisType.ATTRIBUTION,
                results={
                    "attributions": attributions,
                    "patterns": patterns
                },
                confidence=confidence,
                insights=insights,
                recommendations=recommendations,
                metadata={
                    "method": request.attribution_method,
                    "n_features": len(attributions) if attributions else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Attribution analysis failed: {e}")
            raise
    
    async def _analyze_causal(self, request: BehaviorAnalysisRequest) -> BehaviorAnalysisResult:
        """Analyze causal relationships"""
        try:
            # Perform causal analysis
            causal_graph = await self.causal_analyzer.build_causal_graph(
                model=request.model,
                input_data=request.input_data
            )
            
            # Analyze causal effects
            effects = await self.causal_analyzer.analyze_causal_effects(causal_graph)
            
            # Generate insights
            insights = self._generate_causal_insights(effects)
            recommendations = self._generate_causal_recommendations(effects)
            
            # Calculate confidence
            confidence = self._calculate_causal_confidence(effects)
            
            return BehaviorAnalysisResult(
                analysis_type=AnalysisType.CAUSAL,
                results={
                    "causal_graph": causal_graph,
                    "effects": effects
                },
                confidence=confidence,
                insights=insights,
                recommendations=recommendations,
                metadata={
                    "n_nodes": len(causal_graph.nodes) if causal_graph else 0,
                    "n_edges": len(causal_graph.edges) if causal_graph else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Causal analysis failed: {e}")
            raise
    
    async def _analyze_anomalies(self, request: BehaviorAnalysisRequest) -> BehaviorAnalysisResult:
        """Analyze behavioral anomalies"""
        try:
            # Convert input data to numpy array
            if isinstance(request.input_data, torch.Tensor):
                data = request.input_data.detach().cpu().numpy()
            elif isinstance(request.input_data, list):
                # Convert text to embeddings for anomaly detection
                data = np.random.rand(len(request.input_data), 768)  # Placeholder
            else:
                data = request.input_data
            
            # Detect anomalies
            anomaly_methods = request.anomaly_methods or [AnomalyMethod.ISOLATION_FOREST, AnomalyMethod.STATISTICAL]
            anomaly_data = await self.anomaly_detector.detect_anomalies(data, anomaly_methods)
            
            # Analyze anomalies
            analysis_results = await self.anomaly_detector.analyze_anomalies(anomaly_data)
            
            # Generate insights
            insights = self._generate_anomaly_insights(analysis_results)
            recommendations = self._generate_anomaly_recommendations(analysis_results)
            
            # Calculate confidence
            confidence = self._calculate_anomaly_confidence(analysis_results)
            
            return BehaviorAnalysisResult(
                analysis_type=AnalysisType.ANOMALY,
                results={
                    "anomaly_data": anomaly_data,
                    "analysis_results": analysis_results
                },
                confidence=confidence,
                insights=insights,
                recommendations=recommendations,
                metadata={
                    "n_anomaly_methods": len(anomaly_data),
                    "n_anomalies": sum(len(ad.anomaly_labels) for ad in anomaly_data)
                }
            )
            
        except Exception as e:
            logger.error(f"Anomaly analysis failed: {e}")
            raise
    
    async def _perform_cross_analysis(self, individual_results: List[BehaviorAnalysisResult]) -> Dict[str, Any]:
        """Perform cross-analysis between different analysis types"""
        try:
            cross_analysis = {
                "correlations": {},
                "conflicts": [],
                "reinforcements": [],
                "insights": []
            }
            
            # Find correlations between different analysis types
            for i, result1 in enumerate(individual_results):
                for j, result2 in enumerate(individual_results[i+1:], i+1):
                    correlation = self._calculate_correlation(result1, result2)
                    if correlation is not None:
                        cross_analysis["correlations"][f"{result1.analysis_type.value}_{result2.analysis_type.value}"] = correlation
            
            # Find conflicts and reinforcements
            for i, result1 in enumerate(individual_results):
                for j, result2 in enumerate(individual_results[i+1:], i+1):
                    conflict = self._detect_conflict(result1, result2)
                    if conflict:
                        cross_analysis["conflicts"].append(conflict)
                    
                    reinforcement = self._detect_reinforcement(result1, result2)
                    if reinforcement:
                        cross_analysis["reinforcements"].append(reinforcement)
            
            # Generate cross-analysis insights
            cross_analysis["insights"] = self._generate_cross_analysis_insights(cross_analysis)
            
            return cross_analysis
            
        except Exception as e:
            logger.error(f"Cross-analysis failed: {e}")
            return {}
    
    def _calculate_correlation(self, result1: BehaviorAnalysisResult, result2: BehaviorAnalysisResult) -> Optional[float]:
        """Calculate correlation between two analysis results"""
        try:
            # Extract numerical data for correlation
            data1 = self._extract_numerical_data(result1)
            data2 = self._extract_numerical_data(result2)
            
            if data1 is None or data2 is None or len(data1) != len(data2):
                return None
            
            # Calculate correlation
            correlation = np.corrcoef(data1, data2)[0, 1]
            return float(correlation) if not np.isnan(correlation) else None
            
        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return None
    
    def _extract_numerical_data(self, result: BehaviorAnalysisResult) -> Optional[np.ndarray]:
        """Extract numerical data from analysis result"""
        try:
            if result.analysis_type == AnalysisType.ACTIVATION:
                # Extract activation values
                activations = result.results.get("activations", {})
                if activations:
                    values = []
                    for layer, acts in activations.items():
                        if isinstance(acts, np.ndarray):
                            values.extend(acts.flatten())
                        elif isinstance(acts, list):
                            values.extend(acts)
                    return np.array(values) if values else None
            
            elif result.analysis_type == AnalysisType.ATTRIBUTION:
                # Extract attribution values
                attributions = result.results.get("attributions", {})
                if attributions:
                    values = []
                    for method, attrs in attributions.items():
                        if isinstance(attrs, np.ndarray):
                            values.extend(attrs.flatten())
                        elif isinstance(attrs, list):
                            values.extend(attrs)
                    return np.array(values) if values else None
            
            elif result.analysis_type == AnalysisType.ANOMALY:
                # Extract anomaly scores
                anomaly_data = result.results.get("anomaly_data", [])
                if anomaly_data:
                    scores = []
                    for ad in anomaly_data:
                        scores.extend(ad.anomaly_scores)
                    return np.array(scores) if scores else None
            
            return None
            
        except Exception as e:
            logger.error(f"Numerical data extraction failed: {e}")
            return None
    
    def _detect_conflict(self, result1: BehaviorAnalysisResult, result2: BehaviorAnalysisResult) -> Optional[Dict[str, Any]]:
        """Detect conflicts between analysis results"""
        try:
            # Check for conflicting insights
            insights1 = set(result1.insights)
            insights2 = set(result2.insights)
            
            # Look for contradictory patterns
            conflicts = []
            
            # Check confidence levels
            if abs(result1.confidence - result2.confidence) > 0.5:
                conflicts.append("Conflicting confidence levels")
            
            # Check for contradictory insights
            contradictory_pairs = [
                ("High activation", "Low activation"),
                ("Normal behavior", "Anomalous behavior"),
                ("Strong attribution", "Weak attribution")
            ]
            
            for pair in contradictory_pairs:
                if pair[0] in insights1 and pair[1] in insights2:
                    conflicts.append(f"Contradictory insights: {pair[0]} vs {pair[1]}")
                elif pair[1] in insights1 and pair[0] in insights2:
                    conflicts.append(f"Contradictory insights: {pair[1]} vs {pair[0]}")
            
            if conflicts:
                return {
                    "type": "conflict",
                    "analysis_types": [result1.analysis_type.value, result2.analysis_type.value],
                    "conflicts": conflicts,
                    "severity": "high" if len(conflicts) > 2 else "medium"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Conflict detection failed: {e}")
            return None
    
    def _detect_reinforcement(self, result1: BehaviorAnalysisResult, result2: BehaviorAnalysisResult) -> Optional[Dict[str, Any]]:
        """Detect reinforcement between analysis results"""
        try:
            # Check for reinforcing insights
            insights1 = set(result1.insights)
            insights2 = set(result2.insights)
            
            # Look for reinforcing patterns
            reinforcements = []
            
            # Check for high confidence in both
            if result1.confidence > 0.8 and result2.confidence > 0.8:
                reinforcements.append("High confidence in both analyses")
            
            # Check for similar insights
            common_insights = insights1.intersection(insights2)
            if common_insights:
                reinforcements.append(f"Common insights: {list(common_insights)}")
            
            # Check for complementary insights
            complementary_pairs = [
                ("High activation", "Strong attribution"),
                ("Normal behavior", "Low anomaly score"),
                ("Causal relationship", "High attribution")
            ]
            
            for pair in complementary_pairs:
                if pair[0] in insights1 and pair[1] in insights2:
                    reinforcements.append(f"Complementary insights: {pair[0]} + {pair[1]}")
                elif pair[1] in insights1 and pair[0] in insights2:
                    reinforcements.append(f"Complementary insights: {pair[1]} + {pair[0]}")
            
            if reinforcements:
                return {
                    "type": "reinforcement",
                    "analysis_types": [result1.analysis_type.value, result2.analysis_type.value],
                    "reinforcements": reinforcements,
                    "strength": "strong" if len(reinforcements) > 2 else "medium"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Reinforcement detection failed: {e}")
            return None
    
    def _generate_cross_analysis_insights(self, cross_analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from cross-analysis"""
        try:
            insights = []
            
            # Analyze correlations
            correlations = cross_analysis.get("correlations", {})
            if correlations:
                high_correlations = [k for k, v in correlations.items() if abs(v) > 0.7]
                if high_correlations:
                    insights.append(f"High correlations found: {high_correlations}")
            
            # Analyze conflicts
            conflicts = cross_analysis.get("conflicts", [])
            if conflicts:
                insights.append(f"Found {len(conflicts)} conflicts between analyses")
            
            # Analyze reinforcements
            reinforcements = cross_analysis.get("reinforcements", [])
            if reinforcements:
                insights.append(f"Found {len(reinforcements)} reinforcing patterns")
            
            return insights
            
        except Exception as e:
            logger.error(f"Cross-analysis insight generation failed: {e}")
            return []
    
    def _generate_overall_insights(self, 
                                 individual_results: List[BehaviorAnalysisResult], 
                                 cross_analysis: Dict[str, Any]) -> List[str]:
        """Generate overall insights from all analyses"""
        try:
            insights = []
            
            # Aggregate insights from individual results
            all_insights = []
            for result in individual_results:
                all_insights.extend(result.insights)
            
            # Count insight frequency
            insight_counts = defaultdict(int)
            for insight in all_insights:
                insight_counts[insight] += 1
            
            # Add most frequent insights
            frequent_insights = sorted(insight_counts.items(), key=lambda x: x[1], reverse=True)
            for insight, count in frequent_insights[:5]:
                insights.append(f"{insight} (appears in {count} analyses)")
            
            # Add cross-analysis insights
            cross_insights = cross_analysis.get("insights", [])
            insights.extend(cross_insights)
            
            return insights
            
        except Exception as e:
            logger.error(f"Overall insight generation failed: {e}")
            return []
    
    def _generate_overall_recommendations(self, 
                                        individual_results: List[BehaviorAnalysisResult], 
                                        cross_analysis: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations from all analyses"""
        try:
            recommendations = []
            
            # Aggregate recommendations from individual results
            all_recommendations = []
            for result in individual_results:
                all_recommendations.extend(result.recommendations)
            
            # Count recommendation frequency
            rec_counts = defaultdict(int)
            for rec in all_recommendations:
                rec_counts[rec] += 1
            
            # Add most frequent recommendations
            frequent_recs = sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)
            for rec, count in frequent_recs[:5]:
                recommendations.append(f"{rec} (suggested by {count} analyses)")
            
            # Add cross-analysis recommendations
            if cross_analysis.get("conflicts"):
                recommendations.append("Investigate conflicting analysis results")
            
            if cross_analysis.get("reinforcements"):
                recommendations.append("Leverage reinforcing analysis patterns")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Overall recommendation generation failed: {e}")
            return []
    
    def _calculate_overall_confidence(self, individual_results: List[BehaviorAnalysisResult]) -> float:
        """Calculate overall confidence from individual results"""
        try:
            if not individual_results:
                return 0.0
            
            # Weighted average of individual confidences
            confidences = [result.confidence for result in individual_results]
            weights = [1.0] * len(confidences)  # Equal weights for now
            
            overall_confidence = np.average(confidences, weights=weights)
            return float(overall_confidence)
            
        except Exception as e:
            logger.error(f"Overall confidence calculation failed: {e}")
            return 0.0
    
    # Individual analysis insight generators
    def _generate_activation_insights(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate insights from activation patterns"""
        insights = []
        
        if patterns.get("high_activation_neurons"):
            insights.append("High activation neurons detected")
        
        if patterns.get("sparse_activations"):
            insights.append("Sparse activation pattern observed")
        
        if patterns.get("activation_clusters"):
            insights.append("Neurons form activation clusters")
        
        return insights
    
    def _generate_activation_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate recommendations from activation patterns"""
        recommendations = []
        
        if patterns.get("high_activation_neurons"):
            recommendations.append("Investigate high activation neurons for potential issues")
        
        if patterns.get("sparse_activations"):
            recommendations.append("Consider model pruning for efficiency")
        
        return recommendations
    
    def _generate_attribution_insights(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate insights from attribution patterns"""
        insights = []
        
        if patterns.get("high_attribution_features"):
            insights.append("High attribution features identified")
        
        if patterns.get("attribution_consistency"):
            insights.append("Attribution patterns are consistent")
        
        return insights
    
    def _generate_attribution_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate recommendations from attribution patterns"""
        recommendations = []
        
        if patterns.get("high_attribution_features"):
            recommendations.append("Focus on high attribution features for model improvement")
        
        return recommendations
    
    def _generate_causal_insights(self, effects: Dict[str, Any]) -> List[str]:
        """Generate insights from causal effects"""
        insights = []
        
        if effects.get("strong_causal_effects"):
            insights.append("Strong causal relationships detected")
        
        if effects.get("causal_chains"):
            insights.append("Causal chains identified")
        
        return insights
    
    def _generate_causal_recommendations(self, effects: Dict[str, Any]) -> List[str]:
        """Generate recommendations from causal effects"""
        recommendations = []
        
        if effects.get("strong_causal_effects"):
            recommendations.append("Leverage strong causal relationships for model optimization")
        
        return recommendations
    
    def _generate_anomaly_insights(self, analysis_results: List[Any]) -> List[str]:
        """Generate insights from anomaly analysis"""
        insights = []
        
        for result in analysis_results:
            insights.extend(result.insights)
        
        return insights
    
    def _generate_anomaly_recommendations(self, analysis_results: List[Any]) -> List[str]:
        """Generate recommendations from anomaly analysis"""
        recommendations = []
        
        for result in analysis_results:
            recommendations.extend(result.recommendations)
        
        return recommendations
    
    # Confidence calculators
    def _calculate_activation_confidence(self, patterns: Dict[str, Any]) -> float:
        """Calculate confidence for activation analysis"""
        return 0.8  # Placeholder
    
    def _calculate_attribution_confidence(self, patterns: Dict[str, Any]) -> float:
        """Calculate confidence for attribution analysis"""
        return 0.7  # Placeholder
    
    def _calculate_causal_confidence(self, effects: Dict[str, Any]) -> float:
        """Calculate confidence for causal analysis"""
        return 0.6  # Placeholder
    
    def _calculate_anomaly_confidence(self, analysis_results: List[Any]) -> float:
        """Calculate confidence for anomaly analysis"""
        if not analysis_results:
            return 0.0
        
        confidences = [result.confidence for result in analysis_results]
        return float(np.mean(confidences))
    
    async def get_analysis_history(self, limit: int = 10) -> List[ComprehensiveAnalysis]:
        """Get analysis history"""
        return self.analysis_history[-limit:]
    
    async def export_analysis_data(self, format: str = "json") -> str:
        """Export analysis data"""
        try:
            if format.lower() == "json":
                data = {
                    "analysis_history": [a.__dict__ for a in self.analysis_history],
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Analysis data export failed: {e}")
            return ""
