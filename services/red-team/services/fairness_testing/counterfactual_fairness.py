"""
Counterfactual Fairness Testing
Tests for counterfactual fairness using causal inference
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class CounterfactualFairnessResult:
    """Counterfactual fairness test result"""
    protected_attribute: str
    counterfactual_difference: float
    is_fair: bool
    causal_effects: Dict[str, float]
    recommendations: List[str]
    metadata: Dict[str, Any] = None

class CounterfactualFairnessTester:
    """Tests counterfactual fairness using causal inference"""
    
    def __init__(self, significance_threshold: float = 0.1):
        self.significance_threshold = significance_threshold
        
    def test_counterfactual_fairness(
        self,
        predictions: np.ndarray,
        protected_attributes: np.ndarray,
        features: np.ndarray,
        protected_groups: List[str],
        causal_graph: Optional[nx.DiGraph] = None
    ) -> CounterfactualFairnessResult:
        """
        Test counterfactual fairness across protected groups
        
        Args:
            predictions: Model predictions
            protected_attributes: Protected attribute values
            features: Feature matrix
            protected_groups: List of protected group names
            causal_graph: Optional causal graph for the domain
            
        Returns:
            CounterfactualFairnessResult with test results
        """
        logger.info(f"Testing counterfactual fairness for {len(protected_groups)} groups")
        
        # Estimate causal effects
        causal_effects = self._estimate_causal_effects(
            predictions, protected_attributes, features, protected_groups, causal_graph
        )
        
        # Calculate counterfactual differences
        counterfactual_difference = self._calculate_counterfactual_difference(
            predictions, protected_attributes, features, protected_groups
        )
        
        # Check if counterfactual fairness is satisfied
        is_fair = counterfactual_difference < self.significance_threshold
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            counterfactual_difference, causal_effects, is_fair
        )
        
        result = CounterfactualFairnessResult(
            protected_attribute="counterfactual_fairness",
            counterfactual_difference=counterfactual_difference,
            is_fair=is_fair,
            causal_effects=causal_effects,
            recommendations=recommendations,
            metadata={
                "significance_threshold": self.significance_threshold,
                "protected_groups": protected_groups
            }
        )
        
        logger.info(f"Counterfactual fairness test completed. Fair: {is_fair}")
        return result
    
    def _estimate_causal_effects(
        self,
        predictions: np.ndarray,
        protected_attributes: np.ndarray,
        features: np.ndarray,
        protected_groups: List[str],
        causal_graph: Optional[nx.DiGraph]
    ) -> Dict[str, float]:
        """Estimate causal effects of protected attributes on predictions"""
        causal_effects = {}
        
        try:
            # Create binary protected attribute
            protected_binary = (protected_attributes == protected_groups[0]).astype(int)
            
            # Method 1: Direct regression
            direct_effect = self._estimate_direct_effect(
                predictions, protected_binary, features
            )
            causal_effects["direct_effect"] = direct_effect
            
            # Method 2: Mediation analysis
            if features.shape[1] > 0:
                mediation_effect = self._estimate_mediation_effect(
                    predictions, protected_binary, features
                )
                causal_effects["mediation_effect"] = mediation_effect
            
            # Method 3: Causal graph-based (if provided)
            if causal_graph is not None:
                graph_effect = self._estimate_graph_based_effect(
                    predictions, protected_binary, features, causal_graph
                )
                causal_effects["graph_effect"] = graph_effect
            
        except Exception as e:
            logger.error(f"Error estimating causal effects: {e}")
            causal_effects["error"] = str(e)
        
        return causal_effects
    
    def _estimate_direct_effect(
        self,
        predictions: np.ndarray,
        protected_binary: np.ndarray,
        features: np.ndarray
    ) -> float:
        """Estimate direct causal effect using regression"""
        try:
            # Prepare data
            X = np.column_stack([protected_binary, features])
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(X, predictions)
            
            # Return coefficient for protected attribute
            return model.coef_[0]
            
        except Exception as e:
            logger.error(f"Error in direct effect estimation: {e}")
            return 0.0
    
    def _estimate_mediation_effect(
        self,
        predictions: np.ndarray,
        protected_binary: np.ndarray,
        features: np.ndarray
    ) -> float:
        """Estimate mediation effect through other features"""
        try:
            # Step 1: Predict features from protected attribute
            feature_effects = []
            
            for i in range(features.shape[1]):
                if np.std(features[:, i]) > 0:  # Skip constant features
                    model = LinearRegression()
                    model.fit(protected_binary.reshape(-1, 1), features[:, i])
                    feature_effects.append(model.coef_[0])
            
            # Step 2: Predict outcome from features
            outcome_model = LinearRegression()
            outcome_model.fit(features, predictions)
            
            # Step 3: Calculate mediation effect
            mediation_effect = np.sum(
                np.array(feature_effects) * outcome_model.coef_
            )
            
            return mediation_effect
            
        except Exception as e:
            logger.error(f"Error in mediation effect estimation: {e}")
            return 0.0
    
    def _estimate_graph_based_effect(
        self,
        predictions: np.ndarray,
        protected_binary: np.ndarray,
        features: np.ndarray,
        causal_graph: nx.DiGraph
    ) -> float:
        """Estimate causal effect using causal graph"""
        try:
            # This is a simplified implementation
            # In practice, you would use more sophisticated causal inference methods
            
            # Find paths from protected attribute to outcome
            paths = list(nx.all_simple_paths(
                causal_graph, 
                source="protected_attribute", 
                target="outcome"
            ))
            
            if not paths:
                return 0.0
            
            # Calculate effect along each path
            total_effect = 0.0
            
            for path in paths:
                path_effect = 1.0
                
                for i in range(len(path) - 1):
                    source = path[i]
                    target = path[i + 1]
                    
                    # Estimate edge weight (simplified)
                    if source == "protected_attribute":
                        edge_effect = np.corrcoef(protected_binary, predictions)[0, 1]
                    else:
                        # This would need more sophisticated estimation
                        edge_effect = 0.1  # Placeholder
                    
                    path_effect *= edge_effect
                
                total_effect += path_effect
            
            return total_effect
            
        except Exception as e:
            logger.error(f"Error in graph-based effect estimation: {e}")
            return 0.0
    
    def _calculate_counterfactual_difference(
        self,
        predictions: np.ndarray,
        protected_attributes: np.ndarray,
        features: np.ndarray,
        protected_groups: List[str]
    ) -> float:
        """Calculate counterfactual difference between groups"""
        try:
            # Split data by protected groups
            group_predictions = {}
            group_features = {}
            
            for group in protected_groups:
                group_mask = protected_attributes == group
                group_predictions[group] = predictions[group_mask]
                group_features[group] = features[group_mask]
            
            # Calculate counterfactual predictions
            counterfactual_differences = []
            
            for group in protected_groups:
                if len(group_predictions[group]) == 0:
                    continue
                
                # Train model on other groups
                other_groups = [g for g in protected_groups if g != group]
                other_predictions = np.concatenate([group_predictions[g] for g in other_groups])
                other_features = np.vstack([group_features[g] for g in other_groups])
                
                if len(other_predictions) == 0:
                    continue
                
                # Train model
                model = RandomForestRegressor(n_estimators=10, random_state=42)
                model.fit(other_features, other_predictions)
                
                # Predict on current group
                counterfactual_predictions = model.predict(group_features[group])
                actual_predictions = group_predictions[group]
                
                # Calculate difference
                difference = np.mean(np.abs(counterfactual_predictions - actual_predictions))
                counterfactual_differences.append(difference)
            
            # Return average difference
            if counterfactual_differences:
                return np.mean(counterfactual_differences)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating counterfactual difference: {e}")
            return 0.0
    
    def _generate_recommendations(
        self,
        counterfactual_difference: float,
        causal_effects: Dict[str, float],
        is_fair: bool
    ) -> List[str]:
        """Generate recommendations based on counterfactual fairness test"""
        recommendations = []
        
        if not is_fair:
            recommendations.append(
                f"Counterfactual fairness violation detected. "
                f"Difference: {counterfactual_difference:.3f} "
                f"(threshold: {self.significance_threshold})"
            )
            
            # Analyze causal effects
            if "direct_effect" in causal_effects:
                direct_effect = causal_effects["direct_effect"]
                if abs(direct_effect) > 0.1:
                    recommendations.append(
                        f"Large direct causal effect detected: {direct_effect:.3f}. "
                        "Consider removing direct influence of protected attribute."
                    )
            
            if "mediation_effect" in causal_effects:
                mediation_effect = causal_effects["mediation_effect"]
                if abs(mediation_effect) > 0.1:
                    recommendations.append(
                        f"Large mediation effect detected: {mediation_effect:.3f}. "
                        "Consider addressing indirect bias through other features."
                    )
            
            # Specific recommendations based on violation severity
            if counterfactual_difference > 0.5:
                recommendations.append(
                    "Severe counterfactual fairness violation. "
                    "Consider complete model retraining with fairness constraints."
                )
            elif counterfactual_difference > 0.2:
                recommendations.append(
                    "Moderate counterfactual fairness violation. "
                    "Consider post-processing techniques or feature engineering."
                )
            else:
                recommendations.append(
                    "Minor counterfactual fairness violation. "
                    "Consider fine-tuning model parameters."
                )
        else:
            recommendations.append("Counterfactual fairness is satisfied.")
        
        return recommendations
    
    def test_multiple_attributes(
        self,
        predictions: np.ndarray,
        protected_attributes: Dict[str, np.ndarray],
        features: np.ndarray,
        causal_graphs: Optional[Dict[str, nx.DiGraph]] = None
    ) -> Dict[str, CounterfactualFairnessResult]:
        """Test counterfactual fairness for multiple protected attributes"""
        results = {}
        
        for attr_name, attr_values in protected_attributes.items():
            unique_groups = np.unique(attr_values)
            unique_groups = [str(g) for g in unique_groups]
            
            causal_graph = None
            if causal_graphs and attr_name in causal_graphs:
                causal_graph = causal_graphs[attr_name]
            
            result = self.test_counterfactual_fairness(
                predictions, attr_values, features, unique_groups, causal_graph
            )
            results[attr_name] = result
        
        return results
    
    def get_fairness_score(self, counterfactual_difference: float) -> float:
        """Calculate fairness score based on counterfactual difference"""
        # Fairness score: 1 - normalized difference
        max_possible_difference = 1.0  # Maximum possible difference
        fairness_score = 1.0 - (counterfactual_difference / max_possible_difference)
        return max(0.0, fairness_score)
    
    def generate_fairness_report(
        self,
        results: Dict[str, CounterfactualFairnessResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive counterfactual fairness report"""
        report = {
            "total_attributes_tested": len(results),
            "fair_attributes": 0,
            "unfair_attributes": 0,
            "overall_fairness_score": 0.0,
            "attribute_details": {},
            "recommendations": [],
            "summary": ""
        }
        
        fairness_scores = []
        
        for attr_name, result in results.items():
            is_fair = result.is_fair
            fairness_score = self.get_fairness_score(result.counterfactual_difference)
            
            if is_fair:
                report["fair_attributes"] += 1
            else:
                report["unfair_attributes"] += 1
            
            fairness_scores.append(fairness_score)
            
            report["attribute_details"][attr_name] = {
                "is_fair": is_fair,
                "fairness_score": fairness_score,
                "counterfactual_difference": result.counterfactual_difference,
                "causal_effects": result.causal_effects,
                "recommendations": result.recommendations
            }
            
            report["recommendations"].extend(result.recommendations)
        
        # Calculate overall fairness score
        if fairness_scores:
            report["overall_fairness_score"] = np.mean(fairness_scores)
        
        # Generate summary
        if report["unfair_attributes"] == 0:
            report["summary"] = "All protected attributes show counterfactual fairness."
        elif report["fair_attributes"] == 0:
            report["summary"] = "All protected attributes show counterfactual fairness violations."
        else:
            report["summary"] = (
                f"{report['fair_attributes']} attributes are fair, "
                f"{report['unfair_attributes']} attributes show violations."
            )
        
        return report
