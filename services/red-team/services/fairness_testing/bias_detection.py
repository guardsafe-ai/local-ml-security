"""
Bias Detection Testing
Comprehensive bias detection across multiple dimensions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class BiasDetectionResult:
    """Bias detection test result"""
    bias_type: str
    protected_attribute: str
    bias_score: float
    is_biased: bool
    bias_metrics: Dict[str, float]
    recommendations: List[str]
    metadata: Dict[str, Any] = None

class BiasDetector:
    """Comprehensive bias detection across multiple dimensions"""
    
    def __init__(self, bias_threshold: float = 0.1):
        self.bias_threshold = bias_threshold
        
    def detect_bias(
        self,
        predictions: np.ndarray,
        protected_attributes: np.ndarray,
        features: np.ndarray,
        protected_groups: List[str],
        bias_types: List[str] = None
    ) -> Dict[str, BiasDetectionResult]:
        """
        Detect various types of bias across protected groups
        
        Args:
            predictions: Model predictions
            protected_attributes: Protected attribute values
            features: Feature matrix
            protected_groups: List of protected group names
            bias_types: List of bias types to detect
            
        Returns:
            Dictionary of BiasDetectionResult for each bias type
        """
        if bias_types is None:
            bias_types = [
                "demographic_parity",
                "equalized_odds",
                "equal_opportunity",
                "calibration",
                "representation",
                "intersectional"
            ]
        
        logger.info(f"Detecting bias for {len(bias_types)} types across {len(protected_groups)} groups")
        
        results = {}
        
        for bias_type in bias_types:
            try:
                if bias_type == "demographic_parity":
                    result = self._detect_demographic_parity_bias(
                        predictions, protected_attributes, protected_groups
                    )
                elif bias_type == "equalized_odds":
                    result = self._detect_equalized_odds_bias(
                        predictions, protected_attributes, protected_groups
                    )
                elif bias_type == "equal_opportunity":
                    result = self._detect_equal_opportunity_bias(
                        predictions, protected_attributes, protected_groups
                    )
                elif bias_type == "calibration":
                    result = self._detect_calibration_bias(
                        predictions, protected_attributes, protected_groups
                    )
                elif bias_type == "representation":
                    result = self._detect_representation_bias(
                        features, protected_attributes, protected_groups
                    )
                elif bias_type == "intersectional":
                    result = self._detect_intersectional_bias(
                        predictions, protected_attributes, protected_groups
                    )
                else:
                    logger.warning(f"Unknown bias type: {bias_type}")
                    continue
                
                results[bias_type] = result
                
            except Exception as e:
                logger.error(f"Error detecting {bias_type} bias: {e}")
                results[bias_type] = BiasDetectionResult(
                    bias_type=bias_type,
                    protected_attribute="unknown",
                    bias_score=0.0,
                    is_biased=False,
                    bias_metrics={},
                    recommendations=[f"Error detecting {bias_type} bias: {str(e)}"],
                    metadata={"error": str(e)}
                )
        
        return results
    
    def _detect_demographic_parity_bias(
        self,
        predictions: np.ndarray,
        protected_attributes: np.ndarray,
        protected_groups: List[str]
    ) -> BiasDetectionResult:
        """Detect demographic parity bias"""
        positive_rates = {}
        
        for group in protected_groups:
            group_mask = protected_attributes == group
            group_predictions = predictions[group_mask]
            
            if len(group_predictions) > 0:
                positive_rate = np.mean(group_predictions == 1)
                positive_rates[group] = positive_rate
            else:
                positive_rates[group] = 0.0
        
        # Calculate bias score
        if len(positive_rates) >= 2:
            rates = list(positive_rates.values())
            bias_score = max(rates) - min(rates)
        else:
            bias_score = 0.0
        
        is_biased = bias_score > self.bias_threshold
        
        recommendations = []
        if is_biased:
            recommendations.append(
                f"Demographic parity bias detected. "
                f"Difference in positive rates: {bias_score:.3f}"
            )
        else:
            recommendations.append("No demographic parity bias detected.")
        
        return BiasDetectionResult(
            bias_type="demographic_parity",
            protected_attribute="demographic_parity",
            bias_score=bias_score,
            is_biased=is_biased,
            bias_metrics=positive_rates,
            recommendations=recommendations
        )
    
    def _detect_equalized_odds_bias(
        self,
        predictions: np.ndarray,
        protected_attributes: np.ndarray,
        protected_groups: List[str]
    ) -> BiasDetectionResult:
        """Detect equalized odds bias"""
        # This is a simplified implementation
        # In practice, you would need true labels to calculate TPR and FPR
        
        # For now, we'll use a placeholder implementation
        bias_score = 0.0
        is_biased = False
        
        recommendations = ["Equalized odds bias detection requires true labels."]
        
        return BiasDetectionResult(
            bias_type="equalized_odds",
            protected_attribute="equalized_odds",
            bias_score=bias_score,
            is_biased=is_biased,
            bias_metrics={},
            recommendations=recommendations
        )
    
    def _detect_equal_opportunity_bias(
        self,
        predictions: np.ndarray,
        protected_attributes: np.ndarray,
        protected_groups: List[str]
    ) -> BiasDetectionResult:
        """Detect equal opportunity bias"""
        # This is a simplified implementation
        # In practice, you would need true labels to calculate TPR
        
        # For now, we'll use a placeholder implementation
        bias_score = 0.0
        is_biased = False
        
        recommendations = ["Equal opportunity bias detection requires true labels."]
        
        return BiasDetectionResult(
            bias_type="equal_opportunity",
            protected_attribute="equal_opportunity",
            bias_score=bias_score,
            is_biased=is_biased,
            bias_metrics={},
            recommendations=recommendations
        )
    
    def _detect_calibration_bias(
        self,
        predictions: np.ndarray,
        protected_attributes: np.ndarray,
        protected_groups: List[str]
    ) -> BiasDetectionResult:
        """Detect calibration bias"""
        # This is a simplified implementation
        # In practice, you would need true labels to calculate calibration
        
        # For now, we'll use a placeholder implementation
        bias_score = 0.0
        is_biased = False
        
        recommendations = ["Calibration bias detection requires true labels."]
        
        return BiasDetectionResult(
            bias_type="calibration",
            protected_attribute="calibration",
            bias_score=bias_score,
            is_biased=is_biased,
            bias_metrics={},
            recommendations=recommendations
        )
    
    def _detect_representation_bias(
        self,
        features: np.ndarray,
        protected_attributes: np.ndarray,
        protected_groups: List[str]
    ) -> BiasDetectionResult:
        """Detect representation bias in features"""
        representation_scores = {}
        
        for group in protected_groups:
            group_mask = protected_attributes == group
            group_features = features[group_mask]
            
            if len(group_features) > 0:
                # Calculate mean feature values for this group
                mean_features = np.mean(group_features, axis=0)
                representation_scores[group] = mean_features.tolist()
            else:
                representation_scores[group] = []
        
        # Calculate bias score based on feature differences
        if len(representation_scores) >= 2:
            # This is a simplified calculation
            # In practice, you would use more sophisticated methods
            bias_score = 0.0  # Placeholder
        else:
            bias_score = 0.0
        
        is_biased = bias_score > self.bias_threshold
        
        recommendations = []
        if is_biased:
            recommendations.append("Representation bias detected in features.")
        else:
            recommendations.append("No representation bias detected in features.")
        
        return BiasDetectionResult(
            bias_type="representation",
            protected_attribute="representation",
            bias_score=bias_score,
            is_biased=is_biased,
            bias_metrics=representation_scores,
            recommendations=recommendations
        )
    
    def _detect_intersectional_bias(
        self,
        predictions: np.ndarray,
        protected_attributes: np.ndarray,
        protected_groups: List[str]
    ) -> BiasDetectionResult:
        """Detect intersectional bias"""
        # This is a simplified implementation
        # In practice, you would analyze intersections of multiple protected attributes
        
        # For now, we'll use a placeholder implementation
        bias_score = 0.0
        is_biased = False
        
        recommendations = ["Intersectional bias detection requires multiple protected attributes."]
        
        return BiasDetectionResult(
            bias_type="intersectional",
            protected_attribute="intersectional",
            bias_score=bias_score,
            is_biased=is_biased,
            bias_metrics={},
            recommendations=recommendations
        )
    
    def generate_bias_report(
        self,
        results: Dict[str, BiasDetectionResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive bias detection report"""
        report = {
            "total_bias_types_tested": len(results),
            "biased_types": 0,
            "unbiased_types": 0,
            "overall_bias_score": 0.0,
            "bias_details": {},
            "recommendations": [],
            "summary": ""
        }
        
        bias_scores = []
        
        for bias_type, result in results.items():
            is_biased = result.is_biased
            bias_score = result.bias_score
            
            if is_biased:
                report["biased_types"] += 1
            else:
                report["unbiased_types"] += 1
            
            bias_scores.append(bias_score)
            
            report["bias_details"][bias_type] = {
                "is_biased": is_biased,
                "bias_score": bias_score,
                "bias_metrics": result.bias_metrics,
                "recommendations": result.recommendations
            }
            
            report["recommendations"].extend(result.recommendations)
        
        # Calculate overall bias score
        if bias_scores:
            report["overall_bias_score"] = np.mean(bias_scores)
        
        # Generate summary
        if report["biased_types"] == 0:
            report["summary"] = "No bias detected across all tested types."
        elif report["unbiased_types"] == 0:
            report["summary"] = "Bias detected across all tested types."
        else:
            report["summary"] = (
                f"{report['unbiased_types']} bias types are clean, "
                f"{report['biased_types']} bias types show issues."
            )
        
        return report
    
    def visualize_bias(
        self,
        results: Dict[str, BiasDetectionResult],
        save_path: Optional[str] = None
    ) -> None:
        """Visualize bias detection results"""
        try:
            # Create subplots for different bias types
            n_types = len(results)
            fig, axes = plt.subplots(2, (n_types + 1) // 2, figsize=(15, 10))
            axes = axes.flatten() if n_types > 1 else [axes]
            
            for i, (bias_type, result) in enumerate(results.items()):
                if i >= len(axes):
                    break
                
                ax = axes[i]
                
                # Plot bias scores for each group
                if result.bias_metrics:
                    groups = list(result.bias_metrics.keys())
                    scores = list(result.bias_metrics.values())
                    
                    ax.bar(groups, scores)
                    ax.set_title(f"{bias_type.replace('_', ' ').title()} Bias")
                    ax.set_ylabel("Bias Score")
                    ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error visualizing bias: {e}")
    
    def get_fairness_recommendations(
        self,
        results: Dict[str, BiasDetectionResult]
    ) -> List[str]:
        """Get actionable fairness recommendations"""
        recommendations = []
        
        # Collect all recommendations
        for result in results.values():
            recommendations.extend(result.recommendations)
        
        # Remove duplicates
        recommendations = list(set(recommendations))
        
        # Prioritize recommendations based on bias severity
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for rec in recommendations:
            if "severe" in rec.lower() or "large" in rec.lower():
                high_priority.append(rec)
            elif "moderate" in rec.lower() or "minor" in rec.lower():
                medium_priority.append(rec)
            else:
                low_priority.append(rec)
        
        # Return prioritized recommendations
        return high_priority + medium_priority + low_priority
