"""
Demographic Parity Testing
Tests for demographic parity across different groups
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report

logger = logging.getLogger(__name__)

@dataclass
class DemographicParityResult:
    """Demographic parity test result"""
    protected_attribute: str
    groups: List[str]
    positive_rates: Dict[str, float]
    statistical_significance: float
    is_fair: bool
    recommendations: List[str]
    metadata: Dict[str, Any] = None

class DemographicParityTester:
    """Tests demographic parity across protected attributes"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        
    def test_demographic_parity(
        self,
        predictions: np.ndarray,
        protected_attributes: np.ndarray,
        protected_groups: List[str],
        positive_label: int = 1
    ) -> DemographicParityResult:
        """
        Test demographic parity across protected groups
        
        Args:
            predictions: Model predictions
            protected_attributes: Protected attribute values
            protected_groups: List of protected group names
            positive_label: Label considered positive outcome
            
        Returns:
            DemographicParityResult with test results
        """
        logger.info(f"Testing demographic parity for {len(protected_groups)} groups")
        
        # Calculate positive rates for each group
        positive_rates = {}
        group_sizes = {}
        
        for group in protected_groups:
            group_mask = protected_attributes == group
            group_predictions = predictions[group_mask]
            
            if len(group_predictions) > 0:
                positive_rate = np.mean(group_predictions == positive_label)
                positive_rates[group] = positive_rate
                group_sizes[group] = len(group_predictions)
            else:
                positive_rates[group] = 0.0
                group_sizes[group] = 0
                logger.warning(f"No samples found for group: {group}")
        
        # Statistical significance test
        statistical_significance = self._test_statistical_significance(
            predictions, protected_attributes, protected_groups, positive_label
        )
        
        # Check if demographic parity is satisfied
        is_fair = self._check_demographic_parity(positive_rates)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            positive_rates, group_sizes, is_fair
        )
        
        result = DemographicParityResult(
            protected_attribute="demographic_parity",
            groups=protected_groups,
            positive_rates=positive_rates,
            statistical_significance=statistical_significance,
            is_fair=is_fair,
            recommendations=recommendations,
            metadata={
                "group_sizes": group_sizes,
                "significance_level": self.significance_level,
                "positive_label": positive_label
            }
        )
        
        logger.info(f"Demographic parity test completed. Fair: {is_fair}")
        return result
    
    def _test_statistical_significance(
        self,
        predictions: np.ndarray,
        protected_attributes: np.ndarray,
        protected_groups: List[str],
        positive_label: int
    ) -> float:
        """Test statistical significance of differences in positive rates"""
        try:
            # Prepare data for chi-square test
            contingency_table = []
            
            for group in protected_groups:
                group_mask = protected_attributes == group
                group_predictions = predictions[group_mask]
                
                if len(group_predictions) > 0:
                    positive_count = np.sum(group_predictions == positive_label)
                    negative_count = len(group_predictions) - positive_count
                    contingency_table.append([positive_count, negative_count])
                else:
                    contingency_table.append([0, 0])
            
            # Perform chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            return p_value
            
        except Exception as e:
            logger.error(f"Error in statistical significance test: {e}")
            return 1.0  # Return non-significant if error
    
    def _check_demographic_parity(self, positive_rates: Dict[str, float]) -> bool:
        """Check if demographic parity is satisfied"""
        if len(positive_rates) < 2:
            return True
        
        rates = list(positive_rates.values())
        max_rate = max(rates)
        min_rate = min(rates)
        
        # Demographic parity: difference should be small (e.g., < 0.1)
        difference = max_rate - min_rate
        threshold = 0.1
        
        return difference < threshold
    
    def _generate_recommendations(
        self,
        positive_rates: Dict[str, float],
        group_sizes: Dict[str, int],
        is_fair: bool
    ) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if not is_fair:
            # Find groups with highest and lowest positive rates
            sorted_rates = sorted(positive_rates.items(), key=lambda x: x[1])
            lowest_group, lowest_rate = sorted_rates[0]
            highest_group, highest_rate = sorted_rates[-1]
            
            difference = highest_rate - lowest_rate
            
            recommendations.append(
                f"Demographic parity violation detected. "
                f"Difference between {highest_group} ({highest_rate:.3f}) and "
                f"{lowest_group} ({lowest_rate:.3f}) is {difference:.3f}"
            )
            
            # Check for sample size issues
            min_size = min(group_sizes.values())
            if min_size < 100:
                recommendations.append(
                    f"Small sample size detected for some groups. "
                    f"Minimum group size: {min_size}. Consider collecting more data."
                )
            
            # Specific recommendations based on the violation
            if difference > 0.2:
                recommendations.append(
                    "Large demographic parity violation detected. "
                    "Consider retraining the model with fairness constraints."
                )
            elif difference > 0.1:
                recommendations.append(
                    "Moderate demographic parity violation detected. "
                    "Consider post-processing techniques to improve fairness."
                )
            
            # Recommendations for specific groups
            for group, rate in positive_rates.items():
                if rate < 0.1:
                    recommendations.append(
                        f"Very low positive rate for {group} ({rate:.3f}). "
                        f"Investigate potential bias in training data or model."
                    )
                elif rate > 0.9:
                    recommendations.append(
                        f"Very high positive rate for {group} ({rate:.3f}). "
                        f"Investigate potential overfitting to this group."
                    )
        else:
            recommendations.append("Demographic parity is satisfied across all groups.")
        
        return recommendations
    
    def test_multiple_attributes(
        self,
        predictions: np.ndarray,
        protected_attributes: Dict[str, np.ndarray],
        positive_label: int = 1
    ) -> Dict[str, DemographicParityResult]:
        """Test demographic parity for multiple protected attributes"""
        results = {}
        
        for attr_name, attr_values in protected_attributes.items():
            unique_groups = np.unique(attr_values)
            unique_groups = [str(g) for g in unique_groups]
            
            result = self.test_demographic_parity(
                predictions, attr_values, unique_groups, positive_label
            )
            results[attr_name] = result
        
        return results
    
    def get_fairness_score(self, positive_rates: Dict[str, float]) -> float:
        """Calculate fairness score based on demographic parity"""
        if len(positive_rates) < 2:
            return 1.0
        
        rates = list(positive_rates.values())
        max_rate = max(rates)
        min_rate = min(rates)
        
        # Fairness score: 1 - normalized difference
        difference = max_rate - min_rate
        max_possible_difference = 1.0  # Maximum possible difference
        
        fairness_score = 1.0 - (difference / max_possible_difference)
        return max(0.0, fairness_score)
    
    def generate_fairness_report(
        self,
        results: Dict[str, DemographicParityResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive fairness report"""
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
            fairness_score = self.get_fairness_score(result.positive_rates)
            
            if is_fair:
                report["fair_attributes"] += 1
            else:
                report["unfair_attributes"] += 1
            
            fairness_scores.append(fairness_score)
            
            report["attribute_details"][attr_name] = {
                "is_fair": is_fair,
                "fairness_score": fairness_score,
                "positive_rates": result.positive_rates,
                "statistical_significance": result.statistical_significance,
                "recommendations": result.recommendations
            }
            
            report["recommendations"].extend(result.recommendations)
        
        # Calculate overall fairness score
        if fairness_scores:
            report["overall_fairness_score"] = np.mean(fairness_scores)
        
        # Generate summary
        if report["unfair_attributes"] == 0:
            report["summary"] = "All protected attributes show demographic parity."
        elif report["fair_attributes"] == 0:
            report["summary"] = "All protected attributes show demographic parity violations."
        else:
            report["summary"] = (
                f"{report['fair_attributes']} attributes are fair, "
                f"{report['unfair_attributes']} attributes show violations."
            )
        
        return report
