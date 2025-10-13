"""
Fairness Testing Coordinator
Coordinates all fairness testing modules
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from .demographic_parity import DemographicParityTester, DemographicParityResult
from .counterfactual_fairness import CounterfactualFairnessTester, CounterfactualFairnessResult
from .bias_detection import BiasDetector, BiasDetectionResult

logger = logging.getLogger(__name__)

@dataclass
class FairnessTestResult:
    """Comprehensive fairness test result"""
    demographic_parity: Optional[DemographicParityResult] = None
    counterfactual_fairness: Optional[CounterfactualFairnessResult] = None
    bias_detection: Optional[Dict[str, BiasDetectionResult]] = None
    overall_fairness_score: float = 0.0
    is_fair: bool = False
    recommendations: List[str] = None
    metadata: Dict[str, Any] = None

class FairnessCoordinator:
    """Coordinates all fairness testing modules"""
    
    def __init__(
        self,
        demographic_parity_threshold: float = 0.1,
        counterfactual_threshold: float = 0.1,
        bias_threshold: float = 0.1
    ):
        self.demographic_parity_tester = DemographicParityTester(demographic_parity_threshold)
        self.counterfactual_tester = CounterfactualFairnessTester(counterfactual_threshold)
        self.bias_detector = BiasDetector(bias_threshold)
        
    def test_fairness(
        self,
        predictions: np.ndarray,
        protected_attributes: np.ndarray,
        features: np.ndarray,
        protected_groups: List[str],
        test_types: List[str] = None
    ) -> FairnessTestResult:
        """
        Run comprehensive fairness tests
        
        Args:
            predictions: Model predictions
            protected_attributes: Protected attribute values
            features: Feature matrix
            protected_groups: List of protected group names
            test_types: List of test types to run
            
        Returns:
            FairnessTestResult with comprehensive results
        """
        if test_types is None:
            test_types = ["demographic_parity", "counterfactual_fairness", "bias_detection"]
        
        logger.info(f"Running fairness tests: {test_types}")
        
        result = FairnessTestResult()
        result.metadata = {
            "test_types": test_types,
            "protected_groups": protected_groups,
            "num_samples": len(predictions)
        }
        
        # Run demographic parity tests
        if "demographic_parity" in test_types:
            try:
                result.demographic_parity = self.demographic_parity_tester.test_demographic_parity(
                    predictions, protected_attributes, protected_groups
                )
            except Exception as e:
                logger.error(f"Error in demographic parity test: {e}")
                result.demographic_parity = None
        
        # Run counterfactual fairness tests
        if "counterfactual_fairness" in test_types:
            try:
                result.counterfactual_fairness = self.counterfactual_tester.test_counterfactual_fairness(
                    predictions, protected_attributes, features, protected_groups
                )
            except Exception as e:
                logger.error(f"Error in counterfactual fairness test: {e}")
                result.counterfactual_fairness = None
        
        # Run bias detection tests
        if "bias_detection" in test_types:
            try:
                result.bias_detection = self.bias_detector.detect_bias(
                    predictions, protected_attributes, features, protected_groups
                )
            except Exception as e:
                logger.error(f"Error in bias detection: {e}")
                result.bias_detection = None
        
        # Calculate overall fairness score
        result.overall_fairness_score = self._calculate_overall_fairness_score(result)
        
        # Determine if overall system is fair
        result.is_fair = self._determine_overall_fairness(result)
        
        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)
        
        logger.info(f"Fairness testing completed. Overall fair: {result.is_fair}")
        return result
    
    def _calculate_overall_fairness_score(self, result: FairnessTestResult) -> float:
        """Calculate overall fairness score from all test results"""
        scores = []
        
        # Demographic parity score
        if result.demographic_parity:
            score = self.demographic_parity_tester.get_fairness_score(
                result.demographic_parity.positive_rates
            )
            scores.append(score)
        
        # Counterfactual fairness score
        if result.counterfactual_fairness:
            score = self.counterfactual_tester.get_fairness_score(
                result.counterfactual_fairness.counterfactual_difference
            )
            scores.append(score)
        
        # Bias detection scores
        if result.bias_detection:
            for bias_result in result.bias_detection.values():
                # Convert bias score to fairness score
                fairness_score = 1.0 - min(bias_result.bias_score, 1.0)
                scores.append(fairness_score)
        
        # Return average score
        if scores:
            return np.mean(scores)
        else:
            return 0.0
    
    def _determine_overall_fairness(self, result: FairnessTestResult) -> bool:
        """Determine if overall system is fair based on all test results"""
        # Check demographic parity
        if result.demographic_parity and not result.demographic_parity.is_fair:
            return False
        
        # Check counterfactual fairness
        if result.counterfactual_fairness and not result.counterfactual_fairness.is_fair:
            return False
        
        # Check bias detection
        if result.bias_detection:
            for bias_result in result.bias_detection.values():
                if bias_result.is_biased:
                    return False
        
        return True
    
    def _generate_recommendations(self, result: FairnessTestResult) -> List[str]:
        """Generate comprehensive recommendations from all test results"""
        recommendations = []
        
        # Demographic parity recommendations
        if result.demographic_parity:
            recommendations.extend(result.demographic_parity.recommendations)
        
        # Counterfactual fairness recommendations
        if result.counterfactual_fairness:
            recommendations.extend(result.counterfactual_fairness.recommendations)
        
        # Bias detection recommendations
        if result.bias_detection:
            for bias_result in result.bias_detection.values():
                recommendations.extend(bias_result.recommendations)
        
        # Remove duplicates and prioritize
        recommendations = list(set(recommendations))
        
        # Prioritize based on severity
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for rec in recommendations:
            if "severe" in rec.lower() or "large" in rec.lower() or "violation" in rec.lower():
                high_priority.append(rec)
            elif "moderate" in rec.lower() or "minor" in rec.lower():
                medium_priority.append(rec)
            else:
                low_priority.append(rec)
        
        return high_priority + medium_priority + low_priority
    
    def test_multiple_attributes(
        self,
        predictions: np.ndarray,
        protected_attributes: Dict[str, np.ndarray],
        features: np.ndarray,
        test_types: List[str] = None
    ) -> Dict[str, FairnessTestResult]:
        """Test fairness for multiple protected attributes"""
        results = {}
        
        for attr_name, attr_values in protected_attributes.items():
            unique_groups = np.unique(attr_values)
            unique_groups = [str(g) for g in unique_groups]
            
            result = self.test_fairness(
                predictions, attr_values, features, unique_groups, test_types
            )
            results[attr_name] = result
        
        return results
    
    def generate_comprehensive_report(
        self,
        results: Dict[str, FairnessTestResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive fairness report for multiple attributes"""
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
            fairness_score = result.overall_fairness_score
            
            if is_fair:
                report["fair_attributes"] += 1
            else:
                report["unfair_attributes"] += 1
            
            fairness_scores.append(fairness_score)
            
            report["attribute_details"][attr_name] = {
                "is_fair": is_fair,
                "fairness_score": fairness_score,
                "demographic_parity": result.demographic_parity.is_fair if result.demographic_parity else None,
                "counterfactual_fairness": result.counterfactual_fairness.is_fair if result.counterfactual_fairness else None,
                "bias_detection": {
                    bias_type: bias_result.is_biased 
                    for bias_type, bias_result in (result.bias_detection or {}).items()
                },
                "recommendations": result.recommendations
            }
            
            report["recommendations"].extend(result.recommendations)
        
        # Calculate overall fairness score
        if fairness_scores:
            report["overall_fairness_score"] = np.mean(fairness_scores)
        
        # Generate summary
        if report["unfair_attributes"] == 0:
            report["summary"] = "All protected attributes show fairness across all tested dimensions."
        elif report["fair_attributes"] == 0:
            report["summary"] = "All protected attributes show fairness violations across tested dimensions."
        else:
            report["summary"] = (
                f"{report['fair_attributes']} attributes are fair, "
                f"{report['unfair_attributes']} attributes show violations."
            )
        
        return report
    
    def visualize_fairness_results(
        self,
        results: Dict[str, FairnessTestResult],
        save_path: Optional[str] = None
    ) -> None:
        """Visualize fairness test results"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create subplots for different aspects
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Overall fairness scores
            attr_names = list(results.keys())
            fairness_scores = [result.overall_fairness_score for result in results.values()]
            
            axes[0, 0].bar(attr_names, fairness_scores)
            axes[0, 0].set_title("Overall Fairness Scores")
            axes[0, 0].set_ylabel("Fairness Score")
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Demographic parity results
            demo_scores = []
            for result in results.values():
                if result.demographic_parity:
                    score = self.demographic_parity_tester.get_fairness_score(
                        result.demographic_parity.positive_rates
                    )
                    demo_scores.append(score)
                else:
                    demo_scores.append(0.0)
            
            axes[0, 1].bar(attr_names, demo_scores)
            axes[0, 1].set_title("Demographic Parity Scores")
            axes[0, 1].set_ylabel("Fairness Score")
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Counterfactual fairness results
            counter_scores = []
            for result in results.values():
                if result.counterfactual_fairness:
                    score = self.counterfactual_tester.get_fairness_score(
                        result.counterfactual_fairness.counterfactual_difference
                    )
                    counter_scores.append(score)
                else:
                    counter_scores.append(0.0)
            
            axes[1, 0].bar(attr_names, counter_scores)
            axes[1, 0].set_title("Counterfactual Fairness Scores")
            axes[1, 0].set_ylabel("Fairness Score")
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot 4: Bias detection results
            bias_scores = []
            for result in results.values():
                if result.bias_detection:
                    scores = [1.0 - min(bias_result.bias_score, 1.0) for bias_result in result.bias_detection.values()]
                    bias_scores.append(np.mean(scores) if scores else 0.0)
                else:
                    bias_scores.append(0.0)
            
            axes[1, 1].bar(attr_names, bias_scores)
            axes[1, 1].set_title("Bias Detection Scores")
            axes[1, 1].set_ylabel("Fairness Score")
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error visualizing fairness results: {e}")
    
    def get_fairness_summary(self, results: Dict[str, FairnessTestResult]) -> str:
        """Get a concise fairness summary"""
        total_attrs = len(results)
        fair_attrs = sum(1 for result in results.values() if result.is_fair)
        unfair_attrs = total_attrs - fair_attrs
        
        if unfair_attrs == 0:
            return f"All {total_attrs} protected attributes show fairness across all tested dimensions."
        elif fair_attrs == 0:
            return f"All {total_attrs} protected attributes show fairness violations."
        else:
            return f"{fair_attrs}/{total_attrs} protected attributes are fair, {unfair_attrs} show violations."
