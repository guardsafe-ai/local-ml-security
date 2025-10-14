"""
A/B Testing Framework for Model Governance
Provides champion/challenger testing, gradual rollout, and statistical significance testing
"""

import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import stats
import json

logger = logging.getLogger(__name__)

class TestStatus(Enum):
    """A/B test status"""
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TestResult(Enum):
    """A/B test result"""
    CHAMPION_WINS = "champion_wins"
    CHALLENGER_WINS = "challenger_wins"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"

@dataclass
class ABTestConfig:
    """A/B test configuration"""
    test_id: str
    champion_model: str
    challenger_model: str
    traffic_split: float = 0.5  # 50% to challenger
    min_sample_size: int = 1000
    max_duration_hours: int = 168  # 1 week
    significance_level: float = 0.05
    min_improvement: float = 0.02  # 2% minimum improvement
    primary_metric: str = "accuracy"
    secondary_metrics: List[str] = None
    
    def __post_init__(self):
        if self.secondary_metrics is None:
            self.secondary_metrics = ["f1_score", "precision", "recall"]

@dataclass
class ABTestResult:
    """A/B test result data"""
    test_id: str
    champion_model: str
    challenger_model: str
    champion_metrics: Dict[str, float]
    challenger_metrics: Dict[str, float]
    statistical_significance: Dict[str, bool]
    p_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    sample_sizes: Dict[str, int]
    test_duration_hours: float
    result: TestResult
    recommendation: str
    timestamp: datetime

class ABTestingService:
    """A/B testing service for model comparison"""
    
    def __init__(self):
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.test_results: Dict[str, ABTestResult] = {}
        self.test_metrics: Dict[str, Dict[str, List[float]]] = {}  # test_id -> model -> metrics
        self.test_predictions: Dict[str, Dict[str, List[Dict]]] = {}  # test_id -> model -> predictions
        
    async def create_test(self, config: ABTestConfig) -> str:
        """Create a new A/B test"""
        try:
            # Validate configuration
            if config.traffic_split <= 0 or config.traffic_split >= 1:
                raise ValueError("Traffic split must be between 0 and 1")
            
            if config.champion_model == config.challenger_model:
                raise ValueError("Champion and challenger models must be different")
            
            # Initialize test data structures
            self.active_tests[config.test_id] = config
            self.test_metrics[config.test_id] = {
                "champion": {metric: [] for metric in [config.primary_metric] + config.secondary_metrics},
                "challenger": {metric: [] for metric in [config.primary_metric] + config.secondary_metrics}
            }
            self.test_predictions[config.test_id] = {
                "champion": [],
                "challenger": []
            }
            
            logger.info(f"✅ [AB_TEST] Created test {config.test_id}: {config.champion_model} vs {config.challenger_model}")
            return config.test_id
            
        except Exception as e:
            logger.error(f"❌ [AB_TEST] Failed to create test: {e}")
            raise
    
    async def record_prediction(self, test_id: str, model_name: str, prediction: Dict[str, Any], 
                              ground_truth: Optional[str] = None) -> bool:
        """Record a prediction for A/B test analysis"""
        try:
            if test_id not in self.active_tests:
                logger.warning(f"⚠️ [AB_TEST] Test {test_id} not found")
                return False
            
            config = self.active_tests[test_id]
            
            # Determine which model this prediction belongs to
            if model_name == config.champion_model:
                model_key = "champion"
            elif model_name == config.challenger_model:
                model_key = "challenger"
            else:
                logger.warning(f"⚠️ [AB_TEST] Model {model_name} not part of test {test_id}")
                return False
            
            # Store prediction
            prediction_data = {
                "prediction": prediction.get("prediction", ""),
                "confidence": prediction.get("confidence", 0.0),
                "probabilities": prediction.get("probabilities", {}),
                "ground_truth": ground_truth,
                "timestamp": datetime.now().isoformat()
            }
            
            self.test_predictions[test_id][model_key].append(prediction_data)
            
            # Calculate metrics if ground truth is available
            if ground_truth:
                await self._update_metrics(test_id, model_key, prediction_data)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ [AB_TEST] Failed to record prediction: {e}")
            return False
    
    async def _update_metrics(self, test_id: str, model_key: str, prediction_data: Dict[str, Any]):
        """Update metrics for a model in the test"""
        try:
            config = self.active_tests[test_id]
            predicted = prediction_data["prediction"]
            ground_truth = prediction_data["ground_truth"]
            confidence = prediction_data["confidence"]
            
            # Calculate accuracy
            accuracy = 1.0 if predicted == ground_truth else 0.0
            self.test_metrics[test_id][model_key]["accuracy"].append(accuracy)
            
            # Calculate confidence-based metrics
            if confidence > 0:
                self.test_metrics[test_id][model_key]["confidence"].append(confidence)
            
            # Calculate precision, recall, F1 if we have enough data
            if len(self.test_metrics[test_id][model_key]["accuracy"]) >= 10:
                await self._calculate_advanced_metrics(test_id, model_key)
                
        except Exception as e:
            logger.warning(f"⚠️ [AB_TEST] Failed to update metrics: {e}")
    
    async def _calculate_advanced_metrics(self, test_id: str, model_key: str):
        """Calculate advanced metrics for a model"""
        try:
            predictions = self.test_predictions[test_id][model_key]
            
            if len(predictions) < 10:
                return
            
            # Extract predictions and ground truth
            y_pred = [p["prediction"] for p in predictions if p["ground_truth"]]
            y_true = [p["ground_truth"] for p in predictions if p["ground_truth"]]
            
            if len(y_pred) < 10:
                return
            
            # Calculate precision, recall, F1
            from sklearn.metrics import precision_recall_fscore_support, accuracy_score
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            accuracy = accuracy_score(y_true, y_pred)
            
            # Update metrics
            self.test_metrics[test_id][model_key]["precision"].append(precision)
            self.test_metrics[test_id][model_key]["recall"].append(recall)
            self.test_metrics[test_id][model_key]["f1_score"].append(f1)
            
        except Exception as e:
            logger.warning(f"⚠️ [AB_TEST] Failed to calculate advanced metrics: {e}")
    
    async def analyze_test(self, test_id: str) -> Optional[ABTestResult]:
        """Analyze A/B test results and determine winner"""
        try:
            if test_id not in self.active_tests:
                logger.warning(f"⚠️ [AB_TEST] Test {test_id} not found")
                return None
            
            config = self.active_tests[test_id]
            
            # Check if we have enough data
            champion_data = self.test_metrics[test_id]["champion"]
            challenger_data = self.test_metrics[test_id]["challenger"]
            
            if not champion_data[config.primary_metric] or not challenger_data[config.primary_metric]:
                logger.warning(f"⚠️ [AB_TEST] Insufficient data for analysis")
                return None
            
            # Calculate summary statistics
            champion_metrics = {}
            challenger_metrics = {}
            statistical_significance = {}
            p_values = {}
            confidence_intervals = {}
            
            for metric in [config.primary_metric] + config.secondary_metrics:
                if metric in champion_data and metric in challenger_data:
                    champ_values = champion_data[metric]
                    chall_values = challenger_data[metric]
                    
                    if len(champ_values) < 10 or len(chall_values) < 10:
                        continue
                    
                    # Calculate means
                    champ_mean = np.mean(champ_values)
                    chall_mean = np.mean(chall_values)
                    
                    champion_metrics[metric] = champ_mean
                    challenger_metrics[metric] = chall_mean
                    
                    # Statistical significance test
                    if len(champ_values) >= 30 and len(chall_values) >= 30:
                        # Use t-test for large samples
                        t_stat, p_value = stats.ttest_ind(champ_values, chall_values)
                        statistical_significance[metric] = p_value < config.significance_level
                        p_values[metric] = p_value
                        
                        # Calculate confidence interval for difference
                        diff = chall_mean - champ_mean
                        se_diff = np.sqrt(np.var(champ_values)/len(champ_values) + np.var(chall_values)/len(chall_values))
                        ci_lower = diff - 1.96 * se_diff
                        ci_upper = diff + 1.96 * se_diff
                        confidence_intervals[metric] = (ci_lower, ci_upper)
                    else:
                        # Use Mann-Whitney U test for small samples
                        u_stat, p_value = stats.mannwhitneyu(champ_values, chall_values, alternative='two-sided')
                        statistical_significance[metric] = p_value < config.significance_level
                        p_values[metric] = p_value
                        confidence_intervals[metric] = (0.0, 0.0)  # Not calculated for small samples
            
            # Determine winner based on primary metric
            primary_metric = config.primary_metric
            if primary_metric in champion_metrics and primary_metric in challenger_metrics:
                champ_score = champion_metrics[primary_metric]
                chall_score = challenger_metrics[primary_metric]
                improvement = chall_score - champ_score
                
                is_significant = statistical_significance.get(primary_metric, False)
                meets_min_improvement = improvement >= config.min_improvement
                
                if is_significant and meets_min_improvement:
                    result = TestResult.CHALLENGER_WINS
                    recommendation = f"Deploy challenger model {config.challenger_model} - {improvement:.3f} improvement in {primary_metric}"
                elif is_significant and improvement < -config.min_improvement:
                    result = TestResult.CHAMPION_WINS
                    recommendation = f"Keep champion model {config.champion_model} - challenger is {abs(improvement):.3f} worse in {primary_metric}"
                else:
                    result = TestResult.INCONCLUSIVE
                    recommendation = f"Inconclusive - improvement {improvement:.3f} not significant or below threshold"
            else:
                result = TestResult.ERROR
                recommendation = "Error - insufficient data for primary metric"
            
            # Create result object
            test_result = ABTestResult(
                test_id=test_id,
                champion_model=config.champion_model,
                challenger_model=config.challenger_model,
                champion_metrics=champion_metrics,
                challenger_metrics=challenger_metrics,
                statistical_significance=statistical_significance,
                p_values=p_values,
                confidence_intervals=confidence_intervals,
                sample_sizes={
                    "champion": len(champion_data[primary_metric]) if primary_metric in champion_data else 0,
                    "challenger": len(challenger_data[primary_metric]) if primary_metric in challenger_data else 0
                },
                test_duration_hours=0.0,  # Would need to track start time
                result=result,
                recommendation=recommendation,
                timestamp=datetime.now()
            )
            
            # Store result
            self.test_results[test_id] = test_result
            
            logger.info(f"✅ [AB_TEST] Analysis complete for {test_id}: {result.value}")
            logger.info(f"📊 [AB_TEST] Recommendation: {recommendation}")
            
            return test_result
            
        except Exception as e:
            logger.error(f"❌ [AB_TEST] Failed to analyze test {test_id}: {e}")
            return None
    
    async def get_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an A/B test"""
        try:
            if test_id not in self.active_tests:
                return None
            
            config = self.active_tests[test_id]
            champion_data = self.test_metrics[test_id]["champion"]
            challenger_data = self.test_metrics[test_id]["challenger"]
            
            status = {
                "test_id": test_id,
                "champion_model": config.champion_model,
                "challenger_model": config.challenger_model,
                "traffic_split": config.traffic_split,
                "champion_samples": len(champion_data.get(config.primary_metric, [])),
                "challenger_samples": len(challenger_data.get(config.primary_metric, [])),
                "status": "running" if test_id in self.active_tests else "completed",
                "has_result": test_id in self.test_results
            }
            
            if test_id in self.test_results:
                result = self.test_results[test_id]
                status.update({
                    "result": result.result.value,
                    "recommendation": result.recommendation,
                    "champion_metrics": result.champion_metrics,
                    "challenger_metrics": result.challenger_metrics
                })
            
            return status
            
        except Exception as e:
            logger.error(f"❌ [AB_TEST] Failed to get test status: {e}")
            return None
    
    async def list_tests(self) -> List[Dict[str, Any]]:
        """List all A/B tests"""
        tests = []
        for test_id in self.active_tests:
            status = await self.get_test_status(test_id)
            if status:
                tests.append(status)
        return tests

# Global A/B testing service instance
ab_testing_service = ABTestingService()
