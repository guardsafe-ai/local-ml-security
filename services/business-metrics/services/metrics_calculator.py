"""
Business Metrics Service - Metrics Calculator
Calculates various business metrics and KPIs
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculates business metrics and KPIs"""
    
    def __init__(self):
        self.cost_per_hour = 0.5  # USD per hour for compute
        self.storage_cost_per_gb = 0.023  # USD per GB per month
        self.api_cost_per_1k = 0.002  # USD per 1000 API calls
    
    def calculate_attack_success_rate(self, attacks_data: List[Dict], 
                                    time_range_days: int = 30) -> Dict:
        """Calculate attack success rate metrics"""
        try:
            cutoff_date = datetime.now() - timedelta(days=time_range_days)
            recent_attacks = [
                attack for attack in attacks_data 
                if datetime.fromisoformat(attack.get('timestamp', '')) >= cutoff_date
            ]
            
            if not recent_attacks:
                return {
                    "total_attacks": 0,
                    "successful_attacks": 0,
                    "success_rate": 0.0,
                    "by_category": {},
                    "by_model": {},
                    "trend_7d": 0.0,
                    "trend_30d": 0.0
                }
            
            total_attacks = len(recent_attacks)
            successful_attacks = sum(1 for attack in recent_attacks if attack.get('success', False))
            success_rate = successful_attacks / total_attacks if total_attacks > 0 else 0.0
            
            # Calculate by category
            by_category = {}
            for attack in recent_attacks:
                category = attack.get('category', 'unknown')
                if category not in by_category:
                    by_category[category] = {'total': 0, 'successful': 0}
                by_category[category]['total'] += 1
                if attack.get('success', False):
                    by_category[category]['successful'] += 1
            
            for category in by_category:
                total = by_category[category]['total']
                successful = by_category[category]['successful']
                by_category[category] = successful / total if total > 0 else 0.0
            
            # Calculate by model
            by_model = {}
            for attack in recent_attacks:
                model = attack.get('model_name', 'unknown')
                if model not in by_model:
                    by_model[model] = {'total': 0, 'successful': 0}
                by_model[model]['total'] += 1
                if attack.get('success', False):
                    by_model[model]['successful'] += 1
            
            for model in by_model:
                total = by_model[model]['total']
                successful = by_model[model]['successful']
                by_model[model] = successful / total if total > 0 else 0.0
            
            # Calculate trends (simplified)
            trend_7d = self._calculate_trend(recent_attacks, 7)
            trend_30d = self._calculate_trend(recent_attacks, 30)
            
            return {
                "total_attacks": total_attacks,
                "successful_attacks": successful_attacks,
                "success_rate": success_rate,
                "by_category": by_category,
                "by_model": by_model,
                "trend_7d": trend_7d,
                "trend_30d": trend_30d
            }
            
        except Exception as e:
            logger.error(f"Error calculating attack success rate: {e}")
            return {
                "total_attacks": 0,
                "successful_attacks": 0,
                "success_rate": 0.0,
                "by_category": {},
                "by_model": {},
                "trend_7d": 0.0,
                "trend_30d": 0.0
            }
    
    def calculate_cost_metrics(self, usage_data: List[Dict], 
                             time_range_days: int = 30) -> Dict:
        """Calculate cost metrics"""
        try:
            cutoff_date = datetime.now() - timedelta(days=time_range_days)
            recent_usage = [
                usage for usage in usage_data 
                if datetime.fromisoformat(usage.get('timestamp', '')) >= cutoff_date
            ]
            
            # Calculate compute costs
            total_hours = sum(usage.get('compute_hours', 0) for usage in recent_usage)
            compute_cost = total_hours * self.cost_per_hour
            
            # Calculate storage costs
            total_storage_gb = sum(usage.get('storage_gb', 0) for usage in recent_usage)
            storage_cost = total_storage_gb * self.storage_cost_per_gb
            
            # Calculate API costs
            total_api_calls = sum(usage.get('api_calls', 0) for usage in recent_usage)
            api_calls_cost = (total_api_calls / 1000) * self.api_cost_per_1k
            
            # Calculate training costs (simplified)
            training_hours = sum(usage.get('training_hours', 0) for usage in recent_usage)
            model_training_cost = training_hours * self.cost_per_hour * 2  # 2x for training
            
            total_cost = compute_cost + storage_cost + api_calls_cost + model_training_cost
            
            # Calculate cost per prediction
            total_predictions = sum(usage.get('predictions', 0) for usage in recent_usage)
            cost_per_prediction = total_cost / total_predictions if total_predictions > 0 else 0.0
            
            # Calculate trends (simplified)
            cost_trend_7d = self._calculate_cost_trend(recent_usage, 7)
            cost_trend_30d = self._calculate_cost_trend(recent_usage, 30)
            
            return {
                "total_cost_usd": total_cost,
                "compute_cost": compute_cost,
                "storage_cost": storage_cost,
                "api_calls_cost": api_calls_cost,
                "model_training_cost": model_training_cost,
                "cost_per_prediction": cost_per_prediction,
                "cost_trend_7d": cost_trend_7d,
                "cost_trend_30d": cost_trend_30d
            }
            
        except Exception as e:
            logger.error(f"Error calculating cost metrics: {e}")
            return {
                "total_cost_usd": 0.0,
                "compute_cost": 0.0,
                "storage_cost": 0.0,
                "api_calls_cost": 0.0,
                "model_training_cost": 0.0,
                "cost_per_prediction": 0.0,
                "cost_trend_7d": 0.0,
                "cost_trend_30d": 0.0
            }
    
    def calculate_system_effectiveness(self, performance_data: List[Dict], 
                                     time_range_days: int = 30) -> Dict:
        """Calculate system effectiveness metrics"""
        try:
            cutoff_date = datetime.now() - timedelta(days=time_range_days)
            recent_performance = [
                perf for perf in performance_data 
                if datetime.fromisoformat(perf.get('timestamp', '')) >= cutoff_date
            ]
            
            if not recent_performance:
                return {
                    "overall_effectiveness": 0.0,
                    "detection_accuracy": 0.0,
                    "false_positive_rate": 0.0,
                    "false_negative_rate": 0.0,
                    "response_time_p95": 0.0,
                    "availability_percent": 0.0,
                    "user_satisfaction_score": 0.0
                }
            
            # Calculate detection accuracy
            total_predictions = sum(perf.get('total_predictions', 0) for perf in recent_performance)
            correct_predictions = sum(perf.get('correct_predictions', 0) for perf in recent_performance)
            detection_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
            # Calculate false positive/negative rates
            false_positives = sum(perf.get('false_positives', 0) for perf in recent_performance)
            false_negatives = sum(perf.get('false_negatives', 0) for perf in recent_performance)
            true_negatives = sum(perf.get('true_negatives', 0) for perf in recent_performance)
            true_positives = sum(perf.get('true_positives', 0) for perf in recent_performance)
            
            false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
            false_negative_rate = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0.0
            
            # Calculate response time P95
            response_times = [perf.get('response_time', 0) for perf in recent_performance if perf.get('response_time')]
            response_time_p95 = np.percentile(response_times, 95) if response_times else 0.0
            
            # Calculate availability
            total_uptime = sum(perf.get('uptime_seconds', 0) for perf in recent_performance)
            total_time = time_range_days * 24 * 3600  # seconds
            availability_percent = (total_uptime / total_time) * 100 if total_time > 0 else 0.0
            
            # Calculate user satisfaction (simplified)
            user_satisfaction_score = min(100.0, detection_accuracy * 100 - false_positive_rate * 50)
            
            # Calculate overall effectiveness
            overall_effectiveness = (
                detection_accuracy * 0.4 +
                (1 - false_positive_rate) * 0.3 +
                (1 - false_negative_rate) * 0.2 +
                (availability_percent / 100) * 0.1
            )
            
            return {
                "overall_effectiveness": overall_effectiveness,
                "detection_accuracy": detection_accuracy,
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
                "response_time_p95": response_time_p95,
                "availability_percent": availability_percent,
                "user_satisfaction_score": user_satisfaction_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating system effectiveness: {e}")
            return {
                "overall_effectiveness": 0.0,
                "detection_accuracy": 0.0,
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0,
                "response_time_p95": 0.0,
                "availability_percent": 0.0,
                "user_satisfaction_score": 0.0
            }
    
    def _calculate_trend(self, data: List[Dict], days: int) -> float:
        """Calculate trend for the given number of days (simplified)"""
        # This is a simplified implementation
        # In a real system, you'd calculate actual trends
        return 0.0
    
    def _calculate_cost_trend(self, data: List[Dict], days: int) -> float:
        """Calculate cost trend for the given number of days (simplified)"""
        # This is a simplified implementation
        # In a real system, you'd calculate actual trends
        return 0.0
