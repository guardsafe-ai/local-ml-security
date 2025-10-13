"""
Business Metrics Service - Drift Detection
Model drift detection using statistical and ML methods
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.ensemble import IsolationForest
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects model drift using statistical and ML methods"""
    
    def __init__(self):
        self.reference_data = {}
        self.drift_models = {}
        self.drift_threshold = 0.1
    
    def set_reference_data(self, model_name: str, data: pd.DataFrame):
        """Set reference data for drift detection"""
        self.reference_data[model_name] = data
        self.drift_models[model_name] = IsolationForest(contamination=0.1)
        self.drift_models[model_name].fit(data)
        logger.info(f"Set reference data for model {model_name}")
    
    def detect_drift(self, model_name: str, current_data: pd.DataFrame) -> Dict:
        """Detect drift in model performance"""
        if model_name not in self.reference_data:
            return {
                "model_name": model_name,
                "drift_detected": False,
                "drift_score": 0.0,
                "confidence_interval": (0.0, 0.0),
                "last_drift_check": datetime.now(),
                "features_drifted": [],
                "severity": "low"
            }
        
        try:
            # Statistical drift detection
            reference = self.reference_data[model_name]
            drift_scores = []
            drifted_features = []
            
            # Check each feature for drift
            for column in current_data.columns:
                if column in reference.columns:
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_pvalue = stats.ks_2samp(
                        reference[column].dropna(),
                        current_data[column].dropna()
                    )
                    
                    if ks_pvalue < 0.05:  # Significant drift
                        drift_scores.append(ks_stat)
                        drifted_features.append(column)
            
            # ML-based drift detection
            if model_name in self.drift_models:
                drift_predictions = self.drift_models[model_name].predict(current_data)
                ml_drift_score = np.mean(drift_predictions == -1)  # Outliers
            else:
                ml_drift_score = 0.0
            
            # Combine scores
            overall_drift_score = np.mean(drift_scores) if drift_scores else 0.0
            combined_score = (overall_drift_score + ml_drift_score) / 2
            
            # Determine severity
            if combined_score > 0.3:
                severity = "critical"
            elif combined_score > 0.2:
                severity = "high"
            elif combined_score > 0.1:
                severity = "medium"
            else:
                severity = "low"
            
            return {
                "model_name": model_name,
                "drift_detected": combined_score > self.drift_threshold,
                "drift_score": float(combined_score),
                "confidence_interval": (0.0, 1.0),  # Simplified
                "last_drift_check": datetime.now(),
                "features_drifted": drifted_features,
                "severity": severity
            }
            
        except Exception as e:
            logger.error(f"Error detecting drift for {model_name}: {e}")
            return {
                "model_name": model_name,
                "drift_detected": False,
                "drift_score": 0.0,
                "confidence_interval": (0.0, 0.0),
                "last_drift_check": datetime.now(),
                "features_drifted": [],
                "severity": "low"
            }
