"""
Drift Detection Service
Monitors data and model drift using statistical tests
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from sklearn.exceptions import ConvergenceWarning
# Targeted warning suppression - only suppress specific warnings that are expected
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

# Initialize logger
logger = logging.getLogger(__name__)
logger.debug("🔇 [WARNINGS] Applied targeted warning filters for drift detection")

from .email_notifications import email_service

# Import metrics from main analytics service
import sys
import os
sys.path.append('/app')
try:
    from main import DRIFT_DETECTION_FAILURES, ML_OPERATION_DURATION
except ImportError:
    # Fallback if import fails
    from prometheus_client import Counter, Histogram
    DRIFT_DETECTION_FAILURES = Counter('drift_detection_failures_total', 'Drift detection errors', ['detection_type', 'error_type'])
    ML_OPERATION_DURATION = Histogram('ml_operation_duration_seconds', 'ML operation duration', ['operation_type', 'model_name'], buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0])

def convert_numpy_types(obj):
    """
    Convert numpy types to JSON-serializable Python types
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

logger = logging.getLogger(__name__)

@dataclass
class DriftConfig:
    """Configuration for drift detection"""
    # Statistical test thresholds
    ks_threshold: float = 0.05  # Kolmogorov-Smirnov test p-value threshold
    chi2_threshold: float = 0.05  # Chi-square test p-value threshold
    psi_threshold: float = 0.2  # Population Stability Index threshold
    
    # Drift severity levels
    psi_minor_threshold: float = 0.1
    psi_moderate_threshold: float = 0.2
    psi_severe_threshold: float = 0.25
    
    # Monitoring windows
    reference_window_days: int = 30
    detection_window_days: int = 7
    min_samples: int = 100
    
    # Model performance thresholds
    accuracy_drop_threshold: float = 0.05  # 5% accuracy drop
    f1_drop_threshold: float = 0.05  # 5% F1 score drop

class DriftDetector:
    """Detects data and model drift using statistical methods"""
    
    def __init__(self, config: DriftConfig = None):
        self.config = config or DriftConfig()
        self.reference_data = {}
        self.drift_history = []
        self.sliding_windows = {}  # model_name -> sliding window data
        self.baseline_metrics = {}
        self.baseline_established = False
        self.baseline_sample_count = 0
        self.min_baseline_samples = 100  # Minimum samples needed to establish baseline
    
    async def establish_baseline(self, model_name: str, reference_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Establish baseline metrics for drift detection
        
        Args:
            model_name: Name of the model
            reference_data: Reference dataset (if None, uses production data)
        
        Returns:
            Baseline metrics and status
        """
        try:
            logger.info(f"📊 [BASELINE] Establishing baseline for model {model_name}")
            
            if reference_data is None:
                # Use production data from last 7 days to establish baseline
                reference_data = await self.get_production_inference_data(model_name, hours=168)  # 7 days
            
            if len(reference_data) < self.min_baseline_samples:
                logger.warning(f"⚠️ [BASELINE] Insufficient data for baseline: {len(reference_data)} < {self.min_baseline_samples}")
                return {
                    "status": "insufficient_data",
                    "sample_count": len(reference_data),
                    "min_required": self.min_baseline_samples
                }
            
            # Calculate baseline metrics
            baseline_metrics = {
                "model_name": model_name,
                "sample_count": len(reference_data),
                "established_at": datetime.now().isoformat(),
                "data_distribution": {},
                "performance_metrics": {},
                "feature_statistics": {}
            }
            
            # Calculate data distribution metrics
            if 'input_text' in reference_data.columns:
                text_lengths = reference_data['input_text'].str.len()
                baseline_metrics["data_distribution"] = {
                    "text_length_mean": float(text_lengths.mean()),
                    "text_length_std": float(text_lengths.std()),
                    "text_length_min": int(text_lengths.min()),
                    "text_length_max": int(text_lengths.max()),
                    "text_length_median": float(text_lengths.median())
                }
            
            # Calculate performance metrics if available
            if 'confidence' in reference_data.columns:
                confidences = reference_data['confidence']
                baseline_metrics["performance_metrics"] = {
                    "confidence_mean": float(confidences.mean()),
                    "confidence_std": float(confidences.std()),
                    "confidence_min": float(confidences.min()),
                    "confidence_max": float(confidences.max())
                }
            
            # Calculate prediction distribution
            if 'prediction' in reference_data.columns:
                pred_counts = reference_data['prediction'].value_counts()
                baseline_metrics["prediction_distribution"] = pred_counts.to_dict()
            
            # Store baseline
            self.baseline_metrics[model_name] = baseline_metrics
            self.baseline_established = True
            self.baseline_sample_count = len(reference_data)
            
            # Store reference data for comparison
            self.reference_data[model_name] = reference_data
            
            logger.info(f"✅ [BASELINE] Established baseline for {model_name}: {len(reference_data)} samples")
            logger.info(f"📊 [BASELINE] Text length: {baseline_metrics['data_distribution'].get('text_length_mean', 'N/A'):.1f} ± {baseline_metrics['data_distribution'].get('text_length_std', 'N/A'):.1f}")
            
            return {
                "status": "established",
                "baseline_metrics": baseline_metrics
            }
            
        except Exception as e:
            logger.error(f"❌ [BASELINE] Error establishing baseline for {model_name}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_baseline_status(self, model_name: str) -> Dict[str, Any]:
        """Get current baseline status for a model"""
        if model_name not in self.baseline_metrics:
            return {
                "status": "not_established",
                "baseline_established": False
            }
        
        baseline = self.baseline_metrics[model_name]
        return {
            "status": "established",
            "baseline_established": True,
            "sample_count": baseline["sample_count"],
            "established_at": baseline["established_at"],
            "data_distribution": baseline.get("data_distribution", {}),
            "performance_metrics": baseline.get("performance_metrics", {})
        }
    
    def is_baseline_established(self, model_name: str) -> bool:
        """Check if baseline is established for a model"""
        return model_name in self.baseline_metrics and self.baseline_established
        
    async def get_production_inference_data(self, model_name: str, hours: int = 24) -> pd.DataFrame:
        """Get production inference data from prediction logs"""
        try:
            from database.connection import db_manager
            
            query = """
                SELECT input_text, prediction, confidence, created_at, model_name
                FROM predictions
                WHERE model_name = %s 
                AND created_at > NOW() - INTERVAL '%s hours'
                ORDER BY created_at DESC
                LIMIT 10000
            """
            
            # Use monitored method for database query
            rows = await db_manager.fetch_many(query, model_name, hours)
            
            if not rows:
                logger.warning(f"No production inference data found for {model_name}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for row in rows:
                data.append({
                    "input_text": row["input_text"],
                    "prediction": row["prediction"],
                        "confidence": float(row["confidence"]) if row["confidence"] else 0.0,
                        "created_at": row["created_at"],
                        "model_name": row["model_name"]
                    })
                
                df = pd.DataFrame(data)
                logger.info(f"Retrieved {len(df)} production inference samples for {model_name}")
                return df
                
        except Exception as e:
            logger.error(f"Error getting production inference data: {e}")
            return pd.DataFrame()
    
    def update_sliding_window(self, model_name: str, new_data: pd.DataFrame, window_size: int = 1000):
        """Update sliding window with new data"""
        if model_name not in self.sliding_windows:
            self.sliding_windows[model_name] = []
        
        # Add new data
        self.sliding_windows[model_name].extend(new_data.to_dict('records'))
        
        # Keep only the most recent window_size samples
        if len(self.sliding_windows[model_name]) > window_size:
            self.sliding_windows[model_name] = self.sliding_windows[model_name][-window_size:]
        
        logger.info(f"Updated sliding window for {model_name}: {len(self.sliding_windows[model_name])} samples")
    
    def get_sliding_window_data(self, model_name: str) -> pd.DataFrame:
        """Get current sliding window data for a model"""
        if model_name not in self.sliding_windows:
            return pd.DataFrame()
        
        return pd.DataFrame(self.sliding_windows[model_name])
    
    async def detect_data_drift_with_production_data(self, model_name: str, hours: int = 24) -> Dict[str, Any]:
        """Detect data drift using production inference data"""
        try:
            # Get production inference data
            production_data = await self.get_production_inference_data(model_name, hours)
            
            if production_data.empty:
                return {
                    "error": "No production inference data available",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Update sliding window
            self.update_sliding_window(model_name, production_data)
            
            # Get reference data (use sliding window if available, otherwise stored reference)
            reference_data = self.get_sliding_window_data(model_name)
            if reference_data.empty:
                reference_data = self.reference_data.get('data')
                if reference_data is None:
                    return {"error": "No reference data available for drift detection"}
            
            # Prepare data for drift detection
            # Convert text to numerical features for drift detection
            current_features = self._extract_text_features(production_data)
            reference_features = self._extract_text_features(reference_data)
            
            # Perform drift detection
            drift_results = self.detect_data_drift(current_features, reference_features)
            
            # Add production data context
            drift_results.update({
                "model_name": model_name,
                "production_samples": len(production_data),
                "reference_samples": len(reference_data),
                "data_source": "production_inference",
                "time_window_hours": hours
            })
            
            return drift_results
            
        except Exception as e:
            from utils.enhanced_logging import log_error_with_context
            log_error_with_context(
                error=e,
                operation="drift_detection_production_data",
                model_name=model_name,
                additional_context={"hours": hours, "data_source": "production_inference"}
            )
            return {
                "error": f"Drift detection failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_text_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract numerical features from text data for drift detection with robust error handling"""
        try:
            # Input validation
            if data is None or data.empty:
                logger.warning("⚠️ [DRIFT] Empty data provided for feature extraction")
                return pd.DataFrame()
            
            if len(data) < 10:
                logger.warning(f"⚠️ [DRIFT] Insufficient data for feature extraction: {len(data)} samples")
                return pd.DataFrame()
            
            features = pd.DataFrame()
            
            # Validate required columns
            if 'input_text' not in data.columns:
                logger.warning("⚠️ [DRIFT] Missing 'input_text' column for feature extraction")
                return pd.DataFrame()
            
            # Clean and validate input text
            data = data.copy()
            data['input_text'] = data['input_text'].fillna('').astype(str)
            
            # Filter out empty or very short texts
            valid_text_mask = (data['input_text'].str.len() >= 3) & (data['input_text'].str.strip() != '')
            if valid_text_mask.sum() < 5:
                logger.warning("⚠️ [DRIFT] Insufficient valid text samples after filtering")
                return pd.DataFrame()
            
            data = data[valid_text_mask].reset_index(drop=True)
            
            # Text length features with bounds checking
            features['text_length'] = data['input_text'].str.len().clip(0, 10000)  # Cap at 10k chars
            features['word_count'] = data['input_text'].str.split().str.len().clip(0, 2000)  # Cap at 2k words
            features['sentence_count'] = data['input_text'].str.count(r'[.!?]+').clip(0, 100)  # Cap at 100 sentences
            
            # Character features with safe division
            word_count_safe = features['word_count'].replace(0, 1)  # Avoid division by zero
            text_length_safe = features['text_length'].replace(0, 1)  # Avoid division by zero
            
            features['avg_word_length'] = (features['text_length'] / word_count_safe).clip(0, 50)  # Cap at 50 chars/word
            features['special_char_ratio'] = (data['input_text'].str.count(r'[^a-zA-Z0-9\s]') / text_length_safe).clip(0, 1)
            features['digit_ratio'] = (data['input_text'].str.count(r'\d') / text_length_safe).clip(0, 1)
            features['uppercase_ratio'] = (data['input_text'].str.count(r'[A-Z]') / text_length_safe).clip(0, 1)
            
            # Additional robust features
            features['has_numbers'] = data['input_text'].str.contains(r'\d', na=False).astype(int)
            features['has_special_chars'] = data['input_text'].str.contains(r'[^a-zA-Z0-9\s]', na=False).astype(int)
            features['is_short'] = (features['text_length'] < 20).astype(int)
            features['is_long'] = (features['text_length'] > 500).astype(int)
            
            # Confidence features with validation
            if 'confidence' in data.columns:
                confidence_data = pd.to_numeric(data['confidence'], errors='coerce')
                features['confidence'] = confidence_data.fillna(0.5).clip(0, 1)  # Default to 0.5, clip to [0,1]
            else:
                features['confidence'] = 0.5  # Default confidence
            
            # Prediction distribution features with validation
            if 'prediction' in data.columns:
                prediction_counts = data['prediction'].value_counts()
                total_predictions = len(data)
                
                for pred in prediction_counts.index:
                    if isinstance(pred, str) and len(pred) < 50:  # Sanitize prediction names
                        ratio = prediction_counts[pred] / total_predictions
                        features[f'prediction_{pred}_ratio'] = ratio.clip(0, 1)
            
            # Final validation and cleanup
            features = features.fillna(0)
            
            # Remove any infinite values
            features = features.replace([np.inf, -np.inf], 0)
            
            # Ensure all values are finite
            features = features[np.isfinite(features).all(axis=1)]
            
            if features.empty:
                logger.warning("⚠️ [DRIFT] No valid features extracted after processing")
                return pd.DataFrame()
            
            logger.info(f"✅ [DRIFT] Extracted {len(features)} features from {len(data)} samples")
            return features
            
        except Exception as e:
            logger.error(f"❌ [DRIFT] Error extracting text features: {e}")
            return pd.DataFrame()
        
    def detect_data_drift(self, current_data: pd.DataFrame, 
                         reference_data: pd.DataFrame = None,
                         feature_columns: List[str] = None,
                         model_name: str = None) -> Dict[str, Any]:
        """
        Detect data drift between reference and current data with baseline comparison
        
        Args:
            current_data: Current data to test
            reference_data: Reference data (if None, uses stored reference or baseline)
            feature_columns: Columns to test for drift
            model_name: Model name for baseline lookup
            
        Returns:
            Drift detection results
        """
        import time
        start_time = time.time()
        try:
            # Use baseline data if available and no reference data provided
            if reference_data is None and model_name and self.is_baseline_established(model_name):
                reference_data = self.reference_data.get(model_name)
                logger.info(f"📊 [DRIFT] Using baseline data for {model_name}: {len(reference_data)} samples")
            elif reference_data is None:
                reference_data = self.reference_data.get('data')
                if reference_data is None:
                    return {"error": "No reference data available"}
            
            if feature_columns is None:
                feature_columns = current_data.select_dtypes(include=[np.number]).columns.tolist()
            
            drift_results = {
                "timestamp": datetime.now().isoformat(),
                "total_features": len(feature_columns),
                "drifted_features": [],
                "drift_summary": {},
                "statistical_tests": {}
            }
            
            for feature in feature_columns:
                if feature not in reference_data.columns or feature not in current_data.columns:
                    continue
                
                # Get feature data
                ref_data = reference_data[feature].dropna()
                curr_data = current_data[feature].dropna()
                
                if len(ref_data) < self.config.min_samples or len(curr_data) < self.config.min_samples:
                    continue
                
                # Perform statistical tests
                ks_stat, ks_pvalue = stats.ks_2samp(ref_data, curr_data)
                chi2_stat, chi2_pvalue = self._chi2_test(ref_data, curr_data)
                psi_value = self._calculate_psi(ref_data, curr_data)
                
                # Determine drift
                is_drifted = (
                    ks_pvalue < self.config.ks_threshold or
                    chi2_pvalue < self.config.chi2_threshold or
                    psi_value > self.config.psi_threshold
                )
                
                drift_severity = self._get_drift_severity(psi_value)
                
                feature_result = {
                    "feature": feature,
                    "is_drifted": is_drifted,
                    "drift_severity": drift_severity,
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_pvalue,
                    "chi2_statistic": chi2_stat,
                    "chi2_pvalue": chi2_pvalue,
                    "psi_value": psi_value,
                    "reference_mean": ref_data.mean(),
                    "current_mean": curr_data.mean(),
                    "reference_std": ref_data.std(),
                    "current_std": curr_data.std()
                }
                
                drift_results["statistical_tests"][feature] = feature_result
                
                if is_drifted:
                    drift_results["drifted_features"].append(feature)
            
            # Calculate overall drift summary
            drift_results["drift_summary"] = {
                "total_drifted_features": len(drift_results["drifted_features"]),
                "drift_percentage": len(drift_results["drifted_features"]) / len(feature_columns) * 100 if feature_columns else 0,
                "severe_drift_features": [
                    f for f, result in drift_results["statistical_tests"].items()
                    if result["drift_severity"] == "severe"
                ],
                "moderate_drift_features": [
                    f for f, result in drift_results["statistical_tests"].items()
                    if result["drift_severity"] == "moderate"
                ],
                "minor_drift_features": [
                    f for f, result in drift_results["statistical_tests"].items()
                    if result["drift_severity"] == "minor"
                ]
            }
            
            # Store drift history
            self.drift_history.append(drift_results)
            
            # Send email notification if drift detected
            if len(drift_results['drifted_features']) > 0:
                logger.info(f"📧 [EMAIL] Sending drift alert for {len(drift_results['drifted_features'])} drifted features")
                email_service.send_drift_alert(drift_results, "ML Security Model")
            
            logger.info(f"Data drift detection completed: {len(drift_results['drifted_features'])}/{len(feature_columns)} features drifted")
            
            # Convert numpy types to JSON-serializable types
            return convert_numpy_types(drift_results)
            
        except Exception as e:
            logger.error(f"Error detecting data drift: {e}")
            
            # Record failure metrics
            DRIFT_DETECTION_FAILURES.labels(
                detection_type="data_drift",
                error_type=type(e).__name__
            ).inc()
            ML_OPERATION_DURATION.labels(
                operation_type="drift_detection",
                model_name="unknown"
            ).observe(time.time() - start_time)
            
            return {"error": str(e)}
        finally:
            # Record success metrics
            ML_OPERATION_DURATION.labels(
                operation_type="drift_detection",
                model_name="unknown"
            ).observe(time.time() - start_time)
    
    def detect_semantic_drift(self, current_texts: List[str], reference_texts: List[str] = None) -> Dict[str, Any]:
        """
        Detect semantic drift in text data using embedding-based similarity
        
        Args:
            current_texts: Current text data to test
            reference_texts: Reference text data (if None, uses stored reference)
            
        Returns:
            Semantic drift detection results
        """
        import time
        start_time = time.time()
        
        try:
            if reference_texts is None:
                reference_texts = self.reference_data.get('texts', [])
                if not reference_texts:
                    return {"error": "No reference text data available"}
            
            if len(current_texts) < 10 or len(reference_texts) < 10:
                return {"error": "Insufficient text data for semantic drift detection"}
            
            # Create TF-IDF vectors for both datasets
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            # Fit on reference data and transform both datasets
            ref_vectors = vectorizer.fit_transform(reference_texts)
            curr_vectors = vectorizer.transform(current_texts)
            
            # Calculate mean vectors for each dataset
            ref_mean = np.mean(ref_vectors.toarray(), axis=0)
            curr_mean = np.mean(curr_vectors.toarray(), axis=0)
            
            # Calculate cosine similarity between mean vectors
            cosine_sim = cosine_similarity([ref_mean], [curr_mean])[0][0]
            semantic_drift_score = 1 - cosine_sim  # Higher score = more drift
            
            # Calculate distribution similarity using JS divergence
            js_divergence = self._calculate_js_divergence(ref_vectors, curr_vectors)
            
            # Determine drift severity
            drift_detected = semantic_drift_score > 0.1 or js_divergence > 0.1
            severity = "high" if semantic_drift_score > 0.2 or js_divergence > 0.2 else "medium" if drift_detected else "low"
            
            # Calculate vocabulary overlap
            ref_vocab = set(vectorizer.get_feature_names_out())
            curr_vocab = set(vectorizer.get_feature_names_out())
            vocab_overlap = len(ref_vocab & curr_vocab) / len(ref_vocab | curr_vocab) if ref_vocab | curr_vocab else 0
            
            results = {
                "timestamp": datetime.now().isoformat(),
                "drift_detected": drift_detected,
                "semantic_drift_score": float(semantic_drift_score),
                "cosine_similarity": float(cosine_sim),
                "js_divergence": float(js_divergence),
                "vocab_overlap": float(vocab_overlap),
                "severity": severity,
                "reference_samples": len(reference_texts),
                "current_samples": len(current_texts),
                "processing_time_ms": (time.time() - start_time) * 1000
            }
            
            logger.info(f"✅ [SEMANTIC DRIFT] Score: {semantic_drift_score:.3f}, JS Divergence: {js_divergence:.3f}, Severity: {severity}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ [SEMANTIC DRIFT] Error detecting semantic drift: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_js_divergence(self, ref_vectors, curr_vectors, bins=10):
        """Calculate Jensen-Shannon divergence between vector distributions"""
        try:
            # Convert sparse matrices to dense for calculation
            ref_dense = ref_vectors.toarray()
            curr_dense = curr_vectors.toarray()
            
            # Calculate mean vectors
            ref_mean = np.mean(ref_dense, axis=0)
            curr_mean = np.mean(curr_dense, axis=0)
            
            # Normalize to probability distributions
            ref_prob = ref_mean / (np.sum(ref_mean) + 1e-10)
            curr_prob = curr_mean / (np.sum(curr_mean) + 1e-10)
            
            # Calculate JS divergence
            m = 0.5 * (ref_prob + curr_prob)
            kl_ref = np.sum(ref_prob * np.log((ref_prob + 1e-10) / (m + 1e-10)))
            kl_curr = np.sum(curr_prob * np.log((curr_prob + 1e-10) / (m + 1e-10)))
            js_div = 0.5 * (kl_ref + kl_curr)
            
            return js_div
            
        except Exception as e:
            logger.warning(f"⚠️ [JS DIVERGENCE] Error calculating JS divergence: {e}")
            return 0.0
    
    def detect_model_performance_drift(self, old_model_predictions: List[Dict[str, Any]], 
                                     new_model_predictions: List[Dict[str, Any]], 
                                     ground_truth: List[str] = None) -> Dict[str, Any]:
        """
        Compare performance between old and new models on the same data
        
        Args:
            old_model_predictions: Predictions from the old model
            new_model_predictions: Predictions from the new model  
            ground_truth: True labels for accuracy calculation (optional)
            
        Returns:
            Model performance comparison results
        """
        try:
            # Extract prediction data
            old_preds = [pred.get('prediction', '') for pred in old_model_predictions]
            new_preds = [pred.get('prediction', '') for pred in new_model_predictions]
            old_confidences = [pred.get('confidence', 0.0) for pred in old_model_predictions]
            new_confidences = [pred.get('confidence', 0.0) for pred in new_model_predictions]
            
            # Calculate prediction distribution changes
            from collections import Counter
            old_dist = Counter(old_preds)
            new_dist = Counter(new_preds)
            
            # Calculate confidence statistics
            old_avg_conf = np.mean(old_confidences) if old_confidences else 0.0
            new_avg_conf = np.mean(new_confidences) if new_confidences else 0.0
            confidence_change = new_avg_conf - old_avg_conf
            
            # Calculate prediction agreement
            agreement = sum(1 for old, new in zip(old_preds, new_preds) if old == new) / len(old_preds) if old_preds else 0.0
            
            # Calculate performance metrics if ground truth available
            performance_metrics = {}
            if ground_truth and len(ground_truth) == len(old_preds):
                # Old model performance
                old_accuracy = accuracy_score(ground_truth, old_preds)
                old_precision, old_recall, old_f1, _ = precision_recall_fscore_support(
                    ground_truth, old_preds, average='weighted', zero_division=0
                )
                
                # New model performance
                new_accuracy = accuracy_score(ground_truth, new_preds)
                new_precision, new_recall, new_f1, _ = precision_recall_fscore_support(
                    ground_truth, new_preds, average='weighted', zero_division=0
                )
                
                # Calculate improvements
                accuracy_improvement = new_accuracy - old_accuracy
                f1_improvement = new_f1 - old_f1
                precision_improvement = new_precision - old_precision
                recall_improvement = new_recall - old_recall
                
                performance_metrics = {
                    "old_model": {
                        "accuracy": old_accuracy,
                        "precision": old_precision,
                        "recall": old_recall,
                        "f1_score": old_f1
                    },
                    "new_model": {
                        "accuracy": new_accuracy,
                        "precision": new_precision,
                        "recall": new_recall,
                        "f1_score": new_f1
                    },
                    "improvements": {
                        "accuracy": accuracy_improvement,
                        "precision": precision_improvement,
                        "recall": recall_improvement,
                        "f1_score": f1_improvement
                    }
                }
            
            # Determine if new model is significantly better
            is_improved = False
            if performance_metrics:
                f1_improvement = performance_metrics["improvements"]["f1_score"]
                is_improved = f1_improvement > 0.05  # 5% improvement threshold
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "prediction_agreement": agreement,
                "confidence_change": confidence_change,
                "old_model_avg_confidence": old_avg_conf,
                "new_model_avg_confidence": new_avg_conf,
                "prediction_distributions": {
                    "old_model": dict(old_dist),
                    "new_model": dict(new_dist)
                },
                "performance_metrics": performance_metrics,
                "is_improved": is_improved,
                "recommendation": "use_new_model" if is_improved else "keep_old_model"
            }
            
            # Send email notification for model performance changes
            if result.get("is_improved") or abs(result.get("confidence_change", 0)) > 0.1:
                logger.info(f"📧 [EMAIL] Sending model performance alert")
                email_service.send_model_performance_alert(result, "ML Security Model")
            
            # Convert numpy types to JSON-serializable types
            return convert_numpy_types(result)
            
        except Exception as e:
            logger.error(f"Error detecting model performance drift: {e}")
            return {"error": str(e)}
    
    def calculate_comprehensive_metrics(self, y_true: List[str], y_pred: List[str], 
                                      y_prob: Optional[List[float]] = None,
                                      model_name: str = "unknown") -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics for model evaluation
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            model_name: Name of the model
            
        Returns:
            Dictionary with comprehensive metrics
        """
        try:
            # Basic classification metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            
            # Per-class metrics
            precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            # Classification report
            class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            # Get unique labels for per-class analysis
            unique_labels = sorted(list(set(y_true + y_pred)))
            
            # Per-class analysis
            per_class_metrics = {}
            for i, label in enumerate(unique_labels):
                if i < len(precision_per_class):
                    per_class_metrics[label] = {
                        "precision": float(precision_per_class[i]),
                        "recall": float(recall_per_class[i]),
                        "f1_score": float(f1_per_class[i]),
                        "support": int(support_per_class[i])
                    }
            
            # Confusion matrix as list of lists for JSON serialization
            conf_matrix_list = conf_matrix.tolist()
            
            # Calculate class distribution
            class_distribution = {}
            for label in unique_labels:
                class_distribution[label] = {
                    "true_count": int(sum(1 for y in y_true if y == label)),
                    "pred_count": int(sum(1 for y in y_pred if y == label))
                }
            
            # Probability-based metrics (if probabilities available)
            prob_metrics = {}
            if y_prob is not None and len(y_prob) == len(y_true):
                try:
                    # Convert labels to binary for ROC calculation
                    # Assuming we're doing binary classification or use the most common class
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y_true_binary = le.fit_transform(y_true)
                    y_pred_binary = le.transform(y_pred)
                    
                    # ROC AUC
                    if len(unique_labels) == 2:
                        roc_auc = roc_auc_score(y_true_binary, y_prob)
                        prob_metrics["roc_auc"] = float(roc_auc)
                        
                        # ROC curve
                        fpr, tpr, _ = roc_curve(y_true_binary, y_prob)
                        prob_metrics["roc_curve"] = {
                            "fpr": fpr.tolist(),
                            "tpr": tpr.tolist()
                        }
                    
                    # Precision-Recall curve
                    precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, y_prob)
                    avg_precision = average_precision_score(y_true_binary, y_prob)
                    prob_metrics["precision_recall_curve"] = {
                        "precision": precision_curve.tolist(),
                        "recall": recall_curve.tolist()
                    }
                    prob_metrics["average_precision"] = float(avg_precision)
                    
                    # Calibration curve
                    prob_true, prob_pred = calibration_curve(y_true_binary, y_prob, n_bins=10)
                    prob_metrics["calibration_curve"] = {
                        "prob_true": prob_true.tolist(),
                        "prob_pred": prob_pred.tolist()
                    }
                    
                except Exception as e:
                    logger.warning(f"⚠️ Could not calculate probability metrics: {e}")
                    prob_metrics["error"] = str(e)
            
            # Overall metrics summary
            metrics_summary = {
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
                "overall_metrics": {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "support": int(sum(support))
                },
                "per_class_metrics": per_class_metrics,
                "confusion_matrix": conf_matrix_list,
                "classification_report": class_report,
                "class_distribution": class_distribution,
                "probability_metrics": prob_metrics,
                "data_summary": {
                    "total_samples": len(y_true),
                    "unique_labels": len(unique_labels),
                    "labels": unique_labels
                }
            }
            
            logger.info(f"✅ [METRICS] Calculated comprehensive metrics for {model_name}")
            logger.info(f"📊 [METRICS] Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            return metrics_summary
            
        except Exception as e:
            logger.error(f"❌ [METRICS] Failed to calculate comprehensive metrics: {e}")
            return {
                "error": str(e),
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }

    def detect_model_drift(self, current_predictions: List[Dict[str, Any]], 
                          reference_predictions: List[Dict[str, Any]] = None,
                          current_labels: List[str] = None,
                          reference_labels: List[str] = None) -> Dict[str, Any]:
        """
        Detect model performance drift
        
        Args:
            current_predictions: Current model predictions
            reference_predictions: Reference predictions (if None, uses stored reference)
            current_labels: Current true labels
            reference_labels: Reference true labels
            
        Returns:
            Model drift detection results
        """
        try:
            if reference_predictions is None:
                reference_predictions = self.reference_data.get('predictions')
                if reference_predictions is None:
                    return {"error": "No reference predictions available"}
            
            # Extract prediction data
            curr_preds = [pred.get('prediction', '') for pred in current_predictions]
            ref_preds = [pred.get('prediction', '') for pred in reference_predictions]
            
            # Calculate performance metrics if labels available
            performance_drift = {}
            if current_labels and reference_labels:
                # Current performance
                curr_accuracy = accuracy_score(current_labels, curr_preds)
                curr_precision, curr_recall, curr_f1, _ = precision_recall_fscore_support(
                    current_labels, curr_preds, average='weighted', zero_division=0
                )
                
                # Reference performance
                ref_accuracy = accuracy_score(reference_labels, ref_preds)
                ref_precision, ref_recall, ref_f1, _ = precision_recall_fscore_support(
                    reference_labels, ref_preds, average='weighted', zero_division=0
                )
                
                # Calculate performance drops
                accuracy_drop = ref_accuracy - curr_accuracy
                f1_drop = ref_f1 - curr_f1
                precision_drop = ref_precision - curr_precision
                recall_drop = ref_recall - curr_recall
                
                # Determine if there's significant performance drift
                performance_drift = {
                    "current_accuracy": curr_accuracy,
                    "reference_accuracy": ref_accuracy,
                    "accuracy_drop": accuracy_drop,
                    "current_f1": curr_f1,
                    "reference_f1": ref_f1,
                    "f1_drop": f1_drop,
                    "current_precision": curr_precision,
                    "reference_precision": ref_precision,
                    "precision_drop": precision_drop,
                    "current_recall": curr_recall,
                    "reference_recall": ref_recall,
                    "recall_drop": recall_drop,
                    "significant_accuracy_drop": accuracy_drop > self.config.accuracy_drop_threshold,
                    "significant_f1_drop": f1_drop > self.config.f1_drop_threshold,
                    "overall_performance_drift": (
                        accuracy_drop > self.config.accuracy_drop_threshold or
                        f1_drop > self.config.f1_drop_threshold
                    )
                }
            
            # Calculate prediction distribution drift
            curr_dist = self._calculate_prediction_distribution(curr_preds)
            ref_dist = self._calculate_prediction_distribution(ref_preds)
            
            # Chi-square test for prediction distribution
            chi2_stat, chi2_pvalue = self._chi2_test_distributions(ref_dist, curr_dist)
            
            # Calculate PSI for prediction distribution
            psi_value = self._calculate_psi_distributions(ref_dist, curr_dist)
            
            drift_results = {
                "timestamp": datetime.now().isoformat(),
                "performance_drift": performance_drift,
                "prediction_distribution_drift": {
                    "chi2_statistic": chi2_stat,
                    "chi2_pvalue": chi2_pvalue,
                    "psi_value": psi_value,
                    "is_drifted": chi2_pvalue < self.config.chi2_threshold or psi_value > self.config.psi_threshold,
                    "current_distribution": curr_dist,
                    "reference_distribution": ref_dist
                },
                "overall_model_drift": (
                    performance_drift.get("overall_performance_drift", False) or
                    (chi2_pvalue < self.config.chi2_threshold or psi_value > self.config.psi_threshold)
                )
            }
            
            # Store drift history
            self.drift_history.append(drift_results)
            
            logger.info(f"Model drift detection completed: {drift_results['overall_model_drift']}")
            
            # Convert numpy types to JSON-serializable types
            return convert_numpy_types(drift_results)
            
        except Exception as e:
            logger.error(f"Error detecting model drift: {e}")
            return {"error": str(e)}
    
    def _chi2_test(self, ref_data: np.ndarray, curr_data: np.ndarray) -> Tuple[float, float]:
        """Perform chi-square test for categorical data"""
        try:
            # Create bins for continuous data
            if ref_data.dtype in ['float64', 'int64']:
                bins = np.histogram_bin_edges(np.concatenate([ref_data, curr_data]), bins=10)
                ref_binned = np.histogram(ref_data, bins=bins)[0]
                curr_binned = np.histogram(curr_data, bins=bins)[0]
            else:
                # For categorical data
                ref_binned = ref_data.value_counts().values
                curr_binned = curr_data.value_counts().values
            
            # Ensure same length
            min_len = min(len(ref_binned), len(curr_binned))
            ref_binned = ref_binned[:min_len]
            curr_binned = curr_binned[:min_len]
            
            # Add small value to avoid zero division
            ref_binned = ref_binned + 1e-10
            curr_binned = curr_binned + 1e-10
            
            chi2_stat, p_value = stats.chisquare(curr_binned, ref_binned)
            return chi2_stat, p_value
            
        except Exception as e:
            logger.error(f"Error in chi-square test: {e}")
            return 0.0, 1.0
    
    def _calculate_psi(self, ref_data: np.ndarray, curr_data: np.ndarray) -> float:
        """Calculate Population Stability Index (PSI) - CORRECTED FORMULA"""
        try:
            # Create bins
            bins = np.histogram_bin_edges(np.concatenate([ref_data, curr_data]), bins=10)
            
            # Calculate distributions
            ref_dist = np.histogram(ref_data, bins=bins)[0]
            curr_dist = np.histogram(curr_data, bins=bins)[0]
            
            # Normalize to probabilities (percentages)
            ref_total = np.sum(ref_dist)
            curr_total = np.sum(curr_dist)
            
            if ref_total == 0 or curr_total == 0:
                return 0.0
            
            ref_probs = ref_dist / ref_total
            curr_probs = curr_dist / curr_total
            
            # Add small value to avoid log(0) and division by zero
            ref_probs = ref_probs + 1e-10
            curr_probs = curr_probs + 1e-10
            
            # Calculate PSI using correct formula: PSI = Σ((Actual% - Expected%) * ln(Actual% / Expected%))
            psi = np.sum((curr_probs - ref_probs) * np.log(curr_probs / ref_probs))
            
            return psi
            
        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return 0.0
    
    def _get_drift_severity(self, psi_value: float) -> str:
        """Determine drift severity based on PSI value"""
        if psi_value < self.config.psi_minor_threshold:
            return "none"
        elif psi_value < self.config.psi_moderate_threshold:
            return "minor"
        elif psi_value < self.config.psi_severe_threshold:
            return "moderate"
        else:
            return "severe"
    
    def _calculate_prediction_distribution(self, predictions: List[str]) -> Dict[str, int]:
        """Calculate distribution of predictions"""
        from collections import Counter
        return dict(Counter(predictions))
    
    def _chi2_test_distributions(self, ref_dist: Dict[str, int], curr_dist: Dict[str, int]) -> Tuple[float, float]:
        """Chi-square test for prediction distributions"""
        try:
            # Get all unique labels
            all_labels = set(ref_dist.keys()) | set(curr_dist.keys())
            
            # Create arrays for chi-square test
            ref_array = np.array([ref_dist.get(label, 0) for label in all_labels])
            curr_array = np.array([curr_dist.get(label, 0) for label in all_labels])
            
            # Add small value to avoid zero division
            ref_array = ref_array + 1e-10
            curr_array = curr_array + 1e-10
            
            chi2_stat, p_value = stats.chisquare(curr_array, ref_array)
            return chi2_stat, p_value
            
        except Exception as e:
            logger.error(f"Error in chi-square test for distributions: {e}")
            return 0.0, 1.0
    
    def _calculate_psi_distributions(self, ref_dist: Dict[str, int], curr_dist: Dict[str, int]) -> float:
        """Calculate PSI for prediction distributions"""
        try:
            # Get all unique labels
            all_labels = set(ref_dist.keys()) | set(curr_dist.keys())
            
            # Calculate probabilities
            ref_total = sum(ref_dist.values())
            curr_total = sum(curr_dist.values())
            
            ref_probs = np.array([ref_dist.get(label, 0) / ref_total for label in all_labels])
            curr_probs = np.array([curr_dist.get(label, 0) / curr_total for label in all_labels])
            
            # Add small value to avoid log(0)
            ref_probs = ref_probs + 1e-10
            curr_probs = curr_probs + 1e-10
            
            # Calculate PSI
            psi = np.sum((curr_probs - ref_probs) * np.log(curr_probs / ref_probs))
            
            return psi
            
        except Exception as e:
            logger.error(f"Error calculating PSI for distributions: {e}")
            return 0.0
    
    def set_reference_data(self, data: pd.DataFrame, predictions: List[Dict[str, Any]] = None):
        """Set reference data for drift detection"""
        self.reference_data = {
            "data": data,
            "predictions": predictions,
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"Reference data set with {len(data)} samples")
    
    def get_drift_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get drift detection history for the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        history = []
        for drift_result in self.drift_history:
            if datetime.fromisoformat(drift_result["timestamp"]) >= cutoff_date:
                history.append(drift_result)
        
        return history
    
    def get_drift_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary of drift detection results"""
        history = self.get_drift_history(days)
        
        if not history:
            return {"message": "No drift detection history available"}
        
        # Count drift events
        data_drift_events = sum(1 for result in history if "drifted_features" in result and result["drifted_features"])
        model_drift_events = sum(1 for result in history if result.get("overall_model_drift", False))
        
        # Calculate average drift metrics
        avg_drift_percentage = np.mean([
            result.get("drift_summary", {}).get("drift_percentage", 0)
            for result in history if "drift_summary" in result
        ])
        
        return {
            "period_days": days,
            "total_checks": len(history),
            "data_drift_events": data_drift_events,
            "model_drift_events": model_drift_events,
            "average_drift_percentage": avg_drift_percentage,
            "drift_frequency": {
                "data_drift_rate": data_drift_events / len(history) if history else 0,
                "model_drift_rate": model_drift_events / len(history) if history else 0
            },
            "last_check": history[-1]["timestamp"] if history else None
        }

    async def trigger_retraining_if_drift(
        self, 
        drift_results: Dict,
        model_name: str,
        training_data_path: str = "latest"
    ) -> Dict[str, Any]:
        """
        Automatically trigger retraining if severe drift detected
        
        Args:
            drift_results: Results from detect_data_drift() or detect_model_drift()
            model_name: Name of the model to retrain
            training_data_path: Path to training data (default: "latest")
        
        Returns:
            Dict with retraining status and job details
        """
        import httpx
        
        # Determine drift severity
        psi_max = 0.0
        severe_drift_count = 0
        
        if "statistical_tests" in drift_results:
            # Data drift results
            psi_values = [
                test.get("psi", 0) 
                for test in drift_results["statistical_tests"].values()
            ]
            psi_max = max(psi_values) if psi_values else 0.0
            severe_drift_count = len(drift_results.get("severe_drift_features", []))
        elif "performance_drift" in drift_results:
            # Model drift results
            perf_drift = drift_results["performance_drift"]
            if perf_drift.get("accuracy_drop", 0) > self.config.accuracy_drop_threshold:
                psi_max = 0.3  # Simulate high drift for model performance drop
                severe_drift_count = 1
        
        # Check if retraining is needed
        should_retrain = (
            psi_max > self.config.psi_severe_threshold or 
            severe_drift_count > 2
        )
        
        if not should_retrain:
            return {
                "retraining_triggered": False,
                "reason": "Drift below threshold",
                "psi_max": psi_max,
                "severe_drift_count": severe_drift_count
            }
        
        logger.warning(
            f"🚨 Severe drift detected for {model_name} "
            f"(PSI: {psi_max:.3f}, severe features: {severe_drift_count}). "
            f"Triggering automatic retraining..."
        )
        
        try:
            # Use transaction for drift detection and retraining operations
            from database.connection import db_manager
            
            async with db_manager.transaction() as conn:
                # Store drift detection results
                drift_id = await conn.fetchval(
                    """
                    INSERT INTO analytics.drift_detections 
                    (model_name, drift_score, drift_type, detected_at, details)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                    """,
                    model_name,
                    psi_max,
                    "data_drift",
                    datetime.now(),
                    json.dumps(drift_results)
                )
                
                # Call training service to submit retraining job
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        "http://training:8002/queue/submit",
                        json={
                            "model_name": model_name,
                            "training_data_path": training_data_path,
                            "config": {
                                "reason": "drift_detected",
                                "priority": "HIGH",
                                "drift_info": {
                                    "psi_max": psi_max,
                                    "severe_drift_count": severe_drift_count,
                                    "detection_timestamp": datetime.now().isoformat(),
                                    "drift_id": drift_id
                                }
                            }
                        }
                    )
                    
                    if response.status_code == 200:
                        job_data = response.json()
                        
                        # Store retraining job reference
                        await conn.execute(
                            """
                            INSERT INTO analytics.retrain_jobs 
                            (drift_id, job_id, model_name, status, created_at)
                            VALUES ($1, $2, $3, $4, $5)
                            """,
                            drift_id, job_data.get("job_id"), model_name, "pending", datetime.now()
                        )
                        
                        logger.info(f"✅ Retraining job submitted: {job_data.get('job_id')}")
                        
                        return {
                            "retraining_triggered": True,
                            "job_id": job_data.get("job_id"),
                            "drift_id": drift_id,
                            "psi_max": psi_max,
                            "severe_drift_count": severe_drift_count,
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        logger.error(f"Failed to submit retraining job: {response.text}")
                        return {
                            "retraining_triggered": False,
                            "error": f"Training service returned {response.status_code}",
                            "psi_max": psi_max,
                            "severe_drift_count": severe_drift_count
                        }
                    
        except Exception as e:
            logger.error(f"Error triggering retraining: {e}")
            return {
                "retraining_triggered": False,
                "error": str(e),
                "psi_max": psi_max,
                "severe_drift_count": severe_drift_count
            }

# Global drift detector instance
drift_detector = DriftDetector()
