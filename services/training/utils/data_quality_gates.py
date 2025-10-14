"""
Data Quality Gates
Comprehensive data quality validation with configurable thresholds
"""

import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class QualityThresholds:
    """Configurable quality thresholds for data validation"""
    # Class balance thresholds
    max_imbalance_ratio: float = 10.0  # Max ratio between largest and smallest class
    min_class_samples: int = 10  # Minimum samples per class
    max_class_samples: int = 1000000  # Maximum samples per class (prevent memory issues)
    
    # Duplicate detection
    max_duplicate_rate: float = 5.0  # Max percentage of duplicate samples
    max_exact_duplicates: int = 100  # Max number of exact duplicates
    
    # Text quality
    min_text_length: int = 1  # Minimum text length
    max_text_length: int = 50000  # Maximum text length
    min_unique_words: int = 1  # Minimum unique words per text
    max_avg_text_length: int = 10000  # Maximum average text length
    
    # Data completeness
    max_missing_rate: float = 5.0  # Max percentage of missing values
    min_total_samples: int = 10  # Minimum total samples
    max_total_samples: int = 10000000  # Maximum total samples
    
    # Label quality
    min_label_length: int = 1  # Minimum label length
    max_label_length: int = 100  # Maximum label length
    allowed_labels: Optional[List[str]] = None  # Allowed label values
    
    # Statistical quality
    max_outlier_ratio: float = 10.0  # Max percentage of outliers
    min_variance: float = 0.001  # Minimum variance for numerical features

class DataQualityValidator:
    """Validates data quality against configurable thresholds"""
    
    def __init__(self, thresholds: QualityThresholds = None):
        self.thresholds = thresholds or QualityThresholds()
        self.validation_results = {}
    
    def validate_dataset(self, texts: List[str], labels: List[str], 
                        dataset_name: str = "dataset") -> Dict[str, Any]:
        """
        Comprehensive dataset quality validation
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels
            dataset_name: Name of the dataset for logging
        
        Returns:
            Validation results with pass/fail status and detailed metrics
        """
        logger.info(f"🔍 [QUALITY] Starting quality validation for {dataset_name}")
        
        validation_results = {
            "dataset_name": dataset_name,
            "total_samples": len(texts),
            "validation_passed": True,
            "errors": [],
            "warnings": [],
            "metrics": {},
            "recommendations": []
        }
        
        try:
            # Basic validation
            self._validate_basic_requirements(texts, labels, validation_results)
            if not validation_results["validation_passed"]:
                return validation_results
            
            # Data completeness validation
            self._validate_data_completeness(texts, labels, validation_results)
            
            # Text quality validation
            self._validate_text_quality(texts, validation_results)
            
            # Label quality validation
            self._validate_label_quality(labels, validation_results)
            
            # Class balance validation
            self._validate_class_balance(labels, validation_results)
            
            # Duplicate detection
            self._validate_duplicates(texts, labels, validation_results)
            
            # Statistical quality validation
            self._validate_statistical_quality(texts, labels, validation_results)
            
            # Generate recommendations
            self._generate_recommendations(validation_results)
            
            logger.info(f"✅ [QUALITY] Validation completed for {dataset_name}: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ [QUALITY] Error during validation: {e}")
            validation_results["validation_passed"] = False
            validation_results["errors"].append(f"Validation error: {str(e)}")
            return validation_results
    
    def _validate_basic_requirements(self, texts: List[str], labels: List[str], 
                                   results: Dict[str, Any]):
        """Validate basic requirements"""
        # Check if inputs are lists
        if not isinstance(texts, list) or not isinstance(labels, list):
            results["validation_passed"] = False
            results["errors"].append("Texts and labels must be lists")
            return
        
        # Check if lengths match
        if len(texts) != len(labels):
            results["validation_passed"] = False
            results["errors"].append(f"Length mismatch: {len(texts)} texts vs {len(labels)} labels")
            return
        
        # Check minimum samples
        if len(texts) < self.thresholds.min_total_samples:
            results["validation_passed"] = False
            results["errors"].append(f"Insufficient samples: {len(texts)} < {self.thresholds.min_total_samples}")
            return
        
        # Check maximum samples
        if len(texts) > self.thresholds.max_total_samples:
            results["validation_passed"] = False
            results["errors"].append(f"Too many samples: {len(texts)} > {self.thresholds.max_total_samples}")
            return
        
        results["metrics"]["total_samples"] = len(texts)
        logger.debug(f"✅ [QUALITY] Basic requirements validated: {len(texts)} samples")
    
    def _validate_data_completeness(self, texts: List[str], labels: List[str], 
                                  results: Dict[str, Any]):
        """Validate data completeness"""
        # Check for missing texts
        missing_texts = sum(1 for text in texts if not text or not text.strip())
        missing_text_rate = (missing_texts / len(texts)) * 100
        
        if missing_text_rate > self.thresholds.max_missing_rate:
            results["validation_passed"] = False
            results["errors"].append(f"Too many missing texts: {missing_text_rate:.1f}% > {self.thresholds.max_missing_rate}%")
        
        # Check for missing labels
        missing_labels = sum(1 for label in labels if not label or not str(label).strip())
        missing_label_rate = (missing_labels / len(labels)) * 100
        
        if missing_label_rate > self.thresholds.max_missing_rate:
            results["validation_passed"] = False
            results["errors"].append(f"Too many missing labels: {missing_label_rate:.1f}% > {self.thresholds.max_missing_rate}%")
        
        results["metrics"]["missing_text_rate"] = missing_text_rate
        results["metrics"]["missing_label_rate"] = missing_label_rate
        
        if missing_text_rate > 0 or missing_label_rate > 0:
            results["warnings"].append(f"Missing data detected: {missing_text_rate:.1f}% texts, {missing_label_rate:.1f}% labels")
    
    def _validate_text_quality(self, texts: List[str], results: Dict[str, Any]):
        """Validate text quality"""
        if not texts:
            return
        
        text_lengths = [len(text) for text in texts if text and text.strip()]
        if not text_lengths:
            results["validation_passed"] = False
            results["errors"].append("No valid texts found")
            return
        
        # Length validation
        min_length = min(text_lengths)
        max_length = max(text_lengths)
        avg_length = np.mean(text_lengths)
        
        if min_length < self.thresholds.min_text_length:
            results["validation_passed"] = False
            results["errors"].append(f"Text too short: {min_length} < {self.thresholds.min_text_length}")
        
        if max_length > self.thresholds.max_text_length:
            results["validation_passed"] = False
            results["errors"].append(f"Text too long: {max_length} > {self.thresholds.max_text_length}")
        
        if avg_length > self.thresholds.max_avg_text_length:
            results["validation_passed"] = False
            results["errors"].append(f"Average text too long: {avg_length:.1f} > {self.thresholds.max_avg_text_length}")
        
        # Unique words validation
        unique_words_per_text = []
        for text in texts:
            if text and text.strip():
                words = text.split()
                unique_words_per_text.append(len(set(words)))
        
        if unique_words_per_text:
            min_unique_words = min(unique_words_per_text)
            if min_unique_words < self.thresholds.min_unique_words:
                results["warnings"].append(f"Some texts have very few unique words: {min_unique_words}")
        
        results["metrics"]["text_length_stats"] = {
            "min": min_length,
            "max": max_length,
            "mean": avg_length,
            "std": np.std(text_lengths)
        }
    
    def _validate_label_quality(self, labels: List[str], results: Dict[str, Any]):
        """Validate label quality"""
        if not labels:
            return
        
        # Check label lengths
        label_lengths = [len(str(label)) for label in labels if label and str(label).strip()]
        if not label_lengths:
            results["validation_passed"] = False
            results["errors"].append("No valid labels found")
            return
        
        min_label_length = min(label_lengths)
        max_label_length = max(label_lengths)
        
        if min_label_length < self.thresholds.min_label_length:
            results["validation_passed"] = False
            results["errors"].append(f"Label too short: {min_label_length} < {self.thresholds.min_label_length}")
        
        if max_label_length > self.thresholds.max_label_length:
            results["validation_passed"] = False
            results["errors"].append(f"Label too long: {max_label_length} > {self.thresholds.max_label_length}")
        
        # Check allowed labels
        if self.thresholds.allowed_labels:
            unique_labels = set(labels)
            allowed_set = set(self.thresholds.allowed_labels)
            invalid_labels = unique_labels - allowed_set
            
            if invalid_labels:
                results["validation_passed"] = False
                results["errors"].append(f"Invalid labels found: {list(invalid_labels)}")
        
        results["metrics"]["label_length_stats"] = {
            "min": min_label_length,
            "max": max_label_length,
            "unique_count": len(set(labels))
        }
    
    def _validate_class_balance(self, labels: List[str], results: Dict[str, Any]):
        """Validate class balance"""
        if not labels:
            return
        
        label_counts = Counter(labels)
        unique_labels = list(label_counts.keys())
        
        if len(unique_labels) < 2:
            results["validation_passed"] = False
            results["errors"].append(f"Insufficient classes: {len(unique_labels)} < 2")
            return
        
        # Check minimum samples per class
        min_class_count = min(label_counts.values())
        max_class_count = max(label_counts.values())
        
        if min_class_count < self.thresholds.min_class_samples:
            results["validation_passed"] = False
            results["errors"].append(f"Class has too few samples: {min_class_count} < {self.thresholds.min_class_samples}")
        
        if max_class_count > self.thresholds.max_class_samples:
            results["validation_passed"] = False
            results["errors"].append(f"Class has too many samples: {max_class_count} > {self.thresholds.max_class_samples}")
        
        # Check class imbalance
        imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
        
        if imbalance_ratio > self.thresholds.max_imbalance_ratio:
            results["validation_passed"] = False
            results["errors"].append(f"Severe class imbalance: {imbalance_ratio:.1f}:1 > {self.thresholds.max_imbalance_ratio}:1")
        elif imbalance_ratio > 5.0:  # Warning threshold
            results["warnings"].append(f"Class imbalance detected: {imbalance_ratio:.1f}:1")
        
        results["metrics"]["class_balance"] = {
            "unique_classes": len(unique_labels),
            "min_class_count": min_class_count,
            "max_class_count": max_class_count,
            "imbalance_ratio": imbalance_ratio,
            "class_distribution": dict(label_counts)
        }
    
    def _validate_duplicates(self, texts: List[str], labels: List[str], 
                           results: Dict[str, Any]):
        """Validate for duplicates"""
        if not texts:
            return
        
        # Find exact duplicates
        text_hashes = {}
        duplicates = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in text_hashes:
                    duplicates.append((i, text_hashes[text_hash]))
                else:
                    text_hashes[text_hash] = i
        
        duplicate_count = len(duplicates)
        duplicate_rate = (duplicate_count / len(texts)) * 100
        
        if duplicate_count > self.thresholds.max_exact_duplicates:
            results["validation_passed"] = False
            results["errors"].append(f"Too many exact duplicates: {duplicate_count} > {self.thresholds.max_exact_duplicates}")
        
        if duplicate_rate > self.thresholds.max_duplicate_rate:
            results["validation_passed"] = False
            results["errors"].append(f"Duplicate rate too high: {duplicate_rate:.1f}% > {self.thresholds.max_duplicate_rate}%")
        elif duplicate_rate > 1.0:  # Warning threshold
            results["warnings"].append(f"Duplicate rate detected: {duplicate_rate:.1f}%")
        
        results["metrics"]["duplicates"] = {
            "exact_duplicates": duplicate_count,
            "duplicate_rate": duplicate_rate
        }
    
    def _validate_statistical_quality(self, texts: List[str], labels: List[str], 
                                    results: Dict[str, Any]):
        """Validate statistical quality"""
        if not texts:
            return
        
        # Text length variance
        text_lengths = [len(text) for text in texts if text and text.strip()]
        if len(text_lengths) > 1:
            length_variance = np.var(text_lengths)
            
            if length_variance < self.thresholds.min_variance:
                results["warnings"].append(f"Very low text length variance: {length_variance:.6f}")
            
            # Outlier detection (using IQR method)
            q1 = np.percentile(text_lengths, 25)
            q3 = np.percentile(text_lengths, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = [length for length in text_lengths if length < lower_bound or length > upper_bound]
            outlier_rate = (len(outliers) / len(text_lengths)) * 100
            
            if outlier_rate > self.thresholds.max_outlier_ratio:
                results["warnings"].append(f"High outlier rate: {outlier_rate:.1f}%")
            
            results["metrics"]["statistical_quality"] = {
                "length_variance": length_variance,
                "outlier_count": len(outliers),
                "outlier_rate": outlier_rate
            }
    
    def _generate_recommendations(self, results: Dict[str, Any]):
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Class imbalance recommendations
        if "class_balance" in results["metrics"]:
            imbalance_ratio = results["metrics"]["class_balance"]["imbalance_ratio"]
            if imbalance_ratio > 3.0:
                recommendations.append("Consider data augmentation for minority classes")
                recommendations.append("Consider using class weights during training")
        
        # Duplicate recommendations
        if "duplicates" in results["metrics"]:
            duplicate_rate = results["metrics"]["duplicates"]["duplicate_rate"]
            if duplicate_rate > 1.0:
                recommendations.append("Consider removing or deduplicating data")
        
        # Text length recommendations
        if "text_length_stats" in results["metrics"]:
            max_length = results["metrics"]["text_length_stats"]["max"]
            if max_length > 10000:
                recommendations.append("Consider truncating very long texts")
        
        # Missing data recommendations
        if results["metrics"].get("missing_text_rate", 0) > 0:
            recommendations.append("Consider handling missing text data")
        
        results["recommendations"] = recommendations

def create_quality_thresholds_for_model_type(model_type: str) -> QualityThresholds:
    """Create quality thresholds specific to model type"""
    if model_type == "security_classification":
        return QualityThresholds(
            max_imbalance_ratio=5.0,  # Stricter for security
            min_class_samples=50,  # More samples needed
            max_duplicate_rate=2.0,  # Stricter duplicate control
            min_text_length=10,  # Longer minimum text
            max_text_length=20000,  # Reasonable max for security texts
            allowed_labels=["prompt_injection", "jailbreak", "system_extraction", "code_injection", "benign"]
        )
    elif model_type == "general_classification":
        return QualityThresholds(
            max_imbalance_ratio=10.0,
            min_class_samples=20,
            max_duplicate_rate=5.0,
            min_text_length=1,
            max_text_length=50000
        )
    else:
        return QualityThresholds()  # Default thresholds
