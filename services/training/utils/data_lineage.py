"""
Data Lineage and Validation Utilities
Prevents data leakage and ensures proper train/val/test splits
"""

import logging
import hashlib
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class DataSplitInfo:
    """Information about a data split"""
    split_type: str  # 'train', 'val', 'test'
    sample_ids: List[str]  # Unique identifiers for each sample
    sample_count: int
    label_distribution: Dict[str, int]
    data_source: str  # S3 path or file identifier
    created_at: str  # ISO timestamp

class DataLineageTracker:
    """Tracks data lineage to prevent leakage between splits"""
    
    def __init__(self):
        self.splits: Dict[str, DataSplitInfo] = {}
        self.sample_to_split: Dict[str, str] = {}  # sample_id -> split_type
        self.used_samples: Set[str] = set()
    
    def generate_sample_id(self, text: str, label: str, index: int) -> str:
        """Generate unique sample ID based on content and position"""
        content_hash = hashlib.md5(f"{text}_{label}_{index}".encode()).hexdigest()[:12]
        return f"sample_{index}_{content_hash}"
    
    def validate_no_leakage(self, train_ids: List[str], val_ids: List[str], test_ids: List[str]) -> bool:
        """Validate that there's no overlap between splits"""
        try:
            # Check for overlaps
            train_set = set(train_ids)
            val_set = set(val_ids)
            test_set = set(test_ids)
            
            # Check for overlaps
            train_val_overlap = train_set & val_set
            train_test_overlap = train_set & test_set
            val_test_overlap = val_set & test_set
            
            if train_val_overlap:
                logger.error(f"❌ [DATA LEAKAGE] Train/Val overlap detected: {len(train_val_overlap)} samples")
                logger.error(f"Overlapping samples: {list(train_val_overlap)[:5]}...")
                return False
            
            if train_test_overlap:
                logger.error(f"❌ [DATA LEAKAGE] Train/Test overlap detected: {len(train_test_overlap)} samples")
                logger.error(f"Overlapping samples: {list(train_test_overlap)[:5]}...")
                return False
            
            if val_test_overlap:
                logger.error(f"❌ [DATA LEAKAGE] Val/Test overlap detected: {len(val_test_overlap)} samples")
                logger.error(f"Overlapping samples: {list(val_test_overlap)[:5]}...")
                return False
            
            logger.info("✅ [DATA VALIDATION] No data leakage detected between splits")
            return True
            
        except Exception as e:
            logger.error(f"❌ [DATA VALIDATION] Error validating splits: {e}")
            return False
    
    def register_split(self, split_type: str, sample_ids: List[str], 
                      label_distribution: Dict[str, int], data_source: str) -> bool:
        """Register a data split and validate against existing splits"""
        try:
            # Check for duplicates within the split
            if len(sample_ids) != len(set(sample_ids)):
                logger.error(f"❌ [DATA VALIDATION] Duplicate samples found in {split_type} split")
                return False
            
            # Check for samples already used in other splits
            already_used = set(sample_ids) & self.used_samples
            if already_used:
                logger.error(f"❌ [DATA LEAKAGE] {len(already_used)} samples already used in other splits")
                logger.error(f"Already used samples: {list(already_used)[:5]}...")
                return False
            
            # Register the split
            split_info = DataSplitInfo(
                split_type=split_type,
                sample_ids=sample_ids,
                sample_count=len(sample_ids),
                label_distribution=label_distribution,
                data_source=data_source,
                created_at=json.dumps({"timestamp": "now"})  # Will be replaced with actual timestamp
            )
            
            self.splits[split_type] = split_info
            self.used_samples.update(sample_ids)
            
            # Update sample-to-split mapping
            for sample_id in sample_ids:
                self.sample_to_split[sample_id] = split_type
            
            logger.info(f"✅ [DATA LINEAGE] Registered {split_type} split: {len(sample_ids)} samples")
            logger.info(f"Label distribution: {label_distribution}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ [DATA LINEAGE] Error registering {split_type} split: {e}")
            return False
    
    def get_split_summary(self) -> Dict[str, Any]:
        """Get summary of all registered splits"""
        return {
            "total_splits": len(self.splits),
            "total_samples": len(self.used_samples),
            "splits": {
                split_type: {
                    "sample_count": info.sample_count,
                    "label_distribution": info.label_distribution,
                    "data_source": info.data_source
                }
                for split_type, info in self.splits.items()
            }
        }
    
    def validate_label_consistency(self, texts: List[str], labels: List[str]) -> bool:
        """Validate that labels are consistent and properly distributed"""
        try:
            logger.info(f"🔍 [DATA VALIDATION] Starting validation with {len(texts)} texts and {len(labels)} labels")
            
            if len(texts) != len(labels):
                logger.error(f"❌ [DATA VALIDATION] Mismatch between texts ({len(texts)}) and labels ({len(labels)})")
                return False
            
            # Check for empty texts or labels
            empty_texts = sum(1 for text in texts if not text or not text.strip())
            empty_labels = sum(1 for label in labels if label is None or (isinstance(label, str) and not label.strip()))
            
            logger.info(f"🔍 [DATA VALIDATION] Empty texts: {empty_texts}, Empty labels: {empty_labels}")
            
            if empty_texts > 0:
                logger.warning(f"⚠️ [DATA VALIDATION] {empty_texts} empty texts found")
            
            if empty_labels > 0:
                logger.error(f"❌ [DATA VALIDATION] {empty_labels} empty labels found")
                # Show some examples of empty labels
                empty_label_indices = [i for i, label in enumerate(labels) if not label or not str(label).strip()]
                logger.error(f"❌ [DATA VALIDATION] Empty label indices: {empty_label_indices[:10]}")
                return False
            
            # Check label distribution
            from collections import Counter
            label_counts = Counter(labels)
            
            if len(label_counts) < 2:
                logger.error(f"❌ [DATA VALIDATION] Insufficient label diversity: {list(label_counts.keys())}")
                return False
            
            # Check for severe imbalance
            min_count = min(label_counts.values())
            max_count = max(label_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 100:  # More than 100:1 ratio
                logger.warning(f"⚠️ [DATA VALIDATION] Severe class imbalance: {imbalance_ratio:.1f}:1 ratio")
                logger.warning(f"Label distribution: {dict(label_counts)}")
            
            logger.info(f"✅ [DATA VALIDATION] Label validation passed")
            logger.info(f"Label distribution: {dict(label_counts)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ [DATA VALIDATION] Error validating labels: {e}")
            return False

def create_data_splits_with_lineage(
    texts: List[str], 
    labels: List[str], 
    data_source: str,
    test_size: float = 0.2,
    val_size: float = 0.25,
    random_state: int = 42
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], DataLineageTracker]:
    """
    Create train/val/test splits with data lineage tracking
    
    Returns:
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, lineage_tracker
    """
    from sklearn.model_selection import train_test_split
    from collections import Counter
    import time
    
    logger.info(f"🔄 [DATA SPLIT] Creating splits with lineage tracking for {len(texts)} samples")
    
    # Initialize lineage tracker
    lineage_tracker = DataLineageTracker()
    
    # Validate input data
    if not lineage_tracker.validate_label_consistency(texts, labels):
        raise ValueError("Data validation failed - check logs for details")
    
    # Generate sample IDs
    sample_ids = [
        lineage_tracker.generate_sample_id(text, str(label), i)
        for i, (text, label) in enumerate(zip(texts, labels))
    ]
    
    try:
        # First split: train+val vs test
        train_val_texts, test_texts, train_val_labels, test_labels, train_val_ids, test_ids = train_test_split(
            texts, labels, sample_ids,
            test_size=test_size, 
            random_state=random_state, 
            stratify=labels
        )
        
        # Second split: train vs val from train+val
        train_texts, val_texts, train_labels, val_labels, train_ids, val_ids = train_test_split(
            train_val_texts, train_val_labels, train_val_ids,
            test_size=val_size, 
            random_state=random_state, 
            stratify=train_val_labels
        )
        
        # Register splits with lineage tracker
        train_label_dist = dict(Counter(train_labels))
        val_label_dist = dict(Counter(val_labels))
        test_label_dist = dict(Counter(test_labels))
        
        if not lineage_tracker.register_split("train", train_ids, train_label_dist, data_source):
            raise ValueError("Failed to register train split")
        
        if not lineage_tracker.register_split("val", val_ids, val_label_dist, data_source):
            raise ValueError("Failed to register val split")
        
        if not lineage_tracker.register_split("test", test_ids, test_label_dist, data_source):
            raise ValueError("Failed to register test split")
        
        # Final validation
        if not lineage_tracker.validate_no_leakage(train_ids, val_ids, test_ids):
            raise ValueError("Data leakage detected - check logs for details")
        
        logger.info("✅ [DATA SPLIT] Successfully created splits with lineage tracking")
        logger.info(f"Split summary: {lineage_tracker.get_split_summary()}")
        
        return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, lineage_tracker
        
    except Exception as e:
        logger.error(f"❌ [DATA SPLIT] Error creating splits: {e}")
        raise
