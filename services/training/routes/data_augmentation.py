"""
Data Augmentation Routes
API endpoints for data augmentation functionality
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from services.data_augmentation import augmenter, AugmentationConfig

logger = logging.getLogger(__name__)
router = APIRouter()

class AugmentationRequest(BaseModel):
    """Request model for data augmentation"""
    texts: List[str]
    labels: List[str]
    augmentation_factor: float = 1.0
    config: Optional[Dict[str, Any]] = None

class AugmentationResponse(BaseModel):
    """Response model for data augmentation"""
    original_count: int
    augmented_count: int
    augmentation_factor: float
    augmented_texts: List[str]
    augmented_labels: List[str]

class SyntheticDataRequest(BaseModel):
    """Request model for synthetic data generation"""
    num_samples: int
    label_distribution: Optional[Dict[str, float]] = None

class SyntheticDataResponse(BaseModel):
    """Response model for synthetic data generation"""
    num_samples: int
    generated_texts: List[str]
    generated_labels: List[str]
    label_distribution: Dict[str, int]

@router.post("/augment", response_model=AugmentationResponse)
async def augment_data(request: AugmentationRequest):
    """Augment training data with various techniques"""
    try:
        if len(request.texts) != len(request.labels):
            raise HTTPException(
                status_code=400, 
                detail="Number of texts and labels must match"
            )
        
        if request.config:
            # Create custom augmentation config
            config = AugmentationConfig(**request.config)
            custom_augmenter = augmenter.__class__(config)
        else:
            custom_augmenter = augmenter
        
        # Augment the data
        augmented_texts, augmented_labels = custom_augmenter.augment_dataset(
            request.texts, 
            request.labels, 
            request.augmentation_factor
        )
        
        return AugmentationResponse(
            original_count=len(request.texts),
            augmented_count=len(augmented_texts),
            augmentation_factor=len(augmented_texts) / len(request.texts),
            augmented_texts=augmented_texts,
            augmented_labels=augmented_labels
        )
        
    except Exception as e:
        logger.error(f"Error augmenting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/augment/single")
async def augment_single_text(text: str, label: str, num_augmentations: int = 3):
    """Augment a single text sample"""
    try:
        augmented_samples = augmenter.augment_text(text, label, num_augmentations)
        
        return {
            "original_text": text,
            "original_label": label,
            "augmented_samples": [
                {"text": aug_text, "label": aug_label} 
                for aug_text, aug_label in augmented_samples
            ],
            "count": len(augmented_samples)
        }
        
    except Exception as e:
        logger.error(f"Error augmenting single text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/balance")
async def balance_dataset(texts: List[str], labels: List[str]):
    """Balance dataset by oversampling minority classes"""
    try:
        if len(texts) != len(labels):
            raise HTTPException(
                status_code=400, 
                detail="Number of texts and labels must match"
            )
        
        balanced_texts, balanced_labels = augmenter.balance_dataset(texts, labels)
        
        # Calculate label distribution
        from collections import Counter
        original_dist = Counter(labels)
        balanced_dist = Counter(balanced_labels)
        
        return {
            "original_count": len(texts),
            "balanced_count": len(balanced_texts),
            "original_distribution": dict(original_dist),
            "balanced_distribution": dict(balanced_dist),
            "balanced_texts": balanced_texts,
            "balanced_labels": balanced_labels
        }
        
    except Exception as e:
        logger.error(f"Error balancing dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/synthetic", response_model=SyntheticDataResponse)
async def generate_synthetic_data(request: SyntheticDataRequest):
    """Generate synthetic training data"""
    try:
        synthetic_samples = augmenter.generate_synthetic_samples(
            request.num_samples,
            request.label_distribution
        )
        
        if not synthetic_samples:
            raise HTTPException(status_code=500, detail="Failed to generate synthetic data")
        
        texts, labels = zip(*synthetic_samples)
        
        # Calculate label distribution
        from collections import Counter
        label_dist = Counter(labels)
        
        return SyntheticDataResponse(
            num_samples=len(texts),
            generated_texts=list(texts),
            generated_labels=list(labels),
            label_distribution=dict(label_dist)
        )
        
    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def get_augmentation_config():
    """Get current augmentation configuration"""
    try:
        config = augmenter.config
        
        return {
            "synonym_replacement_prob": config.synonym_replacement_prob,
            "back_translation_prob": config.back_translation_prob,
            "random_insertion_prob": config.random_insertion_prob,
            "random_deletion_prob": config.random_deletion_prob,
            "random_swap_prob": config.random_swap_prob,
            "random_caps_prob": config.random_caps_prob,
            "max_augmentations": config.max_augmentations,
            "preserve_labels": config.preserve_labels
        }
        
    except Exception as e:
        logger.error(f"Error getting augmentation config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/config")
async def update_augmentation_config(config: Dict[str, Any]):
    """Update augmentation configuration"""
    try:
        # Update the global augmenter config
        for key, value in config.items():
            if hasattr(augmenter.config, key):
                setattr(augmenter.config, key, value)
        
        return {
            "message": "Augmentation configuration updated successfully",
            "config": {
                "synonym_replacement_prob": augmenter.config.synonym_replacement_prob,
                "back_translation_prob": augmenter.config.back_translation_prob,
                "random_insertion_prob": augmenter.config.random_insertion_prob,
                "random_deletion_prob": augmenter.config.random_deletion_prob,
                "random_swap_prob": augmenter.config.random_swap_prob,
                "random_caps_prob": augmenter.config.random_caps_prob,
                "max_augmentations": augmenter.config.max_augmentations,
                "preserve_labels": augmenter.config.preserve_labels
            }
        }
        
    except Exception as e:
        logger.error(f"Error updating augmentation config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/techniques")
async def get_available_techniques():
    """Get list of available augmentation techniques"""
    try:
        return {
            "techniques": [
                {
                    "name": "synonym_replacement",
                    "description": "Replace words with synonyms",
                    "probability": augmenter.config.synonym_replacement_prob
                },
                {
                    "name": "random_insertion",
                    "description": "Insert random words or phrases",
                    "probability": augmenter.config.random_insertion_prob
                },
                {
                    "name": "random_deletion",
                    "description": "Randomly delete words",
                    "probability": augmenter.config.random_deletion_prob
                },
                {
                    "name": "random_swap",
                    "description": "Randomly swap adjacent words",
                    "probability": augmenter.config.random_swap_prob
                },
                {
                    "name": "random_caps",
                    "description": "Randomly change capitalization",
                    "probability": augmenter.config.random_caps_prob
                }
            ],
            "synonyms_available": len(augmenter.synonyms),
            "attack_patterns_available": len(augmenter.attack_patterns),
            "benign_patterns_available": len(augmenter.benign_patterns)
        }
        
    except Exception as e:
        logger.error(f"Error getting available techniques: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/preview")
async def preview_augmentation(text: str, label: str, num_samples: int = 5):
    """Preview augmentation results for a single text"""
    try:
        augmented_samples = augmenter.augment_text(text, label, num_samples)
        
        return {
            "original": {"text": text, "label": label},
            "augmented_samples": [
                {"text": aug_text, "label": aug_label} 
                for aug_text, aug_label in augmented_samples
            ],
            "count": len(augmented_samples)
        }
        
    except Exception as e:
        logger.error(f"Error previewing augmentation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
