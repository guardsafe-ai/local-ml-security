"""
Feedback Loop
Implements feedback loop for continuous learning from incidents
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib

from .incident_processor import ProcessedIncident, SecurityIncident
from .pattern_extractor import ExtractedPattern, PatternCluster

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    CORRECTION = "correction"


class LearningPhase(Enum):
    """Learning phases"""
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    PATTERN_EXTRACTION = "pattern_extraction"
    MITIGATION = "mitigation"


@dataclass
class FeedbackEntry:
    """Feedback entry"""
    feedback_id: str
    feedback_type: FeedbackType
    learning_phase: LearningPhase
    incident_id: str
    pattern_id: Optional[str]
    feedback_content: str
    confidence: float
    timestamp: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LearningUpdate:
    """Learning update"""
    update_id: str
    learning_phase: LearningPhase
    update_type: str
    old_value: Any
    new_value: Any
    confidence: float
    timestamp: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FeedbackLoop:
    """
    Feedback Loop
    Implements feedback loop for continuous learning from incidents
    """
    
    def __init__(self, learning_rate: float = 0.1, memory_size: int = 1000):
        """Initialize feedback loop"""
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        
        self.feedback_history: deque = deque(maxlen=memory_size)
        self.learning_updates: List[LearningUpdate] = []
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.adaptation_weights: Dict[str, float] = {}
        
        # Learning models
        self.detection_model = None
        self.classification_model = None
        self.pattern_model = None
        self.mitigation_model = None
        
        logger.info("✅ Initialized Feedback Loop")
    
    async def process_feedback(self, feedback: FeedbackEntry) -> bool:
        """
        Process feedback and update learning models
        """
        try:
            logger.info(f"Processing feedback: {feedback.feedback_id}")
            
            # Store feedback
            self.feedback_history.append(feedback)
            
            # Update performance metrics
            await self._update_performance_metrics(feedback)
            
            # Apply learning updates based on feedback type
            if feedback.feedback_type == FeedbackType.POSITIVE:
                await self._apply_positive_feedback(feedback)
            elif feedback.feedback_type == FeedbackType.NEGATIVE:
                await self._apply_negative_feedback(feedback)
            elif feedback.feedback_type == FeedbackType.CORRECTION:
                await self._apply_correction_feedback(feedback)
            
            # Update adaptation weights
            await self._update_adaptation_weights(feedback)
            
            logger.info(f"Feedback processed successfully: {feedback.feedback_id}")
            return True
            
        except Exception as e:
            logger.error(f"Feedback processing failed: {e}")
            return False
    
    async def _update_performance_metrics(self, feedback: FeedbackEntry):
        """Update performance metrics based on feedback"""
        try:
            phase = feedback.learning_phase.value
            
            # Update accuracy metrics
            if feedback.feedback_type == FeedbackType.POSITIVE:
                self.performance_metrics[f"{phase}_accuracy"].append(1.0)
            elif feedback.feedback_type == FeedbackType.NEGATIVE:
                self.performance_metrics[f"{phase}_accuracy"].append(0.0)
            elif feedback.feedback_type == FeedbackType.CORRECTION:
                self.performance_metrics[f"{phase}_accuracy"].append(0.5)
            
            # Update confidence metrics
            self.performance_metrics[f"{phase}_confidence"].append(feedback.confidence)
            
            # Update response time metrics (if available)
            if "response_time" in feedback.metadata:
                self.performance_metrics[f"{phase}_response_time"].append(feedback.metadata["response_time"])
            
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")
    
    async def _apply_positive_feedback(self, feedback: FeedbackEntry):
        """Apply positive feedback to learning models"""
        try:
            phase = feedback.learning_phase
            
            if phase == LearningPhase.DETECTION:
                await self._update_detection_model(feedback, 1.0)
            elif phase == LearningPhase.CLASSIFICATION:
                await self._update_classification_model(feedback, 1.0)
            elif phase == LearningPhase.PATTERN_EXTRACTION:
                await self._update_pattern_model(feedback, 1.0)
            elif phase == LearningPhase.MITIGATION:
                await self._update_mitigation_model(feedback, 1.0)
            
        except Exception as e:
            logger.error(f"Positive feedback application failed: {e}")
    
    async def _apply_negative_feedback(self, feedback: FeedbackEntry):
        """Apply negative feedback to learning models"""
        try:
            phase = feedback.learning_phase
            
            if phase == LearningPhase.DETECTION:
                await self._update_detection_model(feedback, -1.0)
            elif phase == LearningPhase.CLASSIFICATION:
                await self._update_classification_model(feedback, -1.0)
            elif phase == LearningPhase.PATTERN_EXTRACTION:
                await self._update_pattern_model(feedback, -1.0)
            elif phase == LearningPhase.MITIGATION:
                await self._update_mitigation_model(feedback, -1.0)
            
        except Exception as e:
            logger.error(f"Negative feedback application failed: {e}")
    
    async def _apply_correction_feedback(self, feedback: FeedbackEntry):
        """Apply correction feedback to learning models"""
        try:
            phase = feedback.learning_phase
            
            # Extract correction information
            correction_info = feedback.metadata.get("correction", {})
            old_value = correction_info.get("old_value")
            new_value = correction_info.get("new_value")
            
            if old_value is not None and new_value is not None:
                # Create learning update
                update = LearningUpdate(
                    update_id=f"update_{hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8]}",
                    learning_phase=phase,
                    update_type="correction",
                    old_value=old_value,
                    new_value=new_value,
                    confidence=feedback.confidence,
                    timestamp=datetime.utcnow().isoformat(),
                    metadata={
                        "feedback_id": feedback.feedback_id,
                        "incident_id": feedback.incident_id
                    }
                )
                
                self.learning_updates.append(update)
                
                # Apply correction to appropriate model
                if phase == LearningPhase.DETECTION:
                    await self._correct_detection_model(update)
                elif phase == LearningPhase.CLASSIFICATION:
                    await self._correct_classification_model(update)
                elif phase == LearningPhase.PATTERN_EXTRACTION:
                    await self._correct_pattern_model(update)
                elif phase == LearningPhase.MITIGATION:
                    await self._correct_mitigation_model(update)
            
        except Exception as e:
            logger.error(f"Correction feedback application failed: {e}")
    
    async def _update_detection_model(self, feedback: FeedbackEntry, weight: float):
        """Update detection model based on feedback"""
        try:
            # This is a simplified implementation
            # In practice, you would update actual ML models
            
            if self.detection_model is None:
                self.detection_model = {"weights": {}, "thresholds": {}}
            
            # Update model weights based on feedback
            incident_id = feedback.incident_id
            pattern_id = feedback.pattern_id
            
            if pattern_id:
                if pattern_id not in self.detection_model["weights"]:
                    self.detection_model["weights"][pattern_id] = 0.0
                
                self.detection_model["weights"][pattern_id] += weight * self.learning_rate
            
            # Update detection thresholds
            if "detection_threshold" not in self.detection_model["thresholds"]:
                self.detection_model["thresholds"]["detection_threshold"] = 0.5
            
            # Adjust threshold based on feedback
            if weight > 0:
                self.detection_model["thresholds"]["detection_threshold"] *= 0.99
            else:
                self.detection_model["thresholds"]["detection_threshold"] *= 1.01
            
            # Ensure threshold stays within bounds
            self.detection_model["thresholds"]["detection_threshold"] = max(0.1, min(0.9, 
                self.detection_model["thresholds"]["detection_threshold"]))
            
        except Exception as e:
            logger.error(f"Detection model update failed: {e}")
    
    async def _update_classification_model(self, feedback: FeedbackEntry, weight: float):
        """Update classification model based on feedback"""
        try:
            if self.classification_model is None:
                self.classification_model = {"class_weights": {}, "features": {}}
            
            # Update class weights based on feedback
            incident_id = feedback.incident_id
            
            # Extract incident type from feedback metadata
            incident_type = feedback.metadata.get("incident_type", "unknown")
            
            if incident_type not in self.classification_model["class_weights"]:
                self.classification_model["class_weights"][incident_type] = 0.0
            
            self.classification_model["class_weights"][incident_type] += weight * self.learning_rate
            
        except Exception as e:
            logger.error(f"Classification model update failed: {e}")
    
    async def _update_pattern_model(self, feedback: FeedbackEntry, weight: float):
        """Update pattern model based on feedback"""
        try:
            if self.pattern_model is None:
                self.pattern_model = {"pattern_weights": {}, "similarity_thresholds": {}}
            
            pattern_id = feedback.pattern_id
            
            if pattern_id:
                if pattern_id not in self.pattern_model["pattern_weights"]:
                    self.pattern_model["pattern_weights"][pattern_id] = 0.0
                
                self.pattern_model["pattern_weights"][pattern_id] += weight * self.learning_rate
            
        except Exception as e:
            logger.error(f"Pattern model update failed: {e}")
    
    async def _update_mitigation_model(self, feedback: FeedbackEntry, weight: float):
        """Update mitigation model based on feedback"""
        try:
            if self.mitigation_model is None:
                self.mitigation_model = {"mitigation_weights": {}, "effectiveness": {}}
            
            # Update mitigation weights based on feedback
            mitigation_type = feedback.metadata.get("mitigation_type", "unknown")
            
            if mitigation_type not in self.mitigation_model["mitigation_weights"]:
                self.mitigation_model["mitigation_weights"][mitigation_type] = 0.0
            
            self.mitigation_model["mitigation_weights"][mitigation_type] += weight * self.learning_rate
            
        except Exception as e:
            logger.error(f"Mitigation model update failed: {e}")
    
    async def _correct_detection_model(self, update: LearningUpdate):
        """Apply correction to detection model"""
        try:
            if self.detection_model is None:
                self.detection_model = {"weights": {}, "thresholds": {}}
            
            # Apply correction based on update
            if update.update_type == "correction":
                old_value = update.old_value
                new_value = update.new_value
                
                # Update model parameters
                if isinstance(old_value, dict) and isinstance(new_value, dict):
                    for key, value in new_value.items():
                        self.detection_model["weights"][key] = value
            
        except Exception as e:
            logger.error(f"Detection model correction failed: {e}")
    
    async def _correct_classification_model(self, update: LearningUpdate):
        """Apply correction to classification model"""
        try:
            if self.classification_model is None:
                self.classification_model = {"class_weights": {}, "features": {}}
            
            # Apply correction based on update
            if update.update_type == "correction":
                old_value = update.old_value
                new_value = update.new_value
                
                # Update class weights
                if isinstance(old_value, dict) and isinstance(new_value, dict):
                    for key, value in new_value.items():
                        self.classification_model["class_weights"][key] = value
            
        except Exception as e:
            logger.error(f"Classification model correction failed: {e}")
    
    async def _correct_pattern_model(self, update: LearningUpdate):
        """Apply correction to pattern model"""
        try:
            if self.pattern_model is None:
                self.pattern_model = {"pattern_weights": {}, "similarity_thresholds": {}}
            
            # Apply correction based on update
            if update.update_type == "correction":
                old_value = update.old_value
                new_value = update.new_value
                
                # Update pattern weights
                if isinstance(old_value, dict) and isinstance(new_value, dict):
                    for key, value in new_value.items():
                        self.pattern_model["pattern_weights"][key] = value
            
        except Exception as e:
            logger.error(f"Pattern model correction failed: {e}")
    
    async def _correct_mitigation_model(self, update: LearningUpdate):
        """Apply correction to mitigation model"""
        try:
            if self.mitigation_model is None:
                self.mitigation_model = {"mitigation_weights": {}, "effectiveness": {}}
            
            # Apply correction based on update
            if update.update_type == "correction":
                old_value = update.old_value
                new_value = update.new_value
                
                # Update mitigation weights
                if isinstance(old_value, dict) and isinstance(new_value, dict):
                    for key, value in new_value.items():
                        self.mitigation_model["mitigation_weights"][key] = value
            
        except Exception as e:
            logger.error(f"Mitigation model correction failed: {e}")
    
    async def _update_adaptation_weights(self, feedback: FeedbackEntry):
        """Update adaptation weights based on feedback"""
        try:
            phase = feedback.learning_phase.value
            
            # Initialize weight if not exists
            if phase not in self.adaptation_weights:
                self.adaptation_weights[phase] = 1.0
            
            # Update weight based on feedback type
            if feedback.feedback_type == FeedbackType.POSITIVE:
                self.adaptation_weights[phase] *= 1.01
            elif feedback.feedback_type == FeedbackType.NEGATIVE:
                self.adaptation_weights[phase] *= 0.99
            elif feedback.feedback_type == FeedbackType.CORRECTION:
                self.adaptation_weights[phase] *= 1.005
            
            # Ensure weight stays within bounds
            self.adaptation_weights[phase] = max(0.1, min(2.0, self.adaptation_weights[phase]))
            
        except Exception as e:
            logger.error(f"Adaptation weights update failed: {e}")
    
    async def get_performance_metrics(self, phase: Optional[LearningPhase] = None) -> Dict[str, Any]:
        """Get performance metrics for learning phases"""
        try:
            if phase:
                phase_key = phase.value
                return {
                    f"{phase_key}_accuracy": self.performance_metrics.get(f"{phase_key}_accuracy", []),
                    f"{phase_key}_confidence": self.performance_metrics.get(f"{phase_key}_confidence", []),
                    f"{phase_key}_response_time": self.performance_metrics.get(f"{phase_key}_response_time", [])
                }
            else:
                return dict(self.performance_metrics)
                
        except Exception as e:
            logger.error(f"Performance metrics retrieval failed: {e}")
            return {}
    
    async def get_learning_updates(self, phase: Optional[LearningPhase] = None) -> List[LearningUpdate]:
        """Get learning updates for specific phase or all phases"""
        try:
            if phase:
                return [update for update in self.learning_updates if update.learning_phase == phase]
            else:
                return self.learning_updates
                
        except Exception as e:
            logger.error(f"Learning updates retrieval failed: {e}")
            return []
    
    async def get_adaptation_weights(self) -> Dict[str, float]:
        """Get current adaptation weights"""
        try:
            return dict(self.adaptation_weights)
            
        except Exception as e:
            logger.error(f"Adaptation weights retrieval failed: {e}")
            return {}
    
    async def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        try:
            if not self.feedback_history:
                return {}
            
            # Count feedback by type
            feedback_counts = defaultdict(int)
            for feedback in self.feedback_history:
                feedback_counts[feedback.feedback_type.value] += 1
            
            # Count feedback by phase
            phase_counts = defaultdict(int)
            for feedback in self.feedback_history:
                phase_counts[feedback.learning_phase.value] += 1
            
            # Calculate average confidence
            confidences = [feedback.confidence for feedback in self.feedback_history]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                "total_feedback": len(self.feedback_history),
                "feedback_by_type": dict(feedback_counts),
                "feedback_by_phase": dict(phase_counts),
                "average_confidence": avg_confidence,
                "learning_updates": len(self.learning_updates),
                "adaptation_weights": dict(self.adaptation_weights)
            }
            
        except Exception as e:
            logger.error(f"Feedback statistics retrieval failed: {e}")
            return {}
    
    async def export_feedback_data(self, format: str = "json") -> str:
        """Export feedback data"""
        try:
            if format.lower() == "json":
                data = {
                    "feedback_history": [feedback.__dict__ for feedback in self.feedback_history],
                    "learning_updates": [update.__dict__ for update in self.learning_updates],
                    "performance_metrics": dict(self.performance_metrics),
                    "adaptation_weights": dict(self.adaptation_weights),
                    "models": {
                        "detection_model": self.detection_model,
                        "classification_model": self.classification_model,
                        "pattern_model": self.pattern_model,
                        "mitigation_model": self.mitigation_model
                    },
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Feedback data export failed: {e}")
            return ""
    
    async def reset_learning(self, phase: Optional[LearningPhase] = None):
        """Reset learning for specific phase or all phases"""
        try:
            if phase:
                # Reset specific phase
                phase_key = phase.value
                self.performance_metrics[f"{phase_key}_accuracy"] = []
                self.performance_metrics[f"{phase_key}_confidence"] = []
                self.performance_metrics[f"{phase_key}_response_time"] = []
                
                # Reset phase-specific model
                if phase == LearningPhase.DETECTION:
                    self.detection_model = None
                elif phase == LearningPhase.CLASSIFICATION:
                    self.classification_model = None
                elif phase == LearningPhase.PATTERN_EXTRACTION:
                    self.pattern_model = None
                elif phase == LearningPhase.MITIGATION:
                    self.mitigation_model = None
            else:
                # Reset all learning
                self.feedback_history.clear()
                self.learning_updates.clear()
                self.performance_metrics.clear()
                self.adaptation_weights.clear()
                self.detection_model = None
                self.classification_model = None
                self.pattern_model = None
                self.mitigation_model = None
            
            logger.info(f"Learning reset for phase: {phase.value if phase else 'all'}")
            
        except Exception as e:
            logger.error(f"Learning reset failed: {e}")
