"""
Incident Learning Pipeline
Main pipeline for processing incidents and learning from them
"""

import asyncio
import logging
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib

from .incident_processor import IncidentProcessor, ProcessedIncident, SecurityIncident
from .pattern_extractor import PatternExtractor, ExtractedPattern, PatternCluster
from .feedback_loop import FeedbackLoop, FeedbackEntry, FeedbackType, LearningPhase

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline stages"""
    INCIDENT_PROCESSING = "incident_processing"
    PATTERN_EXTRACTION = "pattern_extraction"
    PATTERN_CLUSTERING = "pattern_clustering"
    FEEDBACK_PROCESSING = "feedback_processing"
    MODEL_UPDATING = "model_updating"
    REPORTING = "reporting"


@dataclass
class PipelineStatus:
    """Pipeline status"""
    stage: PipelineStage
    status: str
    progress: float
    timestamp: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LearningReport:
    """Learning report"""
    report_id: str
    report_type: str
    summary: str
    patterns_learned: int
    clusters_created: int
    feedback_processed: int
    model_updates: int
    performance_metrics: Dict[str, Any]
    recommendations: List[str]
    timestamp: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class IncidentLearningPipeline:
    """
    Incident Learning Pipeline
    Main pipeline for processing incidents and learning from them
    """
    
    def __init__(self, 
                 batch_size: int = 100,
                 learning_rate: float = 0.1,
                 memory_size: int = 1000):
        """Initialize incident learning pipeline"""
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        
        # Initialize components
        self.incident_processor = IncidentProcessor()
        self.pattern_extractor = PatternExtractor()
        self.feedback_loop = FeedbackLoop(learning_rate=learning_rate, memory_size=memory_size)
        
        # Pipeline state
        self.pipeline_status: PipelineStatus = None
        self.learning_reports: List[LearningReport] = []
        self.processed_incidents: List[ProcessedIncident] = []
        self.extracted_patterns: List[ExtractedPattern] = []
        self.pattern_clusters: List[PatternCluster] = []
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=memory_size)
        self.learning_metrics: Dict[str, List[float]] = defaultdict(list)
        
        logger.info("✅ Initialized Incident Learning Pipeline")
    
    async def process_incidents(self, incidents: List[Dict[str, Any]]) -> LearningReport:
        """
        Process incidents through the learning pipeline
        """
        try:
            logger.info(f"Starting incident processing pipeline with {len(incidents)} incidents")
            
            # Stage 1: Incident Processing
            await self._update_pipeline_status(PipelineStage.INCIDENT_PROCESSING, "processing", 0.0)
            processed_incidents = await self._process_incidents_stage(incidents)
            
            # Stage 2: Pattern Extraction
            await self._update_pipeline_status(PipelineStage.PATTERN_EXTRACTION, "processing", 0.2)
            extracted_patterns = await self._extract_patterns_stage(processed_incidents)
            
            # Stage 3: Pattern Clustering
            await self._update_pipeline_status(PipelineStage.PATTERN_CLUSTERING, "processing", 0.4)
            pattern_clusters = await self._cluster_patterns_stage(extracted_patterns)
            
            # Stage 4: Feedback Processing
            await self._update_pipeline_status(PipelineStage.FEEDBACK_PROCESSING, "processing", 0.6)
            feedback_results = await self._process_feedback_stage(processed_incidents, extracted_patterns)
            
            # Stage 5: Model Updating
            await self._update_pipeline_status(PipelineStage.MODEL_UPDATING, "processing", 0.8)
            model_updates = await self._update_models_stage(feedback_results)
            
            # Stage 6: Reporting
            await self._update_pipeline_status(PipelineStage.REPORTING, "processing", 1.0)
            learning_report = await self._generate_report_stage(
                processed_incidents, extracted_patterns, pattern_clusters, feedback_results, model_updates
            )
            
            # Update pipeline state
            self.processed_incidents.extend(processed_incidents)
            self.extracted_patterns.extend(extracted_patterns)
            self.pattern_clusters.extend(pattern_clusters)
            self.learning_reports.append(learning_report)
            
            # Update performance metrics
            await self._update_performance_metrics(learning_report)
            
            await self._update_pipeline_status(PipelineStage.REPORTING, "completed", 1.0)
            
            logger.info("Incident processing pipeline completed successfully")
            return learning_report
            
        except Exception as e:
            logger.error(f"Incident processing pipeline failed: {e}")
            await self._update_pipeline_status(PipelineStage.REPORTING, "failed", 1.0)
            raise
    
    async def _process_incidents_stage(self, incidents: List[Dict[str, Any]]) -> List[ProcessedIncident]:
        """Process incidents stage"""
        try:
            logger.info("Processing incidents stage")
            
            # Process incidents in batches
            processed_incidents = []
            for i in range(0, len(incidents), self.batch_size):
                batch = incidents[i:i + self.batch_size]
                batch_processed = await self.incident_processor.batch_process_incidents(batch)
                processed_incidents.extend(batch_processed)
                
                # Update progress
                progress = (i + len(batch)) / len(incidents) * 0.2
                await self._update_pipeline_status(PipelineStage.INCIDENT_PROCESSING, "processing", progress)
            
            logger.info(f"Processed {len(processed_incidents)} incidents")
            return processed_incidents
            
        except Exception as e:
            logger.error(f"Incident processing stage failed: {e}")
            raise
    
    async def _extract_patterns_stage(self, processed_incidents: List[ProcessedIncident]) -> List[ExtractedPattern]:
        """Extract patterns stage"""
        try:
            logger.info("Extracting patterns stage")
            
            # Extract patterns from processed incidents
            extracted_patterns = await self.pattern_extractor.extract_patterns(processed_incidents)
            
            logger.info(f"Extracted {len(extracted_patterns)} patterns")
            return extracted_patterns
            
        except Exception as e:
            logger.error(f"Pattern extraction stage failed: {e}")
            raise
    
    async def _cluster_patterns_stage(self, extracted_patterns: List[ExtractedPattern]) -> List[PatternCluster]:
        """Cluster patterns stage"""
        try:
            logger.info("Clustering patterns stage")
            
            # Cluster extracted patterns
            pattern_clusters = await self.pattern_extractor.cluster_patterns(extracted_patterns)
            
            logger.info(f"Created {len(pattern_clusters)} pattern clusters")
            return pattern_clusters
            
        except Exception as e:
            logger.error(f"Pattern clustering stage failed: {e}")
            raise
    
    async def _process_feedback_stage(self, 
                                    processed_incidents: List[ProcessedIncident], 
                                    extracted_patterns: List[ExtractedPattern]) -> List[FeedbackEntry]:
        """Process feedback stage"""
        try:
            logger.info("Processing feedback stage")
            
            feedback_results = []
            
            # Generate feedback based on incident analysis
            for incident in processed_incidents:
                feedback = await self._generate_feedback(incident, extracted_patterns)
                if feedback:
                    # Process feedback through feedback loop
                    success = await self.feedback_loop.process_feedback(feedback)
                    if success:
                        feedback_results.append(feedback)
            
            logger.info(f"Processed {len(feedback_results)} feedback entries")
            return feedback_results
            
        except Exception as e:
            logger.error(f"Feedback processing stage failed: {e}")
            raise
    
    async def _generate_feedback(self, 
                               incident: ProcessedIncident, 
                               patterns: List[ExtractedPattern]) -> Optional[FeedbackEntry]:
        """Generate feedback for incident"""
        try:
            # Determine feedback type based on incident analysis
            feedback_type = await self._determine_feedback_type(incident)
            
            # Determine learning phase
            learning_phase = await self._determine_learning_phase(incident)
            
            # Find relevant patterns
            relevant_patterns = await self._find_relevant_patterns(incident, patterns)
            
            # Generate feedback content
            feedback_content = await self._generate_feedback_content(incident, relevant_patterns)
            
            # Calculate confidence
            confidence = await self._calculate_feedback_confidence(incident, relevant_patterns)
            
            feedback = FeedbackEntry(
                feedback_id=f"feedback_{hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8]}",
                feedback_type=feedback_type,
                learning_phase=learning_phase,
                incident_id=incident.incident.incident_id,
                pattern_id=relevant_patterns[0].pattern_id if relevant_patterns else None,
                feedback_content=feedback_content,
                confidence=confidence,
                timestamp=datetime.utcnow().isoformat(),
                metadata={
                    "incident_type": incident.incident.incident_type.value,
                    "severity": incident.incident.severity.value,
                    "success": incident.incident.success,
                    "risk_score": incident.risk_score,
                    "relevant_patterns": [p.pattern_id for p in relevant_patterns]
                }
            )
            
            return feedback
            
        except Exception as e:
            logger.error(f"Feedback generation failed: {e}")
            return None
    
    async def _determine_feedback_type(self, incident: ProcessedIncident) -> FeedbackType:
        """Determine feedback type based on incident"""
        try:
            # Simple heuristic for feedback type determination
            if incident.incident.success:
                if incident.risk_score > 0.7:
                    return FeedbackType.NEGATIVE
                else:
                    return FeedbackType.POSITIVE
            else:
                return FeedbackType.NEUTRAL
                
        except Exception as e:
            logger.error(f"Feedback type determination failed: {e}")
            return FeedbackType.NEUTRAL
    
    async def _determine_learning_phase(self, incident: ProcessedIncident) -> LearningPhase:
        """Determine learning phase based on incident"""
        try:
            # Simple heuristic for learning phase determination
            if incident.incident.incident_type.value in ["prompt_injection", "jailbreak"]:
                return LearningPhase.DETECTION
            elif incident.incident.incident_type.value in ["adversarial_attack", "data_poisoning"]:
                return LearningPhase.CLASSIFICATION
            else:
                return LearningPhase.PATTERN_EXTRACTION
                
        except Exception as e:
            logger.error(f"Learning phase determination failed: {e}")
            return LearningPhase.PATTERN_EXTRACTION
    
    async def _find_relevant_patterns(self, 
                                    incident: ProcessedIncident, 
                                    patterns: List[ExtractedPattern]) -> List[ExtractedPattern]:
        """Find relevant patterns for incident"""
        try:
            relevant_patterns = []
            
            # Find patterns that match incident characteristics
            for pattern in patterns:
                if self._is_pattern_relevant(incident, pattern):
                    relevant_patterns.append(pattern)
            
            # Sort by relevance score
            relevant_patterns.sort(key=lambda p: self._calculate_pattern_relevance(incident, p), reverse=True)
            
            return relevant_patterns[:5]  # Return top 5 relevant patterns
            
        except Exception as e:
            logger.error(f"Relevant pattern finding failed: {e}")
            return []
    
    def _is_pattern_relevant(self, incident: ProcessedIncident, pattern: ExtractedPattern) -> bool:
        """Check if pattern is relevant to incident"""
        try:
            # Check if pattern matches incident type
            if pattern.metadata.get("incident_type") == incident.incident.incident_type.value:
                return True
            
            # Check if pattern matches incident patterns
            for incident_pattern in incident.patterns:
                if pattern.pattern_content in incident_pattern:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _calculate_pattern_relevance(self, incident: ProcessedIncident, pattern: ExtractedPattern) -> float:
        """Calculate pattern relevance score"""
        try:
            score = 0.0
            
            # Type matching score
            if pattern.metadata.get("incident_type") == incident.incident.incident_type.value:
                score += 0.5
            
            # Pattern content matching score
            for incident_pattern in incident.patterns:
                if pattern.pattern_content in incident_pattern:
                    score += 0.3
            
            # Frequency score
            score += min(pattern.frequency / 10.0, 0.2)
            
            return score
            
        except Exception:
            return 0.0
    
    async def _generate_feedback_content(self, 
                                       incident: ProcessedIncident, 
                                       patterns: List[ExtractedPattern]) -> str:
        """Generate feedback content"""
        try:
            content = f"Incident {incident.incident.incident_id} analysis:\n"
            content += f"Type: {incident.incident.incident_type.value}\n"
            content += f"Severity: {incident.incident.severity.value}\n"
            content += f"Success: {incident.incident.success}\n"
            content += f"Risk Score: {incident.risk_score:.2f}\n"
            
            if patterns:
                content += f"Relevant patterns: {len(patterns)}\n"
                for pattern in patterns[:3]:
                    content += f"- {pattern.pattern_content}\n"
            
            return content
            
        except Exception as e:
            logger.error(f"Feedback content generation failed: {e}")
            return "Feedback content generation failed"
    
    async def _calculate_feedback_confidence(self, 
                                           incident: ProcessedIncident, 
                                           patterns: List[ExtractedPattern]) -> float:
        """Calculate feedback confidence"""
        try:
            confidence = 0.0
            
            # Base confidence from incident
            confidence += incident.confidence * 0.4
            
            # Pattern confidence
            if patterns:
                avg_pattern_confidence = np.mean([p.confidence for p in patterns])
                confidence += avg_pattern_confidence * 0.3
            
            # Risk score confidence
            confidence += incident.risk_score * 0.3
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Feedback confidence calculation failed: {e}")
            return 0.5
    
    async def _update_models_stage(self, feedback_results: List[FeedbackEntry]) -> Dict[str, Any]:
        """Update models stage"""
        try:
            logger.info("Updating models stage")
            
            model_updates = {
                "detection_model": self.feedback_loop.detection_model,
                "classification_model": self.feedback_loop.classification_model,
                "pattern_model": self.feedback_loop.pattern_model,
                "mitigation_model": self.feedback_loop.mitigation_model
            }
            
            logger.info("Models updated successfully")
            return model_updates
            
        except Exception as e:
            logger.error(f"Model updating stage failed: {e}")
            raise
    
    async def _generate_report_stage(self, 
                                   processed_incidents: List[ProcessedIncident],
                                   extracted_patterns: List[ExtractedPattern],
                                   pattern_clusters: List[PatternCluster],
                                   feedback_results: List[FeedbackEntry],
                                   model_updates: Dict[str, Any]) -> LearningReport:
        """Generate report stage"""
        try:
            logger.info("Generating learning report")
            
            # Get performance metrics
            performance_metrics = await self.feedback_loop.get_performance_metrics()
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                processed_incidents, extracted_patterns, pattern_clusters, performance_metrics
            )
            
            # Create learning report
            report = LearningReport(
                report_id=f"report_{hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8]}",
                report_type="incident_learning",
                summary=f"Processed {len(processed_incidents)} incidents, extracted {len(extracted_patterns)} patterns, created {len(pattern_clusters)} clusters",
                patterns_learned=len(extracted_patterns),
                clusters_created=len(pattern_clusters),
                feedback_processed=len(feedback_results),
                model_updates=len(model_updates),
                performance_metrics=performance_metrics,
                recommendations=recommendations,
                timestamp=datetime.utcnow().isoformat(),
                metadata={
                    "pipeline_version": "1.0.0",
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate
                }
            )
            
            logger.info("Learning report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Report generation stage failed: {e}")
            raise
    
    async def _generate_recommendations(self, 
                                      processed_incidents: List[ProcessedIncident],
                                      extracted_patterns: List[ExtractedPattern],
                                      pattern_clusters: List[PatternCluster],
                                      performance_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        try:
            recommendations = []
            
            # Pattern-based recommendations
            if len(extracted_patterns) > 100:
                recommendations.append("Consider implementing pattern-based detection rules")
            
            if len(pattern_clusters) > 10:
                recommendations.append("Implement cluster-based attack classification")
            
            # Performance-based recommendations
            for phase, metrics in performance_metrics.items():
                if "accuracy" in phase and metrics:
                    avg_accuracy = np.mean(metrics)
                    if avg_accuracy < 0.7:
                        recommendations.append(f"Improve {phase} accuracy (current: {avg_accuracy:.2f})")
            
            # Incident-based recommendations
            high_risk_incidents = [inc for inc in processed_incidents if inc.risk_score > 0.8]
            if len(high_risk_incidents) > len(processed_incidents) * 0.3:
                recommendations.append("High number of high-risk incidents detected - review security measures")
            
            # General recommendations
            recommendations.append("Regular review of learned patterns recommended")
            recommendations.append("Consider implementing automated response based on learned patterns")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Error generating recommendations"]
    
    async def _update_performance_metrics(self, learning_report: LearningReport):
        """Update performance metrics"""
        try:
            # Track learning metrics
            self.learning_metrics["patterns_learned"].append(learning_report.patterns_learned)
            self.learning_metrics["clusters_created"].append(learning_report.clusters_created)
            self.learning_metrics["feedback_processed"].append(learning_report.feedback_processed)
            
            # Track performance history
            self.performance_history.append({
                "timestamp": learning_report.timestamp,
                "patterns_learned": learning_report.patterns_learned,
                "clusters_created": learning_report.clusters_created,
                "feedback_processed": learning_report.feedback_processed
            })
            
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")
    
    async def _update_pipeline_status(self, stage: PipelineStage, status: str, progress: float):
        """Update pipeline status"""
        try:
            self.pipeline_status = PipelineStatus(
                stage=stage,
                status=status,
                progress=progress,
                timestamp=datetime.utcnow().isoformat(),
                metadata={
                    "pipeline_version": "1.0.0"
                }
            )
            
        except Exception as e:
            logger.error(f"Pipeline status update failed: {e}")
    
    async def get_pipeline_status(self) -> Optional[PipelineStatus]:
        """Get current pipeline status"""
        try:
            return self.pipeline_status
            
        except Exception as e:
            logger.error(f"Pipeline status retrieval failed: {e}")
            return None
    
    async def get_learning_reports(self) -> List[LearningReport]:
        """Get all learning reports"""
        try:
            return self.learning_reports
            
        except Exception as e:
            logger.error(f"Learning reports retrieval failed: {e}")
            return []
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            return {
                "learning_metrics": dict(self.learning_metrics),
                "performance_history": list(self.performance_history),
                "feedback_metrics": await self.feedback_loop.get_performance_metrics(),
                "pattern_metrics": await self.pattern_extractor.get_pattern_statistics()
            }
            
        except Exception as e:
            logger.error(f"Performance metrics retrieval failed: {e}")
            return {}
    
    async def export_pipeline_data(self, format: str = "json") -> str:
        """Export pipeline data"""
        try:
            if format.lower() == "json":
                data = {
                    "pipeline_status": self.pipeline_status.__dict__ if self.pipeline_status else None,
                    "learning_reports": [report.__dict__ for report in self.learning_reports],
                    "performance_metrics": await self.get_performance_metrics(),
                    "feedback_data": await self.feedback_loop.export_feedback_data(),
                    "pattern_data": await self.pattern_extractor.export_patterns(),
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Pipeline data export failed: {e}")
            return ""
