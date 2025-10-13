"""
Privacy Attack Manager
Coordinates different privacy attack methods and provides unified interface
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
from collections import defaultdict

from .membership_inference import MembershipInferenceAttacker, MIAttackConfig, MIAttackType, MIAttackResult
from .model_inversion import ModelInversionAttacker, InversionConfig, InversionMethod, InversionResult
from .data_extraction import DataExtractionAttacker, ExtractionConfig, ExtractionMethod, ExtractionResult

logger = logging.getLogger(__name__)


class PrivacyAttackType(Enum):
    """Types of privacy attacks"""
    MEMBERSHIP_INFERENCE = "membership_inference"
    MODEL_INVERSION = "model_inversion"
    DATA_EXTRACTION = "data_extraction"
    COMPREHENSIVE = "comprehensive"


@dataclass
class PrivacyAttackRequest:
    """Request for privacy attack"""
    model: nn.Module
    attack_types: List[PrivacyAttackType]
    target_data: Optional[torch.Tensor] = None
    target_labels: Optional[torch.Tensor] = None
    non_member_data: Optional[torch.Tensor] = None
    non_member_labels: Optional[torch.Tensor] = None
    target_class: Optional[int] = None
    target_layer: Optional[str] = None
    mi_config: Optional[MIAttackConfig] = None
    inversion_config: Optional[InversionConfig] = None
    extraction_config: Optional[ExtractionConfig] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class UnifiedPrivacyResult:
    """Unified privacy attack result"""
    attack_type: PrivacyAttackType
    success: bool
    confidence: float
    privacy_risk: float
    raw_result: Any
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ComprehensivePrivacyAnalysis:
    """Comprehensive privacy analysis results"""
    request: PrivacyAttackRequest
    individual_results: List[UnifiedPrivacyResult]
    combined_risk_assessment: Dict[str, Any]
    overall_insights: List[str]
    overall_recommendations: List[str]
    analysis_timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PrivacyAttackManager:
    """
    Privacy Attack Manager
    Coordinates different privacy attack methods and provides unified interface
    """
    
    def __init__(self):
        """Initialize privacy attack manager"""
        self.mi_attacker = MembershipInferenceAttacker()
        self.inversion_attacker = ModelInversionAttacker()
        self.extraction_attacker = DataExtractionAttacker()
        
        self.privacy_analyses: List[ComprehensivePrivacyAnalysis] = []
        
        logger.info("✅ Initialized Privacy Attack Manager")
    
    async def perform_privacy_analysis(self, request: PrivacyAttackRequest) -> ComprehensivePrivacyAnalysis:
        """
        Perform comprehensive privacy analysis
        """
        try:
            logger.info(f"Starting privacy analysis: {[t.value for t in request.attack_types]}")
            
            individual_results = []
            
            # Perform individual privacy attacks
            for attack_type in request.attack_types:
                try:
                    result = await self._perform_individual_privacy_attack(request, attack_type)
                    if result:
                        individual_results.append(result)
                except Exception as e:
                    logger.warning(f"Privacy attack failed for {attack_type.value}: {e}")
                    continue
            
            # Perform combined risk assessment
            combined_risk = await self._perform_combined_risk_assessment(individual_results)
            
            # Generate overall insights and recommendations
            overall_insights = self._generate_overall_insights(individual_results, combined_risk)
            overall_recommendations = self._generate_overall_recommendations(individual_results, combined_risk)
            
            # Create comprehensive analysis
            comprehensive_analysis = ComprehensivePrivacyAnalysis(
                request=request,
                individual_results=individual_results,
                combined_risk_assessment=combined_risk,
                overall_insights=overall_insights,
                overall_recommendations=overall_recommendations,
                analysis_timestamp=datetime.utcnow(),
                metadata={
                    "n_attacks": len(request.attack_types),
                    "successful_attacks": len(individual_results)
                }
            )
            
            # Store in history
            self.privacy_analyses.append(comprehensive_analysis)
            
            logger.info(f"Privacy analysis completed: {len(individual_results)} successful attacks")
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"Privacy analysis failed: {e}")
            raise
    
    async def _perform_individual_privacy_attack(self, 
                                               request: PrivacyAttackRequest, 
                                               attack_type: PrivacyAttackType) -> Optional[UnifiedPrivacyResult]:
        """Perform individual privacy attack based on type"""
        try:
            if attack_type == PrivacyAttackType.MEMBERSHIP_INFERENCE:
                return await self._perform_membership_inference_attack(request)
            elif attack_type == PrivacyAttackType.MODEL_INVERSION:
                return await self._perform_model_inversion_attack(request)
            elif attack_type == PrivacyAttackType.DATA_EXTRACTION:
                return await self._perform_data_extraction_attack(request)
            else:
                logger.warning(f"Unknown privacy attack type: {attack_type}")
                return None
                
        except Exception as e:
            logger.error(f"Individual privacy attack failed for {attack_type.value}: {e}")
            return None
    
    async def _perform_membership_inference_attack(self, request: PrivacyAttackRequest) -> UnifiedPrivacyResult:
        """Perform membership inference attack"""
        try:
            # Use provided config or create default
            if request.mi_config is None:
                mi_config = MIAttackConfig(
                    attack_type=MIAttackType.SHADOW_MODEL,
                    shadow_models=5,
                    shadow_epochs=10,
                    attack_epochs=10,
                    learning_rate=0.001,
                    batch_size=32,
                    threshold=0.5,
                    confidence_threshold=0.8
                )
            else:
                mi_config = request.mi_config
            
            # Check if we have required data
            if request.target_data is None or request.target_labels is None or request.non_member_data is None or request.non_member_labels is None:
                raise ValueError("Membership inference requires target_data, target_labels, non_member_data, and non_member_labels")
            
            # Perform attack
            result = await self.mi_attacker.perform_attack(
                target_model=request.model,
                target_data=request.target_data,
                target_labels=request.target_labels,
                non_member_data=request.non_member_data,
                non_member_labels=request.non_member_labels,
                attack_config=mi_config
            )
            
            # Calculate privacy risk
            privacy_risk = self._calculate_mi_privacy_risk(result)
            
            # Convert to unified format
            return UnifiedPrivacyResult(
                attack_type=PrivacyAttackType.MEMBERSHIP_INFERENCE,
                success=result.accuracy > 0.5,
                confidence=result.accuracy,
                privacy_risk=privacy_risk,
                raw_result=result,
                metadata={
                    "attack_type": result.attack_type.value,
                    "accuracy": result.accuracy,
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1_score": result.f1_score,
                    "auc_score": result.auc_score
                }
            )
            
        except Exception as e:
            logger.error(f"Membership inference attack failed: {e}")
            raise
    
    async def _perform_model_inversion_attack(self, request: PrivacyAttackRequest) -> UnifiedPrivacyResult:
        """Perform model inversion attack"""
        try:
            # Use provided config or create default
            if request.inversion_config is None:
                inversion_config = InversionConfig(
                    method=InversionMethod.GRADIENT_DESCENT,
                    target_class=request.target_class or 0,
                    num_iterations=1000,
                    learning_rate=0.01,
                    regularization_weight=0.01,
                    noise_level=0.1,
                    image_size=(224, 224),
                    num_channels=3
                )
            else:
                inversion_config = request.inversion_config
            
            # Perform attack
            result = await self.inversion_attacker.perform_inversion(
                target_model=request.model,
                target_class=inversion_config.target_class,
                inversion_config=inversion_config
            )
            
            # Calculate privacy risk
            privacy_risk = self._calculate_inversion_privacy_risk(result)
            
            # Convert to unified format
            return UnifiedPrivacyResult(
                attack_type=PrivacyAttackType.MODEL_INVERSION,
                success=result.confidence > 0.5,
                confidence=result.confidence,
                privacy_risk=privacy_risk,
                raw_result=result,
                metadata={
                    "method": result.method.value,
                    "target_class": result.target_class,
                    "confidence": result.confidence,
                    "reconstruction_loss": result.reconstruction_loss,
                    "num_iterations": result.num_iterations
                }
            )
            
        except Exception as e:
            logger.error(f"Model inversion attack failed: {e}")
            raise
    
    async def _perform_data_extraction_attack(self, request: PrivacyAttackRequest) -> UnifiedPrivacyResult:
        """Perform data extraction attack"""
        try:
            # Use provided config or create default
            if request.extraction_config is None:
                extraction_config = ExtractionConfig(
                    method=ExtractionMethod.GRADIENT_BASED,
                    target_layer=request.target_layer,
                    num_samples=100,
                    num_clusters=10,
                    learning_rate=0.01,
                    num_iterations=1000,
                    regularization_weight=0.01,
                    noise_level=0.1
                )
            else:
                extraction_config = request.extraction_config
            
            # Perform attack
            result = await self.extraction_attacker.perform_extraction(
                target_model=request.model,
                extraction_config=extraction_config
            )
            
            # Calculate privacy risk
            privacy_risk = self._calculate_extraction_privacy_risk(result)
            
            # Convert to unified format
            return UnifiedPrivacyResult(
                attack_type=PrivacyAttackType.DATA_EXTRACTION,
                success=result.reconstruction_quality > 0.5,
                confidence=result.extraction_confidence,
                privacy_risk=privacy_risk,
                raw_result=result,
                metadata={
                    "method": result.method.value,
                    "reconstruction_quality": result.reconstruction_quality,
                    "extraction_confidence": result.extraction_confidence,
                    "num_extracted": result.num_extracted
                }
            )
            
        except Exception as e:
            logger.error(f"Data extraction attack failed: {e}")
            raise
    
    def _calculate_mi_privacy_risk(self, result: MIAttackResult) -> float:
        """Calculate privacy risk from membership inference result"""
        try:
            # Higher accuracy means higher privacy risk
            accuracy_risk = result.accuracy
            
            # Higher AUC means higher privacy risk
            auc_risk = result.auc_score
            
            # Combine risks
            privacy_risk = (accuracy_risk + auc_risk) / 2.0
            
            return min(1.0, max(0.0, privacy_risk))
            
        except Exception as e:
            logger.error(f"MI privacy risk calculation failed: {e}")
            return 0.0
    
    def _calculate_inversion_privacy_risk(self, result: InversionResult) -> float:
        """Calculate privacy risk from model inversion result"""
        try:
            # Higher confidence means higher privacy risk
            confidence_risk = result.confidence
            
            # Lower reconstruction loss means higher privacy risk
            loss_risk = 1.0 - min(1.0, result.reconstruction_loss / 10.0)  # Normalize loss
            
            # Combine risks
            privacy_risk = (confidence_risk + loss_risk) / 2.0
            
            return min(1.0, max(0.0, privacy_risk))
            
        except Exception as e:
            logger.error(f"Inversion privacy risk calculation failed: {e}")
            return 0.0
    
    def _calculate_extraction_privacy_risk(self, result: ExtractionResult) -> float:
        """Calculate privacy risk from data extraction result"""
        try:
            # Higher reconstruction quality means higher privacy risk
            quality_risk = result.reconstruction_quality
            
            # Higher extraction confidence means higher privacy risk
            confidence_risk = result.extraction_confidence
            
            # Combine risks
            privacy_risk = (quality_risk + confidence_risk) / 2.0
            
            return min(1.0, max(0.0, privacy_risk))
            
        except Exception as e:
            logger.error(f"Extraction privacy risk calculation failed: {e}")
            return 0.0
    
    async def _perform_combined_risk_assessment(self, individual_results: List[UnifiedPrivacyResult]) -> Dict[str, Any]:
        """Perform combined risk assessment"""
        try:
            if not individual_results:
                return {"overall_risk": 0.0, "risk_level": "LOW"}
            
            # Calculate overall risk
            privacy_risks = [r.privacy_risk for r in individual_results]
            overall_risk = np.mean(privacy_risks)
            
            # Determine risk level
            if overall_risk >= 0.8:
                risk_level = "CRITICAL"
            elif overall_risk >= 0.6:
                risk_level = "HIGH"
            elif overall_risk >= 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            # Calculate attack-specific risks
            attack_risks = {}
            for result in individual_results:
                attack_risks[result.attack_type.value] = {
                    "risk": result.privacy_risk,
                    "success": result.success,
                    "confidence": result.confidence
                }
            
            # Calculate risk distribution
            risk_distribution = {
                "mean": overall_risk,
                "std": np.std(privacy_risks),
                "min": np.min(privacy_risks),
                "max": np.max(privacy_risks)
            }
            
            return {
                "overall_risk": overall_risk,
                "risk_level": risk_level,
                "attack_risks": attack_risks,
                "risk_distribution": risk_distribution,
                "n_attacks": len(individual_results),
                "successful_attacks": sum(1 for r in individual_results if r.success)
            }
            
        except Exception as e:
            logger.error(f"Combined risk assessment failed: {e}")
            return {"overall_risk": 0.0, "risk_level": "UNKNOWN"}
    
    def _generate_overall_insights(self, 
                                 individual_results: List[UnifiedPrivacyResult], 
                                 combined_risk: Dict[str, Any]) -> List[str]:
        """Generate overall insights from privacy analysis"""
        try:
            insights = []
            
            # Analyze overall risk
            overall_risk = combined_risk.get("overall_risk", 0.0)
            risk_level = combined_risk.get("risk_level", "UNKNOWN")
            
            insights.append(f"Overall privacy risk: {overall_risk:.2f} ({risk_level})")
            
            # Analyze successful attacks
            successful_attacks = [r for r in individual_results if r.success]
            if successful_attacks:
                insights.append(f"{len(successful_attacks)} out of {len(individual_results)} privacy attacks were successful")
                
                # Analyze attack types
                attack_types = [r.attack_type.value for r in successful_attacks]
                unique_attack_types = list(set(attack_types))
                insights.append(f"Successful attack types: {', '.join(unique_attack_types)}")
            
            # Analyze risk levels
            high_risk_attacks = [r for r in individual_results if r.privacy_risk > 0.7]
            if high_risk_attacks:
                insights.append(f"{len(high_risk_attacks)} attacks pose high privacy risk")
            
            # Analyze confidence levels
            high_confidence_attacks = [r for r in individual_results if r.confidence > 0.8]
            if high_confidence_attacks:
                insights.append(f"{len(high_confidence_attacks)} attacks have high confidence")
            
            # Analyze specific attack insights
            for result in individual_results:
                if result.attack_type == PrivacyAttackType.MEMBERSHIP_INFERENCE:
                    if result.success:
                        insights.append("Model is vulnerable to membership inference attacks")
                    else:
                        insights.append("Model appears resistant to membership inference attacks")
                
                elif result.attack_type == PrivacyAttackType.MODEL_INVERSION:
                    if result.success:
                        insights.append("Model is vulnerable to model inversion attacks")
                    else:
                        insights.append("Model appears resistant to model inversion attacks")
                
                elif result.attack_type == PrivacyAttackType.DATA_EXTRACTION:
                    if result.success:
                        insights.append("Model is vulnerable to data extraction attacks")
                    else:
                        insights.append("Model appears resistant to data extraction attacks")
            
            return insights
            
        except Exception as e:
            logger.error(f"Overall insight generation failed: {e}")
            return []
    
    def _generate_overall_recommendations(self, 
                                        individual_results: List[UnifiedPrivacyResult], 
                                        combined_risk: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations from privacy analysis"""
        try:
            recommendations = []
            
            # Analyze overall risk level
            risk_level = combined_risk.get("risk_level", "UNKNOWN")
            
            if risk_level == "CRITICAL":
                recommendations.append("CRITICAL: Immediate action required to address privacy vulnerabilities")
                recommendations.append("Consider implementing differential privacy or federated learning")
                recommendations.append("Review and update data handling practices")
            
            elif risk_level == "HIGH":
                recommendations.append("HIGH: Significant privacy risks detected - implement countermeasures")
                recommendations.append("Consider using privacy-preserving techniques")
                recommendations.append("Regular privacy audits recommended")
            
            elif risk_level == "MEDIUM":
                recommendations.append("MEDIUM: Moderate privacy risks - monitor and improve")
                recommendations.append("Consider implementing additional privacy protections")
            
            else:
                recommendations.append("LOW: Privacy risks are manageable")
                recommendations.append("Continue monitoring for privacy vulnerabilities")
            
            # Analyze specific attack recommendations
            for result in individual_results:
                if result.attack_type == PrivacyAttackType.MEMBERSHIP_INFERENCE and result.success:
                    recommendations.append("Implement membership inference defenses (e.g., regularization, noise)")
                
                elif result.attack_type == PrivacyAttackType.MODEL_INVERSION and result.success:
                    recommendations.append("Implement model inversion defenses (e.g., gradient masking)")
                
                elif result.attack_type == PrivacyAttackType.DATA_EXTRACTION and result.success:
                    recommendations.append("Implement data extraction defenses (e.g., feature obfuscation)")
            
            # General recommendations
            if any(r.success for r in individual_results):
                recommendations.append("Consider using privacy-preserving machine learning techniques")
                recommendations.append("Implement comprehensive privacy testing in CI/CD pipeline")
                recommendations.append("Regular privacy risk assessments recommended")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Overall recommendation generation failed: {e}")
            return []
    
    async def get_privacy_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all privacy analyses"""
        try:
            if not self.privacy_analyses:
                return {"message": "No privacy analyses available"}
            
            # Calculate overall statistics
            total_analyses = len(self.privacy_analyses)
            high_risk_analyses = sum(1 for a in self.privacy_analyses 
                                   if a.combined_risk_assessment.get("risk_level") in ["HIGH", "CRITICAL"])
            high_risk_rate = high_risk_analyses / total_analyses
            
            # Calculate attack-specific statistics
            attack_stats = defaultdict(lambda: {"count": 0, "successes": 0, "risks": []})
            
            for analysis in self.privacy_analyses:
                for result in analysis.individual_results:
                    attack_type = result.attack_type.value
                    attack_stats[attack_type]["count"] += 1
                    if result.success:
                        attack_stats[attack_type]["successes"] += 1
                    attack_stats[attack_type]["risks"].append(result.privacy_risk)
            
            # Calculate averages for each attack type
            for attack_type, stats in attack_stats.items():
                stats["success_rate"] = stats["successes"] / stats["count"] if stats["count"] > 0 else 0
                stats["mean_risk"] = np.mean(stats["risks"]) if stats["risks"] else 0
                stats["std_risk"] = np.std(stats["risks"]) if stats["risks"] else 0
            
            return {
                "total_analyses": total_analyses,
                "high_risk_analyses": high_risk_analyses,
                "high_risk_rate": high_risk_rate,
                "attack_statistics": dict(attack_stats),
                "summary_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Privacy analysis summary generation failed: {e}")
            return {}
    
    async def get_privacy_analysis_history(self, limit: int = 10) -> List[ComprehensivePrivacyAnalysis]:
        """Get privacy analysis history"""
        return self.privacy_analyses[-limit:]
    
    async def export_privacy_analysis_data(self, format: str = "json") -> str:
        """Export privacy analysis data"""
        try:
            if format.lower() == "json":
                data = {
                    "privacy_analyses": [a.__dict__ for a in self.privacy_analyses],
                    "summary": await self.get_privacy_analysis_summary(),
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Privacy analysis data export failed: {e}")
            return ""
