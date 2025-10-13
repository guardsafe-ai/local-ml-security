"""
Certification Manager
Coordinates different certification methods and provides unified interface
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

from .randomized_smoothing import RandomizedSmoothingCertifier, SmoothingConfig, SmoothingType, CertificationResult
from .interval_bound_propagation import IBPCertifier, IBPConfig, IBPType, IBPCertificationResult

logger = logging.getLogger(__name__)


class CertificationMethod(Enum):
    """Certification methods"""
    RANDOMIZED_SMOOTHING = "randomized_smoothing"
    IBP = "ibp"
    COMBINED = "combined"


@dataclass
class CertificationRequest:
    """Request for certification"""
    model: nn.Module
    input_data: torch.Tensor
    true_label: int
    methods: List[CertificationMethod]
    target_radius: Optional[float] = None
    smoothing_config: Optional[SmoothingConfig] = None
    ibp_config: Optional[IBPConfig] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class UnifiedCertificationResult:
    """Unified certification result"""
    method: CertificationMethod
    is_certified: bool
    certified_radius: float
    confidence: float
    verification_time: float
    raw_result: Any
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ComprehensiveCertification:
    """Comprehensive certification results"""
    request: CertificationRequest
    individual_results: List[UnifiedCertificationResult]
    combined_result: Optional[UnifiedCertificationResult]
    overall_insights: List[str]
    overall_recommendations: List[str]
    analysis_timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CertificationManager:
    """
    Certification Manager
    Coordinates different certification methods and provides unified interface
    """
    
    def __init__(self):
        """Initialize certification manager"""
        self.smoothing_certifier = RandomizedSmoothingCertifier()
        self.ibp_certifier = IBPCertifier()
        
        self.certification_history: List[ComprehensiveCertification] = []
        
        logger.info("✅ Initialized Certification Manager")
    
    async def certify_robustness(self, request: CertificationRequest) -> ComprehensiveCertification:
        """
        Perform comprehensive robustness certification
        """
        try:
            logger.info(f"Starting comprehensive certification: {[m.value for m in request.methods]}")
            
            individual_results = []
            
            # Perform individual certifications
            for method in request.methods:
                try:
                    result = await self._perform_individual_certification(request, method)
                    if result:
                        individual_results.append(result)
                except Exception as e:
                    logger.warning(f"Certification failed for {method.value}: {e}")
                    continue
            
            # Perform combined analysis
            combined_result = await self._perform_combined_analysis(individual_results)
            
            # Generate overall insights and recommendations
            overall_insights = self._generate_overall_insights(individual_results, combined_result)
            overall_recommendations = self._generate_overall_recommendations(individual_results, combined_result)
            
            # Create comprehensive certification
            comprehensive_certification = ComprehensiveCertification(
                request=request,
                individual_results=individual_results,
                combined_result=combined_result,
                overall_insights=overall_insights,
                overall_recommendations=overall_recommendations,
                analysis_timestamp=datetime.utcnow(),
                metadata={
                    "n_methods": len(request.methods),
                    "successful_certifications": len(individual_results)
                }
            )
            
            # Store in history
            self.certification_history.append(comprehensive_certification)
            
            logger.info(f"Comprehensive certification completed: {len(individual_results)} successful certifications")
            return comprehensive_certification
            
        except Exception as e:
            logger.error(f"Comprehensive certification failed: {e}")
            raise
    
    async def _perform_individual_certification(self, 
                                              request: CertificationRequest, 
                                              method: CertificationMethod) -> Optional[UnifiedCertificationResult]:
        """Perform individual certification based on method"""
        try:
            if method == CertificationMethod.RANDOMIZED_SMOOTHING:
                return await self._certify_with_smoothing(request)
            elif method == CertificationMethod.IBP:
                return await self._certify_with_ibp(request)
            else:
                logger.warning(f"Unknown certification method: {method}")
                return None
                
        except Exception as e:
            logger.error(f"Individual certification failed for {method.value}: {e}")
            return None
    
    async def _certify_with_smoothing(self, request: CertificationRequest) -> UnifiedCertificationResult:
        """Certify using randomized smoothing"""
        try:
            # Use provided config or create default
            if request.smoothing_config is None:
                smoothing_config = SmoothingConfig(
                    smoothing_type=SmoothingType.GAUSSIAN,
                    noise_std=0.25,
                    n_samples=1000,
                    alpha=0.001,
                    confidence_level=0.99
                )
            else:
                smoothing_config = request.smoothing_config
            
            # Perform certification
            result = await self.smoothing_certifier.certify_robustness(
                model=request.model,
                input_data=request.input_data,
                true_label=request.true_label,
                smoothing_config=smoothing_config,
                target_radius=request.target_radius
            )
            
            # Convert to unified format
            return UnifiedCertificationResult(
                method=CertificationMethod.RANDOMIZED_SMOOTHING,
                is_certified=result.is_certified,
                certified_radius=result.certified_radius,
                confidence=result.confidence,
                verification_time=0.0,  # Not tracked in smoothing
                raw_result=result,
                metadata={
                    "n_samples": result.n_samples,
                    "p_value": result.p_value,
                    "smoothing_type": result.smoothing_config.smoothing_type.value
                }
            )
            
        except Exception as e:
            logger.error(f"Smoothing certification failed: {e}")
            raise
    
    async def _certify_with_ibp(self, request: CertificationRequest) -> UnifiedCertificationResult:
        """Certify using IBP"""
        try:
            # Use provided config or create default
            if request.ibp_config is None:
                ibp_config = IBPConfig(
                    ibp_type=IBPType.CROWN_IBP,
                    epsilon=0.1,
                    method="backward",
                    use_alpha=True,
                    alpha=0.0,
                    beta=1.0,
                    gamma=1.0,
                    use_ibp=True,
                    use_alpha_crown=True
                )
            else:
                ibp_config = request.ibp_config
            
            # Perform certification
            result = await self.ibp_certifier.certify_robustness(
                model=request.model,
                input_data=request.input_data,
                true_label=request.true_label,
                ibp_config=ibp_config
            )
            
            # Convert to unified format
            return UnifiedCertificationResult(
                method=CertificationMethod.IBP,
                is_certified=result.is_certified,
                certified_radius=result.certified_radius,
                confidence=1.0,  # IBP provides exact bounds
                verification_time=result.verification_time,
                raw_result=result,
                metadata={
                    "n_layers": len(result.bounds),
                    "ibp_type": result.ibp_config.ibp_type.value,
                    "epsilon": result.ibp_config.epsilon
                }
            )
            
        except Exception as e:
            logger.error(f"IBP certification failed: {e}")
            raise
    
    async def _perform_combined_analysis(self, 
                                       individual_results: List[UnifiedCertificationResult]) -> Optional[UnifiedCertificationResult]:
        """Perform combined analysis of multiple certification methods"""
        try:
            if len(individual_results) < 2:
                return None
            
            # Check if all methods agree on certification
            certified_methods = [r for r in individual_results if r.is_certified]
            not_certified_methods = [r for r in individual_results if not r.is_certified]
            
            # Calculate combined radius (conservative approach)
            if certified_methods:
                combined_radius = min(r.certified_radius for r in certified_methods)
            else:
                combined_radius = 0.0
            
            # Determine combined certification status
            if len(certified_methods) == len(individual_results):
                # All methods agree on certification
                combined_certified = True
                combined_confidence = np.mean([r.confidence for r in individual_results])
            elif len(not_certified_methods) == len(individual_results):
                # All methods agree on no certification
                combined_certified = False
                combined_confidence = np.mean([r.confidence for r in individual_results])
            else:
                # Methods disagree - use conservative approach
                combined_certified = False
                combined_confidence = 0.5
            
            # Calculate combined verification time
            combined_time = np.mean([r.verification_time for r in individual_results])
            
            return UnifiedCertificationResult(
                method=CertificationMethod.COMBINED,
                is_certified=combined_certified,
                certified_radius=combined_radius,
                confidence=combined_confidence,
                verification_time=combined_time,
                raw_result=individual_results,
                metadata={
                    "n_methods": len(individual_results),
                    "certified_methods": len(certified_methods),
                    "not_certified_methods": len(not_certified_methods),
                    "agreement": len(certified_methods) == len(individual_results) or len(not_certified_methods) == len(individual_results)
                }
            )
            
        except Exception as e:
            logger.error(f"Combined analysis failed: {e}")
            return None
    
    def _generate_overall_insights(self, 
                                 individual_results: List[UnifiedCertificationResult], 
                                 combined_result: Optional[UnifiedCertificationResult]) -> List[str]:
        """Generate overall insights from certification results"""
        try:
            insights = []
            
            # Analyze individual results
            certified_methods = [r for r in individual_results if r.is_certified]
            not_certified_methods = [r for r in individual_results if not r.is_certified]
            
            if certified_methods:
                insights.append(f"{len(certified_methods)} out of {len(individual_results)} methods certified robustness")
                
                # Analyze radius consistency
                radii = [r.certified_radius for r in certified_methods]
                if len(radii) > 1:
                    radius_std = np.std(radii)
                    if radius_std < 0.01:
                        insights.append("Certified radii are highly consistent across methods")
                    elif radius_std < 0.05:
                        insights.append("Certified radii show moderate variation across methods")
                    else:
                        insights.append("Certified radii show significant variation across methods")
            
            if not_certified_methods:
                insights.append(f"{len(not_certified_methods)} methods failed to certify robustness")
            
            # Analyze method agreement
            if len(certified_methods) == len(individual_results):
                insights.append("All certification methods agree on robustness")
            elif len(not_certified_methods) == len(individual_results):
                insights.append("All certification methods agree on lack of robustness")
            else:
                insights.append("Certification methods show disagreement on robustness")
            
            # Analyze confidence levels
            confidences = [r.confidence for r in individual_results]
            mean_confidence = np.mean(confidences)
            if mean_confidence > 0.9:
                insights.append("High confidence in certification results")
            elif mean_confidence > 0.7:
                insights.append("Moderate confidence in certification results")
            else:
                insights.append("Low confidence in certification results")
            
            # Analyze verification times
            times = [r.verification_time for r in individual_results if r.verification_time > 0]
            if times:
                mean_time = np.mean(times)
                if mean_time < 1.0:
                    insights.append("Fast verification times across methods")
                elif mean_time < 5.0:
                    insights.append("Moderate verification times across methods")
                else:
                    insights.append("Slow verification times across methods")
            
            return insights
            
        except Exception as e:
            logger.error(f"Overall insight generation failed: {e}")
            return []
    
    def _generate_overall_recommendations(self, 
                                        individual_results: List[UnifiedCertificationResult], 
                                        combined_result: Optional[UnifiedCertificationResult]) -> List[str]:
        """Generate overall recommendations from certification results"""
        try:
            recommendations = []
            
            # Analyze certification status
            certified_methods = [r for r in individual_results if r.is_certified]
            not_certified_methods = [r for r in individual_results if not r.is_certified]
            
            if certified_methods:
                recommendations.append("Model shows robustness according to certified methods")
                
                # Recommend based on radius
                radii = [r.certified_radius for r in certified_methods]
                min_radius = min(radii)
                if min_radius > 0.1:
                    recommendations.append("Model has strong robustness - suitable for high-security applications")
                elif min_radius > 0.05:
                    recommendations.append("Model has moderate robustness - suitable for standard applications")
                else:
                    recommendations.append("Model has weak robustness - consider additional hardening")
            
            if not_certified_methods:
                recommendations.append("Model failed robustness certification - investigate vulnerabilities")
                
                # Recommend based on failure reasons
                for result in not_certified_methods:
                    if result.method == CertificationMethod.RANDOMIZED_SMOOTHING:
                        recommendations.append("Consider increasing noise level or sample size for randomized smoothing")
                    elif result.method == CertificationMethod.IBP:
                        recommendations.append("Consider using more sophisticated IBP methods or reducing epsilon")
            
            # Analyze method disagreement
            if len(certified_methods) > 0 and len(not_certified_methods) > 0:
                recommendations.append("Methods disagree on robustness - conduct additional analysis")
                recommendations.append("Consider using more sophisticated certification methods")
            
            # Recommend based on confidence
            confidences = [r.confidence for r in individual_results]
            mean_confidence = np.mean(confidences)
            if mean_confidence < 0.7:
                recommendations.append("Low confidence in results - increase sample size or use more methods")
            
            # Recommend based on verification time
            times = [r.verification_time for r in individual_results if r.verification_time > 0]
            if times:
                mean_time = np.mean(times)
                if mean_time > 10.0:
                    recommendations.append("Slow verification - consider optimizing certification methods")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Overall recommendation generation failed: {e}")
            return []
    
    async def batch_certify(self, 
                          model: nn.Module,
                          input_batch: torch.Tensor,
                          true_labels: torch.Tensor,
                          methods: List[CertificationMethod],
                          smoothing_config: Optional[SmoothingConfig] = None,
                          ibp_config: Optional[IBPConfig] = None) -> List[ComprehensiveCertification]:
        """Certify robustness for a batch of inputs"""
        try:
            logger.info(f"Batch certifying {len(input_batch)} inputs with {len(methods)} methods")
            
            results = []
            
            for i in range(len(input_batch)):
                try:
                    request = CertificationRequest(
                        model=model,
                        input_data=input_batch[i],
                        true_label=true_labels[i].item(),
                        methods=methods,
                        smoothing_config=smoothing_config,
                        ibp_config=ibp_config
                    )
                    
                    result = await self.certify_robustness(request)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Batch certification failed for input {i}: {e}")
                    continue
            
            logger.info(f"Batch certification completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Batch certification failed: {e}")
            return []
    
    async def get_certification_summary(self) -> Dict[str, Any]:
        """Get summary of all certification results"""
        try:
            if not self.certification_history:
                return {"message": "No certification results available"}
            
            # Calculate overall statistics
            total_certifications = len(self.certification_history)
            successful_certifications = sum(1 for c in self.certification_history if c.combined_result and c.combined_result.is_certified)
            success_rate = successful_certifications / total_certifications
            
            # Calculate method-specific statistics
            method_stats = defaultdict(lambda: {"total": 0, "successful": 0, "radii": []})
            
            for certification in self.certification_history:
                for result in certification.individual_results:
                    method_stats[result.method.value]["total"] += 1
                    if result.is_certified:
                        method_stats[result.method.value]["successful"] += 1
                        method_stats[result.method.value]["radii"].append(result.certified_radius)
            
            # Calculate success rates and radius statistics for each method
            for method, stats in method_stats.items():
                stats["success_rate"] = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
                if stats["radii"]:
                    stats["mean_radius"] = np.mean(stats["radii"])
                    stats["std_radius"] = np.std(stats["radii"])
                    stats["max_radius"] = np.max(stats["radii"])
                    stats["min_radius"] = np.min(stats["radii"])
                else:
                    stats["mean_radius"] = stats["std_radius"] = stats["max_radius"] = stats["min_radius"] = 0.0
            
            return {
                "total_certifications": total_certifications,
                "successful_certifications": successful_certifications,
                "overall_success_rate": success_rate,
                "method_statistics": dict(method_stats),
                "summary_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Certification summary generation failed: {e}")
            return {}
    
    async def get_certification_history(self, limit: int = 10) -> List[ComprehensiveCertification]:
        """Get certification history"""
        return self.certification_history[-limit:]
    
    async def export_certification_data(self, format: str = "json") -> str:
        """Export certification data"""
        try:
            if format.lower() == "json":
                data = {
                    "certification_history": [c.__dict__ for c in self.certification_history],
                    "summary": await self.get_certification_summary(),
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Certification data export failed: {e}")
            return ""
