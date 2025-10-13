"""
CVSS v3.1 Calculator for Red Team Service
Implements Common Vulnerability Scoring System for ML security vulnerabilities
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CVSSMetrics:
    """CVSS v3.1 metrics structure"""
    attack_vector: str = "network"  # network, adjacent, local, physical
    attack_complexity: str = "low"  # low, high
    privileges_required: str = "none"  # none, low, high
    user_interaction: str = "none"  # none, required
    scope: str = "unchanged"  # unchanged, changed
    confidentiality_impact: str = "none"  # none, low, high
    integrity_impact: str = "none"  # none, low, high
    availability_impact: str = "none"  # none, low, high
    exploit_code_maturity: str = "functional"  # not_defined, unproven, proof_of_concept, functional, high
    remediation_level: str = "unavailable"  # not_defined, official_fix, temporary_fix, workaround, unavailable
    report_confidence: str = "confirmed"  # not_defined, unknown, reasonable, confirmed

class CVSSCalculator:
    """Calculate CVSS v3.1 scores for ML security vulnerabilities"""
    
    def __init__(self):
        self.base_metrics = {
            "attack_vector": {"network": 0.85, "adjacent": 0.62, "local": 0.55, "physical": 0.2},
            "attack_complexity": {"low": 0.77, "high": 0.44},
            "privileges_required": {"none": 0.85, "low": 0.62, "high": 0.27},
            "user_interaction": {"none": 0.85, "required": 0.62},
            "confidentiality_impact": {"none": 0, "low": 0.22, "high": 0.56},
            "integrity_impact": {"none": 0, "low": 0.22, "high": 0.56},
            "availability_impact": {"none": 0, "low": 0.22, "high": 0.56}
        }
        
        self.temporal_metrics = {
            "exploit_code_maturity": {"not_defined": 1.0, "unproven": 0.91, "proof_of_concept": 0.94, 
                                    "functional": 0.97, "high": 1.0},
            "remediation_level": {"not_defined": 1.0, "official_fix": 0.95, "temporary_fix": 0.96, 
                                "workaround": 0.97, "unavailable": 1.0},
            "report_confidence": {"not_defined": 1.0, "unknown": 0.92, "reasonable": 0.96, "confirmed": 1.0}
        }
    
    def calculate_cvss_score(self, vulnerability: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate CVSS v3.1 score for a vulnerability
        
        Args:
            vulnerability: Dictionary containing vulnerability metrics
            
        Returns:
            Dictionary with CVSS scores and severity rating
        """
        try:
            # Extract metrics with defaults
            metrics = CVSSMetrics(
                attack_vector=vulnerability.get("attack_vector", "network"),
                attack_complexity=vulnerability.get("attack_complexity", "low"),
                privileges_required=vulnerability.get("privileges_required", "none"),
                user_interaction=vulnerability.get("user_interaction", "none"),
                scope=vulnerability.get("scope", "unchanged"),
                confidentiality_impact=vulnerability.get("confidentiality_impact", "none"),
                integrity_impact=vulnerability.get("integrity_impact", "none"),
                availability_impact=vulnerability.get("availability_impact", "none"),
                exploit_code_maturity=vulnerability.get("exploit_code_maturity", "functional"),
                remediation_level=vulnerability.get("remediation_level", "unavailable"),
                report_confidence=vulnerability.get("report_confidence", "confirmed")
            )
            
            # Calculate base score
            base_score = self._calculate_base_score(metrics)
            
            # Calculate temporal score
            temporal_score = self._calculate_temporal_score(metrics, base_score)
            
            # Calculate environmental score (simplified for ML context)
            environmental_score = self._calculate_environmental_score(metrics, base_score)
            
            # Overall score (simplified calculation)
            overall_score = min(10.0, (base_score + temporal_score + environmental_score) / 3)
            
            return {
                "cvss_version": "3.1",
                "base_score": round(base_score, 1),
                "temporal_score": round(temporal_score, 1),
                "environmental_score": round(environmental_score, 1),
                "overall_score": round(overall_score, 1),
                "severity": self._get_severity_rating(overall_score),
                "vector_string": self._generate_vector_string(metrics),
                "base_metrics": {
                    "attack_vector": metrics.attack_vector,
                    "attack_complexity": metrics.attack_complexity,
                    "privileges_required": metrics.privileges_required,
                    "user_interaction": metrics.user_interaction,
                    "scope": metrics.scope,
                    "confidentiality_impact": metrics.confidentiality_impact,
                    "integrity_impact": metrics.integrity_impact,
                    "availability_impact": metrics.availability_impact
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating CVSS score: {e}")
            return self._get_default_score()
    
    def _calculate_base_score(self, metrics: CVSSMetrics) -> float:
        """Calculate CVSS v3.1 base score"""
        try:
            # Get base metric values
            av = self.base_metrics["attack_vector"][metrics.attack_vector]
            ac = self.base_metrics["attack_complexity"][metrics.attack_complexity]
            pr = self.base_metrics["privileges_required"][metrics.privileges_required]
            ui = self.base_metrics["user_interaction"][metrics.user_interaction]
            
            # Adjust privileges required based on scope
            if metrics.scope == "changed":
                pr_values = {"none": 0.85, "low": 0.68, "high": 0.50}
                pr = pr_values.get(metrics.privileges_required, pr)
            
            # Calculate exploitability
            exploitability = 8.22 * av * ac * pr * ui
            
            # Calculate impact
            c = self.base_metrics["confidentiality_impact"][metrics.confidentiality_impact]
            i = self.base_metrics["integrity_impact"][metrics.integrity_impact]
            a = self.base_metrics["availability_impact"][metrics.availability_impact]
            
            impact_subscore = 1 - ((1 - c) * (1 - i) * (1 - a))
            
            if metrics.scope == "unchanged":
                impact = 6.42 * impact_subscore
            else:  # scope == "changed"
                impact = 7.52 * (impact_subscore - 0.029) - 3.25 * ((impact_subscore - 0.02) ** 15)
            
            if impact <= 0:
                return 0.0
            
            # Calculate base score
            if exploitability + impact <= 0:
                return 0.0
            
            base_score = min(10.0, 1.08 * (exploitability + impact))
            return base_score
            
        except Exception as e:
            logger.error(f"Error calculating base score: {e}")
            return 0.0
    
    def _calculate_temporal_score(self, metrics: CVSSMetrics, base_score: float) -> float:
        """Calculate CVSS v3.1 temporal score"""
        try:
            ecm = self.temporal_metrics["exploit_code_maturity"][metrics.exploit_code_maturity]
            rl = self.temporal_metrics["remediation_level"][metrics.remediation_level]
            rc = self.temporal_metrics["report_confidence"][metrics.report_confidence]
            
            temporal_score = base_score * ecm * rl * rc
            return min(10.0, temporal_score)
            
        except Exception as e:
            logger.error(f"Error calculating temporal score: {e}")
            return base_score
    
    def _calculate_environmental_score(self, metrics: CVSSMetrics, base_score: float) -> float:
        """Calculate CVSS v3.1 environmental score (simplified for ML context)"""
        try:
            # For ML systems, environmental factors are often similar
            # This is a simplified calculation
            environmental_multiplier = 1.0
            
            # Adjust based on ML-specific factors
            if metrics.confidentiality_impact == "high":
                environmental_multiplier *= 1.1  # Data breach risk
            if metrics.integrity_impact == "high":
                environmental_multiplier *= 1.05  # Model corruption risk
            if metrics.availability_impact == "high":
                environmental_multiplier *= 1.02  # Service disruption risk
            
            environmental_score = base_score * environmental_multiplier
            return min(10.0, environmental_score)
            
        except Exception as e:
            logger.error(f"Error calculating environmental score: {e}")
            return base_score
    
    def _get_severity_rating(self, score: float) -> str:
        """Map CVSS score to severity rating"""
        if score == 0.0:
            return "NONE"
        elif score < 4.0:
            return "LOW"
        elif score < 7.0:
            return "MEDIUM"
        elif score < 9.0:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _generate_vector_string(self, metrics: CVSSMetrics) -> str:
        """Generate CVSS vector string"""
        try:
            vector = f"CVSS:3.1/AV:{metrics.attack_vector[0].upper()}/AC:{metrics.attack_complexity[0].upper()}/PR:{metrics.privileges_required[0].upper()}/UI:{metrics.user_interaction[0].upper()}/S:{metrics.scope[0].upper()}/C:{metrics.confidentiality_impact[0].upper()}/I:{metrics.integrity_impact[0].upper()}/A:{metrics.availability_impact[0].upper()}"
            return vector
        except Exception as e:
            logger.error(f"Error generating vector string: {e}")
            return "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N"
    
    def _get_default_score(self) -> Dict[str, Any]:
        """Return default CVSS score for errors"""
        return {
            "cvss_version": "3.1",
            "base_score": 0.0,
            "temporal_score": 0.0,
            "environmental_score": 0.0,
            "overall_score": 0.0,
            "severity": "NONE",
            "vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N",
            "base_metrics": {}
        }
    
    def assess_ml_vulnerability(self, attack_category: str, detected: bool, 
                              confidence: float, model_type: str = "production") -> Dict[str, Any]:
        """
        Assess ML-specific vulnerability with appropriate CVSS metrics
        
        Args:
            attack_category: Type of attack (prompt_injection, etc.)
            detected: Whether attack was detected
            confidence: Detection confidence (0-1)
            model_type: Type of model (production, staging, development)
            
        Returns:
            CVSS assessment tailored for ML vulnerabilities
        """
        # Map attack categories to CVSS metrics
        vulnerability_mapping = {
            "prompt_injection": {
                "attack_vector": "network",
                "attack_complexity": "low",
                "privileges_required": "none",
                "user_interaction": "none",
                "scope": "unchanged",
                "confidentiality_impact": "high" if not detected else "none",
                "integrity_impact": "high" if not detected else "none",
                "availability_impact": "low" if not detected else "none"
            },
            "jailbreak": {
                "attack_vector": "network",
                "attack_complexity": "low",
                "privileges_required": "none",
                "user_interaction": "none",
                "scope": "unchanged",
                "confidentiality_impact": "high" if not detected else "none",
                "integrity_impact": "high" if not detected else "none",
                "availability_impact": "low" if not detected else "none"
            },
            "system_extraction": {
                "attack_vector": "network",
                "attack_complexity": "low",
                "privileges_required": "none",
                "user_interaction": "none",
                "scope": "unchanged",
                "confidentiality_impact": "high" if not detected else "none",
                "integrity_impact": "none",
                "availability_impact": "none"
            },
            "code_injection": {
                "attack_vector": "network",
                "attack_complexity": "low",
                "privileges_required": "none",
                "user_interaction": "none",
                "scope": "changed",
                "confidentiality_impact": "high" if not detected else "none",
                "integrity_impact": "high" if not detected else "none",
                "availability_impact": "high" if not detected else "none"
            },
            "dos_attack": {
                "attack_vector": "network",
                "attack_complexity": "low",
                "privileges_required": "none",
                "user_interaction": "none",
                "scope": "unchanged",
                "confidentiality_impact": "none",
                "integrity_impact": "none",
                "availability_impact": "high" if not detected else "none"
            }
        }
        
        # Get base metrics for attack category
        base_metrics = vulnerability_mapping.get(attack_category, {
            "attack_vector": "network",
            "attack_complexity": "low",
            "privileges_required": "none",
            "user_interaction": "none",
            "scope": "unchanged",
            "confidentiality_impact": "medium" if not detected else "none",
            "integrity_impact": "medium" if not detected else "none",
            "availability_impact": "low" if not detected else "none"
        })
        
        # Adjust based on model type
        if model_type == "production":
            base_metrics["confidentiality_impact"] = "high" if base_metrics["confidentiality_impact"] != "none" else "none"
            base_metrics["integrity_impact"] = "high" if base_metrics["integrity_impact"] != "none" else "none"
        
        # Add temporal metrics
        base_metrics.update({
            "exploit_code_maturity": "functional",
            "remediation_level": "unavailable",
            "report_confidence": "confirmed"
        })
        
        return self.calculate_cvss_score(base_metrics)