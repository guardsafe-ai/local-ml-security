"""
EU AI Act Compliance
Implementation of EU AI Act compliance testing and validation
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd

logger = logging.getLogger(__name__)


class AIRiskLevel(Enum):
    """EU AI Act Risk Levels"""
    MINIMAL_RISK = "minimal_risk"
    LIMITED_RISK = "limited_risk"
    HIGH_RISK = "high_risk"
    PROHIBITED = "prohibited"


class AIUseCase(Enum):
    """EU AI Act AI Use Cases"""
    BIOMETRIC_IDENTIFICATION = "biometric_identification"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"
    EDUCATION_VOCATIONAL = "education_vocational"
    EMPLOYMENT_WORKER_MANAGEMENT = "employment_worker_management"
    ESSENTIAL_SERVICES = "essential_services"
    LAW_ENFORCEMENT = "law_enforcement"
    MIGRATION_ASYLUM_BORDER = "migration_asylum_border"
    ADMINISTRATIVE_JUDICIAL = "administrative_judicial"
    SOCIAL_SCORING = "social_scoring"
    MANIPULATION = "manipulation"
    EXPLOITATION_VULNERABLE = "exploitation_vulnerable"
    GENERAL_PURPOSE = "general_purpose"


@dataclass
class EUAIActComplianceResult:
    """Result of EU AI Act compliance assessment"""
    risk_level: AIRiskLevel
    use_case: AIUseCase
    compliance_score: float
    requirements_met: Dict[str, bool]
    violations: List[str]
    recommendations: List[str]
    conformity_assessment: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EUAIActCompliance:
    """
    EU AI Act Compliance Testing
    Implements comprehensive testing for EU AI Act compliance
    """
    
    def __init__(self):
        """Initialize EU AI Act compliance tester"""
        self.risk_classifications = self._load_risk_classifications()
        self.compliance_requirements = self._load_compliance_requirements()
        self.conformity_assessment_criteria = self._load_conformity_criteria()
        
        logger.info("✅ Initialized EU AI Act Compliance")
    
    def _load_risk_classifications(self) -> Dict[str, Dict[str, Any]]:
        """Load EU AI Act risk classifications"""
        return {
            "minimal_risk": {
                "description": "AI systems with minimal risk",
                "requirements": ["transparency"],
                "examples": ["spam filters", "video games", "recommendation systems"]
            },
            "limited_risk": {
                "description": "AI systems with limited risk",
                "requirements": ["transparency", "user_awareness"],
                "examples": ["chatbots", "deepfakes", "emotion recognition"]
            },
            "high_risk": {
                "description": "AI systems with high risk",
                "requirements": [
                    "risk_management_system",
                    "data_governance",
                    "technical_documentation",
                    "record_keeping",
                    "transparency",
                    "human_oversight",
                    "accuracy_robustness_cybersecurity",
                    "conformity_assessment"
                ],
                "examples": [
                    "biometric identification",
                    "critical infrastructure",
                    "education",
                    "employment",
                    "essential services",
                    "law enforcement",
                    "migration",
                    "administrative"
                ]
            },
            "prohibited": {
                "description": "Prohibited AI practices",
                "requirements": ["prohibition"],
                "examples": [
                    "social scoring",
                    "manipulation",
                    "exploitation of vulnerable groups",
                    "subliminal techniques",
                    "real-time biometric identification"
                ]
            }
        }
    
    def _load_compliance_requirements(self) -> Dict[str, Dict[str, Any]]:
        """Load compliance requirements for each risk level"""
        return {
            "risk_management_system": {
                "description": "Risk management system for AI systems",
                "requirements": [
                    "risk_identification",
                    "risk_estimation",
                    "risk_evaluation",
                    "risk_mitigation",
                    "risk_monitoring",
                    "risk_documentation"
                ],
                "assessment_criteria": [
                    "systematic_approach",
                    "continuous_monitoring",
                    "stakeholder_involvement",
                    "documentation_quality"
                ]
            },
            "data_governance": {
                "description": "Data governance and quality management",
                "requirements": [
                    "data_quality_assessment",
                    "data_collection_procedures",
                    "data_preprocessing",
                    "data_validation",
                    "bias_mitigation",
                    "data_protection"
                ],
                "assessment_criteria": [
                    "data_quality_metrics",
                    "bias_detection",
                    "privacy_compliance",
                    "data_lineage"
                ]
            },
            "technical_documentation": {
                "description": "Technical documentation requirements",
                "requirements": [
                    "system_description",
                    "technical_specifications",
                    "performance_metrics",
                    "testing_procedures",
                    "validation_results",
                    "risk_assessment"
                ],
                "assessment_criteria": [
                    "completeness",
                    "accuracy",
                    "clarity",
                    "traceability"
                ]
            },
            "record_keeping": {
                "description": "Record keeping and logging requirements",
                "requirements": [
                    "automated_logging",
                    "human_oversight_logs",
                    "decision_records",
                    "audit_trails",
                    "data_retention",
                    "access_controls"
                ],
                "assessment_criteria": [
                    "log_completeness",
                    "retention_compliance",
                    "access_controls",
                    "audit_trail_quality"
                ]
            },
            "transparency": {
                "description": "Transparency and explainability requirements",
                "requirements": [
                    "system_disclosure",
                    "decision_explainability",
                    "user_information",
                    "contact_details",
                    "purpose_clarity",
                    "limitation_disclosure"
                ],
                "assessment_criteria": [
                    "information_completeness",
                    "user_understanding",
                    "explainability_quality",
                    "accessibility"
                ]
            },
            "human_oversight": {
                "description": "Human oversight and control requirements",
                "requirements": [
                    "human_in_loop",
                    "oversight_mechanisms",
                    "intervention_capabilities",
                    "decision_override",
                    "monitoring_systems",
                    "alert_mechanisms"
                ],
                "assessment_criteria": [
                    "oversight_effectiveness",
                    "intervention_capability",
                    "monitoring_quality",
                    "human_control"
                ]
            },
            "accuracy_robustness_cybersecurity": {
                "description": "Accuracy, robustness, and cybersecurity requirements",
                "requirements": [
                    "accuracy_standards",
                    "robustness_testing",
                    "cybersecurity_measures",
                    "adversarial_testing",
                    "performance_monitoring",
                    "incident_response"
                ],
                "assessment_criteria": [
                    "accuracy_metrics",
                    "robustness_scores",
                    "security_measures",
                    "incident_response"
                ]
            },
            "conformity_assessment": {
                "description": "Conformity assessment procedures",
                "requirements": [
                    "internal_audit",
                    "external_verification",
                    "certification_process",
                    "ongoing_monitoring",
                    "compliance_reporting",
                    "remediation_plans"
                ],
                "assessment_criteria": [
                    "audit_quality",
                    "verification_completeness",
                    "certification_validity",
                    "monitoring_effectiveness"
                ]
            }
        }
    
    def _load_conformity_criteria(self) -> Dict[str, List[str]]:
        """Load conformity assessment criteria"""
        return {
            "high_risk_ai_systems": [
                "risk_management_system_compliance",
                "data_governance_compliance",
                "technical_documentation_compliance",
                "record_keeping_compliance",
                "transparency_compliance",
                "human_oversight_compliance",
                "accuracy_robustness_cybersecurity_compliance",
                "conformity_assessment_compliance"
            ],
            "limited_risk_ai_systems": [
                "transparency_compliance",
                "user_awareness_compliance"
            ],
            "minimal_risk_ai_systems": [
                "transparency_compliance"
            ],
            "prohibited_ai_systems": [
                "prohibition_compliance"
            ]
        }
    
    async def classify_ai_system(self, 
                               ai_system_info: Dict[str, Any]) -> Tuple[AIRiskLevel, AIUseCase]:
        """
        Classify AI system according to EU AI Act risk levels
        """
        try:
            logger.info("Starting AI system classification")
            
            # Extract system characteristics
            use_case = ai_system_info.get("use_case", "general_purpose")
            domain = ai_system_info.get("domain", "general")
            biometric_identification = ai_system_info.get("biometric_identification", False)
            real_time = ai_system_info.get("real_time", False)
            social_scoring = ai_system_info.get("social_scoring", False)
            manipulation = ai_system_info.get("manipulation", False)
            vulnerable_groups = ai_system_info.get("vulnerable_groups", False)
            critical_infrastructure = ai_system_info.get("critical_infrastructure", False)
            
            # Check for prohibited practices
            if social_scoring or manipulation or vulnerable_groups:
                return AIRiskLevel.PROHIBITED, AIUseCase.SOCIAL_SCORING
            
            # Check for high-risk use cases
            if biometric_identification and real_time:
                return AIRiskLevel.PROHIBITED, AIUseCase.BIOMETRIC_IDENTIFICATION
            
            if biometric_identification:
                return AIRiskLevel.HIGH_RISK, AIUseCase.BIOMETRIC_IDENTIFICATION
            
            if critical_infrastructure:
                return AIRiskLevel.HIGH_RISK, AIUseCase.CRITICAL_INFRASTRUCTURE
            
            if domain in ["education", "vocational"]:
                return AIRiskLevel.HIGH_RISK, AIUseCase.EDUCATION_VOCATIONAL
            
            if domain in ["employment", "worker_management"]:
                return AIRiskLevel.HIGH_RISK, AIUseCase.EMPLOYMENT_WORKER_MANAGEMENT
            
            if domain in ["essential_services", "utilities", "healthcare"]:
                return AIRiskLevel.HIGH_RISK, AIUseCase.ESSENTIAL_SERVICES
            
            if domain in ["law_enforcement", "security"]:
                return AIRiskLevel.HIGH_RISK, AIUseCase.LAW_ENFORCEMENT
            
            if domain in ["migration", "asylum", "border_control"]:
                return AIRiskLevel.HIGH_RISK, AIUseCase.MIGRATION_ASYLUM_BORDER
            
            if domain in ["administrative", "judicial", "public_services"]:
                return AIRiskLevel.HIGH_RISK, AIUseCase.ADMINISTRATIVE_JUDICIAL
            
            # Check for limited risk
            if use_case in ["chatbot", "deepfake", "emotion_recognition", "recommendation"]:
                return AIRiskLevel.LIMITED_RISK, AIUseCase.GENERAL_PURPOSE
            
            # Default to minimal risk
            return AIRiskLevel.MINIMAL_RISK, AIUseCase.GENERAL_PURPOSE
            
        except Exception as e:
            logger.error(f"AI system classification failed: {e}")
            return AIRiskLevel.MINIMAL_RISK, AIUseCase.GENERAL_PURPOSE
    
    async def assess_compliance(self, 
                              ai_system_info: Dict[str, Any]) -> EUAIActComplianceResult:
        """
        Assess EU AI Act compliance for an AI system
        """
        try:
            logger.info("Starting EU AI Act compliance assessment")
            
            # Classify the AI system
            risk_level, use_case = await self.classify_ai_system(ai_system_info)
            
            # Get applicable requirements
            requirements = self._get_applicable_requirements(risk_level)
            
            # Assess each requirement
            requirements_met = {}
            violations = []
            recommendations = []
            
            for requirement in requirements:
                compliance_result = await self._assess_requirement(
                    requirement, ai_system_info, risk_level
                )
                requirements_met[requirement] = compliance_result["met"]
                
                if not compliance_result["met"]:
                    violations.extend(compliance_result["violations"])
                    recommendations.extend(compliance_result["recommendations"])
            
            # Calculate overall compliance score
            compliance_score = sum(requirements_met.values()) / len(requirements_met)
            
            # Perform conformity assessment
            conformity_assessment = await self._perform_conformity_assessment(
                risk_level, requirements_met, ai_system_info
            )
            
            return EUAIActComplianceResult(
                risk_level=risk_level,
                use_case=use_case,
                compliance_score=compliance_score,
                requirements_met=requirements_met,
                violations=violations,
                recommendations=recommendations,
                conformity_assessment=conformity_assessment,
                metadata={"ai_system_info": ai_system_info}
            )
            
        except Exception as e:
            logger.error(f"EU AI Act compliance assessment failed: {e}")
            return EUAIActComplianceResult(
                risk_level=AIRiskLevel.MINIMAL_RISK,
                use_case=AIUseCase.GENERAL_PURPOSE,
                compliance_score=0.0,
                requirements_met={},
                violations=[f"Assessment failed: {str(e)}"],
                recommendations=["Fix assessment system"],
                conformity_assessment={},
                metadata={"error": str(e)}
            )
    
    async def assess_high_risk_requirements(self, 
                                          ai_system_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess specific high-risk AI system requirements
        """
        try:
            logger.info("Starting high-risk requirements assessment")
            
            results = {}
            
            # Risk Management System
            risk_mgmt_result = await self._assess_risk_management_system(ai_system_info)
            results["risk_management_system"] = risk_mgmt_result
            
            # Data Governance
            data_gov_result = await self._assess_data_governance(ai_system_info)
            results["data_governance"] = data_gov_result
            
            # Technical Documentation
            tech_doc_result = await self._assess_technical_documentation(ai_system_info)
            results["technical_documentation"] = tech_doc_result
            
            # Record Keeping
            record_keeping_result = await self._assess_record_keeping(ai_system_info)
            results["record_keeping"] = record_keeping_result
            
            # Transparency
            transparency_result = await self._assess_transparency(ai_system_info)
            results["transparency"] = transparency_result
            
            # Human Oversight
            human_oversight_result = await self._assess_human_oversight(ai_system_info)
            results["human_oversight"] = human_oversight_result
            
            # Accuracy, Robustness, Cybersecurity
            accuracy_result = await self._assess_accuracy_robustness_cybersecurity(ai_system_info)
            results["accuracy_robustness_cybersecurity"] = accuracy_result
            
            # Conformity Assessment
            conformity_result = await self._assess_conformity_assessment(ai_system_info)
            results["conformity_assessment"] = conformity_result
            
            return results
            
        except Exception as e:
            logger.error(f"High-risk requirements assessment failed: {e}")
            return {"error": str(e)}
    
    def _get_applicable_requirements(self, risk_level: AIRiskLevel) -> List[str]:
        """Get applicable requirements for risk level"""
        if risk_level == AIRiskLevel.HIGH_RISK:
            return [
                "risk_management_system",
                "data_governance",
                "technical_documentation",
                "record_keeping",
                "transparency",
                "human_oversight",
                "accuracy_robustness_cybersecurity",
                "conformity_assessment"
            ]
        elif risk_level == AIRiskLevel.LIMITED_RISK:
            return ["transparency", "user_awareness"]
        elif risk_level == AIRiskLevel.MINIMAL_RISK:
            return ["transparency"]
        else:  # PROHIBITED
            return ["prohibition"]
    
    async def _assess_requirement(self, 
                                requirement: str, 
                                ai_system_info: Dict[str, Any],
                                risk_level: AIRiskLevel) -> Dict[str, Any]:
        """Assess specific requirement compliance"""
        try:
            if requirement == "risk_management_system":
                return await self._assess_risk_management_system(ai_system_info)
            elif requirement == "data_governance":
                return await self._assess_data_governance(ai_system_info)
            elif requirement == "technical_documentation":
                return await self._assess_technical_documentation(ai_system_info)
            elif requirement == "record_keeping":
                return await self._assess_record_keeping(ai_system_info)
            elif requirement == "transparency":
                return await self._assess_transparency(ai_system_info)
            elif requirement == "human_oversight":
                return await self._assess_human_oversight(ai_system_info)
            elif requirement == "accuracy_robustness_cybersecurity":
                return await self._assess_accuracy_robustness_cybersecurity(ai_system_info)
            elif requirement == "conformity_assessment":
                return await self._assess_conformity_assessment(ai_system_info)
            elif requirement == "user_awareness":
                return await self._assess_user_awareness(ai_system_info)
            elif requirement == "prohibition":
                return await self._assess_prohibition(ai_system_info)
            else:
                return {"met": False, "violations": ["Unknown requirement"], "recommendations": ["Implement requirement"]}
                
        except Exception as e:
            logger.error(f"Requirement assessment failed for {requirement}: {e}")
            return {"met": False, "violations": [f"Assessment failed: {str(e)}"], "recommendations": ["Fix assessment"]}
    
    async def _assess_risk_management_system(self, ai_system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk management system compliance"""
        violations = []
        recommendations = []
        
        # Check for risk management system
        if not ai_system_info.get("risk_management_system"):
            violations.append("No risk management system implemented")
            recommendations.append("Implement comprehensive risk management system")
        
        # Check for risk identification
        if not ai_system_info.get("risk_identification"):
            violations.append("No risk identification process")
            recommendations.append("Implement systematic risk identification")
        
        # Check for risk monitoring
        if not ai_system_info.get("risk_monitoring"):
            violations.append("No risk monitoring system")
            recommendations.append("Implement continuous risk monitoring")
        
        met = len(violations) == 0
        return {"met": met, "violations": violations, "recommendations": recommendations}
    
    async def _assess_data_governance(self, ai_system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data governance compliance"""
        violations = []
        recommendations = []
        
        # Check for data quality assessment
        if not ai_system_info.get("data_quality_assessment"):
            violations.append("No data quality assessment")
            recommendations.append("Implement data quality assessment procedures")
        
        # Check for bias mitigation
        if not ai_system_info.get("bias_mitigation"):
            violations.append("No bias mitigation measures")
            recommendations.append("Implement bias detection and mitigation")
        
        # Check for data protection
        if not ai_system_info.get("data_protection"):
            violations.append("Insufficient data protection measures")
            recommendations.append("Implement comprehensive data protection")
        
        met = len(violations) == 0
        return {"met": met, "violations": violations, "recommendations": recommendations}
    
    async def _assess_technical_documentation(self, ai_system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess technical documentation compliance"""
        violations = []
        recommendations = []
        
        # Check for system description
        if not ai_system_info.get("system_description"):
            violations.append("Missing system description")
            recommendations.append("Create comprehensive system description")
        
        # Check for technical specifications
        if not ai_system_info.get("technical_specifications"):
            violations.append("Missing technical specifications")
            recommendations.append("Document technical specifications")
        
        # Check for performance metrics
        if not ai_system_info.get("performance_metrics"):
            violations.append("Missing performance metrics")
            recommendations.append("Define and document performance metrics")
        
        met = len(violations) == 0
        return {"met": met, "violations": violations, "recommendations": recommendations}
    
    async def _assess_record_keeping(self, ai_system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess record keeping compliance"""
        violations = []
        recommendations = []
        
        # Check for automated logging
        if not ai_system_info.get("automated_logging"):
            violations.append("No automated logging system")
            recommendations.append("Implement automated logging")
        
        # Check for audit trails
        if not ai_system_info.get("audit_trails"):
            violations.append("No audit trail system")
            recommendations.append("Implement comprehensive audit trails")
        
        # Check for data retention
        if not ai_system_info.get("data_retention_policy"):
            violations.append("No data retention policy")
            recommendations.append("Implement data retention policy")
        
        met = len(violations) == 0
        return {"met": met, "violations": violations, "recommendations": recommendations}
    
    async def _assess_transparency(self, ai_system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess transparency compliance"""
        violations = []
        recommendations = []
        
        # Check for system disclosure
        if not ai_system_info.get("system_disclosure"):
            violations.append("No system disclosure")
            recommendations.append("Implement system disclosure")
        
        # Check for explainability
        if not ai_system_info.get("explainability"):
            violations.append("No explainability features")
            recommendations.append("Implement explainability features")
        
        # Check for user information
        if not ai_system_info.get("user_information"):
            violations.append("Insufficient user information")
            recommendations.append("Provide comprehensive user information")
        
        met = len(violations) == 0
        return {"met": met, "violations": violations, "recommendations": recommendations}
    
    async def _assess_human_oversight(self, ai_system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess human oversight compliance"""
        violations = []
        recommendations = []
        
        # Check for human in loop
        if not ai_system_info.get("human_in_loop"):
            violations.append("No human in loop mechanism")
            recommendations.append("Implement human in loop")
        
        # Check for oversight mechanisms
        if not ai_system_info.get("oversight_mechanisms"):
            violations.append("No oversight mechanisms")
            recommendations.append("Implement oversight mechanisms")
        
        # Check for intervention capabilities
        if not ai_system_info.get("intervention_capabilities"):
            violations.append("No intervention capabilities")
            recommendations.append("Implement intervention capabilities")
        
        met = len(violations) == 0
        return {"met": met, "violations": violations, "recommendations": recommendations}
    
    async def _assess_accuracy_robustness_cybersecurity(self, ai_system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess accuracy, robustness, and cybersecurity compliance"""
        violations = []
        recommendations = []
        
        # Check for accuracy standards
        if not ai_system_info.get("accuracy_standards"):
            violations.append("No accuracy standards defined")
            recommendations.append("Define accuracy standards")
        
        # Check for robustness testing
        if not ai_system_info.get("robustness_testing"):
            violations.append("No robustness testing")
            recommendations.append("Implement robustness testing")
        
        # Check for cybersecurity measures
        if not ai_system_info.get("cybersecurity_measures"):
            violations.append("Insufficient cybersecurity measures")
            recommendations.append("Implement comprehensive cybersecurity measures")
        
        met = len(violations) == 0
        return {"met": met, "violations": violations, "recommendations": recommendations}
    
    async def _assess_conformity_assessment(self, ai_system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess conformity assessment compliance"""
        violations = []
        recommendations = []
        
        # Check for internal audit
        if not ai_system_info.get("internal_audit"):
            violations.append("No internal audit process")
            recommendations.append("Implement internal audit process")
        
        # Check for external verification
        if not ai_system_info.get("external_verification"):
            violations.append("No external verification")
            recommendations.append("Implement external verification")
        
        # Check for certification
        if not ai_system_info.get("certification"):
            violations.append("No certification process")
            recommendations.append("Implement certification process")
        
        met = len(violations) == 0
        return {"met": met, "violations": violations, "recommendations": recommendations}
    
    async def _assess_user_awareness(self, ai_system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess user awareness compliance"""
        violations = []
        recommendations = []
        
        # Check for user awareness measures
        if not ai_system_info.get("user_awareness"):
            violations.append("No user awareness measures")
            recommendations.append("Implement user awareness measures")
        
        met = len(violations) == 0
        return {"met": met, "violations": violations, "recommendations": recommendations}
    
    async def _assess_prohibition(self, ai_system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess prohibition compliance"""
        violations = []
        recommendations = []
        
        # Check for prohibited practices
        if ai_system_info.get("social_scoring"):
            violations.append("Social scoring detected - prohibited practice")
            recommendations.append("Remove social scoring functionality")
        
        if ai_system_info.get("manipulation"):
            violations.append("Manipulation detected - prohibited practice")
            recommendations.append("Remove manipulation functionality")
        
        if ai_system_info.get("exploitation_vulnerable"):
            violations.append("Exploitation of vulnerable groups detected - prohibited practice")
            recommendations.append("Remove exploitation functionality")
        
        met = len(violations) == 0
        return {"met": met, "violations": violations, "recommendations": recommendations}
    
    async def _perform_conformity_assessment(self, 
                                           risk_level: AIRiskLevel,
                                           requirements_met: Dict[str, bool],
                                           ai_system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Perform conformity assessment"""
        try:
            conformity_score = sum(requirements_met.values()) / len(requirements_met)
            
            if risk_level == AIRiskLevel.HIGH_RISK:
                if conformity_score >= 0.9:
                    conformity_status = "compliant"
                elif conformity_score >= 0.7:
                    conformity_status = "partially_compliant"
                else:
                    conformity_status = "non_compliant"
            else:
                if conformity_score >= 0.8:
                    conformity_status = "compliant"
                elif conformity_score >= 0.6:
                    conformity_status = "partially_compliant"
                else:
                    conformity_status = "non_compliant"
            
            return {
                "conformity_score": conformity_score,
                "conformity_status": conformity_status,
                "risk_level": risk_level.value,
                "requirements_assessed": len(requirements_met),
                "requirements_met": sum(requirements_met.values()),
                "next_assessment_due": "12_months" if conformity_status == "compliant" else "6_months"
            }
            
        except Exception as e:
            logger.error(f"Conformity assessment failed: {e}")
            return {
                "conformity_score": 0.0,
                "conformity_status": "assessment_failed",
                "error": str(e)
            }
    
    async def run_comprehensive_eu_assessment(self, 
                                            ai_system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive EU AI Act assessment"""
        try:
            # Main compliance assessment
            compliance_result = await self.assess_compliance(ai_system_info)
            
            # High-risk specific assessment if applicable
            high_risk_results = {}
            if compliance_result.risk_level == AIRiskLevel.HIGH_RISK:
                high_risk_results = await self.assess_high_risk_requirements(ai_system_info)
            
            return {
                "compliance_result": compliance_result,
                "high_risk_assessment": high_risk_results,
                "assessment_timestamp": asyncio.get_event_loop().time(),
                "assessment_version": "1.0.0"
            }
            
        except Exception as e:
            logger.error(f"Comprehensive EU assessment failed: {e}")
            return {
                "error": str(e),
                "assessment_timestamp": asyncio.get_event_loop().time()
            }
