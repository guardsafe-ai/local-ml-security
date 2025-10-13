"""
Executive Reporter
Creates high-level executive reports with AI risk scores, compliance posture, and ROI analysis.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import math

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceStatus(Enum):
    """Compliance status"""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"

@dataclass
class AIRiskScore:
    """AI Risk Score calculation"""
    overall_score: float
    security_risk: float
    privacy_risk: float
    compliance_risk: float
    operational_risk: float
    business_risk: float
    confidence_level: float
    factors: List[str] = field(default_factory=list)

@dataclass
class CompliancePosture:
    """Compliance posture assessment"""
    overall_score: float
    frameworks: Dict[str, float] = field(default_factory=dict)
    critical_violations: int = 0
    high_violations: int = 0
    medium_violations: int = 0
    low_violations: int = 0
    recommendations: List[str] = field(default_factory=list)

@dataclass
class ROIAnalysis:
    """ROI analysis for security investments"""
    total_investment: float
    cost_savings: float
    risk_reduction: float
    roi_percentage: float
    payback_period_months: float
    net_present_value: float
    break_even_point: datetime
    benefits: List[str] = field(default_factory=list)

@dataclass
class ExecutiveSummary:
    """Executive summary data"""
    report_date: datetime
    organization: str
    ai_risk_score: AIRiskScore
    compliance_posture: CompliancePosture
    roi_analysis: ROIAnalysis
    key_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)

class ExecutiveReporter:
    """Creates executive-level security reports"""
    
    def __init__(self):
        self.risk_weights = {
            "security_risk": 0.3,
            "privacy_risk": 0.25,
            "compliance_risk": 0.2,
            "operational_risk": 0.15,
            "business_risk": 0.1
        }
        
        self.compliance_frameworks = [
            "SOC2", "ISO27001", "OWASP_LLM", "NIST", "PCI_DSS", 
            "HIPAA", "ISO42001", "EU_AI_ACT", "GDPR", "CCPA"
        ]
    
    def calculate_ai_risk_score(self, security_data: Dict[str, Any]) -> AIRiskScore:
        """Calculate comprehensive AI risk score"""
        try:
            # Security risk calculation
            security_risk = self._calculate_security_risk(security_data)
            
            # Privacy risk calculation
            privacy_risk = self._calculate_privacy_risk(security_data)
            
            # Compliance risk calculation
            compliance_risk = self._calculate_compliance_risk(security_data)
            
            # Operational risk calculation
            operational_risk = self._calculate_operational_risk(security_data)
            
            # Business risk calculation
            business_risk = self._calculate_business_risk(security_data)
            
            # Overall weighted score
            overall_score = (
                security_risk * self.risk_weights["security_risk"] +
                privacy_risk * self.risk_weights["privacy_risk"] +
                compliance_risk * self.risk_weights["compliance_risk"] +
                operational_risk * self.risk_weights["operational_risk"] +
                business_risk * self.risk_weights["business_risk"]
            )
            
            # Confidence level based on data completeness
            confidence_level = self._calculate_confidence_level(security_data)
            
            # Risk factors
            factors = self._identify_risk_factors(security_data)
            
            return AIRiskScore(
                overall_score=overall_score,
                security_risk=security_risk,
                privacy_risk=privacy_risk,
                compliance_risk=compliance_risk,
                operational_risk=operational_risk,
                business_risk=business_risk,
                confidence_level=confidence_level,
                factors=factors
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate AI risk score: {e}")
            return AIRiskScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def _calculate_security_risk(self, data: Dict[str, Any]) -> float:
        """Calculate security risk score"""
        try:
            vulnerabilities = data.get("vulnerabilities", [])
            attacks = data.get("attacks", [])
            
            # Base risk from vulnerabilities
            vuln_risk = 0.0
            if vulnerabilities:
                severity_weights = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2}
                for vuln in vulnerabilities:
                    severity = vuln.get("severity", "medium").lower()
                    vuln_risk += severity_weights.get(severity, 0.5)
                vuln_risk = min(vuln_risk / len(vulnerabilities), 1.0)
            
            # Attack success rate
            attack_risk = 0.0
            if attacks:
                success_rates = [attack.get("success_rate", 0.0) for attack in attacks]
                attack_risk = sum(success_rates) / len(success_rates)
            
            # Combine risks
            security_risk = (vuln_risk * 0.6 + attack_risk * 0.4)
            return min(security_risk, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate security risk: {e}")
            return 0.5
    
    def _calculate_privacy_risk(self, data: Dict[str, Any]) -> float:
        """Calculate privacy risk score"""
        try:
            privacy_breaches = data.get("privacy_breaches", [])
            data_types = data.get("sensitive_data_types", [])
            
            # Breach impact
            breach_risk = 0.0
            if privacy_breaches:
                breach_scores = [breach.get("severity_score", 0.5) for breach in privacy_breaches]
                breach_risk = sum(breach_scores) / len(breach_scores)
            
            # Data sensitivity
            sensitivity_risk = 0.0
            if data_types:
                sensitivity_weights = {
                    "PII": 1.0, "PHI": 0.9, "financial": 0.8, 
                    "biometric": 0.9, "location": 0.6, "behavioral": 0.7
                }
                for data_type in data_types:
                    sensitivity_risk += sensitivity_weights.get(data_type.lower(), 0.5)
                sensitivity_risk = min(sensitivity_risk / len(data_types), 1.0)
            
            # Combine risks
            privacy_risk = (breach_risk * 0.7 + sensitivity_risk * 0.3)
            return min(privacy_risk, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate privacy risk: {e}")
            return 0.5
    
    def _calculate_compliance_risk(self, data: Dict[str, Any]) -> float:
        """Calculate compliance risk score"""
        try:
            compliance_data = data.get("compliance", {})
            violations = data.get("compliance_violations", [])
            
            # Framework compliance scores
            framework_risk = 0.0
            if compliance_data:
                scores = [score for score in compliance_data.values() if isinstance(score, (int, float))]
                if scores:
                    framework_risk = 1.0 - (sum(scores) / len(scores))
            
            # Violation impact
            violation_risk = 0.0
            if violations:
                severity_weights = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2}
                for violation in violations:
                    severity = violation.get("severity", "medium").lower()
                    violation_risk += severity_weights.get(severity, 0.5)
                violation_risk = min(violation_risk / len(violations), 1.0)
            
            # Combine risks
            compliance_risk = (framework_risk * 0.6 + violation_risk * 0.4)
            return min(compliance_risk, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate compliance risk: {e}")
            return 0.5
    
    def _calculate_operational_risk(self, data: Dict[str, Any]) -> float:
        """Calculate operational risk score"""
        try:
            system_metrics = data.get("system_metrics", {})
            worker_status = data.get("worker_status", {})
            
            # System performance risk
            perf_risk = 0.0
            if system_metrics:
                cpu_usage = system_metrics.get("cpu_usage", 0)
                memory_usage = system_metrics.get("memory_usage", 0)
                perf_risk = max(cpu_usage, memory_usage) / 100.0
            
            # Worker availability risk
            worker_risk = 0.0
            if worker_status:
                total_workers = worker_status.get("total", 1)
                active_workers = worker_status.get("active", 0)
                worker_risk = 1.0 - (active_workers / total_workers)
            
            # Combine risks
            operational_risk = (perf_risk * 0.5 + worker_risk * 0.5)
            return min(operational_risk, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate operational risk: {e}")
            return 0.5
    
    def _calculate_business_risk(self, data: Dict[str, Any]) -> float:
        """Calculate business risk score"""
        try:
            # Reputation impact from security incidents
            incidents = data.get("security_incidents", [])
            incident_risk = 0.0
            if incidents:
                impact_scores = [incident.get("business_impact", 0.5) for incident in incidents]
                incident_risk = sum(impact_scores) / len(impact_scores)
            
            # Financial impact
            financial_impact = data.get("financial_impact", 0.0)
            financial_risk = min(financial_impact / 1000000.0, 1.0)  # Normalize to millions
            
            # Combine risks
            business_risk = (incident_risk * 0.6 + financial_risk * 0.4)
            return min(business_risk, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate business risk: {e}")
            return 0.5
    
    def _calculate_confidence_level(self, data: Dict[str, Any]) -> float:
        """Calculate confidence level based on data completeness"""
        try:
            required_fields = [
                "vulnerabilities", "attacks", "compliance", 
                "system_metrics", "privacy_breaches"
            ]
            
            present_fields = sum(1 for field in required_fields if field in data and data[field])
            confidence = present_fields / len(required_fields)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence level: {e}")
            return 0.5
    
    def _identify_risk_factors(self, data: Dict[str, Any]) -> List[str]:
        """Identify key risk factors"""
        factors = []
        
        try:
            # Security factors
            vulnerabilities = data.get("vulnerabilities", [])
            if vulnerabilities:
                critical_vulns = [v for v in vulnerabilities if v.get("severity", "").lower() == "critical"]
                if critical_vulns:
                    factors.append(f"{len(critical_vulns)} critical vulnerabilities identified")
            
            # Privacy factors
            privacy_breaches = data.get("privacy_breaches", [])
            if privacy_breaches:
                factors.append(f"{len(privacy_breaches)} privacy breaches detected")
            
            # Compliance factors
            violations = data.get("compliance_violations", [])
            if violations:
                high_violations = [v for v in violations if v.get("severity", "").lower() in ["high", "critical"]]
                if high_violations:
                    factors.append(f"{len(high_violations)} high-severity compliance violations")
            
            # Operational factors
            system_metrics = data.get("system_metrics", {})
            if system_metrics:
                cpu_usage = system_metrics.get("cpu_usage", 0)
                if cpu_usage > 90:
                    factors.append("High CPU usage detected")
                
                memory_usage = system_metrics.get("memory_usage", 0)
                if memory_usage > 90:
                    factors.append("High memory usage detected")
            
            return factors
            
        except Exception as e:
            logger.error(f"Failed to identify risk factors: {e}")
            return []
    
    def assess_compliance_posture(self, compliance_data: Dict[str, Any]) -> CompliancePosture:
        """Assess overall compliance posture"""
        try:
            # Calculate framework scores
            framework_scores = {}
            for framework in self.compliance_frameworks:
                score = compliance_data.get(framework, {}).get("score", 0.0)
                framework_scores[framework] = score
            
            # Overall compliance score
            overall_score = sum(framework_scores.values()) / len(framework_scores) if framework_scores else 0.0
            
            # Count violations by severity
            violations = compliance_data.get("violations", [])
            critical_violations = len([v for v in violations if v.get("severity", "").lower() == "critical"])
            high_violations = len([v for v in violations if v.get("severity", "").lower() == "high"])
            medium_violations = len([v for v in violations if v.get("severity", "").lower() == "medium"])
            low_violations = len([v for v in violations if v.get("severity", "").lower() == "low"])
            
            # Generate recommendations
            recommendations = self._generate_compliance_recommendations(framework_scores, violations)
            
            return CompliancePosture(
                overall_score=overall_score,
                frameworks=framework_scores,
                critical_violations=critical_violations,
                high_violations=high_violations,
                medium_violations=medium_violations,
                low_violations=low_violations,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Failed to assess compliance posture: {e}")
            return CompliancePosture(0.0)
    
    def _generate_compliance_recommendations(self, framework_scores: Dict[str, float], 
                                           violations: List[Dict[str, Any]]) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        try:
            # Framework-specific recommendations
            for framework, score in framework_scores.items():
                if score < 0.7:
                    recommendations.append(f"Improve {framework} compliance (current score: {score:.1%})")
            
            # Violation-based recommendations
            critical_violations = [v for v in violations if v.get("severity", "").lower() == "critical"]
            if critical_violations:
                recommendations.append(f"Address {len(critical_violations)} critical compliance violations immediately")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Maintain current compliance posture")
            else:
                recommendations.append("Implement continuous compliance monitoring")
                recommendations.append("Conduct regular compliance assessments")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate compliance recommendations: {e}")
            return ["Review compliance status"]
    
    def calculate_roi(self, investment_data: Dict[str, Any]) -> ROIAnalysis:
        """Calculate ROI for security investments"""
        try:
            total_investment = investment_data.get("total_investment", 0.0)
            cost_savings = investment_data.get("cost_savings", 0.0)
            risk_reduction = investment_data.get("risk_reduction", 0.0)
            
            # Calculate ROI percentage
            if total_investment > 0:
                roi_percentage = ((cost_savings + risk_reduction) / total_investment) * 100
            else:
                roi_percentage = 0.0
            
            # Calculate payback period
            if cost_savings > 0:
                payback_period_months = (total_investment / cost_savings) * 12
            else:
                payback_period_months = float('inf')
            
            # Calculate NPV (simplified)
            discount_rate = investment_data.get("discount_rate", 0.1)
            years = investment_data.get("analysis_period_years", 3)
            npv = 0.0
            for year in range(1, years + 1):
                annual_benefit = (cost_savings + risk_reduction) / years
                npv += annual_benefit / ((1 + discount_rate) ** year)
            npv -= total_investment
            
            # Calculate break-even point
            if cost_savings > 0:
                months_to_break_even = total_investment / (cost_savings / 12)
                break_even_point = datetime.now() + timedelta(days=months_to_break_even * 30)
            else:
                break_even_point = datetime.now() + timedelta(days=365)
            
            # Generate benefits list
            benefits = self._generate_roi_benefits(cost_savings, risk_reduction)
            
            return ROIAnalysis(
                total_investment=total_investment,
                cost_savings=cost_savings,
                risk_reduction=risk_reduction,
                roi_percentage=roi_percentage,
                payback_period_months=payback_period_months,
                net_present_value=npv,
                break_even_point=break_even_point,
                benefits=benefits
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate ROI: {e}")
            return ROIAnalysis(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, datetime.now())
    
    def _generate_roi_benefits(self, cost_savings: float, risk_reduction: float) -> List[str]:
        """Generate ROI benefits list"""
        benefits = []
        
        if cost_savings > 0:
            benefits.append(f"Direct cost savings: ${cost_savings:,.2f}")
        
        if risk_reduction > 0:
            benefits.append(f"Risk reduction value: ${risk_reduction:,.2f}")
        
        if cost_savings > 0 or risk_reduction > 0:
            benefits.append("Improved security posture")
            benefits.append("Reduced compliance risk")
            benefits.append("Enhanced business continuity")
            benefits.append("Better stakeholder confidence")
        
        return benefits
    
    def generate_executive_summary(self, security_data: Dict[str, Any], 
                                 compliance_data: Dict[str, Any],
                                 investment_data: Dict[str, Any]) -> ExecutiveSummary:
        """Generate comprehensive executive summary"""
        try:
            # Calculate AI risk score
            ai_risk_score = self.calculate_ai_risk_score(security_data)
            
            # Assess compliance posture
            compliance_posture = self.assess_compliance_posture(compliance_data)
            
            # Calculate ROI
            roi_analysis = self.calculate_roi(investment_data)
            
            # Generate key findings
            key_findings = self._generate_key_findings(ai_risk_score, compliance_posture, security_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(ai_risk_score, compliance_posture, security_data)
            
            # Generate next steps
            next_steps = self._generate_next_steps(ai_risk_score, compliance_posture, recommendations)
            
            return ExecutiveSummary(
                report_date=datetime.now(),
                organization=security_data.get("organization", "Unknown"),
                ai_risk_score=ai_risk_score,
                compliance_posture=compliance_posture,
                roi_analysis=roi_analysis,
                key_findings=key_findings,
                recommendations=recommendations,
                next_steps=next_steps
            )
            
        except Exception as e:
            logger.error(f"Failed to generate executive summary: {e}")
            return ExecutiveSummary(
                datetime.now(), "Unknown", 
                AIRiskScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                CompliancePosture(0.0),
                ROIAnalysis(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, datetime.now())
            )
    
    def _generate_key_findings(self, ai_risk_score: AIRiskScore, 
                             compliance_posture: CompliancePosture,
                             security_data: Dict[str, Any]) -> List[str]:
        """Generate key findings"""
        findings = []
        
        try:
            # Risk level findings
            if ai_risk_score.overall_score > 0.8:
                findings.append("Critical AI risk level detected - immediate action required")
            elif ai_risk_score.overall_score > 0.6:
                findings.append("High AI risk level - priority attention needed")
            elif ai_risk_score.overall_score > 0.4:
                findings.append("Medium AI risk level - monitoring recommended")
            else:
                findings.append("Low AI risk level - maintain current controls")
            
            # Compliance findings
            if compliance_posture.overall_score > 0.8:
                findings.append("Strong compliance posture across frameworks")
            elif compliance_posture.overall_score > 0.6:
                findings.append("Moderate compliance posture - some improvements needed")
            else:
                findings.append("Weak compliance posture - significant improvements required")
            
            # Security findings
            vulnerabilities = security_data.get("vulnerabilities", [])
            if vulnerabilities:
                critical_vulns = [v for v in vulnerabilities if v.get("severity", "").lower() == "critical"]
                if critical_vulns:
                    findings.append(f"{len(critical_vulns)} critical vulnerabilities require immediate remediation")
            
            # ROI findings
            if ai_risk_score.overall_score > 0.5:
                findings.append("Security investments show positive ROI potential")
            
            return findings
            
        except Exception as e:
            logger.error(f"Failed to generate key findings: {e}")
            return ["Analysis in progress"]
    
    def _generate_recommendations(self, ai_risk_score: AIRiskScore,
                                compliance_posture: CompliancePosture,
                                security_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        
        try:
            # Risk-based recommendations
            if ai_risk_score.security_risk > 0.7:
                recommendations.append("Implement additional security controls and monitoring")
            
            if ai_risk_score.privacy_risk > 0.7:
                recommendations.append("Enhance data privacy protections and access controls")
            
            if ai_risk_score.compliance_risk > 0.7:
                recommendations.append("Address compliance gaps and implement remediation plans")
            
            # Compliance recommendations
            recommendations.extend(compliance_posture.recommendations)
            
            # General recommendations
            if ai_risk_score.overall_score > 0.6:
                recommendations.append("Conduct regular security assessments and penetration testing")
                recommendations.append("Implement continuous monitoring and threat detection")
                recommendations.append("Develop incident response and business continuity plans")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return ["Review security posture"]
    
    def _generate_next_steps(self, ai_risk_score: AIRiskScore,
                           compliance_posture: CompliancePosture,
                           recommendations: List[str]) -> List[str]:
        """Generate next steps"""
        next_steps = []
        
        try:
            # Immediate actions
            if ai_risk_score.overall_score > 0.8:
                next_steps.append("Convene emergency security committee meeting")
                next_steps.append("Implement immediate risk mitigation measures")
            
            # Short-term actions
            if compliance_posture.critical_violations > 0:
                next_steps.append("Address critical compliance violations within 30 days")
            
            if ai_risk_score.overall_score > 0.6:
                next_steps.append("Develop comprehensive security improvement plan")
                next_steps.append("Allocate budget for security enhancements")
            
            # Medium-term actions
            next_steps.append("Schedule quarterly security reviews")
            next_steps.append("Implement continuous compliance monitoring")
            next_steps.append("Conduct staff security training")
            
            # Long-term actions
            next_steps.append("Develop long-term security strategy")
            next_steps.append("Establish security metrics and KPIs")
            
            return next_steps
            
        except Exception as e:
            logger.error(f"Failed to generate next steps: {e}")
            return ["Review and plan security improvements"]
    
    def export_executive_report(self, summary: ExecutiveSummary, 
                              format_type: str = "json") -> str:
        """Export executive report"""
        try:
            report_data = {
                "executive_summary": {
                    "report_date": summary.report_date.isoformat(),
                    "organization": summary.organization,
                    "ai_risk_score": {
                        "overall_score": summary.ai_risk_score.overall_score,
                        "security_risk": summary.ai_risk_score.security_risk,
                        "privacy_risk": summary.ai_risk_score.privacy_risk,
                        "compliance_risk": summary.ai_risk_score.compliance_risk,
                        "operational_risk": summary.ai_risk_score.operational_risk,
                        "business_risk": summary.ai_risk_score.business_risk,
                        "confidence_level": summary.ai_risk_score.confidence_level,
                        "factors": summary.ai_risk_score.factors
                    },
                    "compliance_posture": {
                        "overall_score": summary.compliance_posture.overall_score,
                        "frameworks": summary.compliance_posture.frameworks,
                        "critical_violations": summary.compliance_posture.critical_violations,
                        "high_violations": summary.compliance_posture.high_violations,
                        "medium_violations": summary.compliance_posture.medium_violations,
                        "low_violations": summary.compliance_posture.low_violations,
                        "recommendations": summary.compliance_posture.recommendations
                    },
                    "roi_analysis": {
                        "total_investment": summary.roi_analysis.total_investment,
                        "cost_savings": summary.roi_analysis.cost_savings,
                        "risk_reduction": summary.roi_analysis.risk_reduction,
                        "roi_percentage": summary.roi_analysis.roi_percentage,
                        "payback_period_months": summary.roi_analysis.payback_period_months,
                        "net_present_value": summary.roi_analysis.net_present_value,
                        "break_even_point": summary.roi_analysis.break_even_point.isoformat(),
                        "benefits": summary.roi_analysis.benefits
                    },
                    "key_findings": summary.key_findings,
                    "recommendations": summary.recommendations,
                    "next_steps": summary.next_steps
                }
            }
            
            if format_type == "json":
                return json.dumps(report_data, indent=2)
            else:
                return str(report_data)
                
        except Exception as e:
            logger.error(f"Failed to export executive report: {e}")
            return "{}"
