"""
Audit Report Generator for Compliance and Regulatory Reporting
Generates comprehensive audit reports for SOC 2, ISO 27001, and other compliance frameworks
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Compliance frameworks"""
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    OWASP_LLM = "owasp_llm"
    NIST = "nist"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    ISO42001 = "iso42001"
    EU_AI_ACT = "eu_ai_act"
    GDPR = "gdpr"
    CCPA = "ccpa"


class AuditStatus(Enum):
    """Audit status"""
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"


@dataclass
class AuditControl:
    """Audit control data structure"""
    id: str
    name: str
    description: str
    framework: ComplianceFramework
    category: str
    status: AuditStatus
    evidence: List[str]
    findings: List[str]
    remediation: Optional[str] = None
    last_tested: Optional[datetime] = None
    tester: Optional[str] = None


@dataclass
class AuditFinding:
    """Audit finding data structure"""
    id: str
    control_id: str
    severity: str  # "critical", "high", "medium", "low"
    title: str
    description: str
    evidence: List[str]
    remediation: str
    due_date: Optional[datetime] = None
    assigned_to: Optional[str] = None
    status: str = "open"  # "open", "in_progress", "resolved", "closed"


class AuditReportGenerator:
    """
    Generates comprehensive audit reports for compliance frameworks
    """
    
    def __init__(self):
        """Initialize audit report generator"""
        self.controls: Dict[str, AuditControl] = {}
        self.findings: List[AuditFinding] = []
        self.audit_metadata: Dict[str, Any] = {}
        logger.info("✅ Audit Report Generator initialized")
    
    async def initialize_audit(self, 
                             audit_scope: str,
                             frameworks: List[ComplianceFramework],
                             auditor: str = "ML Security Platform") -> bool:
        """
        Initialize audit with scope and frameworks
        
        Args:
            audit_scope: Description of audit scope
            frameworks: List of compliance frameworks to audit
            auditor: Name of auditor
            
        Returns:
            True if successful
        """
        try:
            self.audit_metadata = {
                'audit_id': f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'audit_scope': audit_scope,
                'frameworks': [f.value for f in frameworks],
                'auditor': auditor,
                'start_date': datetime.now(),
                'end_date': None,
                'status': 'in_progress'
            }
            
            # Initialize controls for each framework
            await self._initialize_framework_controls(frameworks)
            
            logger.info(f"✅ Audit initialized for frameworks: {[f.value for f in frameworks]}")
            return True
            
        except Exception as e:
            logger.error(f"Audit initialization failed: {e}")
            return False
    
    async def _initialize_framework_controls(self, frameworks: List[ComplianceFramework]):
        """Initialize controls for specified frameworks"""
        try:
            for framework in frameworks:
                if framework == ComplianceFramework.SOC2:
                    await self._initialize_soc2_controls()
                elif framework == ComplianceFramework.ISO27001:
                    await self._initialize_iso27001_controls()
                elif framework == ComplianceFramework.OWASP_LLM:
                    await self._initialize_owasp_llm_controls()
                elif framework == ComplianceFramework.NIST:
                    await self._initialize_nist_controls()
                elif framework == ComplianceFramework.PCI_DSS:
                    await self._initialize_pci_dss_controls()
                elif framework == ComplianceFramework.HIPAA:
                    await self._initialize_hipaa_controls()
                elif framework == ComplianceFramework.ISO42001:
                    await self._initialize_iso42001_controls()
                elif framework == ComplianceFramework.EU_AI_ACT:
                    await self._initialize_eu_ai_act_controls()
                elif framework == ComplianceFramework.GDPR:
                    await self._initialize_gdpr_controls()
                elif framework == ComplianceFramework.CCPA:
                    await self._initialize_ccpa_controls()
            
        except Exception as e:
            logger.error(f"Framework controls initialization failed: {e}")
    
    async def _initialize_soc2_controls(self):
        """Initialize SOC 2 Type II controls"""
        try:
            soc2_controls = [
                {
                    'id': 'CC6.1',
                    'name': 'Logical Access Security',
                    'description': 'The entity implements logical access security software, infrastructure, and architectures over protected information assets to protect them from security events to meet the entity\'s objectives.',
                    'category': 'Access Control'
                },
                {
                    'id': 'CC6.2',
                    'name': 'Access Restriction',
                    'description': 'Prior to issuing system credentials and granting system access, the entity registers and authorizes new internal and external users whose access is administered by the entity.',
                    'category': 'Access Control'
                },
                {
                    'id': 'CC6.3',
                    'name': 'Access Credentials',
                    'description': 'The entity authorizes, modifies, or removes access to data, software, functions, and other protected information assets based on a user\'s role and responsibilities.',
                    'category': 'Access Control'
                },
                {
                    'id': 'CC6.4',
                    'name': 'Access Removal',
                    'description': 'The entity restricts access to information assets including hardware, data, software, mobile devices, output, and offline elements.',
                    'category': 'Access Control'
                },
                {
                    'id': 'CC6.5',
                    'name': 'Access Review',
                    'description': 'The entity discontinues or removes access to protected information assets when access is no longer required.',
                    'category': 'Access Control'
                },
                {
                    'id': 'CC6.6',
                    'name': 'Access Restriction',
                    'description': 'The entity restricts access to information assets including hardware, data, software, mobile devices, output, and offline elements.',
                    'category': 'Access Control'
                },
                {
                    'id': 'CC6.7',
                    'name': 'Access Restriction',
                    'description': 'The entity restricts access to information assets including hardware, data, software, mobile devices, output, and offline elements.',
                    'category': 'Access Control'
                },
                {
                    'id': 'CC6.8',
                    'name': 'Access Restriction',
                    'description': 'The entity restricts access to information assets including hardware, data, software, mobile devices, output, and offline elements.',
                    'category': 'Access Control'
                }
            ]
            
            for control_data in soc2_controls:
                control = AuditControl(
                    id=control_data['id'],
                    name=control_data['name'],
                    description=control_data['description'],
                    framework=ComplianceFramework.SOC2,
                    category=control_data['category'],
                    status=AuditStatus.PENDING,
                    evidence=[],
                    findings=[]
                )
                self.controls[control.id] = control
            
        except Exception as e:
            logger.error(f"SOC 2 controls initialization failed: {e}")
    
    async def _initialize_iso27001_controls(self):
        """Initialize ISO 27001 controls"""
        try:
            iso27001_controls = [
                {
                    'id': 'A.5.1',
                    'name': 'Information Security Policies',
                    'description': 'Management direction and support for information security in accordance with business requirements and relevant laws and regulations.',
                    'category': 'Information Security Policies'
                },
                {
                    'id': 'A.6.1',
                    'name': 'Internal Organization',
                    'description': 'A management framework to initiate and control the implementation and operation of information security within the organization.',
                    'category': 'Organization of Information Security'
                },
                {
                    'id': 'A.7.1',
                    'name': 'Prior to Employment',
                    'description': 'Ensure that employees and contractors understand their responsibilities and are suitable for the roles for which they are considered.',
                    'category': 'Human Resource Security'
                },
                {
                    'id': 'A.8.1',
                    'name': 'Responsibility for Assets',
                    'description': 'Achieve and maintain appropriate protection of organizational assets.',
                    'category': 'Asset Management'
                },
                {
                    'id': 'A.9.1',
                    'name': 'Business Requirement for Access Control',
                    'description': 'Limit access to information and information processing facilities.',
                    'category': 'Access Control'
                },
                {
                    'id': 'A.10.1',
                    'name': 'Cryptographic Controls',
                    'description': 'Ensure proper and effective use of cryptography to protect the confidentiality, authenticity and/or integrity of information.',
                    'category': 'Cryptography'
                },
                {
                    'id': 'A.11.1',
                    'name': 'Physical and Environmental Security',
                    'description': 'Prevent unauthorized physical access, damage and interference to the organization\'s information and information processing facilities.',
                    'category': 'Physical and Environmental Security'
                },
                {
                    'id': 'A.12.1',
                    'name': 'Operational Procedures and Responsibilities',
                    'description': 'Ensure the correct and secure operation of information processing facilities.',
                    'category': 'Operations Security'
                }
            ]
            
            for control_data in iso27001_controls:
                control = AuditControl(
                    id=control_data['id'],
                    name=control_data['name'],
                    description=control_data['description'],
                    framework=ComplianceFramework.ISO27001,
                    category=control_data['category'],
                    status=AuditStatus.PENDING,
                    evidence=[],
                    findings=[]
                )
                self.controls[control.id] = control
            
        except Exception as e:
            logger.error(f"ISO 27001 controls initialization failed: {e}")
    
    async def _initialize_owasp_llm_controls(self):
        """Initialize OWASP LLM Top 10 controls"""
        try:
            owasp_controls = [
                {
                    'id': 'LLM01',
                    'name': 'Prompt Injection',
                    'description': 'Prompt injection vulnerabilities occur when an attacker can manipulate the input to a language model to cause it to behave in unintended ways.',
                    'category': 'Injection'
                },
                {
                    'id': 'LLM02',
                    'name': 'Insecure Output Handling',
                    'description': 'Insecure output handling occurs when the output of a language model is not properly sanitized before being used in other systems.',
                    'category': 'Output Handling'
                },
                {
                    'id': 'LLM03',
                    'name': 'Training Data Poisoning',
                    'description': 'Training data poisoning occurs when an attacker can manipulate the training data to cause the model to behave in unintended ways.',
                    'category': 'Data Security'
                },
                {
                    'id': 'LLM04',
                    'name': 'Model Denial of Service',
                    'description': 'Model denial of service occurs when an attacker can cause a language model to consume excessive resources or become unavailable.',
                    'category': 'Availability'
                },
                {
                    'id': 'LLM05',
                    'name': 'Supply Chain Vulnerabilities',
                    'description': 'Supply chain vulnerabilities occur when an attacker can compromise the supply chain of a language model or its dependencies.',
                    'category': 'Supply Chain'
                },
                {
                    'id': 'LLM06',
                    'name': 'Sensitive Information Disclosure',
                    'description': 'Sensitive information disclosure occurs when a language model reveals sensitive information that should not be accessible.',
                    'category': 'Data Protection'
                },
                {
                    'id': 'LLM07',
                    'name': 'Insecure Plugin Design',
                    'description': 'Insecure plugin design occurs when plugins or extensions for language models are not properly secured.',
                    'category': 'Plugin Security'
                },
                {
                    'id': 'LLM08',
                    'name': 'Excessive Agency',
                    'description': 'Excessive agency occurs when a language model is given too much autonomy to make decisions that could have significant consequences.',
                    'category': 'Agency Control'
                },
                {
                    'id': 'LLM09',
                    'name': 'Overreliance',
                    'description': 'Overreliance occurs when users or systems become too dependent on language models for critical decisions.',
                    'category': 'Dependency Management'
                },
                {
                    'id': 'LLM10',
                    'name': 'Model Theft',
                    'description': 'Model theft occurs when an attacker can steal or replicate a language model without authorization.',
                    'category': 'Intellectual Property'
                }
            ]
            
            for control_data in owasp_controls:
                control = AuditControl(
                    id=control_data['id'],
                    name=control_data['name'],
                    description=control_data['description'],
                    framework=ComplianceFramework.OWASP_LLM,
                    category=control_data['category'],
                    status=AuditStatus.PENDING,
                    evidence=[],
                    findings=[]
                )
                self.controls[control.id] = control
            
        except Exception as e:
            logger.error(f"OWASP LLM controls initialization failed: {e}")
    
    async def _initialize_nist_controls(self):
        """Initialize NIST Cybersecurity Framework controls"""
        try:
            nist_controls = [
                {
                    'id': 'ID.AM-1',
                    'name': 'Physical Devices and Systems',
                    'description': 'Physical devices and systems within the organization are inventoried.',
                    'category': 'Identify'
                },
                {
                    'id': 'ID.AM-2',
                    'name': 'Software Platforms and Applications',
                    'description': 'Software platforms and applications within the organization are inventoried.',
                    'category': 'Identify'
                },
                {
                    'id': 'PR.AC-1',
                    'name': 'Identities and Credentials',
                    'description': 'Identities and credentials are issued, managed, verified, revoked, and audited for authorized devices, users and processes.',
                    'category': 'Protect'
                },
                {
                    'id': 'PR.AC-2',
                    'name': 'Physical Access',
                    'description': 'Physical access to assets is managed and protected.',
                    'category': 'Protect'
                },
                {
                    'id': 'DE.CM-1',
                    'name': 'Network Monitoring',
                    'description': 'The network is monitored to detect potential cybersecurity events.',
                    'category': 'Detect'
                },
                {
                    'id': 'RS.RP-1',
                    'name': 'Response Plan',
                    'description': 'Response plan is executed during or after a cybersecurity incident.',
                    'category': 'Respond'
                },
                {
                    'id': 'RC.RP-1',
                    'name': 'Recovery Plan',
                    'description': 'Recovery plan is executed during or after a cybersecurity incident.',
                    'category': 'Recover'
                }
            ]
            
            for control_data in nist_controls:
                control = AuditControl(
                    id=control_data['id'],
                    name=control_data['name'],
                    description=control_data['description'],
                    framework=ComplianceFramework.NIST,
                    category=control_data['category'],
                    status=AuditStatus.PENDING,
                    evidence=[],
                    findings=[]
                )
                self.controls[control.id] = control
            
        except Exception as e:
            logger.error(f"NIST controls initialization failed: {e}")
    
    async def _initialize_pci_dss_controls(self):
        """Initialize PCI DSS controls"""
        try:
            pci_controls = [
                {
                    'id': 'PCI-1',
                    'name': 'Firewall Configuration',
                    'description': 'Install and maintain a firewall configuration to protect cardholder data.',
                    'category': 'Build and Maintain'
                },
                {
                    'id': 'PCI-2',
                    'name': 'Default Passwords',
                    'description': 'Do not use vendor-supplied defaults for system passwords and other security parameters.',
                    'category': 'Build and Maintain'
                },
                {
                    'id': 'PCI-3',
                    'name': 'Cardholder Data Protection',
                    'description': 'Protect stored cardholder data.',
                    'category': 'Build and Maintain'
                },
                {
                    'id': 'PCI-4',
                    'name': 'Data Transmission',
                    'description': 'Encrypt transmission of cardholder data across open, public networks.',
                    'category': 'Build and Maintain'
                },
                {
                    'id': 'PCI-5',
                    'name': 'Antivirus Software',
                    'description': 'Use and regularly update anti-virus software or programs.',
                    'category': 'Build and Maintain'
                },
                {
                    'id': 'PCI-6',
                    'name': 'Secure Systems',
                    'description': 'Develop and maintain secure systems and applications.',
                    'category': 'Build and Maintain'
                }
            ]
            
            for control_data in pci_controls:
                control = AuditControl(
                    id=control_data['id'],
                    name=control_data['name'],
                    description=control_data['description'],
                    framework=ComplianceFramework.PCI_DSS,
                    category=control_data['category'],
                    status=AuditStatus.PENDING,
                    evidence=[],
                    findings=[]
                )
                self.controls[control.id] = control
            
        except Exception as e:
            logger.error(f"PCI DSS controls initialization failed: {e}")
    
    async def _initialize_hipaa_controls(self):
        """Initialize HIPAA controls"""
        try:
            hipaa_controls = [
                {
                    'id': 'HIPAA-164.308',
                    'name': 'Administrative Safeguards',
                    'description': 'Administrative actions, and policies and procedures, to manage the selection, development, implementation, and maintenance of security measures.',
                    'category': 'Administrative Safeguards'
                },
                {
                    'id': 'HIPAA-164.310',
                    'name': 'Physical Safeguards',
                    'description': 'Physical measures, policies, and procedures to protect a covered entity\'s electronic information systems and related buildings and equipment.',
                    'category': 'Physical Safeguards'
                },
                {
                    'id': 'HIPAA-164.312',
                    'name': 'Technical Safeguards',
                    'description': 'Technology and the policy and procedures for its use that protect electronic protected health information and control access to it.',
                    'category': 'Technical Safeguards'
                }
            ]
            
            for control_data in hipaa_controls:
                control = AuditControl(
                    id=control_data['id'],
                    name=control_data['name'],
                    description=control_data['description'],
                    framework=ComplianceFramework.HIPAA,
                    category=control_data['category'],
                    status=AuditStatus.PENDING,
                    evidence=[],
                    findings=[]
                )
                self.controls[control.id] = control
            
        except Exception as e:
            logger.error(f"HIPAA controls initialization failed: {e}")
    
    async def test_control(self, 
                          control_id: str, 
                          test_results: Dict[str, Any],
                          evidence: List[str] = None,
                          tester: str = "ML Security Platform") -> bool:
        """
        Test a specific control and record results
        
        Args:
            control_id: ID of the control to test
            test_results: Results of the control test
            evidence: List of evidence files/records
            tester: Name of the tester
            
        Returns:
            True if successful
        """
        try:
            if control_id not in self.controls:
                logger.error(f"Control {control_id} not found")
                return False
            
            control = self.controls[control_id]
            
            # Update control status based on test results
            if test_results.get('passed', False):
                control.status = AuditStatus.PASS
            elif test_results.get('partial', False):
                control.status = AuditStatus.PARTIAL
            else:
                control.status = AuditStatus.FAIL
            
            # Add evidence
            if evidence:
                control.evidence.extend(evidence)
            
            # Add findings if any
            findings = test_results.get('findings', [])
            if findings:
                control.findings.extend(findings)
                
                # Create audit findings for failed controls
                for finding in findings:
                    if finding.get('severity') in ['critical', 'high', 'medium', 'low']:
                        audit_finding = AuditFinding(
                            id=f"finding_{control_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            control_id=control_id,
                            severity=finding.get('severity', 'medium'),
                            title=finding.get('title', f"Finding in {control.name}"),
                            description=finding.get('description', 'No description provided'),
                            evidence=finding.get('evidence', []),
                            remediation=finding.get('remediation', 'TBD'),
                            due_date=datetime.now() + timedelta(days=30),
                            assigned_to=finding.get('assigned_to', 'TBD')
                        )
                        self.findings.append(audit_finding)
            
            # Update metadata
            control.last_tested = datetime.now()
            control.tester = tester
            
            logger.info(f"✅ Control {control_id} tested with status: {control.status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Control testing failed for {control_id}: {e}")
            return False
    
    async def generate_audit_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive audit report
        
        Returns:
            Audit report dictionary
        """
        try:
            # Calculate compliance scores
            compliance_scores = await self._calculate_compliance_scores()
            
            # Generate executive summary
            executive_summary = await self._generate_executive_summary()
            
            # Generate detailed findings
            detailed_findings = await self._generate_detailed_findings()
            
            # Generate recommendations
            recommendations = await self._generate_recommendations()
            
            # Update audit metadata
            self.audit_metadata['end_date'] = datetime.now()
            self.audit_metadata['status'] = 'completed'
            
            report = {
                'audit_metadata': self.audit_metadata,
                'executive_summary': executive_summary,
                'compliance_scores': compliance_scores,
                'controls_summary': await self._get_controls_summary(),
                'detailed_findings': detailed_findings,
                'recommendations': recommendations,
                'evidence_summary': await self._get_evidence_summary(),
                'remediation_plan': await self._generate_remediation_plan()
            }
            
            logger.info("✅ Audit report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Audit report generation failed: {e}")
            return {"error": str(e)}
    
    async def _calculate_compliance_scores(self) -> Dict[str, Any]:
        """Calculate compliance scores for each framework"""
        try:
            scores = {}
            
            # Group controls by framework
            framework_controls = {}
            for control in self.controls.values():
                framework = control.framework.value
                if framework not in framework_controls:
                    framework_controls[framework] = []
                framework_controls[framework].append(control)
            
            # Calculate scores for each framework
            for framework, controls in framework_controls.items():
                total_controls = len(controls)
                passed_controls = len([c for c in controls if c.status == AuditStatus.PASS])
                partial_controls = len([c for c in controls if c.status == AuditStatus.PARTIAL])
                failed_controls = len([c for c in controls if c.status == AuditStatus.FAIL])
                
                # Calculate score (pass = 1, partial = 0.5, fail = 0)
                score = (passed_controls + (partial_controls * 0.5)) / total_controls * 100 if total_controls > 0 else 0
                
                scores[framework] = {
                    'score': round(score, 2),
                    'total_controls': total_controls,
                    'passed': passed_controls,
                    'partial': partial_controls,
                    'failed': failed_controls,
                    'status': 'compliant' if score >= 95 else 'partially_compliant' if score >= 80 else 'non_compliant'
                }
            
            return scores
            
        except Exception as e:
            logger.error(f"Compliance scores calculation failed: {e}")
            return {}
    
    async def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary"""
        try:
            total_controls = len(self.controls)
            passed_controls = len([c for c in self.controls.values() if c.status == AuditStatus.PASS])
            failed_controls = len([c for c in self.controls.values() if c.status == AuditStatus.FAIL])
            critical_findings = len([f for f in self.findings if f.severity == 'critical'])
            high_findings = len([f for f in self.findings if f.severity == 'high'])
            
            overall_score = (passed_controls / total_controls * 100) if total_controls > 0 else 0
            
            return {
                'overall_score': round(overall_score, 2),
                'total_controls_tested': total_controls,
                'controls_passed': passed_controls,
                'controls_failed': failed_controls,
                'critical_findings': critical_findings,
                'high_findings': high_findings,
                'overall_status': 'compliant' if overall_score >= 95 else 'partially_compliant' if overall_score >= 80 else 'non_compliant',
                'key_risks': await self._identify_key_risks(),
                'immediate_actions': await self._identify_immediate_actions()
            }
            
        except Exception as e:
            logger.error(f"Executive summary generation failed: {e}")
            return {}
    
    async def _generate_detailed_findings(self) -> List[Dict[str, Any]]:
        """Generate detailed findings"""
        try:
            findings_data = []
            
            for finding in self.findings:
                finding_data = {
                    'id': finding.id,
                    'control_id': finding.control_id,
                    'control_name': self.controls.get(finding.control_id, {}).name if finding.control_id in self.controls else 'Unknown',
                    'severity': finding.severity,
                    'title': finding.title,
                    'description': finding.description,
                    'evidence': finding.evidence,
                    'remediation': finding.remediation,
                    'due_date': finding.due_date.isoformat() if finding.due_date else None,
                    'assigned_to': finding.assigned_to,
                    'status': finding.status
                }
                findings_data.append(finding_data)
            
            # Sort by severity
            severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            findings_data.sort(key=lambda x: severity_order.get(x['severity'], 4))
            
            return findings_data
            
        except Exception as e:
            logger.error(f"Detailed findings generation failed: {e}")
            return []
    
    async def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations based on findings"""
        try:
            recommendations = []
            
            # Group findings by severity
            critical_findings = [f for f in self.findings if f.severity == 'critical']
            high_findings = [f for f in self.findings if f.severity == 'high']
            medium_findings = [f for f in self.findings if f.severity == 'medium']
            
            # Immediate actions (critical findings)
            if critical_findings:
                recommendations.append({
                    'priority': 'immediate',
                    'title': 'Address Critical Security Findings',
                    'description': f'Immediately address {len(critical_findings)} critical security findings',
                    'timeline': '0-7 days',
                    'resources_required': 'High',
                    'findings_count': len(critical_findings)
                })
            
            # Short-term actions (high findings)
            if high_findings:
                recommendations.append({
                    'priority': 'short_term',
                    'title': 'Address High Priority Security Findings',
                    'description': f'Address {len(high_findings)} high priority security findings',
                    'timeline': '1-30 days',
                    'resources_required': 'Medium',
                    'findings_count': len(high_findings)
                })
            
            # Medium-term actions (medium findings)
            if medium_findings:
                recommendations.append({
                    'priority': 'medium_term',
                    'title': 'Address Medium Priority Security Findings',
                    'description': f'Address {len(medium_findings)} medium priority security findings',
                    'timeline': '1-90 days',
                    'resources_required': 'Low',
                    'findings_count': len(medium_findings)
                })
            
            # General recommendations
            recommendations.extend([
                {
                    'priority': 'ongoing',
                    'title': 'Implement Continuous Monitoring',
                    'description': 'Implement continuous security monitoring and automated testing',
                    'timeline': 'Ongoing',
                    'resources_required': 'Medium',
                    'findings_count': 0
                },
                {
                    'priority': 'ongoing',
                    'title': 'Regular Security Training',
                    'description': 'Conduct regular security awareness training for all staff',
                    'timeline': 'Quarterly',
                    'resources_required': 'Low',
                    'findings_count': 0
                }
            ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendations generation failed: {e}")
            return []
    
    async def _get_controls_summary(self) -> Dict[str, Any]:
        """Get controls summary"""
        try:
            total_controls = len(self.controls)
            passed_controls = len([c for c in self.controls.values() if c.status == AuditStatus.PASS])
            partial_controls = len([c for c in self.controls.values() if c.status == AuditStatus.PARTIAL])
            failed_controls = len([c for c in self.controls.values() if c.status == AuditStatus.FAIL])
            pending_controls = len([c for c in self.controls.values() if c.status == AuditStatus.PENDING])
            
            return {
                'total_controls': total_controls,
                'passed': passed_controls,
                'partial': partial_controls,
                'failed': failed_controls,
                'pending': pending_controls,
                'completion_rate': round(((passed_controls + partial_controls + failed_controls) / total_controls * 100), 2) if total_controls > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Controls summary generation failed: {e}")
            return {}
    
    async def _get_evidence_summary(self) -> Dict[str, Any]:
        """Get evidence summary"""
        try:
            total_evidence = sum(len(control.evidence) for control in self.controls.values())
            evidence_by_framework = {}
            
            for control in self.controls.values():
                framework = control.framework.value
                if framework not in evidence_by_framework:
                    evidence_by_framework[framework] = 0
                evidence_by_framework[framework] += len(control.evidence)
            
            return {
                'total_evidence_items': total_evidence,
                'evidence_by_framework': evidence_by_framework,
                'average_evidence_per_control': round(total_evidence / len(self.controls), 2) if self.controls else 0
            }
            
        except Exception as e:
            logger.error(f"Evidence summary generation failed: {e}")
            return {}
    
    async def _generate_remediation_plan(self) -> Dict[str, Any]:
        """Generate remediation plan"""
        try:
            plan = {
                'immediate_actions': [],
                'short_term_actions': [],
                'long_term_actions': [],
                'timeline': {},
                'resources_required': {}
            }
            
            # Categorize findings by timeline
            for finding in self.findings:
                if finding.severity == 'critical':
                    plan['immediate_actions'].append({
                        'finding_id': finding.id,
                        'title': finding.title,
                        'due_date': finding.due_date.isoformat() if finding.due_date else None,
                        'assigned_to': finding.assigned_to
                    })
                elif finding.severity == 'high':
                    plan['short_term_actions'].append({
                        'finding_id': finding.id,
                        'title': finding.title,
                        'due_date': finding.due_date.isoformat() if finding.due_date else None,
                        'assigned_to': finding.assigned_to
                    })
                else:
                    plan['long_term_actions'].append({
                        'finding_id': finding.id,
                        'title': finding.title,
                        'due_date': finding.due_date.isoformat() if finding.due_date else None,
                        'assigned_to': finding.assigned_to
                    })
            
            # Set timeline
            plan['timeline'] = {
                'immediate': '0-7 days',
                'short_term': '1-30 days',
                'long_term': '1-90 days'
            }
            
            # Set resources
            plan['resources_required'] = {
                'immediate': 'High',
                'short_term': 'Medium',
                'long_term': 'Low'
            }
            
            return plan
            
        except Exception as e:
            logger.error(f"Remediation plan generation failed: {e}")
            return {}
    
    async def _identify_key_risks(self) -> List[str]:
        """Identify key risks from findings"""
        try:
            risks = []
            
            critical_findings = [f for f in self.findings if f.severity == 'critical']
            if critical_findings:
                risks.append(f"{len(critical_findings)} critical security findings require immediate attention")
            
            high_findings = [f for f in self.findings if f.severity == 'high']
            if high_findings:
                risks.append(f"{len(high_findings)} high priority security findings need to be addressed")
            
            failed_controls = [c for c in self.controls.values() if c.status == AuditStatus.FAIL]
            if failed_controls:
                risks.append(f"{len(failed_controls)} security controls failed testing")
            
            return risks
            
        except Exception as e:
            logger.error(f"Key risks identification failed: {e}")
            return []
    
    async def _identify_immediate_actions(self) -> List[str]:
        """Identify immediate actions required"""
        try:
            actions = []
            
            critical_findings = [f for f in self.findings if f.severity == 'critical']
            if critical_findings:
                actions.append("Address all critical security findings immediately")
            
            failed_controls = [c for c in self.controls.values() if c.status == AuditStatus.FAIL]
            if failed_controls:
                actions.append("Implement remediation for failed security controls")
            
            actions.append("Review and update security policies and procedures")
            actions.append("Conduct security awareness training for all staff")
            
            return actions
            
        except Exception as e:
            logger.error(f"Immediate actions identification failed: {e}")
            return []
    
    async def get_audit_status(self) -> Dict[str, Any]:
        """Get current audit status"""
        try:
            controls_summary = await self._get_controls_summary()
            return {
                'audit_id': self.audit_metadata.get('audit_id'),
                'status': self.audit_metadata.get('status'),
                'start_date': self.audit_metadata.get('start_date').isoformat() if self.audit_metadata.get('start_date') else None,
                'end_date': self.audit_metadata.get('end_date').isoformat() if self.audit_metadata.get('end_date') else None,
                'frameworks': self.audit_metadata.get('frameworks', []),
                'total_controls': len(self.controls),
                'total_findings': len(self.findings),
                'completion_rate': controls_summary.get('completion_rate', 0)
            }
            
        except Exception as e:
            logger.error(f"Audit status retrieval failed: {e}")
            return {"error": str(e)}
    
    async def _initialize_iso42001_controls(self):
        """Initialize ISO 42001 AI Management System controls"""
        try:
            iso42001_controls = [
                {
                    'id': 'ISO42001-4.1',
                    'name': 'Understanding the Organization and its Context',
                    'description': 'The organization shall determine external and internal issues that are relevant to its purpose and that affect its ability to achieve the intended outcomes of its AI management system.',
                    'category': 'Context of the Organization'
                },
                {
                    'id': 'ISO42001-4.2',
                    'name': 'Understanding the Needs and Expectations of Interested Parties',
                    'description': 'The organization shall determine the interested parties that are relevant to the AI management system and the relevant requirements of these interested parties.',
                    'category': 'Context of the Organization'
                },
                {
                    'id': 'ISO42001-5.1',
                    'name': 'Leadership and Commitment',
                    'description': 'Top management shall demonstrate leadership and commitment with respect to the AI management system.',
                    'category': 'Leadership'
                },
                {
                    'id': 'ISO42001-6.1',
                    'name': 'Actions to Address Risks and Opportunities',
                    'description': 'When planning for the AI management system, the organization shall consider the issues and requirements and determine the risks and opportunities that need to be addressed.',
                    'category': 'Planning'
                },
                {
                    'id': 'ISO42001-7.1',
                    'name': 'Resources',
                    'description': 'The organization shall determine and provide the resources needed for the establishment, implementation, maintenance and continual improvement of the AI management system.',
                    'category': 'Support'
                },
                {
                    'id': 'ISO42001-8.1',
                    'name': 'Operational Planning and Control',
                    'description': 'The organization shall plan, implement and control the processes needed to meet requirements for the AI management system.',
                    'category': 'Operation'
                },
                {
                    'id': 'ISO42001-9.1',
                    'name': 'Monitoring, Measurement, Analysis and Evaluation',
                    'description': 'The organization shall determine what needs to be monitored and measured, the methods for monitoring, measurement, analysis and evaluation.',
                    'category': 'Performance Evaluation'
                },
                {
                    'id': 'ISO42001-10.1',
                    'name': 'Nonconformity and Corrective Action',
                    'description': 'When a nonconformity occurs, the organization shall react to the nonconformity and take action to control and correct it.',
                    'category': 'Improvement'
                }
            ]
            
            for control_data in iso42001_controls:
                control = AuditControl(
                    id=control_data['id'],
                    name=control_data['name'],
                    description=control_data['description'],
                    framework=ComplianceFramework.ISO42001,
                    category=control_data['category'],
                    status=AuditStatus.PENDING,
                    evidence=[],
                    findings=[]
                )
                self.controls[control.id] = control
                
        except Exception as e:
            logger.error(f"ISO 42001 controls initialization failed: {e}")
    
    async def _initialize_eu_ai_act_controls(self):
        """Initialize EU AI Act controls"""
        try:
            eu_ai_act_controls = [
                {
                    'id': 'EU-AI-ACT-1',
                    'name': 'Risk Management System',
                    'description': 'High-risk AI systems shall have a risk management system in place.',
                    'category': 'High-Risk AI Systems'
                },
                {
                    'id': 'EU-AI-ACT-2',
                    'name': 'Data Governance',
                    'description': 'High-risk AI systems shall have appropriate data governance and management practices.',
                    'category': 'High-Risk AI Systems'
                },
                {
                    'id': 'EU-AI-ACT-3',
                    'name': 'Technical Documentation',
                    'description': 'High-risk AI systems shall have technical documentation demonstrating compliance.',
                    'category': 'High-Risk AI Systems'
                },
                {
                    'id': 'EU-AI-ACT-4',
                    'name': 'Record Keeping',
                    'description': 'High-risk AI systems shall have automated logging capabilities.',
                    'category': 'High-Risk AI Systems'
                },
                {
                    'id': 'EU-AI-ACT-5',
                    'name': 'Transparency and Provision of Information',
                    'description': 'High-risk AI systems shall be designed and developed with capabilities enabling users to interpret the system\'s output.',
                    'category': 'High-Risk AI Systems'
                },
                {
                    'id': 'EU-AI-ACT-6',
                    'name': 'Human Oversight',
                    'description': 'High-risk AI systems shall be designed and developed with human oversight capabilities.',
                    'category': 'High-Risk AI Systems'
                },
                {
                    'id': 'EU-AI-ACT-7',
                    'name': 'Accuracy, Robustness and Cybersecurity',
                    'description': 'High-risk AI systems shall be designed and developed with accuracy, robustness and cybersecurity capabilities.',
                    'category': 'High-Risk AI Systems'
                },
                {
                    'id': 'EU-AI-ACT-8',
                    'name': 'Conformity Assessment',
                    'description': 'High-risk AI systems shall undergo conformity assessment procedures.',
                    'category': 'High-Risk AI Systems'
                }
            ]
            
            for control_data in eu_ai_act_controls:
                control = AuditControl(
                    id=control_data['id'],
                    name=control_data['name'],
                    description=control_data['description'],
                    framework=ComplianceFramework.EU_AI_ACT,
                    category=control_data['category'],
                    status=AuditStatus.PENDING,
                    evidence=[],
                    findings=[]
                )
                self.controls[control.id] = control
                
        except Exception as e:
            logger.error(f"EU AI Act controls initialization failed: {e}")
    
    async def _initialize_gdpr_controls(self):
        """Initialize GDPR controls"""
        try:
            gdpr_controls = [
                {
                    'id': 'GDPR-5',
                    'name': 'Principles Relating to Processing of Personal Data',
                    'description': 'Personal data shall be processed lawfully, fairly and in a transparent manner.',
                    'category': 'Data Protection Principles'
                },
                {
                    'id': 'GDPR-6',
                    'name': 'Lawfulness of Processing',
                    'description': 'Processing shall be lawful only if and to the extent that at least one of the legal bases applies.',
                    'category': 'Data Protection Principles'
                },
                {
                    'id': 'GDPR-7',
                    'name': 'Conditions for Consent',
                    'description': 'Consent shall be freely given, specific, informed and unambiguous.',
                    'category': 'Data Subject Rights'
                },
                {
                    'id': 'GDPR-12',
                    'name': 'Transparent Information and Communication',
                    'description': 'The controller shall take appropriate measures to provide information in a concise, transparent, intelligible and easily accessible form.',
                    'category': 'Data Subject Rights'
                },
                {
                    'id': 'GDPR-13',
                    'name': 'Information to be Provided When Personal Data are Collected',
                    'description': 'The controller shall provide information to the data subject at the time when personal data are obtained.',
                    'category': 'Data Subject Rights'
                },
                {
                    'id': 'GDPR-17',
                    'name': 'Right to Erasure',
                    'description': 'The data subject shall have the right to obtain from the controller the erasure of personal data.',
                    'category': 'Data Subject Rights'
                },
                {
                    'id': 'GDPR-25',
                    'name': 'Data Protection by Design and by Default',
                    'description': 'The controller shall implement appropriate technical and organisational measures to ensure data protection by design and by default.',
                    'category': 'Technical and Organisational Measures'
                },
                {
                    'id': 'GDPR-32',
                    'name': 'Security of Processing',
                    'description': 'The controller and the processor shall implement appropriate technical and organisational measures to ensure a level of security appropriate to the risk.',
                    'category': 'Technical and Organisational Measures'
                }
            ]
            
            for control_data in gdpr_controls:
                control = AuditControl(
                    id=control_data['id'],
                    name=control_data['name'],
                    description=control_data['description'],
                    framework=ComplianceFramework.GDPR,
                    category=control_data['category'],
                    status=AuditStatus.PENDING,
                    evidence=[],
                    findings=[]
                )
                self.controls[control.id] = control
                
        except Exception as e:
            logger.error(f"GDPR controls initialization failed: {e}")
    
    async def _initialize_ccpa_controls(self):
        """Initialize CCPA controls"""
        try:
            ccpa_controls = [
                {
                    'id': 'CCPA-1798.100',
                    'name': 'Right to Know About Personal Information Collected',
                    'description': 'A consumer has the right to request that a business disclose what personal information it collects.',
                    'category': 'Consumer Rights'
                },
                {
                    'id': 'CCPA-1798.105',
                    'name': 'Right to Delete Personal Information',
                    'description': 'A consumer has the right to request that a business delete any personal information about the consumer.',
                    'category': 'Consumer Rights'
                },
                {
                    'id': 'CCPA-1798.110',
                    'name': 'Right to Opt-Out of Sale of Personal Information',
                    'description': 'A consumer has the right to opt-out of the sale of personal information.',
                    'category': 'Consumer Rights'
                },
                {
                    'id': 'CCPA-1798.115',
                    'name': 'Right to Non-Discrimination',
                    'description': 'A consumer has the right to not be discriminated against for exercising any of the consumer\'s privacy rights.',
                    'category': 'Consumer Rights'
                },
                {
                    'id': 'CCPA-1798.120',
                    'name': 'Right to Correct Inaccurate Personal Information',
                    'description': 'A consumer has the right to request that a business correct inaccurate personal information.',
                    'category': 'Consumer Rights'
                },
                {
                    'id': 'CCPA-1798.130',
                    'name': 'Methods for Submitting Consumer Requests',
                    'description': 'A business shall provide two or more designated methods for submitting requests.',
                    'category': 'Business Obligations'
                },
                {
                    'id': 'CCPA-1798.135',
                    'name': 'Opt-Out Link or Button',
                    'description': 'A business that sells personal information shall provide a clear and conspicuous link on its homepage.',
                    'category': 'Business Obligations'
                },
                {
                    'id': 'CCPA-1798.150',
                    'name': 'Civil Penalties',
                    'description': 'Any business that violates this title shall be subject to an injunction and liable for a civil penalty.',
                    'category': 'Enforcement'
                }
            ]
            
            for control_data in ccpa_controls:
                control = AuditControl(
                    id=control_data['id'],
                    name=control_data['name'],
                    description=control_data['description'],
                    framework=ComplianceFramework.CCPA,
                    category=control_data['category'],
                    status=AuditStatus.PENDING,
                    evidence=[],
                    findings=[]
                )
                self.controls[control.id] = control
                
        except Exception as e:
            logger.error(f"CCPA controls initialization failed: {e}")
