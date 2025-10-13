"""
Technical Reporter
Creates detailed technical reports with attack reproduction steps, vulnerability fingerprinting, and remediation code.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import hashlib
import base64

logger = logging.getLogger(__name__)

class VulnerabilityType(Enum):
    """Types of vulnerabilities"""
    INJECTION = "injection"
    XSS = "xss"
    CSRF = "csrf"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_EXPOSURE = "data_exposure"
    CONFIGURATION = "configuration"
    CRYPTOGRAPHIC = "cryptographic"
    LOGIC = "logic"
    INPUT_VALIDATION = "input_validation"

class AttackComplexity(Enum):
    """Attack complexity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class VulnerabilityFingerprint:
    """Vulnerability fingerprint for identification"""
    vulnerability_id: str
    name: str
    description: str
    vulnerability_type: VulnerabilityType
    severity: str
    cvss_score: float
    fingerprint_hash: str
    attack_vectors: List[str] = field(default_factory=list)
    affected_components: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AttackReproductionStep:
    """Step in attack reproduction"""
    step_number: int
    description: str
    command: str
    expected_output: str
    actual_output: str
    success: bool
    notes: str = ""
    screenshots: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)

@dataclass
class RemediationCode:
    """Code for vulnerability remediation"""
    vulnerability_id: str
    language: str
    code_type: str  # "patch", "configuration", "script"
    code: str
    description: str
    testing_instructions: str
    rollback_instructions: str
    dependencies: List[str] = field(default_factory=list)

@dataclass
class TechnicalReport:
    """Technical security report"""
    report_id: str
    title: str
    created_at: datetime
    vulnerabilities: List[VulnerabilityFingerprint]
    attack_reproductions: List[AttackReproductionStep]
    remediation_codes: List[RemediationCode]
    technical_details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

class TechnicalReporter:
    """Creates detailed technical security reports"""
    
    def __init__(self):
        self.vulnerability_templates = self._load_vulnerability_templates()
        self.attack_patterns = self._load_attack_patterns()
        self.remediation_templates = self._load_remediation_templates()
    
    def _load_vulnerability_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load vulnerability templates"""
        return {
            "injection": {
                "description": "Code injection vulnerability",
                "indicators": ["unexpected code execution", "system command execution", "database queries"],
                "attack_vectors": ["SQL injection", "command injection", "LDAP injection"]
            },
            "xss": {
                "description": "Cross-site scripting vulnerability",
                "indicators": ["script execution in browser", "DOM manipulation", "cookie theft"],
                "attack_vectors": ["reflected XSS", "stored XSS", "DOM-based XSS"]
            },
            "csrf": {
                "description": "Cross-site request forgery vulnerability",
                "indicators": ["unauthorized actions", "state change requests", "session hijacking"],
                "attack_vectors": ["form submission", "API calls", "state changes"]
            },
            "authentication": {
                "description": "Authentication bypass vulnerability",
                "indicators": ["unauthorized access", "privilege escalation", "session bypass"],
                "attack_vectors": ["credential bypass", "session manipulation", "token manipulation"]
            },
            "authorization": {
                "description": "Authorization bypass vulnerability",
                "indicators": ["unauthorized resource access", "privilege escalation", "data exposure"],
                "attack_vectors": ["direct object reference", "horizontal privilege escalation", "vertical privilege escalation"]
            }
        }
    
    def _load_attack_patterns(self) -> Dict[str, List[str]]:
        """Load attack patterns"""
        return {
            "injection": [
                "'; DROP TABLE users; --",
                "'; SELECT * FROM users WHERE '1'='1",
                "'; INSERT INTO users (username, password) VALUES ('hacker', 'password'); --",
                "| cat /etc/passwd",
                "`whoami`",
                "$(id)"
            ],
            "xss": [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert('XSS')>",
                "javascript:alert('XSS')",
                "<svg onload=alert('XSS')>",
                "<iframe src=javascript:alert('XSS')></iframe>"
            ],
            "csrf": [
                "<form action='http://target.com/transfer' method='POST'>",
                "<input type='hidden' name='amount' value='1000'>",
                "<input type='hidden' name='to' value='attacker'>",
                "<input type='submit' value='Click me'>",
                "</form>"
            ]
        }
    
    def _load_remediation_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load remediation templates"""
        return {
            "injection": {
                "python": {
                    "code": "import sqlite3\nfrom sqlite3 import Cursor\n\ndef safe_query(cursor: Cursor, query: str, params: tuple):\n    return cursor.execute(query, params)\n\n# Example usage\n# cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
                    "description": "Use parameterized queries to prevent SQL injection"
                },
                "javascript": {
                    "code": "// Sanitize input\nfunction sanitizeInput(input) {\n    return input.replace(/[<>\"'&]/g, function(match) {\n        return {\n            '<': '&lt;',\n            '>': '&gt;',\n            '\"': '&quot;',\n            \"'\": '&#x27;',\n            '&': '&amp;'\n        }[match];\n    });\n}",
                    "description": "Sanitize user input to prevent injection attacks"
                }
            },
            "xss": {
                "html": {
                    "code": "<div>{{ user_input | escape }}</div>\n<!-- Or use CSP -->\n<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'self'; script-src 'self' 'unsafe-inline'\">",
                    "description": "Escape output and implement Content Security Policy"
                },
                "javascript": {
                    "code": "// Use textContent instead of innerHTML\ndocument.getElementById('output').textContent = userInput;\n\n// Or use DOMPurify for HTML content\nconst clean = DOMPurify.sanitize(userInput);\ndocument.getElementById('output').innerHTML = clean;",
                    "description": "Use safe DOM manipulation methods"
                }
            }
        }
    
    def create_vulnerability_fingerprint(self, vulnerability_data: Dict[str, Any]) -> VulnerabilityFingerprint:
        """Create vulnerability fingerprint"""
        try:
            # Extract basic information
            vuln_id = vulnerability_data.get("id", f"vuln_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            name = vulnerability_data.get("name", "Unknown Vulnerability")
            description = vulnerability_data.get("description", "")
            vuln_type = VulnerabilityType(vulnerability_data.get("type", "injection"))
            severity = vulnerability_data.get("severity", "medium")
            cvss_score = vulnerability_data.get("cvss_score", 5.0)
            
            # Generate fingerprint hash
            fingerprint_data = f"{name}{description}{vuln_type.value}{severity}"
            fingerprint_hash = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
            
            # Get attack vectors and indicators from template
            template = self.vulnerability_templates.get(vuln_type.value, {})
            attack_vectors = template.get("attack_vectors", [])
            indicators = template.get("indicators", [])
            
            # Add custom attack vectors and indicators
            attack_vectors.extend(vulnerability_data.get("attack_vectors", []))
            indicators.extend(vulnerability_data.get("indicators", []))
            
            return VulnerabilityFingerprint(
                vulnerability_id=vuln_id,
                name=name,
                description=description,
                vulnerability_type=vuln_type,
                severity=severity,
                cvss_score=cvss_score,
                fingerprint_hash=fingerprint_hash,
                attack_vectors=attack_vectors,
                affected_components=vulnerability_data.get("affected_components", []),
                indicators=indicators,
                metadata=vulnerability_data.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Failed to create vulnerability fingerprint: {e}")
            return VulnerabilityFingerprint("", "", "", VulnerabilityType.INJECTION, "medium", 0.0, "")
    
    def create_attack_reproduction_steps(self, attack_data: Dict[str, Any]) -> List[AttackReproductionStep]:
        """Create attack reproduction steps"""
        try:
            steps = []
            attack_type = attack_data.get("type", "injection")
            attack_patterns = self.attack_patterns.get(attack_type, [])
            
            # Step 1: Reconnaissance
            steps.append(AttackReproductionStep(
                step_number=1,
                description="Reconnaissance and target identification",
                command="nmap -sS -O target.com",
                expected_output="Open ports and services identified",
                actual_output=attack_data.get("reconnaissance_output", "Ports 80, 443, 22 open"),
                success=True,
                notes="Identify target services and potential attack vectors"
            ))
            
            # Step 2: Vulnerability discovery
            steps.append(AttackReproductionStep(
                step_number=2,
                description="Vulnerability discovery and validation",
                command="python3 exploit.py --target target.com --vuln injection",
                expected_output="Vulnerability confirmed",
                actual_output=attack_data.get("vulnerability_output", "Vulnerability confirmed"),
                success=True,
                notes="Confirm vulnerability exists and is exploitable"
            ))
            
            # Step 3: Exploit development
            for i, pattern in enumerate(attack_patterns[:3]):  # Limit to 3 patterns
                steps.append(AttackReproductionStep(
                    step_number=3 + i,
                    description=f"Exploit attempt {i + 1}",
                    command=f"curl -X POST 'http://target.com/api' -d 'input={pattern}'",
                    expected_output="Command execution or data extraction",
                    actual_output=attack_data.get(f"exploit_output_{i}", "Exploit successful"),
                    success=attack_data.get(f"exploit_success_{i}", True),
                    notes=f"Testing attack pattern: {pattern[:50]}..."
                ))
            
            # Step 4: Post-exploitation
            steps.append(AttackReproductionStep(
                step_number=len(steps) + 1,
                description="Post-exploitation and persistence",
                command="python3 post_exploit.py --target target.com",
                expected_output="Persistence established",
                actual_output=attack_data.get("post_exploit_output", "Persistence established"),
                success=attack_data.get("post_exploit_success", True),
                notes="Establish persistence and gather additional information"
            ))
            
            return steps
            
        except Exception as e:
            logger.error(f"Failed to create attack reproduction steps: {e}")
            return []
    
    def create_remediation_code(self, vulnerability: VulnerabilityFingerprint, 
                              language: str = "python") -> RemediationCode:
        """Create remediation code for vulnerability"""
        try:
            vuln_type = vulnerability.vulnerability_type.value
            template = self.remediation_templates.get(vuln_type, {})
            lang_template = template.get(language, {})
            
            if not lang_template:
                # Fallback to generic remediation
                lang_template = {
                    "code": f"# Remediation for {vulnerability.name}\n# TODO: Implement proper input validation and sanitization",
                    "description": f"Implement proper security controls for {vuln_type} vulnerability"
                }
            
            # Generate testing instructions
            testing_instructions = self._generate_testing_instructions(vulnerability, language)
            
            # Generate rollback instructions
            rollback_instructions = self._generate_rollback_instructions(vulnerability, language)
            
            return RemediationCode(
                vulnerability_id=vulnerability.vulnerability_id,
                language=language,
                code_type="patch",
                code=lang_template["code"],
                description=lang_template["description"],
                testing_instructions=testing_instructions,
                rollback_instructions=rollback_instructions,
                dependencies=self._get_dependencies(language)
            )
            
        except Exception as e:
            logger.error(f"Failed to create remediation code: {e}")
            return RemediationCode(vulnerability.vulnerability_id, language, "patch", "", "", "", "", [])
    
    def _generate_testing_instructions(self, vulnerability: VulnerabilityFingerprint, 
                                     language: str) -> str:
        """Generate testing instructions for remediation"""
        try:
            instructions = [
                f"Testing instructions for {vulnerability.name} remediation:",
                "",
                "1. Unit Testing:",
                f"   - Create test cases for all input validation functions",
                f"   - Test with malicious inputs: {', '.join(vulnerability.attack_vectors[:3])}",
                f"   - Verify that vulnerable patterns are blocked",
                "",
                "2. Integration Testing:",
                f"   - Test the complete {vulnerability.vulnerability_type.value} protection flow",
                f"   - Verify that legitimate inputs still work correctly",
                f"   - Test error handling and edge cases",
                "",
                "3. Security Testing:",
                f"   - Run automated security scans",
                f"   - Perform manual penetration testing",
                f"   - Verify that {vulnerability.vulnerability_type.value} attacks are blocked",
                "",
                "4. Performance Testing:",
                "   - Ensure remediation doesn't impact performance",
                "   - Test with high load and concurrent users",
                "   - Monitor memory usage and response times"
            ]
            
            return "\n".join(instructions)
            
        except Exception as e:
            logger.error(f"Failed to generate testing instructions: {e}")
            return "Follow standard testing procedures"
    
    def _generate_rollback_instructions(self, vulnerability: VulnerabilityFingerprint, 
                                      language: str) -> str:
        """Generate rollback instructions for remediation"""
        try:
            instructions = [
                f"Rollback instructions for {vulnerability.name} remediation:",
                "",
                "1. Immediate Rollback:",
                "   - Revert code changes to previous version",
                "   - Restore configuration files from backup",
                "   - Restart affected services",
                "",
                "2. Verification:",
                "   - Verify that rollback was successful",
                "   - Check that services are running normally",
                "   - Monitor for any issues or errors",
                "",
                "3. Communication:",
                "   - Notify stakeholders of rollback",
                "   - Document rollback reason and timeline",
                "   - Schedule remediation review meeting",
                "",
                "4. Next Steps:",
                "   - Investigate why remediation failed",
                "   - Develop improved remediation plan",
                "   - Schedule re-deployment after fixes"
            ]
            
            return "\n".join(instructions)
            
        except Exception as e:
            logger.error(f"Failed to generate rollback instructions: {e}")
            return "Follow standard rollback procedures"
    
    def _get_dependencies(self, language: str) -> List[str]:
        """Get dependencies for remediation code"""
        dependencies = {
            "python": ["sqlite3", "re", "html", "urllib.parse"],
            "javascript": ["DOMPurify", "validator"],
            "java": ["org.owasp.encoder", "org.apache.commons.validator"],
            "php": ["htmlspecialchars", "filter_var", "mysqli"],
            "csharp": ["System.Web.Security", "System.Text.RegularExpressions"]
        }
        
        return dependencies.get(language, [])
    
    def generate_technical_report(self, security_data: Dict[str, Any]) -> TechnicalReport:
        """Generate comprehensive technical report"""
        try:
            report_id = f"tech_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            title = security_data.get("title", "Technical Security Assessment Report")
            
            # Create vulnerability fingerprints
            vulnerabilities = []
            vuln_data = security_data.get("vulnerabilities", [])
            for vuln in vuln_data:
                fingerprint = self.create_vulnerability_fingerprint(vuln)
                vulnerabilities.append(fingerprint)
            
            # Create attack reproduction steps
            attack_reproductions = []
            attack_data = security_data.get("attacks", [])
            for attack in attack_data:
                steps = self.create_attack_reproduction_steps(attack)
                attack_reproductions.extend(steps)
            
            # Create remediation codes
            remediation_codes = []
            for vuln in vulnerabilities:
                for language in ["python", "javascript", "java"]:
                    remediation = self.create_remediation_code(vuln, language)
                    remediation_codes.append(remediation)
            
            # Generate technical details
            technical_details = self._generate_technical_details(security_data)
            
            # Generate recommendations
            recommendations = self._generate_technical_recommendations(vulnerabilities, attack_reproductions)
            
            return TechnicalReport(
                report_id=report_id,
                title=title,
                created_at=datetime.now(),
                vulnerabilities=vulnerabilities,
                attack_reproductions=attack_reproductions,
                remediation_codes=remediation_codes,
                technical_details=technical_details,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Failed to generate technical report: {e}")
            return TechnicalReport("", "", datetime.now(), [], [], [], {})
    
    def _generate_technical_details(self, security_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate technical details section"""
        try:
            details = {
                "assessment_scope": security_data.get("scope", "Full security assessment"),
                "methodology": security_data.get("methodology", "OWASP Testing Guide"),
                "tools_used": security_data.get("tools", ["Burp Suite", "OWASP ZAP", "Nmap"]),
                "environment": security_data.get("environment", "Production"),
                "testing_period": security_data.get("testing_period", "1 week"),
                "total_vulnerabilities": len(security_data.get("vulnerabilities", [])),
                "critical_vulnerabilities": len([v for v in security_data.get("vulnerabilities", []) 
                                               if v.get("severity", "").lower() == "critical"]),
                "high_vulnerabilities": len([v for v in security_data.get("vulnerabilities", []) 
                                           if v.get("severity", "").lower() == "high"]),
                "medium_vulnerabilities": len([v for v in security_data.get("vulnerabilities", []) 
                                             if v.get("severity", "").lower() == "medium"]),
                "low_vulnerabilities": len([v for v in security_data.get("vulnerabilities", []) 
                                          if v.get("severity", "").lower() == "low"]),
                "attack_success_rate": security_data.get("attack_success_rate", 0.0),
                "compliance_score": security_data.get("compliance_score", 0.0)
            }
            
            return details
            
        except Exception as e:
            logger.error(f"Failed to generate technical details: {e}")
            return {}
    
    def _generate_technical_recommendations(self, vulnerabilities: List[VulnerabilityFingerprint],
                                          attack_reproductions: List[AttackReproductionStep]) -> List[str]:
        """Generate technical recommendations"""
        try:
            recommendations = []
            
            # Vulnerability-based recommendations
            vuln_types = [v.vulnerability_type.value for v in vulnerabilities]
            unique_vuln_types = list(set(vuln_types))
            
            for vuln_type in unique_vuln_types:
                count = vuln_types.count(vuln_type)
                recommendations.append(f"Address {count} {vuln_type} vulnerabilities with appropriate controls")
            
            # Attack reproduction-based recommendations
            failed_attacks = [a for a in attack_reproductions if not a.success]
            if failed_attacks:
                recommendations.append(f"Investigate {len(failed_attacks)} failed attack attempts for potential improvements")
            
            # General recommendations
            recommendations.extend([
                "Implement input validation and output encoding",
                "Use parameterized queries for database operations",
                "Implement proper authentication and authorization controls",
                "Enable security headers (CSP, HSTS, X-Frame-Options)",
                "Regular security testing and code reviews",
                "Implement logging and monitoring for security events",
                "Keep all dependencies and frameworks updated",
                "Conduct regular penetration testing"
            ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate technical recommendations: {e}")
            return ["Review security posture and implement appropriate controls"]
    
    def export_technical_report(self, report: TechnicalReport, 
                              format_type: str = "json") -> str:
        """Export technical report"""
        try:
            report_data = {
                "report_id": report.report_id,
                "title": report.title,
                "created_at": report.created_at.isoformat(),
                "vulnerabilities": [
                    {
                        "vulnerability_id": v.vulnerability_id,
                        "name": v.name,
                        "description": v.description,
                        "type": v.vulnerability_type.value,
                        "severity": v.severity,
                        "cvss_score": v.cvss_score,
                        "fingerprint_hash": v.fingerprint_hash,
                        "attack_vectors": v.attack_vectors,
                        "affected_components": v.affected_components,
                        "indicators": v.indicators,
                        "metadata": v.metadata
                    }
                    for v in report.vulnerabilities
                ],
                "attack_reproductions": [
                    {
                        "step_number": s.step_number,
                        "description": s.description,
                        "command": s.command,
                        "expected_output": s.expected_output,
                        "actual_output": s.actual_output,
                        "success": s.success,
                        "notes": s.notes,
                        "screenshots": s.screenshots,
                        "artifacts": s.artifacts
                    }
                    for s in report.attack_reproductions
                ],
                "remediation_codes": [
                    {
                        "vulnerability_id": r.vulnerability_id,
                        "language": r.language,
                        "code_type": r.code_type,
                        "code": r.code,
                        "description": r.description,
                        "testing_instructions": r.testing_instructions,
                        "rollback_instructions": r.rollback_instructions,
                        "dependencies": r.dependencies
                    }
                    for r in report.remediation_codes
                ],
                "technical_details": report.technical_details,
                "recommendations": report.recommendations
            }
            
            if format_type == "json":
                return json.dumps(report_data, indent=2)
            else:
                return str(report_data)
                
        except Exception as e:
            logger.error(f"Failed to export technical report: {e}")
            return "{}"
