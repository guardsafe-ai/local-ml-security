"""
Red Team Audit Logger
Comprehensive audit logging for SOC 2, ISO 27001, and GDPR compliance
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import httpx

logger = logging.getLogger(__name__)

class RedTeamAuditLogger:
    """Comprehensive audit logging for red team security testing"""
    
    def __init__(self):
        self.analytics_service_url = "http://analytics:8006"
        self.business_metrics_url = "http://business-metrics:8004"
        self.audit_events_buffer = []
        self.buffer_size = 50
        self.flush_interval = 30  # seconds
        self._flush_task = None
        
    async def initialize(self):
        """Initialize audit logger"""
        try:
            # Start background flush task
            self._flush_task = asyncio.create_task(self._periodic_flush())
            logger.info("✅ Red Team Audit Logger initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize audit logger: {e}")
            raise
    
    async def close(self):
        """Close audit logger and flush remaining events"""
        try:
            if self._flush_task:
                self._flush_task.cancel()
            await self._flush_events()
            logger.info("✅ Red Team Audit Logger closed")
        except Exception as e:
            logger.error(f"❌ Error closing audit logger: {e}")
    
    async def log_test_execution(self, test_session: Dict[str, Any]):
        """Log red team test execution for audit trail"""
        try:
            audit_event = {
                "event_id": str(uuid.uuid4()),
                "event_type": "RED_TEAM_TEST_EXECUTION",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "test_id": test_session.get("test_id"),
                "model_tested": test_session.get("model_name", "unknown"),
                "user_id": test_session.get("initiated_by", "system"),
                "session_id": test_session.get("session_id"),
                "ip_address": test_session.get("source_ip", "127.0.0.1"),
                "user_agent": test_session.get("user_agent", "red-team-service"),
                "resource": f"model/{test_session.get('model_name', 'unknown')}",
                "action": "security_testing",
                "details": {
                    "attack_categories": test_session.get("attack_categories", []),
                    "total_attacks": test_session.get("total_attacks", 0),
                    "vulnerabilities_found": test_session.get("vulnerabilities_found", 0),
                    "detection_rate": test_session.get("detection_rate", 0.0),
                    "test_duration_seconds": test_session.get("test_duration_seconds", 0),
                    "compliance_frameworks": ["OWASP_LLM", "NIST_AI_RMF", "ISO_27001"],
                    "test_type": "automated_security_testing"
                },
                "severity": "INFO",
                "success": True,
                "error_message": None,
                "data_classification": "CONFIDENTIAL",
                "retention_period": 2555,  # 7 years for SOC 2
                "compliance_tags": ["SOC2", "ISO27001", "GDPR"]
            }
            
            await self._queue_audit_event(audit_event)
            logger.info(f"📝 [AUDIT] Logged test execution: {test_session.get('test_id')}")
            
        except Exception as e:
            logger.error(f"❌ [AUDIT] Failed to log test execution: {e}")
    
    async def log_vulnerability_discovery(self, vulnerability: Dict[str, Any]):
        """Log vulnerability discovery for security incident tracking"""
        try:
            severity = "CRITICAL" if vulnerability.get("cvss_score", 0) >= 9.0 else "HIGH"
            
            audit_event = {
                "event_id": str(uuid.uuid4()),
                "event_type": "VULNERABILITY_DISCOVERED",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "test_id": vulnerability.get("test_id"),
                "model_tested": vulnerability.get("model_name", "unknown"),
                "user_id": "red-team-service",
                "session_id": vulnerability.get("session_id"),
                "ip_address": "127.0.0.1",
                "user_agent": "red-team-service",
                "resource": f"model/{vulnerability.get('model_name', 'unknown')}",
                "action": "vulnerability_detection",
                "details": {
                    "vulnerability_id": str(uuid.uuid4()),
                    "attack_category": vulnerability.get("attack_category"),
                    "attack_pattern": vulnerability.get("attack_pattern", "")[:100],  # Truncate for security
                    "cvss_score": vulnerability.get("cvss_score", 0.0),
                    "severity_rating": vulnerability.get("severity_rating", "UNKNOWN"),
                    "owasp_category": vulnerability.get("owasp_mapping", "UNKNOWN"),
                    "detection_confidence": vulnerability.get("confidence", 0.0),
                    "remediation_required": True,
                    "compliance_impact": self._assess_compliance_impact(vulnerability),
                    "business_impact": self._assess_business_impact(vulnerability)
                },
                "severity": severity,
                "success": True,
                "error_message": None,
                "data_classification": "CONFIDENTIAL",
                "retention_period": 2555,  # 7 years for SOC 2
                "compliance_tags": ["SOC2", "ISO27001", "GDPR", "SECURITY_INCIDENT"]
            }
            
            await self._queue_audit_event(audit_event)
            logger.warning(f"🚨 [AUDIT] Logged vulnerability discovery: {vulnerability.get('attack_category')}")
            
        except Exception as e:
            logger.error(f"❌ [AUDIT] Failed to log vulnerability discovery: {e}")
    
    async def log_compliance_check(self, compliance_report: Dict[str, Any]):
        """Log compliance verification for regulators"""
        try:
            audit_event = {
                "event_id": str(uuid.uuid4()),
                "event_type": "COMPLIANCE_VERIFICATION",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "test_id": compliance_report.get("test_id"),
                "model_tested": compliance_report.get("model_name", "all_models"),
                "user_id": compliance_report.get("auditor_id", "system"),
                "session_id": compliance_report.get("session_id"),
                "ip_address": compliance_report.get("auditor_ip", "127.0.0.1"),
                "user_agent": compliance_report.get("auditor_agent", "compliance-tool"),
                "resource": "compliance_framework",
                "action": "compliance_verification",
                "details": {
                    "compliance_frameworks": compliance_report.get("frameworks", []),
                    "overall_compliance_status": compliance_report.get("status", "UNKNOWN"),
                    "owasp_coverage_percentage": compliance_report.get("owasp_coverage", 0),
                    "nist_compliance_percentage": compliance_report.get("nist_compliance", 0),
                    "iso27001_compliance_percentage": compliance_report.get("iso27001_compliance", 0),
                    "soc2_compliance_percentage": compliance_report.get("soc2_compliance", 0),
                    "non_compliant_areas": compliance_report.get("gaps", []),
                    "auditor_notes": compliance_report.get("notes", ""),
                    "next_audit_due": compliance_report.get("next_audit_date"),
                    "evidence_stored": compliance_report.get("evidence_location", ""),
                    "attestation_signed": compliance_report.get("attestation_signed", False)
                },
                "severity": "INFO",
                "success": True,
                "error_message": None,
                "data_classification": "CONFIDENTIAL",
                "retention_period": 2555,  # 7 years for SOC 2
                "compliance_tags": ["SOC2", "ISO27001", "COMPLIANCE_AUDIT"]
            }
            
            await self._queue_audit_event(audit_event)
            logger.info(f"📋 [AUDIT] Logged compliance check: {compliance_report.get('status')}")
            
        except Exception as e:
            logger.error(f"❌ [AUDIT] Failed to log compliance check: {e}")
    
    async def log_model_security_assessment(self, model_name: str, security_score: float, 
                                          vulnerabilities: List[Dict[str, Any]]):
        """Log model security assessment for tracking"""
        try:
            audit_event = {
                "event_id": str(uuid.uuid4()),
                "event_type": "MODEL_SECURITY_ASSESSMENT",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "test_id": f"model_assessment_{model_name}_{int(datetime.now().timestamp())}",
                "model_tested": model_name,
                "user_id": "red-team-service",
                "session_id": f"assessment_{model_name}",
                "ip_address": "127.0.0.1",
                "user_agent": "red-team-service",
                "resource": f"model/{model_name}",
                "action": "security_assessment",
                "details": {
                    "security_score": security_score,
                    "vulnerability_count": len(vulnerabilities),
                    "critical_vulnerabilities": len([v for v in vulnerabilities if v.get("severity") == "CRITICAL"]),
                    "high_vulnerabilities": len([v for v in vulnerabilities if v.get("severity") == "HIGH"]),
                    "medium_vulnerabilities": len([v for v in vulnerabilities if v.get("severity") == "MEDIUM"]),
                    "low_vulnerabilities": len([v for v in vulnerabilities if v.get("severity") == "LOW"]),
                    "assessment_timestamp": datetime.now(timezone.utc).isoformat(),
                    "model_version": "latest",
                    "assessment_type": "comprehensive_security_testing"
                },
                "severity": "INFO",
                "success": True,
                "error_message": None,
                "data_classification": "CONFIDENTIAL",
                "retention_period": 2555,  # 7 years for SOC 2
                "compliance_tags": ["SOC2", "ISO27001", "MODEL_SECURITY"]
            }
            
            await self._queue_audit_event(audit_event)
            logger.info(f"🔍 [AUDIT] Logged model security assessment: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ [AUDIT] Failed to log model security assessment: {e}")
    
    async def _queue_audit_event(self, event: Dict[str, Any]):
        """Queue audit event for batch processing"""
        self.audit_events_buffer.append(event)
        
        # Flush if buffer is full
        if len(self.audit_events_buffer) >= self.buffer_size:
            await self._flush_events()
    
    async def _periodic_flush(self):
        """Periodically flush audit events"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ [AUDIT] Error in periodic flush: {e}")
    
    async def _flush_events(self):
        """Flush queued audit events to analytics service"""
        if not self.audit_events_buffer:
            return
        
        try:
            events_to_flush = self.audit_events_buffer.copy()
            self.audit_events_buffer.clear()
            
            # Send to analytics service
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.analytics_service_url}/audit/events",
                    json={"events": events_to_flush}
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ [AUDIT] Flushed {len(events_to_flush)} audit events")
                else:
                    logger.error(f"❌ [AUDIT] Failed to flush events: {response.status_code}")
                    # Re-queue events for retry
                    self.audit_events_buffer.extend(events_to_flush)
            
            # Send business metrics
            await self._send_business_metrics(events_to_flush)
            
        except Exception as e:
            logger.error(f"❌ [AUDIT] Error flushing events: {e}")
            # Re-queue events for retry
            self.audit_events_buffer.extend(events_to_flush)
    
    async def _send_business_metrics(self, events: List[Dict[str, Any]]):
        """Send business metrics for audit events"""
        try:
            metrics = []
            for event in events:
                if event.get("event_type") == "VULNERABILITY_DISCOVERED":
                    metrics.append({
                        "metric_name": "vulnerabilities_discovered",
                        "value": 1.0,
                        "timestamp": event["timestamp"],
                        "tags": {
                            "severity": event.get("severity", "UNKNOWN"),
                            "attack_category": event.get("details", {}).get("attack_category", "unknown"),
                            "model": event.get("model_tested", "unknown")
                        },
                        "metadata": {
                            "cvss_score": event.get("details", {}).get("cvss_score", 0.0),
                            "compliance_impact": event.get("details", {}).get("compliance_impact", {})
                        }
                    })
                elif event.get("event_type") == "RED_TEAM_TEST_EXECUTION":
                    metrics.append({
                        "metric_name": "security_tests_executed",
                        "value": 1.0,
                        "timestamp": event["timestamp"],
                        "tags": {
                            "model": event.get("model_tested", "unknown"),
                            "test_type": "red_team"
                        },
                        "metadata": {
                            "total_attacks": event.get("details", {}).get("total_attacks", 0),
                            "vulnerabilities_found": event.get("details", {}).get("vulnerabilities_found", 0)
                        }
                    })
            
            if metrics:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    for metric in metrics:
                        await client.post(
                            f"{self.business_metrics_url}/metrics",
                            json=metric
                        )
            
        except Exception as e:
            logger.error(f"❌ [AUDIT] Error sending business metrics: {e}")
    
    def _assess_compliance_impact(self, vulnerability: Dict[str, Any]) -> Dict[str, Any]:
        """Assess compliance impact of vulnerability"""
        attack_category = vulnerability.get("attack_category", "unknown")
        cvss_score = vulnerability.get("cvss_score", 0.0)
        
        compliance_impact = {
            "owasp_llm": {
                "affected": True,
                "category": self._map_to_owasp_category(attack_category),
                "severity": "HIGH" if cvss_score >= 7.0 else "MEDIUM"
            },
            "nist_ai_rmf": {
                "affected": True,
                "category": "GOVERN-1.2: Risk Management",
                "severity": "HIGH" if cvss_score >= 7.0 else "MEDIUM"
            },
            "iso27001": {
                "affected": True,
                "category": "A.12.6.1: Technical Vulnerability Management",
                "severity": "HIGH" if cvss_score >= 7.0 else "MEDIUM"
            },
            "soc2": {
                "affected": True,
                "category": "CC6.1: Logical and Physical Access Controls",
                "severity": "HIGH" if cvss_score >= 7.0 else "MEDIUM"
            }
        }
        
        return compliance_impact
    
    def _assess_business_impact(self, vulnerability: Dict[str, Any]) -> Dict[str, Any]:
        """Assess business impact of vulnerability"""
        cvss_score = vulnerability.get("cvss_score", 0.0)
        attack_category = vulnerability.get("attack_category", "unknown")
        
        # Estimate financial impact based on CVSS score and attack type
        base_financial_impact = cvss_score * 10000  # $10K per CVSS point
        
        if attack_category in ["prompt_injection", "jailbreak"]:
            base_financial_impact *= 1.5  # Higher impact for injection attacks
        
        return {
            "estimated_financial_impact": base_financial_impact,
            "reputation_risk": "HIGH" if cvss_score >= 7.0 else "MEDIUM",
            "regulatory_risk": "HIGH" if cvss_score >= 7.0 else "MEDIUM",
            "operational_risk": "HIGH" if cvss_score >= 7.0 else "MEDIUM",
            "data_breach_risk": "HIGH" if attack_category in ["system_extraction", "data_poisoning"] else "MEDIUM"
        }
    
    def _map_to_owasp_category(self, attack_category: str) -> str:
        """Map attack category to OWASP LLM Top 10"""
        mapping = {
            "prompt_injection": "LLM01",
            "jailbreak": "LLM01",
            "output_security": "LLM02",
            "data_poisoning": "LLM03",
            "dos_attack": "LLM04",
            "supply_chain": "LLM05",
            "system_extraction": "LLM06",
            "code_injection": "LLM07",
            "excessive_agency": "LLM08",
            "hallucination": "LLM09",
            "model_theft": "LLM10"
        }
        return mapping.get(attack_category, "UNKNOWN")
