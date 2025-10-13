"""
Executive Dashboard for Real-time Security Metrics
Provides real-time dashboard integration for security metrics
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import asyncio
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DashboardType(Enum):
    """Dashboard types"""
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    COMPLIANCE = "compliance"
    REAL_TIME = "real_time"


@dataclass
class DashboardMetric:
    """Dashboard metric data structure"""
    name: str
    value: float
    unit: str
    trend: str  # "up", "down", "stable"
    status: str  # "good", "warning", "critical"
    last_updated: datetime
    target_value: Optional[float] = None
    description: Optional[str] = None


@dataclass
class DashboardAlert:
    """Dashboard alert data structure"""
    id: str
    title: str
    message: str
    severity: str  # "info", "warning", "critical"
    timestamp: datetime
    category: str
    resolved: bool = False
    action_required: bool = False


class ExecutiveDashboard:
    """
    Real-time executive dashboard for security metrics
    """
    
    def __init__(self):
        """Initialize executive dashboard"""
        self.metrics: Dict[str, DashboardMetric] = {}
        self.alerts: List[DashboardAlert] = []
        self.last_update = datetime.now()
        logger.info("✅ Executive Dashboard initialized")
    
    async def update_metrics(self, security_data: Dict[str, Any]) -> bool:
        """
        Update dashboard metrics with latest security data
        
        Args:
            security_data: Latest security assessment data
            
        Returns:
            True if successful
        """
        try:
            # Update core security metrics
            await self._update_security_metrics(security_data)
            
            # Update compliance metrics
            await self._update_compliance_metrics(security_data)
            
            # Update performance metrics
            await self._update_performance_metrics(security_data)
            
            # Update risk metrics
            await self._update_risk_metrics(security_data)
            
            # Generate alerts based on metrics
            await self._generate_alerts()
            
            self.last_update = datetime.now()
            logger.info("✅ Dashboard metrics updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Dashboard metrics update failed: {e}")
            return False
    
    async def _update_security_metrics(self, data: Dict[str, Any]):
        """Update security-related metrics"""
        try:
            # Detection rate
            detection_rate = data.get('attack_statistics', {}).get('detection_rate', 0)
            self.metrics['detection_rate'] = DashboardMetric(
                name="Detection Rate",
                value=detection_rate,
                unit="%",
                trend=self._calculate_trend(detection_rate, data.get('prev_detection_rate', 0)),
                status=self._get_status(detection_rate, 95, 80),
                last_updated=datetime.now(),
                target_value=95.0,
                description="Percentage of attacks successfully detected"
            )
            
            # False positive rate
            false_positive_rate = data.get('attack_statistics', {}).get('false_positive_rate', 0)
            self.metrics['false_positive_rate'] = DashboardMetric(
                name="False Positive Rate",
                value=false_positive_rate,
                unit="%",
                trend=self._calculate_trend(false_positive_rate, data.get('prev_false_positive_rate', 0)),
                status=self._get_status(false_positive_rate, 5, 10, reverse=True),
                last_updated=datetime.now(),
                target_value=5.0,
                description="Percentage of false positive detections"
            )
            
            # Response time
            response_time = data.get('attack_statistics', {}).get('avg_response_time', 0)
            self.metrics['response_time'] = DashboardMetric(
                name="Response Time",
                value=response_time,
                unit="ms",
                trend=self._calculate_trend(response_time, data.get('prev_avg_response_time', 0)),
                status=self._get_status(response_time, 100, 500, reverse=True),
                last_updated=datetime.now(),
                target_value=100.0,
                description="Average response time for security decisions"
            )
            
            # Coverage
            coverage = data.get('coverage_metrics', {}).get('overall_coverage', 0)
            self.metrics['coverage'] = DashboardMetric(
                name="Security Coverage",
                value=coverage,
                unit="%",
                trend=self._calculate_trend(coverage, data.get('prev_coverage', 0)),
                status=self._get_status(coverage, 100, 90),
                last_updated=datetime.now(),
                target_value=100.0,
                description="Percentage of attack vectors covered"
            )
            
        except Exception as e:
            logger.error(f"Security metrics update failed: {e}")
    
    async def _update_compliance_metrics(self, data: Dict[str, Any]):
        """Update compliance-related metrics"""
        try:
            # SOC 2 compliance
            soc2_score = data.get('compliance_status', {}).get('soc2', {}).get('score', 0)
            self.metrics['soc2_compliance'] = DashboardMetric(
                name="SOC 2 Compliance",
                value=soc2_score,
                unit="%",
                trend=self._calculate_trend(soc2_score, data.get('prev_soc2_score', 0)),
                status=self._get_status(soc2_score, 95, 80),
                last_updated=datetime.now(),
                target_value=95.0,
                description="SOC 2 Type II compliance score"
            )
            
            # ISO 27001 compliance
            iso27001_score = data.get('compliance_status', {}).get('iso27001', {}).get('score', 0)
            self.metrics['iso27001_compliance'] = DashboardMetric(
                name="ISO 27001 Compliance",
                value=iso27001_score,
                unit="%",
                trend=self._calculate_trend(iso27001_score, data.get('prev_iso27001_score', 0)),
                status=self._get_status(iso27001_score, 95, 80),
                last_updated=datetime.now(),
                target_value=95.0,
                description="ISO 27001 compliance score"
            )
            
            # OWASP LLM Top 10 compliance
            owasp_score = data.get('compliance_status', {}).get('owasp', {}).get('score', 0)
            self.metrics['owasp_compliance'] = DashboardMetric(
                name="OWASP LLM Top 10",
                value=owasp_score,
                unit="%",
                trend=self._calculate_trend(owasp_score, data.get('prev_owasp_score', 0)),
                status=self._get_status(owasp_score, 95, 80),
                last_updated=datetime.now(),
                target_value=95.0,
                description="OWASP LLM Top 10 compliance score"
            )
            
        except Exception as e:
            logger.error(f"Compliance metrics update failed: {e}")
    
    async def _update_performance_metrics(self, data: Dict[str, Any]):
        """Update performance-related metrics"""
        try:
            # Model accuracy
            model_performance = data.get('model_performance', {})
            if model_performance:
                avg_accuracy = sum(perf.get('accuracy', 0) for perf in model_performance.values()) / len(model_performance)
                self.metrics['model_accuracy'] = DashboardMetric(
                    name="Model Accuracy",
                    value=avg_accuracy * 100,
                    unit="%",
                    trend=self._calculate_trend(avg_accuracy * 100, data.get('prev_avg_accuracy', 0)),
                    status=self._get_status(avg_accuracy * 100, 95, 85),
                    last_updated=datetime.now(),
                    target_value=95.0,
                    description="Average model accuracy across all models"
                )
            
            # Training success rate
            training_success_rate = data.get('training_metrics', {}).get('success_rate', 0)
            self.metrics['training_success'] = DashboardMetric(
                name="Training Success Rate",
                value=training_success_rate,
                unit="%",
                trend=self._calculate_trend(training_success_rate, data.get('prev_training_success_rate', 0)),
                status=self._get_status(training_success_rate, 95, 80),
                last_updated=datetime.now(),
                target_value=95.0,
                description="Percentage of successful training jobs"
            )
            
            # System uptime
            uptime = data.get('system_metrics', {}).get('uptime', 0)
            self.metrics['uptime'] = DashboardMetric(
                name="System Uptime",
                value=uptime,
                unit="%",
                trend=self._calculate_trend(uptime, data.get('prev_uptime', 0)),
                status=self._get_status(uptime, 99.9, 99.0),
                last_updated=datetime.now(),
                target_value=99.9,
                description="System availability percentage"
            )
            
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")
    
    async def _update_risk_metrics(self, data: Dict[str, Any]):
        """Update risk-related metrics"""
        try:
            # Overall risk score
            risk_summary = data.get('risk_summary', {})
            overall_risk = risk_summary.get('overall_score', 0)
            self.metrics['overall_risk'] = DashboardMetric(
                name="Overall Risk Score",
                value=overall_risk,
                unit="/10",
                trend=self._calculate_trend(overall_risk, data.get('prev_overall_risk', 0)),
                status=self._get_status(overall_risk, 3, 5, reverse=True),
                last_updated=datetime.now(),
                target_value=3.0,
                description="Overall security risk score (lower is better)"
            )
            
            # Critical vulnerabilities
            critical_vulns = risk_summary.get('critical', 0)
            self.metrics['critical_vulnerabilities'] = DashboardMetric(
                name="Critical Vulnerabilities",
                value=critical_vulns,
                unit="count",
                trend=self._calculate_trend(critical_vulns, data.get('prev_critical_vulns', 0)),
                status=self._get_status(critical_vulns, 0, 5, reverse=True),
                last_updated=datetime.now(),
                target_value=0.0,
                description="Number of critical vulnerabilities"
            )
            
            # High risk issues
            high_risk = risk_summary.get('high', 0)
            self.metrics['high_risk_issues'] = DashboardMetric(
                name="High Risk Issues",
                value=high_risk,
                unit="count",
                trend=self._calculate_trend(high_risk, data.get('prev_high_risk', 0)),
                status=self._get_status(high_risk, 0, 10, reverse=True),
                last_updated=datetime.now(),
                target_value=0.0,
                description="Number of high risk security issues"
            )
            
        except Exception as e:
            logger.error(f"Risk metrics update failed: {e}")
    
    def _calculate_trend(self, current: float, previous: float) -> str:
        """Calculate trend direction"""
        if previous == 0:
            return "stable"
        
        change_percent = ((current - previous) / previous) * 100
        
        if change_percent > 5:
            return "up"
        elif change_percent < -5:
            return "down"
        else:
            return "stable"
    
    def _get_status(self, value: float, good_threshold: float, warning_threshold: float, reverse: bool = False) -> str:
        """Get status based on value and thresholds"""
        if reverse:
            if value <= good_threshold:
                return "good"
            elif value <= warning_threshold:
                return "warning"
            else:
                return "critical"
        else:
            if value >= good_threshold:
                return "good"
            elif value >= warning_threshold:
                return "warning"
            else:
                return "critical"
    
    async def _generate_alerts(self):
        """Generate alerts based on current metrics"""
        try:
            # Clear old alerts
            self.alerts = []
            
            # Check each metric for alerts
            for metric_name, metric in self.metrics.items():
                if metric.status == "critical":
                    alert = DashboardAlert(
                        id=f"critical_{metric_name}_{datetime.now().timestamp()}",
                        title=f"Critical: {metric.name}",
                        message=f"{metric.name} is at {metric.value}{metric.unit} (Status: {metric.status})",
                        severity="critical",
                        timestamp=datetime.now(),
                        category="metric",
                        action_required=True
                    )
                    self.alerts.append(alert)
                
                elif metric.status == "warning":
                    alert = DashboardAlert(
                        id=f"warning_{metric_name}_{datetime.now().timestamp()}",
                        title=f"Warning: {metric.name}",
                        message=f"{metric.name} is at {metric.value}{metric.unit} (Status: {metric.status})",
                        severity="warning",
                        timestamp=datetime.now(),
                        category="metric",
                        action_required=False
                    )
                    self.alerts.append(alert)
            
            # Add system alerts
            await self._add_system_alerts()
            
        except Exception as e:
            logger.error(f"Alert generation failed: {e}")
    
    async def _add_system_alerts(self):
        """Add system-level alerts"""
        try:
            # Check for recent security incidents
            recent_incidents = await self._check_recent_incidents()
            if recent_incidents:
                alert = DashboardAlert(
                    id=f"incident_{datetime.now().timestamp()}",
                    title="Security Incident Detected",
                    message=f"{recent_incidents} security incidents detected in the last 24 hours",
                    severity="critical",
                    timestamp=datetime.now(),
                    category="incident",
                    action_required=True
                )
                self.alerts.append(alert)
            
            # Check for compliance gaps
            compliance_gaps = await self._check_compliance_gaps()
            if compliance_gaps:
                alert = DashboardAlert(
                    id=f"compliance_{datetime.now().timestamp()}",
                    title="Compliance Gap Detected",
                    message=f"{compliance_gaps} compliance gaps identified",
                    severity="warning",
                    timestamp=datetime.now(),
                    category="compliance",
                    action_required=True
                )
                self.alerts.append(alert)
            
        except Exception as e:
            logger.error(f"System alerts generation failed: {e}")
    
    async def _check_recent_incidents(self) -> int:
        """Check for recent security incidents"""
        # This would typically query a security incident database
        # For now, return a mock value
        return 0
    
    async def _check_compliance_gaps(self) -> int:
        """Check for compliance gaps"""
        # This would typically query compliance tracking systems
        # For now, return a mock value
        return 0
    
    def get_dashboard_data(self, dashboard_type: DashboardType = DashboardType.EXECUTIVE) -> Dict[str, Any]:
        """
        Get dashboard data for specified type
        
        Args:
            dashboard_type: Type of dashboard to generate
            
        Returns:
            Dashboard data dictionary
        """
        try:
            if dashboard_type == DashboardType.EXECUTIVE:
                return self._get_executive_dashboard_data()
            elif dashboard_type == DashboardType.TECHNICAL:
                return self._get_technical_dashboard_data()
            elif dashboard_type == DashboardType.COMPLIANCE:
                return self._get_compliance_dashboard_data()
            elif dashboard_type == DashboardType.REAL_TIME:
                return self._get_realtime_dashboard_data()
            else:
                return {"error": "Invalid dashboard type"}
                
        except Exception as e:
            logger.error(f"Dashboard data generation failed: {e}")
            return {"error": str(e)}
    
    def _get_executive_dashboard_data(self) -> Dict[str, Any]:
        """Get executive dashboard data"""
        try:
            # Filter metrics for executive view
            executive_metrics = {
                'detection_rate': self.metrics.get('detection_rate'),
                'false_positive_rate': self.metrics.get('false_positive_rate'),
                'overall_risk': self.metrics.get('overall_risk'),
                'soc2_compliance': self.metrics.get('soc2_compliance'),
                'uptime': self.metrics.get('uptime')
            }
            
            # Filter alerts for executive view
            executive_alerts = [alert for alert in self.alerts if alert.severity in ['critical', 'warning']]
            
            return {
                'dashboard_type': 'executive',
                'last_updated': self.last_update.isoformat(),
                'metrics': {k: v.__dict__ for k, v in executive_metrics.items() if v},
                'alerts': [alert.__dict__ for alert in executive_alerts],
                'summary': {
                    'total_metrics': len([m for m in executive_metrics.values() if m]),
                    'critical_alerts': len([a for a in executive_alerts if a.severity == 'critical']),
                    'warning_alerts': len([a for a in executive_alerts if a.severity == 'warning']),
                    'overall_status': self._get_overall_status()
                }
            }
            
        except Exception as e:
            logger.error(f"Executive dashboard data generation failed: {e}")
            return {"error": str(e)}
    
    def _get_technical_dashboard_data(self) -> Dict[str, Any]:
        """Get technical dashboard data"""
        try:
            # Include all metrics for technical view
            technical_metrics = self.metrics.copy()
            
            # Include all alerts for technical view
            technical_alerts = self.alerts.copy()
            
            return {
                'dashboard_type': 'technical',
                'last_updated': self.last_update.isoformat(),
                'metrics': {k: v.__dict__ for k, v in technical_metrics.items()},
                'alerts': [alert.__dict__ for alert in technical_alerts],
                'summary': {
                    'total_metrics': len(technical_metrics),
                    'critical_alerts': len([a for a in technical_alerts if a.severity == 'critical']),
                    'warning_alerts': len([a for a in technical_alerts if a.severity == 'warning']),
                    'info_alerts': len([a for a in technical_alerts if a.severity == 'info']),
                    'overall_status': self._get_overall_status()
                }
            }
            
        except Exception as e:
            logger.error(f"Technical dashboard data generation failed: {e}")
            return {"error": str(e)}
    
    def _get_compliance_dashboard_data(self) -> Dict[str, Any]:
        """Get compliance dashboard data"""
        try:
            # Filter metrics for compliance view
            compliance_metrics = {
                'soc2_compliance': self.metrics.get('soc2_compliance'),
                'iso27001_compliance': self.metrics.get('iso27001_compliance'),
                'owasp_compliance': self.metrics.get('owasp_compliance'),
                'overall_risk': self.metrics.get('overall_risk')
            }
            
            # Filter alerts for compliance view
            compliance_alerts = [alert for alert in self.alerts if alert.category in ['compliance', 'metric']]
            
            return {
                'dashboard_type': 'compliance',
                'last_updated': self.last_update.isoformat(),
                'metrics': {k: v.__dict__ for k, v in compliance_metrics.items() if v},
                'alerts': [alert.__dict__ for alert in compliance_alerts],
                'summary': {
                    'total_metrics': len([m for m in compliance_metrics.values() if m]),
                    'critical_alerts': len([a for a in compliance_alerts if a.severity == 'critical']),
                    'warning_alerts': len([a for a in compliance_alerts if a.severity == 'warning']),
                    'overall_status': self._get_overall_status()
                }
            }
            
        except Exception as e:
            logger.error(f"Compliance dashboard data generation failed: {e}")
            return {"error": str(e)}
    
    def _get_realtime_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        try:
            # Include all metrics and alerts for real-time view
            return {
                'dashboard_type': 'real_time',
                'last_updated': self.last_update.isoformat(),
                'metrics': {k: v.__dict__ for k, v in self.metrics.items()},
                'alerts': [alert.__dict__ for alert in self.alerts],
                'summary': {
                    'total_metrics': len(self.metrics),
                    'critical_alerts': len([a for a in self.alerts if a.severity == 'critical']),
                    'warning_alerts': len([a for a in self.alerts if a.severity == 'warning']),
                    'info_alerts': len([a for a in self.alerts if a.severity == 'info']),
                    'overall_status': self._get_overall_status()
                }
            }
            
        except Exception as e:
            logger.error(f"Real-time dashboard data generation failed: {e}")
            return {"error": str(e)}
    
    def _get_overall_status(self) -> str:
        """Get overall dashboard status"""
        try:
            critical_alerts = len([a for a in self.alerts if a.severity == 'critical'])
            warning_alerts = len([a for a in self.alerts if a.severity == 'warning'])
            
            if critical_alerts > 0:
                return "critical"
            elif warning_alerts > 3:
                return "warning"
            else:
                return "good"
                
        except Exception as e:
            logger.error(f"Overall status calculation failed: {e}")
            return "unknown"
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        try:
            return {
                'total_metrics': len(self.metrics),
                'metrics_by_status': {
                    'good': len([m for m in self.metrics.values() if m.status == 'good']),
                    'warning': len([m for m in self.metrics.values() if m.status == 'warning']),
                    'critical': len([m for m in self.metrics.values() if m.status == 'critical'])
                },
                'total_alerts': len(self.alerts),
                'alerts_by_severity': {
                    'critical': len([a for a in self.alerts if a.severity == 'critical']),
                    'warning': len([a for a in self.alerts if a.severity == 'warning']),
                    'info': len([a for a in self.alerts if a.severity == 'info'])
                },
                'last_update': self.last_update.isoformat(),
                'overall_status': self._get_overall_status()
            }
            
        except Exception as e:
            logger.error(f"Metrics summary generation failed: {e}")
            return {"error": str(e)}
    
    def clear_alerts(self, alert_ids: List[str] = None):
        """Clear alerts by IDs or all alerts"""
        try:
            if alert_ids:
                self.alerts = [alert for alert in self.alerts if alert.id not in alert_ids]
            else:
                self.alerts = []
            
            logger.info(f"Cleared {len(alert_ids) if alert_ids else 'all'} alerts")
            
        except Exception as e:
            logger.error(f"Alert clearing failed: {e}")
    
    def mark_alert_resolved(self, alert_id: str):
        """Mark a specific alert as resolved"""
        try:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    alert.action_required = False
                    break
            
            logger.info(f"Marked alert {alert_id} as resolved")
            
        except Exception as e:
            logger.error(f"Alert resolution marking failed: {e}")
