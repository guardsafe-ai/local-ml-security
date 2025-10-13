"""
Alert Manager
Manages alerts, notifications, and escalation for red team testing events.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
import json

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class AlertChannel(Enum):
    """Alert notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    CONSOLE = "console"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    description: str
    condition: str  # Expression to evaluate
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown: int = 300  # Seconds between alerts
    enabled: bool = True
    tags: Set[str] = field(default_factory=set)

@dataclass
class Alert:
    """Alert instance"""
    alert_id: str
    rule_id: str
    title: str
    message: str
    severity: AlertSeverity
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

@dataclass
class NotificationConfig:
    """Notification configuration"""
    channel: AlertChannel
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)

class AlertManager:
    """Alert management system"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_configs: Dict[AlertChannel, NotificationConfig] = {}
        self.cooldowns: Dict[str, datetime] = {}
        self.notification_handlers: Dict[AlertChannel, Callable] = {}
        self.alert_handlers: List[Callable] = []
        
        # Initialize default notification configs
        self._setup_default_configs()
        
        # Initialize default rules
        self._setup_default_rules()
    
    def _setup_default_configs(self):
        """Setup default notification configurations"""
        self.notification_configs = {
            AlertChannel.CONSOLE: NotificationConfig(
                channel=AlertChannel.CONSOLE,
                enabled=True
            ),
            AlertChannel.EMAIL: NotificationConfig(
                channel=AlertChannel.EMAIL,
                enabled=False,
                config={
                    "smtp_server": "localhost",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "from_email": "alerts@redteam.local",
                    "to_emails": []
                }
            ),
            AlertChannel.SLACK: NotificationConfig(
                channel=AlertChannel.SLACK,
                enabled=False,
                config={
                    "webhook_url": "",
                    "channel": "#alerts"
                }
            ),
            AlertChannel.WEBHOOK: NotificationConfig(
                channel=AlertChannel.WEBHOOK,
                enabled=False,
                config={
                    "url": "",
                    "headers": {},
                    "timeout": 30
                }
            ),
            AlertChannel.SMS: NotificationConfig(
                channel=AlertChannel.SMS,
                enabled=False,
                config={
                    "provider": "twilio",
                    "account_sid": "",
                    "auth_token": "",
                    "from_number": "",
                    "to_numbers": []
                }
            )
        }
    
    def _setup_default_rules(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                rule_id="high_vulnerability",
                name="High Severity Vulnerability Detected",
                description="Alert when high severity vulnerability is detected",
                condition="severity == 'high' and event_type == 'vulnerability_detected'",
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL],
                cooldown=600
            ),
            AlertRule(
                rule_id="critical_vulnerability",
                name="Critical Severity Vulnerability Detected",
                description="Alert when critical severity vulnerability is detected",
                condition="severity == 'critical' and event_type == 'vulnerability_detected'",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL, AlertChannel.SLACK],
                cooldown=300
            ),
            AlertRule(
                rule_id="compliance_violation",
                name="Compliance Violation Detected",
                description="Alert when compliance violation is detected",
                condition="event_type == 'compliance_violation'",
                severity=AlertSeverity.MEDIUM,
                channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL],
                cooldown=900
            ),
            AlertRule(
                rule_id="privacy_breach",
                name="Privacy Breach Detected",
                description="Alert when privacy breach is detected",
                condition="event_type == 'privacy_breach'",
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL, AlertChannel.SLACK],
                cooldown=600
            ),
            AlertRule(
                rule_id="system_error",
                name="System Error",
                description="Alert when system error occurs",
                condition="severity == 'error' and event_type == 'system_alert'",
                severity=AlertSeverity.MEDIUM,
                channels=[AlertChannel.CONSOLE],
                cooldown=300
            ),
            AlertRule(
                rule_id="worker_failure",
                name="Worker Process Failure",
                description="Alert when worker process fails",
                condition="event_type == 'worker_status' and status == 'failed'",
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL],
                cooldown=600
            ),
            AlertRule(
                rule_id="queue_backlog",
                name="Queue Backlog",
                description="Alert when queue size exceeds threshold",
                condition="event_type == 'queue_update' and size > 1000",
                severity=AlertSeverity.MEDIUM,
                channels=[AlertChannel.CONSOLE],
                cooldown=1800
            ),
            AlertRule(
                rule_id="high_cpu_usage",
                name="High CPU Usage",
                description="Alert when CPU usage is high",
                condition="event_type == 'metrics_update' and cpu_usage > 90",
                severity=AlertSeverity.MEDIUM,
                channels=[AlertChannel.CONSOLE],
                cooldown=1800
            ),
            AlertRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                description="Alert when memory usage is high",
                condition="event_type == 'metrics_update' and memory_usage > 90",
                severity=AlertSeverity.MEDIUM,
                channels=[AlertChannel.CONSOLE],
                cooldown=1800
            ),
            AlertRule(
                rule_id="attack_failure_rate",
                name="High Attack Failure Rate",
                description="Alert when attack failure rate is high",
                condition="event_type == 'attack_complete' and success == false and failure_rate > 0.5",
                severity=AlertSeverity.MEDIUM,
                channels=[AlertChannel.CONSOLE],
                cooldown=1800
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_id: str):
        """Remove alert rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
    
    def update_rule(self, rule_id: str, **updates):
        """Update alert rule"""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            logger.info(f"Updated alert rule: {rule_id}")
    
    def configure_notification(self, channel: AlertChannel, config: NotificationConfig):
        """Configure notification channel"""
        self.notification_configs[channel] = config
        logger.info(f"Configured notification channel: {channel.value}")
    
    def register_notification_handler(self, channel: AlertChannel, handler: Callable):
        """Register notification handler for channel"""
        self.notification_handlers[channel] = handler
        logger.info(f"Registered notification handler for: {channel.value}")
    
    def register_alert_handler(self, handler: Callable):
        """Register alert handler"""
        self.alert_handlers.append(handler)
        logger.info("Registered alert handler")
    
    async def evaluate_event(self, event_data: Dict[str, Any]):
        """Evaluate event against alert rules"""
        try:
            for rule in self.rules.values():
                if not rule.enabled:
                    continue
                
                # Check cooldown
                cooldown_key = f"{rule.rule_id}_{self._get_event_key(event_data)}"
                if cooldown_key in self.cooldowns:
                    if datetime.now() - self.cooldowns[cooldown_key] < timedelta(seconds=rule.cooldown):
                        continue
                
                # Evaluate condition
                if self._evaluate_condition(rule.condition, event_data):
                    await self._trigger_alert(rule, event_data)
                    self.cooldowns[cooldown_key] = datetime.now()
                    
        except Exception as e:
            logger.error(f"Failed to evaluate event: {e}")
    
    def _evaluate_condition(self, condition: str, event_data: Dict[str, Any]) -> bool:
        """Evaluate alert condition"""
        try:
            # Simple condition evaluation
            # In production, use a proper expression evaluator
            local_vars = event_data.copy()
            local_vars.update({
                'severity': event_data.get('severity', ''),
                'event_type': event_data.get('event_type', ''),
                'status': event_data.get('status', ''),
                'size': event_data.get('size', 0),
                'cpu_usage': event_data.get('cpu_usage', 0),
                'memory_usage': event_data.get('memory_usage', 0),
                'success': event_data.get('success', True),
                'failure_rate': event_data.get('failure_rate', 0)
            })
            
            # Basic condition evaluation (simplified)
            if 'severity == "critical"' in condition and local_vars.get('severity') == 'critical':
                return True
            elif 'severity == "high"' in condition and local_vars.get('severity') == 'high':
                return True
            elif 'event_type == "vulnerability_detected"' in condition and local_vars.get('event_type') == 'vulnerability_detected':
                return True
            elif 'event_type == "compliance_violation"' in condition and local_vars.get('event_type') == 'compliance_violation':
                return True
            elif 'event_type == "privacy_breach"' in condition and local_vars.get('event_type') == 'privacy_breach':
                return True
            elif 'size > 1000' in condition and local_vars.get('size', 0) > 1000:
                return True
            elif 'cpu_usage > 90' in condition and local_vars.get('cpu_usage', 0) > 90:
                return True
            elif 'memory_usage > 90' in condition and local_vars.get('memory_usage', 0) > 90:
                return True
            elif 'success == false' in condition and local_vars.get('success') == False:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to evaluate condition: {e}")
            return False
    
    def _get_event_key(self, event_data: Dict[str, Any]) -> str:
        """Get unique key for event (for cooldown tracking)"""
        return f"{event_data.get('event_type', 'unknown')}_{event_data.get('source', 'unknown')}"
    
    async def _trigger_alert(self, rule: AlertRule, event_data: Dict[str, Any]):
        """Trigger alert for rule"""
        try:
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                title=f"Alert: {rule.name}",
                message=f"{rule.description}\nEvent: {json.dumps(event_data, indent=2)}",
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metadata=event_data,
                tags=rule.tags
            )
            
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            # Call alert handlers
            for handler in self.alert_handlers:
                try:
                    await handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")
            
            # Send notifications
            await self._send_notifications(alert, rule)
            
            logger.info(f"Triggered alert: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")
    
    async def _send_notifications(self, alert: Alert, rule: AlertRule):
        """Send notifications for alert"""
        try:
            for channel in rule.channels:
                if channel in self.notification_configs:
                    config = self.notification_configs[channel]
                    if config.enabled:
                        if channel in self.notification_handlers:
                            handler = self.notification_handlers[channel]
                            await handler(alert, config)
                        else:
                            await self._send_default_notification(alert, channel, config)
                            
        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")
    
    async def _send_default_notification(self, alert: Alert, channel: AlertChannel, config: NotificationConfig):
        """Send default notification"""
        try:
            if channel == AlertChannel.CONSOLE:
                print(f"\n🚨 ALERT: {alert.title}")
                print(f"Severity: {alert.severity.value.upper()}")
                print(f"Message: {alert.message}")
                print(f"Time: {alert.created_at}")
                print("-" * 50)
                
            elif channel == AlertChannel.EMAIL:
                await self._send_email_notification(alert, config)
                
            elif channel == AlertChannel.SLACK:
                await self._send_slack_notification(alert, config)
                
            elif channel == AlertChannel.WEBHOOK:
                await self._send_webhook_notification(alert, config)
                
            elif channel == AlertChannel.SMS:
                await self._send_sms_notification(alert, config)
                
        except Exception as e:
            logger.error(f"Failed to send {channel.value} notification: {e}")
    
    async def _send_email_notification(self, alert: Alert, config: NotificationConfig):
        """Send email notification"""
        # Implementation would depend on email service
        logger.info(f"Email notification sent for alert: {alert.alert_id}")
    
    async def _send_slack_notification(self, alert: Alert, config: NotificationConfig):
        """Send Slack notification"""
        # Implementation would depend on Slack API
        logger.info(f"Slack notification sent for alert: {alert.alert_id}")
    
    async def _send_webhook_notification(self, alert: Alert, config: NotificationConfig):
        """Send webhook notification"""
        # Implementation would depend on webhook service
        logger.info(f"Webhook notification sent for alert: {alert.alert_id}")
    
    async def _send_sms_notification(self, alert: Alert, config: NotificationConfig):
        """Send SMS notification"""
        # Implementation would depend on SMS service
        logger.info(f"SMS notification sent for alert: {alert.alert_id}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.updated_at = datetime.now()
            logger.info(f"Alert acknowledged: {alert_id}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            alert.updated_at = datetime.now()
            logger.info(f"Alert resolved: {alert_id}")
    
    def suppress_alert(self, alert_id: str):
        """Suppress alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            alert.updated_at = datetime.now()
            logger.info(f"Alert suppressed: {alert_id}")
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts"""
        alerts = list(self.active_alerts.values())
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history"""
        return self.alert_history[-limit:] if limit else self.alert_history
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)
        
        severity_counts = {}
        for alert in self.alert_history:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        status_counts = {}
        for alert in self.active_alerts.values():
            status = alert.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "severity_counts": severity_counts,
            "status_counts": status_counts,
            "rules_count": len(self.rules)
        }
    
    def cleanup_old_alerts(self, days: int = 30):
        """Clean up old alerts"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            old_alerts = [a for a in self.alert_history if a.created_at < cutoff_date]
            
            for alert in old_alerts:
                if alert.alert_id in self.active_alerts:
                    del self.active_alerts[alert.alert_id]
                self.alert_history.remove(alert)
            
            logger.info(f"Cleaned up {len(old_alerts)} old alerts")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old alerts: {e}")
    
    async def start_cleanup_task(self, interval: int = 3600):
        """Start periodic cleanup task"""
        try:
            while True:
                self.cleanup_old_alerts()
                await asyncio.sleep(interval)
        except Exception as e:
            logger.error(f"Cleanup task failed: {e}")
