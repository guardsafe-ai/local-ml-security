"""
Splunk Integration
Integrates with Splunk for security event management and analysis.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import requests
import base64

logger = logging.getLogger(__name__)

class SplunkEventType(Enum):
    """Splunk event types"""
    SECURITY_ALERT = "security_alert"
    VULNERABILITY = "vulnerability"
    ATTACK_EVENT = "attack_event"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SYSTEM_EVENT = "system_event"
    CUSTOM = "custom"

class SplunkSeverity(Enum):
    """Splunk severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class SplunkEvent:
    """Splunk event data"""
    event_type: SplunkEventType
    severity: SplunkSeverity
    source: str
    sourcetype: str
    index: str
    host: str
    timestamp: datetime
    data: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    fields: Dict[str, str] = field(default_factory=dict)

@dataclass
class SplunkConfig:
    """Splunk configuration"""
    host: str
    port: int = 8089
    username: str = ""
    password: str = ""
    token: str = ""
    verify_ssl: bool = True
    timeout: int = 30
    index: str = "main"
    sourcetype: str = "redteam"
    source: str = "redteam_service"

class SplunkIntegration:
    """Integrates with Splunk for security event management"""
    
    def __init__(self, config: SplunkConfig):
        self.config = config
        self.session = requests.Session()
        self.session.verify = config.verify_ssl
        self.session.timeout = config.timeout
        
        # Setup authentication
        self._setup_authentication()
    
    def _setup_authentication(self):
        """Setup Splunk authentication"""
        try:
            if self.config.token:
                # Use token authentication
                self.session.headers.update({
                    'Authorization': f'Bearer {self.config.token}'
                })
            elif self.config.username and self.config.password:
                # Use basic authentication
                auth_string = f"{self.config.username}:{self.config.password}"
                auth_bytes = auth_string.encode('ascii')
                auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
                self.session.headers.update({
                    'Authorization': f'Basic {auth_b64}'
                })
            else:
                logger.warning("No authentication method configured for Splunk")
                
        except Exception as e:
            logger.error(f"Failed to setup Splunk authentication: {e}")
    
    def send_event(self, event: SplunkEvent) -> bool:
        """Send event to Splunk"""
        try:
            # Prepare event data
            event_data = self._prepare_event_data(event)
            
            # Send to Splunk
            url = f"https://{self.config.host}:{self.config.port}/services/collector/event"
            
            response = self.session.post(url, json=event_data)
            response.raise_for_status()
            
            logger.info(f"Successfully sent event to Splunk: {event.event_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send event to Splunk: {e}")
            return False
    
    def send_events_batch(self, events: List[SplunkEvent]) -> Dict[str, Any]:
        """Send multiple events to Splunk in batch"""
        try:
            results = {
                "total_events": len(events),
                "successful": 0,
                "failed": 0,
                "errors": []
            }
            
            # Prepare batch data
            batch_data = []
            for event in events:
                try:
                    event_data = self._prepare_event_data(event)
                    batch_data.append(event_data)
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"Failed to prepare event: {e}")
            
            if not batch_data:
                return results
            
            # Send batch to Splunk
            url = f"https://{self.config.host}:{self.config.port}/services/collector/event"
            
            response = self.session.post(url, json=batch_data)
            response.raise_for_status()
            
            results["successful"] = len(batch_data)
            logger.info(f"Successfully sent {len(batch_data)} events to Splunk")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to send events batch to Splunk: {e}")
            return {"total_events": len(events), "successful": 0, "failed": len(events), "errors": [str(e)]}
    
    def _prepare_event_data(self, event: SplunkEvent) -> Dict[str, Any]:
        """Prepare event data for Splunk"""
        try:
            # Convert timestamp to Unix timestamp
            timestamp = int(event.timestamp.timestamp())
            
            # Prepare event payload
            event_payload = {
                "time": timestamp,
                "source": event.source,
                "sourcetype": event.sourcetype,
                "index": event.index,
                "host": event.host,
                "event": event.data
            }
            
            # Add tags if present
            if event.tags:
                event_payload["tags"] = event.tags
            
            # Add fields if present
            if event.fields:
                event_payload["fields"] = event.fields
            
            return event_payload
            
        except Exception as e:
            logger.error(f"Failed to prepare event data: {e}")
            return {}
    
    def create_security_alert(self, alert_data: Dict[str, Any]) -> SplunkEvent:
        """Create security alert event"""
        try:
            return SplunkEvent(
                event_type=SplunkEventType.SECURITY_ALERT,
                severity=SplunkSeverity(alert_data.get("severity", "medium")),
                source=self.config.source,
                sourcetype=self.config.sourcetype,
                index=self.config.index,
                host=alert_data.get("host", "unknown"),
                timestamp=datetime.now(),
                data=alert_data,
                tags=["security", "alert"],
                fields={
                    "alert_id": alert_data.get("alert_id", ""),
                    "rule_name": alert_data.get("rule_name", ""),
                    "threat_level": alert_data.get("threat_level", "medium")
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create security alert: {e}")
            return None
    
    def create_vulnerability_event(self, vuln_data: Dict[str, Any]) -> SplunkEvent:
        """Create vulnerability event"""
        try:
            return SplunkEvent(
                event_type=SplunkEventType.VULNERABILITY,
                severity=SplunkSeverity(vuln_data.get("severity", "medium")),
                source=self.config.source,
                sourcetype=self.config.sourcetype,
                index=self.config.index,
                host=vuln_data.get("host", "unknown"),
                timestamp=datetime.now(),
                data=vuln_data,
                tags=["vulnerability", "security"],
                fields={
                    "vuln_id": vuln_data.get("vuln_id", ""),
                    "cve_id": vuln_data.get("cve_id", ""),
                    "cvss_score": str(vuln_data.get("cvss_score", 0)),
                    "vuln_type": vuln_data.get("vuln_type", "")
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create vulnerability event: {e}")
            return None
    
    def create_attack_event(self, attack_data: Dict[str, Any]) -> SplunkEvent:
        """Create attack event"""
        try:
            return SplunkEvent(
                event_type=SplunkEventType.ATTACK_EVENT,
                severity=SplunkSeverity(attack_data.get("severity", "medium")),
                source=self.config.source,
                sourcetype=self.config.sourcetype,
                index=self.config.index,
                host=attack_data.get("host", "unknown"),
                timestamp=datetime.now(),
                data=attack_data,
                tags=["attack", "security"],
                fields={
                    "attack_id": attack_data.get("attack_id", ""),
                    "attack_type": attack_data.get("attack_type", ""),
                    "success": str(attack_data.get("success", False)),
                    "model_name": attack_data.get("model_name", "")
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create attack event: {e}")
            return None
    
    def create_compliance_violation_event(self, compliance_data: Dict[str, Any]) -> SplunkEvent:
        """Create compliance violation event"""
        try:
            return SplunkEvent(
                event_type=SplunkEventType.COMPLIANCE_VIOLATION,
                severity=SplunkSeverity(compliance_data.get("severity", "medium")),
                source=self.config.source,
                sourcetype=self.config.sourcetype,
                index=self.config.index,
                host=compliance_data.get("host", "unknown"),
                timestamp=datetime.now(),
                data=compliance_data,
                tags=["compliance", "violation"],
                fields={
                    "framework": compliance_data.get("framework", ""),
                    "control": compliance_data.get("control", ""),
                    "violation_type": compliance_data.get("violation_type", ""),
                    "remediation": compliance_data.get("remediation", "")
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create compliance violation event: {e}")
            return None
    
    def create_system_event(self, system_data: Dict[str, Any]) -> SplunkEvent:
        """Create system event"""
        try:
            return SplunkEvent(
                event_type=SplunkEventType.SYSTEM_EVENT,
                severity=SplunkSeverity(system_data.get("severity", "info")),
                source=self.config.source,
                sourcetype=self.config.sourcetype,
                index=self.config.index,
                host=system_data.get("host", "unknown"),
                timestamp=datetime.now(),
                data=system_data,
                tags=["system", "monitoring"],
                fields={
                    "component": system_data.get("component", ""),
                    "status": system_data.get("status", ""),
                    "message": system_data.get("message", "")
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create system event: {e}")
            return None
    
    def search_events(self, query: str, start_time: Optional[datetime] = None, 
                     end_time: Optional[datetime] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Search events in Splunk"""
        try:
            # Prepare search query
            search_query = self._prepare_search_query(query, start_time, end_time)
            
            # Execute search
            url = f"https://{self.config.host}:{self.config.port}/services/search/jobs"
            
            search_data = {
                "search": search_query,
                "output_mode": "json",
                "count": limit
            }
            
            response = self.session.post(url, data=search_data)
            response.raise_for_status()
            
            # Parse results
            results = response.json()
            events = results.get("results", [])
            
            logger.info(f"Found {len(events)} events in Splunk")
            return events
            
        except Exception as e:
            logger.error(f"Failed to search events in Splunk: {e}")
            return []
    
    def _prepare_search_query(self, query: str, start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None) -> str:
        """Prepare search query for Splunk"""
        try:
            # Add time range if specified
            if start_time and end_time:
                start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
                end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
                time_range = f"earliest={start_str} latest={end_str}"
                return f"{query} | {time_range}"
            elif start_time:
                start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
                time_range = f"earliest={start_str}"
                return f"{query} | {time_range}"
            else:
                return query
                
        except Exception as e:
            logger.error(f"Failed to prepare search query: {e}")
            return query
    
    def get_dashboard_data(self, dashboard_query: str) -> Dict[str, Any]:
        """Get dashboard data from Splunk"""
        try:
            # Execute dashboard query
            events = self.search_events(dashboard_query, limit=1000)
            
            # Process events for dashboard
            dashboard_data = {
                "total_events": len(events),
                "events_by_type": {},
                "events_by_severity": {},
                "events_by_host": {},
                "timeline": [],
                "top_events": []
            }
            
            # Group events by type
            for event in events:
                event_type = event.get("event_type", "unknown")
                dashboard_data["events_by_type"][event_type] = dashboard_data["events_by_type"].get(event_type, 0) + 1
                
                severity = event.get("severity", "info")
                dashboard_data["events_by_severity"][severity] = dashboard_data["events_by_severity"].get(severity, 0) + 1
                
                host = event.get("host", "unknown")
                dashboard_data["events_by_host"][host] = dashboard_data["events_by_host"].get(host, 0) + 1
            
            # Create timeline (last 24 hours)
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            
            timeline_events = self.search_events(
                f"index={self.config.index} sourcetype={self.config.sourcetype}",
                start_time=start_time,
                end_time=end_time,
                limit=100
            )
            
            dashboard_data["timeline"] = timeline_events
            
            # Get top events
            dashboard_data["top_events"] = events[:10]
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {}
    
    def create_alert_rule(self, rule_data: Dict[str, Any]) -> bool:
        """Create alert rule in Splunk"""
        try:
            # This would typically involve creating a saved search or alert in Splunk
            # For now, we'll just log the rule creation
            logger.info(f"Creating alert rule: {rule_data.get('name', 'Unknown')}")
            
            # In a real implementation, this would:
            # 1. Create a saved search in Splunk
            # 2. Configure alert conditions
            # 3. Set up notification actions
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create alert rule: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test connection to Splunk"""
        try:
            url = f"https://{self.config.host}:{self.config.port}/services/server/info"
            response = self.session.get(url)
            response.raise_for_status()
            
            logger.info("Successfully connected to Splunk")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Splunk: {e}")
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information"""
        try:
            return {
                "host": self.config.host,
                "port": self.config.port,
                "index": self.config.index,
                "sourcetype": self.config.sourcetype,
                "source": self.config.source,
                "connected": self.test_connection()
            }
            
        except Exception as e:
            logger.error(f"Failed to get connection info: {e}")
            return {}
    
    def export_events(self, query: str, format_type: str = "json") -> str:
        """Export events from Splunk"""
        try:
            events = self.search_events(query, limit=1000)
            
            if format_type == "json":
                return json.dumps(events, indent=2)
            else:
                return str(events)
                
        except Exception as e:
            logger.error(f"Failed to export events: {e}")
            return "[]"
