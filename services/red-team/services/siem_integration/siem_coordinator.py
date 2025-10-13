"""
SIEM Coordinator
Coordinates all SIEM integrations for comprehensive security event management.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from .splunk_integration import SplunkIntegration, SplunkConfig, SplunkEvent
from .elastic_integration import ElasticIntegration, ElasticConfig, ElasticEvent
from .azure_sentinel_integration import AzureSentinelIntegration, AzureSentinelConfig, AzureSentinelEvent

logger = logging.getLogger(__name__)

class SIEMProvider(Enum):
    """SIEM providers"""
    SPLUNK = "splunk"
    ELASTIC = "elastic"
    AZURE_SENTINEL = "azure_sentinel"
    ALL = "all"

class EventPriority(Enum):
    """Event priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class SIEMConfig:
    """SIEM configuration"""
    providers: List[SIEMProvider] = field(default_factory=lambda: [SIEMProvider.SPLUNK])
    splunk_config: Optional[SplunkConfig] = None
    elastic_config: Optional[ElasticConfig] = None
    azure_sentinel_config: Optional[AzureSentinelConfig] = None
    default_priority: EventPriority = EventPriority.MEDIUM
    enable_duplicate_detection: bool = True
    duplicate_window_minutes: int = 5

@dataclass
class SecurityEvent:
    """Unified security event"""
    event_id: str
    event_type: str
    severity: str
    priority: EventPriority
    source: str
    timestamp: datetime
    data: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class SIEMCoordinator:
    """Coordinates all SIEM integrations"""
    
    def __init__(self, config: SIEMConfig):
        self.config = config
        self.integrations: Dict[SIEMProvider, Any] = {}
        self.event_cache: Dict[str, datetime] = {}
        
        # Initialize integrations
        self._initialize_integrations()
    
    def _initialize_integrations(self):
        """Initialize SIEM integrations"""
        try:
            if SIEMProvider.SPLUNK in self.config.providers and self.config.splunk_config:
                self.integrations[SIEMProvider.SPLUNK] = SplunkIntegration(self.config.splunk_config)
                logger.info("Splunk integration initialized")
            
            if SIEMProvider.ELASTIC in self.config.providers and self.config.elastic_config:
                self.integrations[SIEMProvider.ELASTIC] = ElasticIntegration(self.config.elastic_config)
                logger.info("Elastic integration initialized")
            
            if SIEMProvider.AZURE_SENTINEL in self.config.providers and self.config.azure_sentinel_config:
                self.integrations[SIEMProvider.AZURE_SENTINEL] = AzureSentinelIntegration(self.config.azure_sentinel_config)
                logger.info("Azure Sentinel integration initialized")
            
            logger.info(f"Initialized {len(self.integrations)} SIEM integrations")
            
        except Exception as e:
            logger.error(f"Failed to initialize SIEM integrations: {e}")
    
    def send_event(self, event: SecurityEvent, providers: List[SIEMProvider] = None) -> Dict[str, bool]:
        """Send event to specified SIEM providers"""
        try:
            if providers is None:
                providers = self.config.providers
            
            # Check for duplicates if enabled
            if self.config.enable_duplicate_detection:
                if self._is_duplicate_event(event):
                    logger.info(f"Duplicate event detected, skipping: {event.event_id}")
                    return {provider.value: False for provider in providers}
            
            results = {}
            
            for provider in providers:
                if provider in self.integrations:
                    try:
                        success = self._send_event_to_provider(event, provider)
                        results[provider.value] = success
                        
                        if success:
                            logger.info(f"Successfully sent event to {provider.value}")
                        else:
                            logger.error(f"Failed to send event to {provider.value}")
                            
                    except Exception as e:
                        logger.error(f"Error sending event to {provider.value}: {e}")
                        results[provider.value] = False
                else:
                    logger.warning(f"Provider {provider.value} not configured")
                    results[provider.value] = False
            
            # Cache event if duplicate detection is enabled
            if self.config.enable_duplicate_detection:
                self.event_cache[event.event_id] = event.timestamp
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to send event: {e}")
            return {provider.value: False for provider in providers}
    
    def send_events_batch(self, events: List[SecurityEvent], 
                         providers: List[SIEMProvider] = None) -> Dict[str, Dict[str, Any]]:
        """Send multiple events to SIEM providers"""
        try:
            if providers is None:
                providers = self.config.providers
            
            results = {}
            
            for provider in providers:
                if provider in self.integrations:
                    try:
                        provider_events = self._convert_events_for_provider(events, provider)
                        result = self._send_events_batch_to_provider(provider_events, provider)
                        results[provider.value] = result
                        
                    except Exception as e:
                        logger.error(f"Error sending events batch to {provider.value}: {e}")
                        results[provider.value] = {"successful": 0, "failed": len(events), "error": str(e)}
                else:
                    logger.warning(f"Provider {provider.value} not configured")
                    results[provider.value] = {"successful": 0, "failed": len(events), "error": "Provider not configured"}
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to send events batch: {e}")
            return {}
    
    def _send_event_to_provider(self, event: SecurityEvent, provider: SIEMProvider) -> bool:
        """Send event to specific provider"""
        try:
            if provider == SIEMProvider.SPLUNK:
                splunk_event = self._convert_to_splunk_event(event)
                return self.integrations[provider].send_event(splunk_event)
            elif provider == SIEMProvider.ELASTIC:
                elastic_event = self._convert_to_elastic_event(event)
                return self.integrations[provider].send_event(elastic_event)
            elif provider == SIEMProvider.AZURE_SENTINEL:
                azure_event = self._convert_to_azure_sentinel_event(event)
                return self.integrations[provider].send_event(azure_event)
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to send event to {provider.value}: {e}")
            return False
    
    def _send_events_batch_to_provider(self, events: List[Any], provider: SIEMProvider) -> Dict[str, Any]:
        """Send events batch to specific provider"""
        try:
            if provider == SIEMProvider.SPLUNK:
                return self.integrations[provider].send_events_batch(events)
            elif provider == SIEMProvider.ELASTIC:
                return self.integrations[provider].send_events_batch(events)
            elif provider == SIEMProvider.AZURE_SENTINEL:
                return self.integrations[provider].send_events_batch(events)
            else:
                return {"successful": 0, "failed": len(events), "error": "Unknown provider"}
                
        except Exception as e:
            logger.error(f"Failed to send events batch to {provider.value}: {e}")
            return {"successful": 0, "failed": len(events), "error": str(e)}
    
    def _convert_to_splunk_event(self, event: SecurityEvent) -> SplunkEvent:
        """Convert SecurityEvent to SplunkEvent"""
        try:
            from .splunk_integration import SplunkEvent, SplunkEventType, SplunkSeverity
            
            return SplunkEvent(
                event_type=SplunkEventType(event.event_type),
                severity=SplunkSeverity(event.severity),
                source=event.source,
                sourcetype="redteam",
                index="main",
                host=event.data.get("host", "unknown"),
                timestamp=event.timestamp,
                data=event.data,
                tags=event.tags,
                fields=event.metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to convert to Splunk event: {e}")
            return None
    
    def _convert_to_elastic_event(self, event: SecurityEvent) -> ElasticEvent:
        """Convert SecurityEvent to ElasticEvent"""
        try:
            from .elastic_integration import ElasticEvent, ElasticEventType, ElasticSeverity
            
            return ElasticEvent(
                event_type=ElasticEventType(event.event_type),
                severity=ElasticSeverity(event.severity),
                source=event.source,
                index="redteam-events",
                timestamp=event.timestamp,
                data=event.data,
                tags=event.tags,
                metadata=event.metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to convert to Elastic event: {e}")
            return None
    
    def _convert_to_azure_sentinel_event(self, event: SecurityEvent) -> AzureSentinelEvent:
        """Convert SecurityEvent to AzureSentinelEvent"""
        try:
            from .azure_sentinel_integration import AzureSentinelEvent, AzureSentinelEventType, AzureSentinelSeverity
            
            return AzureSentinelEvent(
                event_type=AzureSentinelEventType(event.event_type),
                severity=AzureSentinelSeverity(event.severity),
                source=event.source,
                workspace_id="",
                timestamp=event.timestamp,
                data=event.data,
                tags=event.tags,
                metadata=event.metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to convert to Azure Sentinel event: {e}")
            return None
    
    def _convert_events_for_provider(self, events: List[SecurityEvent], 
                                   provider: SIEMProvider) -> List[Any]:
        """Convert events for specific provider"""
        try:
            converted_events = []
            
            for event in events:
                if provider == SIEMProvider.SPLUNK:
                    converted_event = self._convert_to_splunk_event(event)
                elif provider == SIEMProvider.ELASTIC:
                    converted_event = self._convert_to_elastic_event(event)
                elif provider == SIEMProvider.AZURE_SENTINEL:
                    converted_event = self._convert_to_azure_sentinel_event(event)
                else:
                    continue
                
                if converted_event:
                    converted_events.append(converted_event)
            
            return converted_events
            
        except Exception as e:
            logger.error(f"Failed to convert events for provider {provider.value}: {e}")
            return []
    
    def _is_duplicate_event(self, event: SecurityEvent) -> bool:
        """Check if event is duplicate"""
        try:
            if event.event_id in self.event_cache:
                cached_time = self.event_cache[event.event_id]
                time_diff = (event.timestamp - cached_time).total_seconds() / 60
                
                if time_diff <= self.config.duplicate_window_minutes:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check duplicate event: {e}")
            return False
    
    def create_security_alert(self, alert_data: Dict[str, Any]) -> SecurityEvent:
        """Create security alert event"""
        try:
            return SecurityEvent(
                event_id=alert_data.get("alert_id", f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                event_type="security_alert",
                severity=alert_data.get("severity", "medium"),
                priority=EventPriority(alert_data.get("priority", "medium")),
                source=alert_data.get("source", "redteam"),
                timestamp=datetime.now(),
                data=alert_data,
                tags=["security", "alert"],
                metadata={
                    "rule_name": alert_data.get("rule_name", ""),
                    "threat_level": alert_data.get("threat_level", "medium")
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create security alert: {e}")
            return None
    
    def create_vulnerability_event(self, vuln_data: Dict[str, Any]) -> SecurityEvent:
        """Create vulnerability event"""
        try:
            return SecurityEvent(
                event_id=vuln_data.get("vuln_id", f"vuln_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                event_type="vulnerability",
                severity=vuln_data.get("severity", "medium"),
                priority=EventPriority(vuln_data.get("priority", "medium")),
                source=vuln_data.get("source", "redteam"),
                timestamp=datetime.now(),
                data=vuln_data,
                tags=["vulnerability", "security"],
                metadata={
                    "cve_id": vuln_data.get("cve_id", ""),
                    "cvss_score": vuln_data.get("cvss_score", 0),
                    "vuln_type": vuln_data.get("vuln_type", "")
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create vulnerability event: {e}")
            return None
    
    def create_attack_event(self, attack_data: Dict[str, Any]) -> SecurityEvent:
        """Create attack event"""
        try:
            return SecurityEvent(
                event_id=attack_data.get("attack_id", f"attack_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                event_type="attack_event",
                severity=attack_data.get("severity", "medium"),
                priority=EventPriority(attack_data.get("priority", "medium")),
                source=attack_data.get("source", "redteam"),
                timestamp=datetime.now(),
                data=attack_data,
                tags=["attack", "security"],
                metadata={
                    "attack_type": attack_data.get("attack_type", ""),
                    "success": attack_data.get("success", False),
                    "model_name": attack_data.get("model_name", "")
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create attack event: {e}")
            return None
    
    def test_connections(self) -> Dict[str, bool]:
        """Test connections to all configured SIEM providers"""
        try:
            results = {}
            
            for provider, integration in self.integrations.items():
                try:
                    if hasattr(integration, 'test_connection'):
                        success = integration.test_connection()
                        results[provider.value] = success
                    else:
                        results[provider.value] = False
                        
                except Exception as e:
                    logger.error(f"Failed to test connection to {provider.value}: {e}")
                    results[provider.value] = False
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to test connections: {e}")
            return {}
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrations"""
        try:
            status = {
                "total_integrations": len(self.integrations),
                "integrations": {},
                "connection_tests": self.test_connections()
            }
            
            for provider, integration in self.integrations.items():
                try:
                    if hasattr(integration, 'get_connection_info'):
                        info = integration.get_connection_info()
                        status["integrations"][provider.value] = info
                    else:
                        status["integrations"][provider.value] = {"status": "unknown"}
                        
                except Exception as e:
                    logger.error(f"Failed to get status for {provider.value}: {e}")
                    status["integrations"][provider.value] = {"status": "error", "error": str(e)}
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get integration status: {e}")
            return {}
    
    def search_events(self, query: str, providers: List[SIEMProvider] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Search events across SIEM providers"""
        try:
            if providers is None:
                providers = self.config.providers
            
            results = {}
            
            for provider in providers:
                if provider in self.integrations:
                    try:
                        if hasattr(self.integrations[provider], 'search_events'):
                            events = self.integrations[provider].search_events(query)
                            results[provider.value] = events
                        else:
                            results[provider.value] = []
                            
                    except Exception as e:
                        logger.error(f"Failed to search events in {provider.value}: {e}")
                        results[provider.value] = []
                else:
                    results[provider.value] = []
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search events: {e}")
            return {}
    
    def get_dashboard_data(self, providers: List[SIEMProvider] = None) -> Dict[str, Any]:
        """Get dashboard data from SIEM providers"""
        try:
            if providers is None:
                providers = self.config.providers
            
            dashboard_data = {
                "providers": {},
                "aggregated": {
                    "total_events": 0,
                    "events_by_type": {},
                    "events_by_severity": {},
                    "events_by_source": {}
                }
            }
            
            for provider in providers:
                if provider in self.integrations:
                    try:
                        if hasattr(self.integrations[provider], 'get_dashboard_data'):
                            data = self.integrations[provider].get_dashboard_data()
                            dashboard_data["providers"][provider.value] = data
                            
                            # Aggregate data
                            dashboard_data["aggregated"]["total_events"] += data.get("total_events", 0)
                            
                            # Aggregate by type
                            for event_type, count in data.get("events_by_type", {}).items():
                                dashboard_data["aggregated"]["events_by_type"][event_type] = \
                                    dashboard_data["aggregated"]["events_by_type"].get(event_type, 0) + count
                            
                            # Aggregate by severity
                            for severity, count in data.get("events_by_severity", {}).items():
                                dashboard_data["aggregated"]["events_by_severity"][severity] = \
                                    dashboard_data["aggregated"]["events_by_severity"].get(severity, 0) + count
                            
                            # Aggregate by source
                            for source, count in data.get("events_by_source", {}).items():
                                dashboard_data["aggregated"]["events_by_source"][source] = \
                                    dashboard_data["aggregated"]["events_by_source"].get(source, 0) + count
                                
                    except Exception as e:
                        logger.error(f"Failed to get dashboard data from {provider.value}: {e}")
                        dashboard_data["providers"][provider.value] = {"error": str(e)}
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {}
    
    def export_events(self, query: str, format_type: str = "json", 
                     providers: List[SIEMProvider] = None) -> Dict[str, str]:
        """Export events from SIEM providers"""
        try:
            if providers is None:
                providers = self.config.providers
            
            results = {}
            
            for provider in providers:
                if provider in self.integrations:
                    try:
                        if hasattr(self.integrations[provider], 'export_events'):
                            data = self.integrations[provider].export_events(query, format_type)
                            results[provider.value] = data
                        else:
                            results[provider.value] = "[]"
                            
                    except Exception as e:
                        logger.error(f"Failed to export events from {provider.value}: {e}")
                        results[provider.value] = "[]"
                else:
                    results[provider.value] = "[]"
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to export events: {e}")
            return {}
    
    def clear_event_cache(self):
        """Clear event cache"""
        self.event_cache.clear()
        logger.info("Event cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            return {
                "cached_events": len(self.event_cache),
                "duplicate_detection_enabled": self.config.enable_duplicate_detection,
                "duplicate_window_minutes": self.config.duplicate_window_minutes
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
