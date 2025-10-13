"""
SIEM Integration Module
Integrates with Splunk, Elastic Security, and Azure Sentinel for security event management.
"""

from .splunk_integration import SplunkIntegration
from .elastic_integration import ElasticIntegration
from .azure_sentinel_integration import AzureSentinelIntegration
from .siem_coordinator import SIEMCoordinator

__all__ = [
    'SplunkIntegration',
    'ElasticIntegration',
    'AzureSentinelIntegration',
    'SIEMCoordinator'
]
