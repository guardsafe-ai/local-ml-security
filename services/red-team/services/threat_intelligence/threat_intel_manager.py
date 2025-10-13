"""
Threat Intelligence Manager
Centralized management of all threat intelligence sources
"""

import asyncio
import logging
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import numpy as np

from .mitre_atlas_scraper import MITREATLASScraper, ATLASEntry
from .cve_scraper import CVEScraper, CVEEntry
from .jailbreak_scraper import JailbreakScraper, JailbreakEntry

logger = logging.getLogger(__name__)


class ThreatSource(Enum):
    """Threat intelligence sources"""
    MITRE_ATLAS = "mitre_atlas"
    CVE = "cve"
    JAILBREAK = "jailbreak"


@dataclass
class ThreatIntelligenceReport:
    """Threat intelligence report"""
    report_id: str
    title: str
    summary: str
    threat_level: str
    affected_systems: List[str]
    attack_vectors: List[str]
    mitigations: List[str]
    references: List[str]
    created_date: str
    updated_date: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ThreatIntelligenceManager:
    """
    Threat Intelligence Manager
    Centralized management of all threat intelligence sources
    """
    
    def __init__(self):
        """Initialize threat intelligence manager"""
        self.mitre_atlas_scraper = MITREATLASScraper()
        self.cve_scraper = CVEScraper()
        self.jailbreak_scraper = JailbreakScraper()
        
        self.atlas_entries: Dict[str, ATLASEntry] = {}
        self.cve_entries: Dict[str, CVEEntry] = {}
        self.jailbreak_entries: Dict[str, JailbreakEntry] = {}
        
        self.threat_reports: Dict[str, ThreatIntelligenceReport] = {}
        
        logger.info("✅ Initialized Threat Intelligence Manager")
    
    async def collect_all_intelligence(self) -> Dict[str, Any]:
        """
        Collect threat intelligence from all sources
        """
        try:
            logger.info("Starting comprehensive threat intelligence collection")
            
            results = {}
            
            # Collect MITRE ATLAS data
            logger.info("Collecting MITRE ATLAS data")
            async with self.mitre_atlas_scraper:
                atlas_entries = await self.mitre_atlas_scraper.scrape_all_techniques()
                self.atlas_entries = {entry.technique_id: entry for entry in atlas_entries}
                results["mitre_atlas"] = {
                    "count": len(atlas_entries),
                    "entries": atlas_entries
                }
            
            # Collect CVE data
            logger.info("Collecting CVE data")
            async with self.cve_scraper:
                cve_entries = await self.cve_scraper.scrape_ai_ml_cves()
                self.cve_entries = {entry.cve_id: entry for entry in cve_entries}
                results["cve"] = {
                    "count": len(cve_entries),
                    "entries": cve_entries
                }
            
            # Collect jailbreak data
            logger.info("Collecting jailbreak data")
            async with self.jailbreak_scraper:
                jailbreak_entries = await self.jailbreak_scraper.scrape_all_jailbreaks()
                self.jailbreak_entries = {entry.jailbreak_id: entry for entry in jailbreak_entries}
                results["jailbreaks"] = {
                    "count": len(jailbreak_entries),
                    "entries": jailbreak_entries
                }
            
            # Generate threat intelligence reports
            logger.info("Generating threat intelligence reports")
            reports = await self._generate_threat_reports()
            results["reports"] = {
                "count": len(reports),
                "reports": reports
            }
            
            logger.info("Threat intelligence collection completed")
            return results
            
        except Exception as e:
            logger.error(f"Threat intelligence collection failed: {e}")
            return {}
    
    async def _generate_threat_reports(self) -> List[ThreatIntelligenceReport]:
        """Generate threat intelligence reports"""
        try:
            reports = []
            
            # Generate reports for different threat categories
            categories = [
                "prompt_injection",
                "adversarial_attacks",
                "data_poisoning",
                "model_extraction",
                "privacy_attacks",
                "bias_attacks",
                "robustness_attacks"
            ]
            
            for category in categories:
                report = await self._generate_category_report(category)
                if report:
                    reports.append(report)
                    self.threat_reports[report.report_id] = report
            
            return reports
            
        except Exception as e:
            logger.error(f"Threat report generation failed: {e}")
            return []
    
    async def _generate_category_report(self, category: str) -> Optional[ThreatIntelligenceReport]:
        """Generate threat report for specific category"""
        try:
            # Collect relevant entries from all sources
            atlas_entries = [entry for entry in self.atlas_entries.values() 
                           if category in entry.category.lower() or category in entry.name.lower()]
            
            cve_entries = [entry for entry in self.cve_entries.values() 
                          if category in entry.description.lower() or 
                          any(category in product.lower() for product in entry.products)]
            
            jailbreak_entries = [entry for entry in self.jailbreak_entries.values() 
                               if category in entry.jailbreak_type.value or 
                               any(category in tag.lower() for tag in entry.tags)]
            
            if not atlas_entries and not cve_entries and not jailbreak_entries:
                return None
            
            # Generate report
            report_id = f"threat_report_{category}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Determine threat level
            threat_level = self._calculate_threat_level(atlas_entries, cve_entries, jailbreak_entries)
            
            # Extract attack vectors
            attack_vectors = self._extract_attack_vectors(atlas_entries, cve_entries, jailbreak_entries)
            
            # Extract mitigations
            mitigations = self._extract_mitigations(atlas_entries, cve_entries, jailbreak_entries)
            
            # Extract affected systems
            affected_systems = self._extract_affected_systems(atlas_entries, cve_entries, jailbreak_entries)
            
            # Extract references
            references = self._extract_references(atlas_entries, cve_entries, jailbreak_entries)
            
            # Generate summary
            summary = self._generate_summary(category, atlas_entries, cve_entries, jailbreak_entries)
            
            return ThreatIntelligenceReport(
                report_id=report_id,
                title=f"Threat Intelligence Report: {category.replace('_', ' ').title()}",
                summary=summary,
                threat_level=threat_level,
                affected_systems=affected_systems,
                attack_vectors=attack_vectors,
                mitigations=mitigations,
                references=references,
                created_date=datetime.utcnow().isoformat(),
                updated_date=datetime.utcnow().isoformat(),
                metadata={
                    "category": category,
                    "atlas_count": len(atlas_entries),
                    "cve_count": len(cve_entries),
                    "jailbreak_count": len(jailbreak_entries),
                    "total_entries": len(atlas_entries) + len(cve_entries) + len(jailbreak_entries)
                }
            )
            
        except Exception as e:
            logger.error(f"Category report generation failed for {category}: {e}")
            return None
    
    def _calculate_threat_level(self, atlas_entries: List[ATLASEntry], 
                               cve_entries: List[CVEEntry], 
                               jailbreak_entries: List[JailbreakEntry]) -> str:
        """Calculate overall threat level"""
        try:
            threat_scores = []
            
            # Calculate scores from different sources
            for entry in atlas_entries:
                if entry.severity == "critical":
                    threat_scores.append(4)
                elif entry.severity == "high":
                    threat_scores.append(3)
                elif entry.severity == "medium":
                    threat_scores.append(2)
                else:
                    threat_scores.append(1)
            
            for entry in cve_entries:
                if entry.cvss_score >= 9.0:
                    threat_scores.append(4)
                elif entry.cvss_score >= 7.0:
                    threat_scores.append(3)
                elif entry.cvss_score >= 4.0:
                    threat_scores.append(2)
                else:
                    threat_scores.append(1)
            
            for entry in jailbreak_entries:
                if entry.success_rate >= 0.8:
                    threat_scores.append(4)
                elif entry.success_rate >= 0.6:
                    threat_scores.append(3)
                elif entry.success_rate >= 0.4:
                    threat_scores.append(2)
                else:
                    threat_scores.append(1)
            
            if not threat_scores:
                return "low"
            
            avg_score = np.mean(threat_scores)
            
            if avg_score >= 3.5:
                return "critical"
            elif avg_score >= 2.5:
                return "high"
            elif avg_score >= 1.5:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            logger.error(f"Threat level calculation failed: {e}")
            return "unknown"
    
    def _extract_attack_vectors(self, atlas_entries: List[ATLASEntry], 
                               cve_entries: List[CVEEntry], 
                               jailbreak_entries: List[JailbreakEntry]) -> List[str]:
        """Extract attack vectors from all sources"""
        try:
            attack_vectors = set()
            
            # From ATLAS entries
            for entry in atlas_entries:
                attack_vectors.update(entry.techniques)
                attack_vectors.update(entry.tactics)
            
            # From CVE entries
            for entry in cve_entries:
                if entry.attack_vector:
                    attack_vectors.add(entry.attack_vector)
                if entry.cwe_id:
                    attack_vectors.add(f"CWE-{entry.cwe_id}")
            
            # From jailbreak entries
            for entry in jailbreak_entries:
                attack_vectors.add(entry.jailbreak_type.value)
                attack_vectors.update(entry.tags)
            
            return list(attack_vectors)
            
        except Exception as e:
            logger.error(f"Attack vector extraction failed: {e}")
            return []
    
    def _extract_mitigations(self, atlas_entries: List[ATLASEntry], 
                            cve_entries: List[CVEEntry], 
                            jailbreak_entries: List[JailbreakEntry]) -> List[str]:
        """Extract mitigations from all sources"""
        try:
            mitigations = set()
            
            # From ATLAS entries
            for entry in atlas_entries:
                mitigations.update(entry.mitigations)
            
            # From CVE entries (references might contain mitigations)
            for entry in cve_entries:
                for ref in entry.references:
                    if "mitigation" in ref.lower() or "fix" in ref.lower():
                        mitigations.add(ref)
            
            # From jailbreak entries (references might contain mitigations)
            for entry in jailbreak_entries:
                for ref in entry.references:
                    if "mitigation" in ref.lower() or "defense" in ref.lower():
                        mitigations.add(ref)
            
            return list(mitigations)
            
        except Exception as e:
            logger.error(f"Mitigation extraction failed: {e}")
            return []
    
    def _extract_affected_systems(self, atlas_entries: List[ATLASEntry], 
                                 cve_entries: List[CVEEntry], 
                                 jailbreak_entries: List[JailbreakEntry]) -> List[str]:
        """Extract affected systems from all sources"""
        try:
            affected_systems = set()
            
            # From CVE entries
            for entry in cve_entries:
                affected_systems.update(entry.products)
                affected_systems.update(entry.vendors)
            
            # From jailbreak entries
            for entry in jailbreak_entries:
                if entry.target_model != "unknown":
                    affected_systems.add(entry.target_model)
            
            return list(affected_systems)
            
        except Exception as e:
            logger.error(f"Affected system extraction failed: {e}")
            return []
    
    def _extract_references(self, atlas_entries: List[ATLASEntry], 
                           cve_entries: List[CVEEntry], 
                           jailbreak_entries: List[JailbreakEntry]) -> List[str]:
        """Extract references from all sources"""
        try:
            references = set()
            
            # From ATLAS entries
            for entry in atlas_entries:
                references.update(entry.references)
            
            # From CVE entries
            for entry in cve_entries:
                references.update(entry.references)
            
            # From jailbreak entries
            for entry in jailbreak_entries:
                references.update(entry.references)
            
            return list(references)
            
        except Exception as e:
            logger.error(f"Reference extraction failed: {e}")
            return []
    
    def _generate_summary(self, category: str, atlas_entries: List[ATLASEntry], 
                         cve_entries: List[CVEEntry], jailbreak_entries: List[JailbreakEntry]) -> str:
        """Generate summary for threat report"""
        try:
            total_entries = len(atlas_entries) + len(cve_entries) + len(jailbreak_entries)
            
            summary = f"Threat intelligence analysis for {category.replace('_', ' ')} category. "
            summary += f"Found {total_entries} total entries across all sources: "
            summary += f"{len(atlas_entries)} MITRE ATLAS techniques, "
            summary += f"{len(cve_entries)} CVE entries, and "
            summary += f"{len(jailbreak_entries)} jailbreak entries. "
            
            if atlas_entries:
                summary += f"MITRE ATLAS techniques include {', '.join([entry.name for entry in atlas_entries[:3]])}. "
            
            if cve_entries:
                high_severity = [entry for entry in cve_entries if entry.cvss_score >= 7.0]
                if high_severity:
                    summary += f"Found {len(high_severity)} high-severity CVEs. "
            
            if jailbreak_entries:
                high_success = [entry for entry in jailbreak_entries if entry.success_rate >= 0.8]
                if high_success:
                    summary += f"Found {len(high_success)} high-success-rate jailbreaks. "
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Threat intelligence analysis for {category.replace('_', ' ')} category."
    
    async def search_threats(self, query: str) -> Dict[str, List[Any]]:
        """
        Search threats across all sources
        """
        try:
            results = {}
            
            # Search MITRE ATLAS
            async with self.mitre_atlas_scraper:
                atlas_results = await self.mitre_atlas_scraper.search_techniques(query)
                results["mitre_atlas"] = atlas_results
            
            # Search CVE
            async with self.cve_scraper:
                cve_results = await self.cve_scraper.search_cves(query)
                results["cve"] = cve_results
            
            # Search jailbreaks
            async with self.jailbreak_scraper:
                jailbreak_results = await self.jailbreak_scraper.search_jailbreaks(query)
                results["jailbreaks"] = jailbreak_results
            
            return results
            
        except Exception as e:
            logger.error(f"Threat search failed: {e}")
            return {}
    
    async def get_threat_report(self, report_id: str) -> Optional[ThreatIntelligenceReport]:
        """
        Get threat report by ID
        """
        try:
            return self.threat_reports.get(report_id)
            
        except Exception as e:
            logger.error(f"Threat report retrieval failed for {report_id}: {e}")
            return None
    
    async def get_all_threat_reports(self) -> List[ThreatIntelligenceReport]:
        """
        Get all threat reports
        """
        try:
            return list(self.threat_reports.values())
            
        except Exception as e:
            logger.error(f"Threat reports retrieval failed: {e}")
            return []
    
    async def export_intelligence(self, format: str = "json") -> str:
        """
        Export all threat intelligence data
        """
        try:
            if format.lower() == "json":
                data = {
                    "mitre_atlas": {k: v.__dict__ for k, v in self.atlas_entries.items()},
                    "cve": {k: v.__dict__ for k, v in self.cve_entries.items()},
                    "jailbreaks": {k: v.__dict__ for k, v in self.jailbreak_entries.items()},
                    "threat_reports": {k: v.__dict__ for k, v in self.threat_reports.items()},
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Intelligence export failed: {e}")
            return ""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get threat intelligence statistics"""
        return {
            "total_atlas_entries": len(self.atlas_entries),
            "total_cve_entries": len(self.cve_entries),
            "total_jailbreak_entries": len(self.jailbreak_entries),
            "total_threat_reports": len(self.threat_reports),
            "atlas_categories": list(set(entry.category for entry in self.atlas_entries.values())),
            "cve_severity_distribution": self._get_cve_severity_distribution(),
            "jailbreak_type_distribution": self._get_jailbreak_type_distribution(),
            "threat_level_distribution": self._get_threat_level_distribution()
        }
    
    def _get_cve_severity_distribution(self) -> Dict[str, int]:
        """Get CVE severity distribution"""
        distribution = {}
        for entry in self.cve_entries.values():
            severity = entry.severity
            distribution[severity] = distribution.get(severity, 0) + 1
        return distribution
    
    def _get_jailbreak_type_distribution(self) -> Dict[str, int]:
        """Get jailbreak type distribution"""
        distribution = {}
        for entry in self.jailbreak_entries.values():
            jb_type = entry.jailbreak_type.value
            distribution[jb_type] = distribution.get(jb_type, 0) + 1
        return distribution
    
    def _get_threat_level_distribution(self) -> Dict[str, int]:
        """Get threat level distribution"""
        distribution = {}
        for report in self.threat_reports.values():
            level = report.threat_level
            distribution[level] = distribution.get(level, 0) + 1
        return distribution
