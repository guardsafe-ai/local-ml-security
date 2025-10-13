"""
MITRE ATLAS Scraper
Scrapes and analyzes MITRE ATLAS (Adversarial Threat Landscape for AI Systems) data
"""

import asyncio
import logging
import aiohttp
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class ATLASTechnique(Enum):
    """MITRE ATLAS techniques"""
    PROMPT_INJECTION = "prompt_injection"
    DATA_POISONING = "data_poisoning"
    MODEL_EXTRACTION = "model_extraction"
    ADVERSARIAL_EXAMPLES = "adversarial_examples"
    BACKDOOR_ATTACKS = "backdoor_attacks"
    MEMBERSHIP_INFERENCE = "membership_inference"
    MODEL_INVERSION = "model_inversion"
    PRIVACY_ATTACKS = "privacy_attacks"
    BIAS_ATTACKS = "bias_attacks"
    ROBUSTNESS_ATTACKS = "robustness_attacks"


@dataclass
class ATLASEntry:
    """MITRE ATLAS entry"""
    technique_id: str
    name: str
    description: str
    category: str
    subcategory: str
    tactics: List[str]
    techniques: List[str]
    mitigations: List[str]
    references: List[str]
    severity: str
    difficulty: str
    detection: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MITREATLASScraper:
    """
    MITRE ATLAS Scraper
    Scrapes and analyzes MITRE ATLAS data for AI security threats
    """
    
    def __init__(self):
        """Initialize MITRE ATLAS scraper"""
        self.base_url = "https://atlas.mitre.org"
        self.api_url = "https://atlas.mitre.org/api"
        self.session = None
        self.techniques = {}
        self.tactics = {}
        self.mitigations = {}
        
        logger.info("✅ Initialized MITRE ATLAS Scraper")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def scrape_all_techniques(self) -> List[ATLASEntry]:
        """
        Scrape all MITRE ATLAS techniques
        """
        try:
            logger.info("Starting MITRE ATLAS techniques scraping")
            
            techniques = []
            
            # Scrape techniques from different categories
            categories = [
                "adversarial-examples",
                "data-poisoning",
                "model-extraction",
                "membership-inference",
                "model-inversion",
                "privacy-attacks",
                "bias-attacks",
                "robustness-attacks"
            ]
            
            for category in categories:
                logger.info(f"Scraping category: {category}")
                category_techniques = await self._scrape_category(category)
                techniques.extend(category_techniques)
            
            # Store techniques
            self.techniques = {tech.technique_id: tech for tech in techniques}
            
            logger.info(f"Scraped {len(techniques)} MITRE ATLAS techniques")
            return techniques
            
        except Exception as e:
            logger.error(f"MITRE ATLAS scraping failed: {e}")
            return []
    
    async def _scrape_category(self, category: str) -> List[ATLASEntry]:
        """Scrape techniques from specific category"""
        try:
            techniques = []
            
            # Get category page
            url = f"{self.base_url}/techniques/{category}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    techniques = await self._parse_category_page(html, category)
                else:
                    logger.warning(f"Failed to fetch category {category}: {response.status}")
            
            return techniques
            
        except Exception as e:
            logger.error(f"Category scraping failed for {category}: {e}")
            return []
    
    async def _parse_category_page(self, html: str, category: str) -> List[ATLASEntry]:
        """Parse category page HTML"""
        try:
            techniques = []
            
            # This is a simplified parser
            # In practice, you would use BeautifulSoup or similar for robust HTML parsing
            
            # Extract technique links
            technique_links = re.findall(r'href="/techniques/([^"]+)"', html)
            
            for link in technique_links:
                technique_id = link.split('/')[-1]
                technique = await self._scrape_technique(technique_id)
                if technique:
                    techniques.append(technique)
            
            return techniques
            
        except Exception as e:
            logger.error(f"Category page parsing failed for {category}: {e}")
            return []
    
    async def _scrape_technique(self, technique_id: str) -> Optional[ATLASEntry]:
        """Scrape individual technique"""
        try:
            url = f"{self.base_url}/techniques/{technique_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    return await self._parse_technique_page(html, technique_id)
                else:
                    logger.warning(f"Failed to fetch technique {technique_id}: {response.status}")
                    return None
            
        except Exception as e:
            logger.error(f"Technique scraping failed for {technique_id}: {e}")
            return None
    
    async def _parse_technique_page(self, html: str, technique_id: str) -> Optional[ATLASEntry]:
        """Parse technique page HTML"""
        try:
            # This is a simplified parser
            # In practice, you would use BeautifulSoup for robust HTML parsing
            
            # Extract technique information
            name = self._extract_text(html, r'<h1[^>]*>([^<]+)</h1>')
            description = self._extract_text(html, r'<div[^>]*class="description"[^>]*>([^<]+)</div>')
            category = self._extract_text(html, r'<span[^>]*class="category"[^>]*>([^<]+)</span>')
            subcategory = self._extract_text(html, r'<span[^>]*class="subcategory"[^>]*>([^<]+)</span>')
            
            # Extract tactics
            tactics = self._extract_list(html, r'<li[^>]*class="tactic"[^>]*>([^<]+)</li>')
            
            # Extract techniques
            techniques = self._extract_list(html, r'<li[^>]*class="technique"[^>]*>([^<]+)</li>')
            
            # Extract mitigations
            mitigations = self._extract_list(html, r'<li[^>]*class="mitigation"[^>]*>([^<]+)</li>')
            
            # Extract references
            references = self._extract_list(html, r'<a[^>]*href="([^"]+)"[^>]*>([^<]+)</a>')
            
            # Extract severity
            severity = self._extract_text(html, r'<span[^>]*class="severity"[^>]*>([^<]+)</span>')
            
            # Extract difficulty
            difficulty = self._extract_text(html, r'<span[^>]*class="difficulty"[^>]*>([^<]+)</span>')
            
            # Extract detection
            detection = self._extract_text(html, r'<div[^>]*class="detection"[^>]*>([^<]+)</div>')
            
            return ATLASEntry(
                technique_id=technique_id,
                name=name or f"Technique {technique_id}",
                description=description or "No description available",
                category=category or "Unknown",
                subcategory=subcategory or "Unknown",
                tactics=tactics,
                techniques=techniques,
                mitigations=mitigations,
                references=references,
                severity=severity or "Unknown",
                difficulty=difficulty or "Unknown",
                detection=detection or "Unknown",
                metadata={
                    "scraped_at": datetime.utcnow().isoformat(),
                    "source": "MITRE ATLAS",
                    "url": f"{self.base_url}/techniques/{technique_id}"
                }
            )
            
        except Exception as e:
            logger.error(f"Technique page parsing failed for {technique_id}: {e}")
            return None
    
    def _extract_text(self, html: str, pattern: str) -> str:
        """Extract text using regex pattern"""
        try:
            match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            return match.group(1).strip() if match else ""
        except Exception:
            return ""
    
    def _extract_list(self, html: str, pattern: str) -> List[str]:
        """Extract list of items using regex pattern"""
        try:
            matches = re.findall(pattern, html, re.IGNORECASE | re.DOTALL)
            return [match.strip() for match in matches if match.strip()]
        except Exception:
            return []
    
    async def scrape_tactics(self) -> Dict[str, Any]:
        """
        Scrape MITRE ATLAS tactics
        """
        try:
            logger.info("Scraping MITRE ATLAS tactics")
            
            tactics = {}
            
            # Get tactics page
            url = f"{self.base_url}/tactics"
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    tactics = await self._parse_tactics_page(html)
                else:
                    logger.warning(f"Failed to fetch tactics: {response.status}")
            
            self.tactics = tactics
            return tactics
            
        except Exception as e:
            logger.error(f"Tactics scraping failed: {e}")
            return {}
    
    async def _parse_tactics_page(self, html: str) -> Dict[str, Any]:
        """Parse tactics page HTML"""
        try:
            tactics = {}
            
            # Extract tactic information
            tactic_links = re.findall(r'href="/tactics/([^"]+)"', html)
            
            for link in tactic_links:
                tactic_id = link.split('/')[-1]
                tactic_info = await self._scrape_tactic(tactic_id)
                if tactic_info:
                    tactics[tactic_id] = tactic_info
            
            return tactics
            
        except Exception as e:
            logger.error(f"Tactics page parsing failed: {e}")
            return {}
    
    async def _scrape_tactic(self, tactic_id: str) -> Optional[Dict[str, Any]]:
        """Scrape individual tactic"""
        try:
            url = f"{self.base_url}/tactics/{tactic_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    return await self._parse_tactic_page(html, tactic_id)
                else:
                    logger.warning(f"Failed to fetch tactic {tactic_id}: {response.status}")
                    return None
            
        except Exception as e:
            logger.error(f"Tactic scraping failed for {tactic_id}: {e}")
            return None
    
    async def _parse_tactic_page(self, html: str, tactic_id: str) -> Optional[Dict[str, Any]]:
        """Parse tactic page HTML"""
        try:
            name = self._extract_text(html, r'<h1[^>]*>([^<]+)</h1>')
            description = self._extract_text(html, r'<div[^>]*class="description"[^>]*>([^<]+)</div>')
            
            return {
                "tactic_id": tactic_id,
                "name": name or f"Tactic {tactic_id}",
                "description": description or "No description available",
                "scraped_at": datetime.utcnow().isoformat(),
                "source": "MITRE ATLAS"
            }
            
        except Exception as e:
            logger.error(f"Tactic page parsing failed for {tactic_id}: {e}")
            return None
    
    async def scrape_mitigations(self) -> Dict[str, Any]:
        """
        Scrape MITRE ATLAS mitigations
        """
        try:
            logger.info("Scraping MITRE ATLAS mitigations")
            
            mitigations = {}
            
            # Get mitigations page
            url = f"{self.base_url}/mitigations"
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    mitigations = await self._parse_mitigations_page(html)
                else:
                    logger.warning(f"Failed to fetch mitigations: {response.status}")
            
            self.mitigations = mitigations
            return mitigations
            
        except Exception as e:
            logger.error(f"Mitigations scraping failed: {e}")
            return {}
    
    async def _parse_mitigations_page(self, html: str) -> Dict[str, Any]:
        """Parse mitigations page HTML"""
        try:
            mitigations = {}
            
            # Extract mitigation information
            mitigation_links = re.findall(r'href="/mitigations/([^"]+)"', html)
            
            for link in mitigation_links:
                mitigation_id = link.split('/')[-1]
                mitigation_info = await self._scrape_mitigation(mitigation_id)
                if mitigation_info:
                    mitigations[mitigation_id] = mitigation_info
            
            return mitigations
            
        except Exception as e:
            logger.error(f"Mitigations page parsing failed: {e}")
            return {}
    
    async def _scrape_mitigation(self, mitigation_id: str) -> Optional[Dict[str, Any]]:
        """Scrape individual mitigation"""
        try:
            url = f"{self.base_url}/mitigations/{mitigation_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    return await self._parse_mitigation_page(html, mitigation_id)
                else:
                    logger.warning(f"Failed to fetch mitigation {mitigation_id}: {response.status}")
                    return None
            
        except Exception as e:
            logger.error(f"Mitigation scraping failed for {mitigation_id}: {e}")
            return None
    
    async def _parse_mitigation_page(self, html: str, mitigation_id: str) -> Optional[Dict[str, Any]]:
        """Parse mitigation page HTML"""
        try:
            name = self._extract_text(html, r'<h1[^>]*>([^<]+)</h1>')
            description = self._extract_text(html, r'<div[^>]*class="description"[^>]*>([^<]+)</div>')
            
            return {
                "mitigation_id": mitigation_id,
                "name": name or f"Mitigation {mitigation_id}",
                "description": description or "No description available",
                "scraped_at": datetime.utcnow().isoformat(),
                "source": "MITRE ATLAS"
            }
            
        except Exception as e:
            logger.error(f"Mitigation page parsing failed for {mitigation_id}: {e}")
            return None
    
    async def search_techniques(self, query: str) -> List[ATLASEntry]:
        """
        Search techniques by query
        """
        try:
            if not self.techniques:
                await self.scrape_all_techniques()
            
            results = []
            query_lower = query.lower()
            
            for technique in self.techniques.values():
                if (query_lower in technique.name.lower() or 
                    query_lower in technique.description.lower() or
                    query_lower in technique.category.lower()):
                    results.append(technique)
            
            return results
            
        except Exception as e:
            logger.error(f"Technique search failed: {e}")
            return []
    
    async def get_technique_by_id(self, technique_id: str) -> Optional[ATLASEntry]:
        """
        Get technique by ID
        """
        try:
            if not self.techniques:
                await self.scrape_all_techniques()
            
            return self.techniques.get(technique_id)
            
        except Exception as e:
            logger.error(f"Technique retrieval failed for {technique_id}: {e}")
            return None
    
    async def get_techniques_by_category(self, category: str) -> List[ATLASEntry]:
        """
        Get techniques by category
        """
        try:
            if not self.techniques:
                await self.scrape_all_techniques()
            
            results = []
            category_lower = category.lower()
            
            for technique in self.techniques.values():
                if category_lower in technique.category.lower():
                    results.append(technique)
            
            return results
            
        except Exception as e:
            logger.error(f"Category-based technique retrieval failed: {e}")
            return []
    
    async def export_data(self, format: str = "json") -> str:
        """
        Export scraped data
        """
        try:
            if format.lower() == "json":
                data = {
                    "techniques": {k: v.__dict__ for k, v in self.techniques.items()},
                    "tactics": self.tactics,
                    "mitigations": self.mitigations,
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            return ""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scraping statistics"""
        return {
            "total_techniques": len(self.techniques),
            "total_tactics": len(self.tactics),
            "total_mitigations": len(self.mitigations),
            "categories": list(set(tech.category for tech in self.techniques.values())),
            "severity_distribution": self._get_severity_distribution(),
            "difficulty_distribution": self._get_difficulty_distribution()
        }
    
    def _get_severity_distribution(self) -> Dict[str, int]:
        """Get severity distribution"""
        distribution = {}
        for technique in self.techniques.values():
            severity = technique.severity
            distribution[severity] = distribution.get(severity, 0) + 1
        return distribution
    
    def _get_difficulty_distribution(self) -> Dict[str, int]:
        """Get difficulty distribution"""
        distribution = {}
        for technique in self.techniques.values():
            difficulty = technique.difficulty
            distribution[difficulty] = distribution.get(difficulty, 0) + 1
        return distribution
