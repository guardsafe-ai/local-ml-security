"""
CVE Scraper
Scrapes and analyzes CVE (Common Vulnerabilities and Exposures) data for AI/ML security
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


class CVESeverity(Enum):
    """CVE severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class CVEEntry:
    """CVE entry"""
    cve_id: str
    description: str
    severity: str
    cvss_score: float
    cvss_vector: str
    published_date: str
    modified_date: str
    references: List[str]
    products: List[str]
    vendors: List[str]
    cwe_id: str
    attack_vector: str
    attack_complexity: str
    privileges_required: str
    user_interaction: str
    scope: str
    confidentiality_impact: str
    integrity_impact: str
    availability_impact: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CVEScraper:
    """
    CVE Scraper
    Scrapes and analyzes CVE data for AI/ML security vulnerabilities
    """
    
    def __init__(self):
        """Initialize CVE scraper"""
        self.base_url = "https://cve.mitre.org"
        self.nvd_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        self.session = None
        self.cves = {}
        self.ai_ml_keywords = [
            "artificial intelligence", "machine learning", "neural network",
            "deep learning", "tensorflow", "pytorch", "scikit-learn",
            "keras", "xgboost", "lightgbm", "catboost",
            "hugging face", "transformers", "bert", "gpt",
            "openai", "anthropic", "claude", "chatgpt",
            "llm", "large language model", "generative ai",
            "adversarial", "poisoning", "injection", "jailbreak"
        ]
        
        logger.info("✅ Initialized CVE Scraper")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def scrape_ai_ml_cves(self, 
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               max_results: int = 1000) -> List[CVEEntry]:
        """
        Scrape AI/ML related CVEs
        """
        try:
            logger.info("Starting AI/ML CVE scraping")
            
            cves = []
            
            # Search for AI/ML related CVEs
            for keyword in self.ai_ml_keywords:
                logger.info(f"Searching for CVEs with keyword: {keyword}")
                keyword_cves = await self._search_cves_by_keyword(keyword, max_results // len(self.ai_ml_keywords))
                cves.extend(keyword_cves)
            
            # Remove duplicates
            unique_cves = {cve.cve_id: cve for cve in cves}
            cves = list(unique_cves.values())
            
            # Filter by date range if specified
            if start_date or end_date:
                cves = self._filter_by_date_range(cves, start_date, end_date)
            
            # Store CVEs
            self.cves = {cve.cve_id: cve for cve in cves}
            
            logger.info(f"Scraped {len(cves)} AI/ML related CVEs")
            return cves
            
        except Exception as e:
            logger.error(f"AI/ML CVE scraping failed: {e}")
            return []
    
    async def _search_cves_by_keyword(self, keyword: str, max_results: int) -> List[CVEEntry]:
        """Search CVEs by keyword"""
        try:
            cves = []
            
            # Search NVD API
            nvd_cves = await self._search_nvd_api(keyword, max_results)
            cves.extend(nvd_cves)
            
            # Search MITRE CVE database
            mitre_cves = await self._search_mitre_cve(keyword, max_results)
            cves.extend(mitre_cves)
            
            return cves
            
        except Exception as e:
            logger.error(f"CVE search failed for keyword {keyword}: {e}")
            return []
    
    async def _search_nvd_api(self, keyword: str, max_results: int) -> List[CVEEntry]:
        """Search NVD API for CVEs"""
        try:
            cves = []
            
            # Build search query
            query = {
                "keywordSearch": keyword,
                "resultsPerPage": min(max_results, 2000),
                "startIndex": 0
            }
            
            # Make API request
            url = f"{self.nvd_url}?keywordSearch={keyword}&resultsPerPage={min(max_results, 2000)}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    cves = await self._parse_nvd_response(data)
                else:
                    logger.warning(f"NVD API request failed: {response.status}")
            
            return cves
            
        except Exception as e:
            logger.error(f"NVD API search failed: {e}")
            return []
    
    async def _parse_nvd_response(self, data: Dict[str, Any]) -> List[CVEEntry]:
        """Parse NVD API response"""
        try:
            cves = []
            
            if "vulnerabilities" in data:
                for vuln in data["vulnerabilities"]:
                    cve = await self._parse_nvd_cve(vuln)
                    if cve:
                        cves.append(cve)
            
            return cves
            
        except Exception as e:
            logger.error(f"NVD response parsing failed: {e}")
            return []
    
    async def _parse_nvd_cve(self, vuln: Dict[str, Any]) -> Optional[CVEEntry]:
        """Parse individual NVD CVE"""
        try:
            cve_data = vuln.get("cve", {})
            
            cve_id = cve_data.get("id", "")
            description = ""
            
            # Extract description
            descriptions = cve_data.get("descriptions", [])
            for desc in descriptions:
                if desc.get("lang") == "en":
                    description = desc.get("value", "")
                    break
            
            # Extract references
            references = []
            refs = cve_data.get("references", [])
            for ref in refs:
                references.append(ref.get("url", ""))
            
            # Extract products and vendors
            products = []
            vendors = []
            
            configurations = vuln.get("configurations", [])
            for config in configurations:
                nodes = config.get("nodes", [])
                for node in nodes:
                    cpe_matches = node.get("cpeMatch", [])
                    for match in cpe_matches:
                        cpe = match.get("criteria", "")
                        if cpe:
                            parts = cpe.split(":")
                            if len(parts) >= 4:
                                vendors.append(parts[3])
                            if len(parts) >= 5:
                                products.append(parts[4])
            
            # Extract CVSS data
            cvss_data = vuln.get("metrics", {})
            cvss_score = 0.0
            cvss_vector = ""
            severity = "unknown"
            
            if "cvssMetricV31" in cvss_data:
                cvss = cvss_data["cvssMetricV31"][0]["cvssData"]
                cvss_score = cvss.get("baseScore", 0.0)
                cvss_vector = cvss.get("vectorString", "")
                severity = cvss.get("baseSeverity", "unknown").lower()
            elif "cvssMetricV30" in cvss_data:
                cvss = cvss_data["cvssMetricV30"][0]["cvssData"]
                cvss_score = cvss.get("baseScore", 0.0)
                cvss_vector = cvss.get("vectorString", "")
                severity = cvss.get("baseSeverity", "unknown").lower()
            elif "cvssMetricV2" in cvss_data:
                cvss = cvss_data["cvssMetricV2"][0]["cvssData"]
                cvss_score = cvss.get("baseScore", 0.0)
                cvss_vector = cvss.get("vectorString", "")
                severity = cvss.get("baseSeverity", "unknown").lower()
            
            # Extract attack vector details
            attack_vector = ""
            attack_complexity = ""
            privileges_required = ""
            user_interaction = ""
            scope = ""
            confidentiality_impact = ""
            integrity_impact = ""
            availability_impact = ""
            
            if "cvssMetricV31" in cvss_data:
                cvss = cvss_data["cvssMetricV31"][0]["cvssData"]
                attack_vector = cvss.get("attackVector", "")
                attack_complexity = cvss.get("attackComplexity", "")
                privileges_required = cvss.get("privilegesRequired", "")
                user_interaction = cvss.get("userInteraction", "")
                scope = cvss.get("scope", "")
                confidentiality_impact = cvss.get("confidentialityImpact", "")
                integrity_impact = cvss.get("integrityImpact", "")
                availability_impact = cvss.get("availabilityImpact", "")
            
            # Extract CWE ID
            cwe_id = ""
            weaknesses = cve_data.get("weaknesses", [])
            for weakness in weaknesses:
                descriptions = weakness.get("description", [])
                for desc in descriptions:
                    if desc.get("lang") == "en":
                        cwe_id = desc.get("value", "")
                        break
            
            # Extract dates
            published_date = cve_data.get("published", "")
            modified_date = cve_data.get("lastModified", "")
            
            return CVEEntry(
                cve_id=cve_id,
                description=description,
                severity=severity,
                cvss_score=cvss_score,
                cvss_vector=cvss_vector,
                published_date=published_date,
                modified_date=modified_date,
                references=references,
                products=products,
                vendors=vendors,
                cwe_id=cwe_id,
                attack_vector=attack_vector,
                attack_complexity=attack_complexity,
                privileges_required=privileges_required,
                user_interaction=user_interaction,
                scope=scope,
                confidentiality_impact=confidentiality_impact,
                integrity_impact=integrity_impact,
                availability_impact=availability_impact,
                metadata={
                    "scraped_at": datetime.utcnow().isoformat(),
                    "source": "NVD",
                    "url": f"https://nvd.nist.gov/vuln/detail/{cve_id}"
                }
            )
            
        except Exception as e:
            logger.error(f"NVD CVE parsing failed: {e}")
            return None
    
    async def _search_mitre_cve(self, keyword: str, max_results: int) -> List[CVEEntry]:
        """Search MITRE CVE database"""
        try:
            cves = []
            
            # Search MITRE CVE database
            url = f"{self.base_url}/cgi-bin/cvekey.cgi?keyword={keyword}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    cves = await self._parse_mitre_cve_page(html)
                else:
                    logger.warning(f"MITRE CVE search failed: {response.status}")
            
            return cves[:max_results]
            
        except Exception as e:
            logger.error(f"MITRE CVE search failed: {e}")
            return []
    
    async def _parse_mitre_cve_page(self, html: str) -> List[CVEEntry]:
        """Parse MITRE CVE search results page"""
        try:
            cves = []
            
            # Extract CVE IDs
            cve_links = re.findall(r'href="([^"]*cvename\.cgi\?name=([^"]+))"', html)
            
            for link, cve_id in cve_links:
                cve = await self._scrape_mitre_cve(cve_id)
                if cve:
                    cves.append(cve)
            
            return cves
            
        except Exception as e:
            logger.error(f"MITRE CVE page parsing failed: {e}")
            return []
    
    async def _scrape_mitre_cve(self, cve_id: str) -> Optional[CVEEntry]:
        """Scrape individual MITRE CVE"""
        try:
            url = f"{self.base_url}/cgi-bin/cvename.cgi?name={cve_id}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    return await self._parse_mitre_cve_page(html, cve_id)
                else:
                    logger.warning(f"Failed to fetch MITRE CVE {cve_id}: {response.status}")
                    return None
            
        except Exception as e:
            logger.error(f"MITRE CVE scraping failed for {cve_id}: {e}")
            return None
    
    async def _parse_mitre_cve_page(self, html: str, cve_id: str) -> Optional[CVEEntry]:
        """Parse MITRE CVE page"""
        try:
            # Extract description
            description = self._extract_text(html, r'<td[^>]*>Description</td>\s*<td[^>]*>([^<]+)</td>')
            
            # Extract references
            references = self._extract_list(html, r'<a[^>]*href="([^"]+)"[^>]*>([^<]+)</a>')
            
            # Extract products and vendors
            products = self._extract_list(html, r'<td[^>]*>Product</td>\s*<td[^>]*>([^<]+)</td>')
            vendors = self._extract_list(html, r'<td[^>]*>Vendor</td>\s*<td[^>]*>([^<]+)</td>')
            
            # Extract dates
            published_date = self._extract_text(html, r'<td[^>]*>Published</td>\s*<td[^>]*>([^<]+)</td>')
            modified_date = self._extract_text(html, r'<td[^>]*>Modified</td>\s*<td[^>]*>([^<]+)</td>')
            
            return CVEEntry(
                cve_id=cve_id,
                description=description or "No description available",
                severity="unknown",
                cvss_score=0.0,
                cvss_vector="",
                published_date=published_date or "",
                modified_date=modified_date or "",
                references=references,
                products=products,
                vendors=vendors,
                cwe_id="",
                attack_vector="",
                attack_complexity="",
                privileges_required="",
                user_interaction="",
                scope="",
                confidentiality_impact="",
                integrity_impact="",
                availability_impact="",
                metadata={
                    "scraped_at": datetime.utcnow().isoformat(),
                    "source": "MITRE CVE",
                    "url": f"{self.base_url}/cgi-bin/cvename.cgi?name={cve_id}"
                }
            )
            
        except Exception as e:
            logger.error(f"MITRE CVE page parsing failed for {cve_id}: {e}")
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
    
    def _filter_by_date_range(self, cves: List[CVEEntry], start_date: str, end_date: str) -> List[CVEEntry]:
        """Filter CVEs by date range"""
        try:
            filtered_cves = []
            
            for cve in cves:
                if cve.published_date:
                    cve_date = datetime.fromisoformat(cve.published_date.replace('Z', '+00:00'))
                    
                    if start_date:
                        start = datetime.fromisoformat(start_date)
                        if cve_date < start:
                            continue
                    
                    if end_date:
                        end = datetime.fromisoformat(end_date)
                        if cve_date > end:
                            continue
                    
                    filtered_cves.append(cve)
            
            return filtered_cves
            
        except Exception as e:
            logger.error(f"Date range filtering failed: {e}")
            return cves
    
    async def search_cves(self, query: str) -> List[CVEEntry]:
        """
        Search CVEs by query
        """
        try:
            if not self.cves:
                await self.scrape_ai_ml_cves()
            
            results = []
            query_lower = query.lower()
            
            for cve in self.cves.values():
                if (query_lower in cve.description.lower() or
                    query_lower in cve.cve_id.lower() or
                    any(query_lower in product.lower() for product in cve.products) or
                    any(query_lower in vendor.lower() for vendor in cve.vendors)):
                    results.append(cve)
            
            return results
            
        except Exception as e:
            logger.error(f"CVE search failed: {e}")
            return []
    
    async def get_cve_by_id(self, cve_id: str) -> Optional[CVEEntry]:
        """
        Get CVE by ID
        """
        try:
            if not self.cves:
                await self.scrape_ai_ml_cves()
            
            return self.cves.get(cve_id)
            
        except Exception as e:
            logger.error(f"CVE retrieval failed for {cve_id}: {e}")
            return None
    
    async def get_cves_by_severity(self, severity: str) -> List[CVEEntry]:
        """
        Get CVEs by severity
        """
        try:
            if not self.cves:
                await self.scrape_ai_ml_cves()
            
            results = []
            severity_lower = severity.lower()
            
            for cve in self.cves.values():
                if severity_lower in cve.severity.lower():
                    results.append(cve)
            
            return results
            
        except Exception as e:
            logger.error(f"Severity-based CVE retrieval failed: {e}")
            return []
    
    async def get_cves_by_vendor(self, vendor: str) -> List[CVEEntry]:
        """
        Get CVEs by vendor
        """
        try:
            if not self.cves:
                await self.scrape_ai_ml_cves()
            
            results = []
            vendor_lower = vendor.lower()
            
            for cve in self.cves.values():
                if any(vendor_lower in v.lower() for v in cve.vendors):
                    results.append(cve)
            
            return results
            
        except Exception as e:
            logger.error(f"Vendor-based CVE retrieval failed: {e}")
            return []
    
    async def export_data(self, format: str = "json") -> str:
        """
        Export scraped data
        """
        try:
            if format.lower() == "json":
                data = {
                    "cves": {k: v.__dict__ for k, v in self.cves.items()},
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
            "total_cves": len(self.cves),
            "severity_distribution": self._get_severity_distribution(),
            "vendor_distribution": self._get_vendor_distribution(),
            "product_distribution": self._get_product_distribution(),
            "cvss_score_distribution": self._get_cvss_score_distribution()
        }
    
    def _get_severity_distribution(self) -> Dict[str, int]:
        """Get severity distribution"""
        distribution = {}
        for cve in self.cves.values():
            severity = cve.severity
            distribution[severity] = distribution.get(severity, 0) + 1
        return distribution
    
    def _get_vendor_distribution(self) -> Dict[str, int]:
        """Get vendor distribution"""
        distribution = {}
        for cve in self.cves.values():
            for vendor in cve.vendors:
                distribution[vendor] = distribution.get(vendor, 0) + 1
        return distribution
    
    def _get_product_distribution(self) -> Dict[str, int]:
        """Get product distribution"""
        distribution = {}
        for cve in self.cves.values():
            for product in cve.products:
                distribution[product] = distribution.get(product, 0) + 1
        return distribution
    
    def _get_cvss_score_distribution(self) -> Dict[str, int]:
        """Get CVSS score distribution"""
        distribution = {
            "0.0-2.0": 0,
            "2.1-4.0": 0,
            "4.1-6.0": 0,
            "6.1-8.0": 0,
            "8.1-10.0": 0
        }
        
        for cve in self.cves.values():
            score = cve.cvss_score
            if score <= 2.0:
                distribution["0.0-2.0"] += 1
            elif score <= 4.0:
                distribution["2.1-4.0"] += 1
            elif score <= 6.0:
                distribution["4.1-6.0"] += 1
            elif score <= 8.0:
                distribution["6.1-8.0"] += 1
            else:
                distribution["8.1-10.0"] += 1
        
        return distribution
