"""
Jailbreak Scraper
Scrapes and analyzes jailbreak databases and repositories for AI security research
"""

import asyncio
import logging
import aiohttp
import json
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class JailbreakType(Enum):
    """Types of jailbreaks"""
    PROMPT_INJECTION = "prompt_injection"
    ROLEPLAY = "roleplay"
    HYPOTHETICAL = "hypothetical"
    CODING = "coding"
    SOCIAL_ENGINEERING = "social_engineering"
    TECHNICAL_EXPLOIT = "technical_exploit"
    CREATIVE_WRITING = "creative_writing"
    RESEARCH = "research"
    GAMING = "gaming"
    OTHER = "other"


@dataclass
class JailbreakEntry:
    """Jailbreak entry"""
    jailbreak_id: str
    name: str
    description: str
    jailbreak_type: JailbreakType
    prompt: str
    target_model: str
    success_rate: float
    difficulty: str
    stealth_score: float
    effectiveness: float
    tags: List[str]
    references: List[str]
    author: str
    created_date: str
    updated_date: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class JailbreakScraper:
    """
    Jailbreak Scraper
    Scrapes and analyzes jailbreak databases and repositories
    """
    
    def __init__(self):
        """Initialize jailbreak scraper"""
        self.session = None
        self.jailbreaks = {}
        self.sources = [
            "https://github.com/llm-attacks/llm-attacks",
            "https://github.com/kevinscruffy/jailbreak-prompt",
            "https://github.com/verazuo/jailbreak_llm",
            "https://github.com/ai-safety/jailbreak-datasets",
            "https://github.com/llm-attacks/llm-attacks/tree/main/data",
            "https://huggingface.co/datasets/llm-attacks/jailbreak-datasets"
        ]
        
        logger.info("✅ Initialized Jailbreak Scraper")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def scrape_all_jailbreaks(self) -> List[JailbreakEntry]:
        """
        Scrape jailbreaks from all sources
        """
        try:
            logger.info("Starting jailbreak scraping from all sources")
            
            jailbreaks = []
            
            for source in self.sources:
                logger.info(f"Scraping from source: {source}")
                source_jailbreaks = await self._scrape_source(source)
                jailbreaks.extend(source_jailbreaks)
            
            # Remove duplicates
            unique_jailbreaks = {jb.jailbreak_id: jb for jb in jailbreaks}
            jailbreaks = list(unique_jailbreaks.values())
            
            # Store jailbreaks
            self.jailbreaks = {jb.jailbreak_id: jb for jb in jailbreaks}
            
            logger.info(f"Scraped {len(jailbreaks)} jailbreaks")
            return jailbreaks
            
        except Exception as e:
            logger.error(f"Jailbreak scraping failed: {e}")
            return []
    
    async def _scrape_source(self, source: str) -> List[JailbreakEntry]:
        """Scrape jailbreaks from specific source"""
        try:
            if "github.com" in source:
                return await self._scrape_github_source(source)
            elif "huggingface.co" in source:
                return await self._scrape_huggingface_source(source)
            else:
                logger.warning(f"Unknown source type: {source}")
                return []
            
        except Exception as e:
            logger.error(f"Source scraping failed for {source}: {e}")
            return []
    
    async def _scrape_github_source(self, source: str) -> List[JailbreakEntry]:
        """Scrape jailbreaks from GitHub source"""
        try:
            jailbreaks = []
            
            # Extract repository information
            repo_match = re.search(r'github\.com/([^/]+)/([^/]+)', source)
            if not repo_match:
                logger.warning(f"Invalid GitHub URL: {source}")
                return []
            
            owner, repo = repo_match.groups()
            
            # Get repository contents
            api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
            
            async with self.session.get(api_url) as response:
                if response.status == 200:
                    contents = await response.json()
                    
                    # Look for data files
                    for item in contents:
                        if item["type"] == "file" and self._is_data_file(item["name"]):
                            file_jailbreaks = await self._scrape_github_file(item["download_url"])
                            jailbreaks.extend(file_jailbreaks)
                        elif item["type"] == "dir" and self._is_data_directory(item["name"]):
                            dir_jailbreaks = await self._scrape_github_directory(f"{api_url}/{item['name']}")
                            jailbreaks.extend(dir_jailbreaks)
                else:
                    logger.warning(f"Failed to fetch GitHub contents: {response.status}")
            
            return jailbreaks
            
        except Exception as e:
            logger.error(f"GitHub source scraping failed: {e}")
            return []
    
    async def _scrape_github_file(self, file_url: str) -> List[JailbreakEntry]:
        """Scrape jailbreaks from GitHub file"""
        try:
            jailbreaks = []
            
            async with self.session.get(file_url) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Parse based on file type
                    if file_url.endswith('.json'):
                        jailbreaks = await self._parse_json_file(content)
                    elif file_url.endswith('.csv'):
                        jailbreaks = await self._parse_csv_file(content)
                    elif file_url.endswith('.txt'):
                        jailbreaks = await self._parse_text_file(content)
                    else:
                        logger.warning(f"Unsupported file type: {file_url}")
                
            return jailbreaks
            
        except Exception as e:
            logger.error(f"GitHub file scraping failed: {e}")
            return []
    
    async def _scrape_github_directory(self, dir_url: str) -> List[JailbreakEntry]:
        """Scrape jailbreaks from GitHub directory"""
        try:
            jailbreaks = []
            
            async with self.session.get(dir_url) as response:
                if response.status == 200:
                    contents = await response.json()
                    
                    for item in contents:
                        if item["type"] == "file" and self._is_data_file(item["name"]):
                            file_jailbreaks = await self._scrape_github_file(item["download_url"])
                            jailbreaks.extend(file_jailbreaks)
                
            return jailbreaks
            
        except Exception as e:
            logger.error(f"GitHub directory scraping failed: {e}")
            return []
    
    async def _scrape_huggingface_source(self, source: str) -> List[JailbreakEntry]:
        """Scrape jailbreaks from Hugging Face source"""
        try:
            jailbreaks = []
            
            # Extract dataset information
            dataset_match = re.search(r'huggingface\.co/datasets/([^/]+)/([^/]+)', source)
            if not dataset_match:
                logger.warning(f"Invalid Hugging Face URL: {source}")
                return []
            
            owner, dataset = dataset_match.groups()
            
            # Get dataset files
            api_url = f"https://huggingface.co/api/datasets/{owner}/{dataset}"
            
            async with self.session.get(api_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Look for data files
                    if "siblings" in data:
                        for sibling in data["siblings"]:
                            if sibling["rfilename"].endswith(('.json', '.csv', '.txt')):
                                file_url = f"https://huggingface.co/datasets/{owner}/{dataset}/resolve/main/{sibling['rfilename']}"
                                file_jailbreaks = await self._scrape_huggingface_file(file_url)
                                jailbreaks.extend(file_jailbreaks)
                else:
                    logger.warning(f"Failed to fetch Hugging Face dataset: {response.status}")
            
            return jailbreaks
            
        except Exception as e:
            logger.error(f"Hugging Face source scraping failed: {e}")
            return []
    
    async def _scrape_huggingface_file(self, file_url: str) -> List[JailbreakEntry]:
        """Scrape jailbreaks from Hugging Face file"""
        try:
            jailbreaks = []
            
            async with self.session.get(file_url) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Parse based on file type
                    if file_url.endswith('.json'):
                        jailbreaks = await self._parse_json_file(content)
                    elif file_url.endswith('.csv'):
                        jailbreaks = await self._parse_csv_file(content)
                    elif file_url.endswith('.txt'):
                        jailbreaks = await self._parse_text_file(content)
                    else:
                        logger.warning(f"Unsupported file type: {file_url}")
                
            return jailbreaks
            
        except Exception as e:
            logger.error(f"Hugging Face file scraping failed: {e}")
            return []
    
    def _is_data_file(self, filename: str) -> bool:
        """Check if file is a data file"""
        data_extensions = ['.json', '.csv', '.txt', '.jsonl', '.tsv']
        return any(filename.endswith(ext) for ext in data_extensions)
    
    def _is_data_directory(self, dirname: str) -> bool:
        """Check if directory is a data directory"""
        data_dirs = ['data', 'datasets', 'jailbreaks', 'prompts', 'examples']
        return dirname.lower() in data_dirs
    
    async def _parse_json_file(self, content: str) -> List[JailbreakEntry]:
        """Parse JSON file content"""
        try:
            data = json.loads(content)
            jailbreaks = []
            
            if isinstance(data, list):
                for item in data:
                    jailbreak = await self._parse_json_item(item)
                    if jailbreak:
                        jailbreaks.append(jailbreak)
            elif isinstance(data, dict):
                jailbreak = await self._parse_json_item(data)
                if jailbreak:
                    jailbreaks.append(jailbreak)
            
            return jailbreaks
            
        except Exception as e:
            logger.error(f"JSON file parsing failed: {e}")
            return []
    
    async def _parse_json_item(self, item: Dict[str, Any]) -> Optional[JailbreakEntry]:
        """Parse individual JSON item"""
        try:
            # Extract fields with fallbacks
            jailbreak_id = item.get("id", item.get("jailbreak_id", self._generate_id(item)))
            name = item.get("name", item.get("title", f"Jailbreak {jailbreak_id}"))
            description = item.get("description", item.get("desc", ""))
            prompt = item.get("prompt", item.get("text", item.get("content", "")))
            target_model = item.get("target_model", item.get("model", "unknown"))
            success_rate = float(item.get("success_rate", item.get("success", 0.5)))
            difficulty = item.get("difficulty", item.get("complexity", "medium"))
            stealth_score = float(item.get("stealth_score", item.get("stealth", 0.5)))
            effectiveness = float(item.get("effectiveness", item.get("effect", 0.5)))
            tags = item.get("tags", item.get("categories", []))
            references = item.get("references", item.get("sources", []))
            author = item.get("author", item.get("creator", "unknown"))
            created_date = item.get("created_date", item.get("date", datetime.utcnow().isoformat()))
            updated_date = item.get("updated_date", item.get("modified", created_date))
            
            # Determine jailbreak type
            jailbreak_type = self._determine_jailbreak_type(prompt, tags)
            
            return JailbreakEntry(
                jailbreak_id=jailbreak_id,
                name=name,
                description=description,
                jailbreak_type=jailbreak_type,
                prompt=prompt,
                target_model=target_model,
                success_rate=success_rate,
                difficulty=difficulty,
                stealth_score=stealth_score,
                effectiveness=effectiveness,
                tags=tags if isinstance(tags, list) else [tags] if tags else [],
                references=references if isinstance(references, list) else [references] if references else [],
                author=author,
                created_date=created_date,
                updated_date=updated_date,
                metadata={
                    "scraped_at": datetime.utcnow().isoformat(),
                    "source": "jailbreak_database"
                }
            )
            
        except Exception as e:
            logger.error(f"JSON item parsing failed: {e}")
            return None
    
    async def _parse_csv_file(self, content: str) -> List[JailbreakEntry]:
        """Parse CSV file content"""
        try:
            lines = content.strip().split('\n')
            if not lines:
                return []
            
            # Parse header
            header = lines[0].split(',')
            jailbreaks = []
            
            for line in lines[1:]:
                if line.strip():
                    values = line.split(',')
                    if len(values) >= len(header):
                        item = dict(zip(header, values))
                        jailbreak = await self._parse_json_item(item)
                        if jailbreak:
                            jailbreaks.append(jailbreak)
            
            return jailbreaks
            
        except Exception as e:
            logger.error(f"CSV file parsing failed: {e}")
            return []
    
    async def _parse_text_file(self, content: str) -> List[JailbreakEntry]:
        """Parse text file content"""
        try:
            jailbreaks = []
            lines = content.strip().split('\n')
            
            current_jailbreak = {}
            for line in lines:
                line = line.strip()
                if not line:
                    if current_jailbreak:
                        jailbreak = await self._parse_json_item(current_jailbreak)
                        if jailbreak:
                            jailbreaks.append(jailbreak)
                        current_jailbreak = {}
                elif ':' in line:
                    key, value = line.split(':', 1)
                    current_jailbreak[key.strip()] = value.strip()
                else:
                    # Assume it's a prompt
                    if 'prompt' not in current_jailbreak:
                        current_jailbreak['prompt'] = line
                    else:
                        current_jailbreak['prompt'] += '\n' + line
            
            # Process last jailbreak
            if current_jailbreak:
                jailbreak = await self._parse_json_item(current_jailbreak)
                if jailbreak:
                    jailbreaks.append(jailbreak)
            
            return jailbreaks
            
        except Exception as e:
            logger.error(f"Text file parsing failed: {e}")
            return []
    
    def _generate_id(self, item: Dict[str, Any]) -> str:
        """Generate ID for jailbreak"""
        try:
            # Use prompt or name to generate ID
            text = item.get("prompt", item.get("name", str(item)))
            return hashlib.md5(text.encode()).hexdigest()[:8]
        except Exception:
            return f"jb_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    def _determine_jailbreak_type(self, prompt: str, tags: List[str]) -> JailbreakType:
        """Determine jailbreak type from prompt and tags"""
        try:
            prompt_lower = prompt.lower()
            tags_lower = [tag.lower() for tag in tags]
            
            # Check for specific patterns
            if any(keyword in prompt_lower for keyword in ["roleplay", "pretend", "act as", "you are"]):
                return JailbreakType.ROLEPLAY
            elif any(keyword in prompt_lower for keyword in ["hypothetical", "imagine", "suppose", "what if"]):
                return JailbreakType.HYPOTHETICAL
            elif any(keyword in prompt_lower for keyword in ["code", "programming", "script", "function"]):
                return JailbreakType.CODING
            elif any(keyword in prompt_lower for keyword in ["social", "psychological", "manipulation"]):
                return JailbreakType.SOCIAL_ENGINEERING
            elif any(keyword in prompt_lower for keyword in ["technical", "exploit", "vulnerability"]):
                return JailbreakType.TECHNICAL_EXPLOIT
            elif any(keyword in prompt_lower for keyword in ["creative", "writing", "story", "poem"]):
                return JailbreakType.CREATIVE_WRITING
            elif any(keyword in prompt_lower for keyword in ["research", "study", "analysis"]):
                return JailbreakType.RESEARCH
            elif any(keyword in prompt_lower for keyword in ["game", "gaming", "play"]):
                return JailbreakType.GAMING
            elif any(keyword in prompt_lower for keyword in ["inject", "injection", "bypass"]):
                return JailbreakType.PROMPT_INJECTION
            else:
                return JailbreakType.OTHER
                
        except Exception:
            return JailbreakType.OTHER
    
    async def search_jailbreaks(self, query: str) -> List[JailbreakEntry]:
        """
        Search jailbreaks by query
        """
        try:
            if not self.jailbreaks:
                await self.scrape_all_jailbreaks()
            
            results = []
            query_lower = query.lower()
            
            for jailbreak in self.jailbreaks.values():
                if (query_lower in jailbreak.name.lower() or
                    query_lower in jailbreak.description.lower() or
                    query_lower in jailbreak.prompt.lower() or
                    any(query_lower in tag.lower() for tag in jailbreak.tags)):
                    results.append(jailbreak)
            
            return results
            
        except Exception as e:
            logger.error(f"Jailbreak search failed: {e}")
            return []
    
    async def get_jailbreak_by_id(self, jailbreak_id: str) -> Optional[JailbreakEntry]:
        """
        Get jailbreak by ID
        """
        try:
            if not self.jailbreaks:
                await self.scrape_all_jailbreaks()
            
            return self.jailbreaks.get(jailbreak_id)
            
        except Exception as e:
            logger.error(f"Jailbreak retrieval failed for {jailbreak_id}: {e}")
            return None
    
    async def get_jailbreaks_by_type(self, jailbreak_type: JailbreakType) -> List[JailbreakEntry]:
        """
        Get jailbreaks by type
        """
        try:
            if not self.jailbreaks:
                await self.scrape_all_jailbreaks()
            
            results = []
            for jailbreak in self.jailbreaks.values():
                if jailbreak.jailbreak_type == jailbreak_type:
                    results.append(jailbreak)
            
            return results
            
        except Exception as e:
            logger.error(f"Type-based jailbreak retrieval failed: {e}")
            return []
    
    async def get_jailbreaks_by_difficulty(self, difficulty: str) -> List[JailbreakEntry]:
        """
        Get jailbreaks by difficulty
        """
        try:
            if not self.jailbreaks:
                await self.scrape_all_jailbreaks()
            
            results = []
            difficulty_lower = difficulty.lower()
            
            for jailbreak in self.jailbreaks.values():
                if difficulty_lower in jailbreak.difficulty.lower():
                    results.append(jailbreak)
            
            return results
            
        except Exception as e:
            logger.error(f"Difficulty-based jailbreak retrieval failed: {e}")
            return []
    
    async def export_data(self, format: str = "json") -> str:
        """
        Export scraped data
        """
        try:
            if format.lower() == "json":
                data = {
                    "jailbreaks": {k: v.__dict__ for k, v in self.jailbreaks.items()},
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
            "total_jailbreaks": len(self.jailbreaks),
            "type_distribution": self._get_type_distribution(),
            "difficulty_distribution": self._get_difficulty_distribution(),
            "target_model_distribution": self._get_target_model_distribution(),
            "average_success_rate": self._get_average_success_rate(),
            "average_stealth_score": self._get_average_stealth_score(),
            "average_effectiveness": self._get_average_effectiveness()
        }
    
    def _get_type_distribution(self) -> Dict[str, int]:
        """Get type distribution"""
        distribution = {}
        for jailbreak in self.jailbreaks.values():
            jb_type = jailbreak.jailbreak_type.value
            distribution[jb_type] = distribution.get(jb_type, 0) + 1
        return distribution
    
    def _get_difficulty_distribution(self) -> Dict[str, int]:
        """Get difficulty distribution"""
        distribution = {}
        for jailbreak in self.jailbreaks.values():
            difficulty = jailbreak.difficulty
            distribution[difficulty] = distribution.get(difficulty, 0) + 1
        return distribution
    
    def _get_target_model_distribution(self) -> Dict[str, int]:
        """Get target model distribution"""
        distribution = {}
        for jailbreak in self.jailbreaks.values():
            model = jailbreak.target_model
            distribution[model] = distribution.get(model, 0) + 1
        return distribution
    
    def _get_average_success_rate(self) -> float:
        """Get average success rate"""
        if not self.jailbreaks:
            return 0.0
        return sum(jb.success_rate for jb in self.jailbreaks.values()) / len(self.jailbreaks)
    
    def _get_average_stealth_score(self) -> float:
        """Get average stealth score"""
        if not self.jailbreaks:
            return 0.0
        return sum(jb.stealth_score for jb in self.jailbreaks.values()) / len(self.jailbreaks)
    
    def _get_average_effectiveness(self) -> float:
        """Get average effectiveness"""
        if not self.jailbreaks:
            return 0.0
        return sum(jb.effectiveness for jb in self.jailbreaks.values()) / len(self.jailbreaks)
