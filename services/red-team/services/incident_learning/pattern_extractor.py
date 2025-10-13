"""
Pattern Extractor
Extracts and analyzes patterns from security incidents for learning
"""

import asyncio
import logging
import json
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from collections import Counter, defaultdict
import hashlib

from .incident_processor import ProcessedIncident, SecurityIncident

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of patterns"""
    TEXT_PATTERN = "text_pattern"
    BEHAVIORAL_PATTERN = "behavioral_pattern"
    TEMPORAL_PATTERN = "temporal_pattern"
    STRUCTURAL_PATTERN = "structural_pattern"
    SEMANTIC_PATTERN = "semantic_pattern"


@dataclass
class ExtractedPattern:
    """Extracted pattern representation"""
    pattern_id: str
    pattern_type: PatternType
    pattern_content: str
    frequency: int
    confidence: float
    support: float
    examples: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PatternCluster:
    """Pattern cluster representation"""
    cluster_id: str
    patterns: List[ExtractedPattern]
    centroid: str
    size: int
    cohesion: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PatternExtractor:
    """
    Pattern Extractor
    Extracts and analyzes patterns from security incidents for learning
    """
    
    def __init__(self):
        """Initialize pattern extractor"""
        self.patterns: Dict[str, ExtractedPattern] = {}
        self.clusters: Dict[str, PatternCluster] = {}
        self.pattern_frequency: Dict[str, int] = defaultdict(int)
        self.pattern_co_occurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        
        logger.info("✅ Initialized Pattern Extractor")
    
    async def extract_patterns(self, processed_incidents: List[ProcessedIncident]) -> List[ExtractedPattern]:
        """
        Extract patterns from processed incidents
        """
        try:
            logger.info(f"Extracting patterns from {len(processed_incidents)} incidents")
            
            all_patterns = []
            
            # Extract different types of patterns
            text_patterns = await self._extract_text_patterns(processed_incidents)
            behavioral_patterns = await self._extract_behavioral_patterns(processed_incidents)
            temporal_patterns = await self._extract_temporal_patterns(processed_incidents)
            structural_patterns = await self._extract_structural_patterns(processed_incidents)
            semantic_patterns = await self._extract_semantic_patterns(processed_incidents)
            
            all_patterns.extend(text_patterns)
            all_patterns.extend(behavioral_patterns)
            all_patterns.extend(temporal_patterns)
            all_patterns.extend(structural_patterns)
            all_patterns.extend(semantic_patterns)
            
            # Update pattern frequency
            for pattern in all_patterns:
                self.pattern_frequency[pattern.pattern_id] = pattern.frequency
                self.patterns[pattern.pattern_id] = pattern
            
            # Calculate co-occurrence
            await self._calculate_co_occurrence(processed_incidents)
            
            logger.info(f"Extracted {len(all_patterns)} patterns")
            return all_patterns
            
        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
            return []
    
    async def _extract_text_patterns(self, processed_incidents: List[ProcessedIncident]) -> List[ExtractedPattern]:
        """Extract text-based patterns"""
        try:
            patterns = []
            
            # Extract common text patterns
            text_samples = [incident.incident.payload for incident in processed_incidents if incident.incident.payload]
            
            if not text_samples:
                return patterns
            
            # Extract n-gram patterns
            ngram_patterns = await self._extract_ngram_patterns(text_samples)
            patterns.extend(ngram_patterns)
            
            # Extract regex patterns
            regex_patterns = await self._extract_regex_patterns(text_samples)
            patterns.extend(regex_patterns)
            
            # Extract keyword patterns
            keyword_patterns = await self._extract_keyword_patterns(text_samples)
            patterns.extend(keyword_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Text pattern extraction failed: {e}")
            return []
    
    async def _extract_ngram_patterns(self, text_samples: List[str]) -> List[ExtractedPattern]:
        """Extract n-gram patterns"""
        try:
            patterns = []
            
            # Extract bigrams and trigrams
            for n in [2, 3]:
                ngrams = []
                for text in text_samples:
                    words = text.lower().split()
                    for i in range(len(words) - n + 1):
                        ngram = " ".join(words[i:i+n])
                        ngrams.append(ngram)
                
                # Count frequency
                ngram_counts = Counter(ngrams)
                
                # Extract frequent patterns
                for ngram, count in ngram_counts.most_common(20):
                    if count >= 2:  # Minimum frequency threshold
                        pattern_id = f"ngram_{n}_{hashlib.md5(ngram.encode()).hexdigest()[:8]}"
                        
                        pattern = ExtractedPattern(
                            pattern_id=pattern_id,
                            pattern_type=PatternType.TEXT_PATTERN,
                            pattern_content=ngram,
                            frequency=count,
                            confidence=min(count / len(text_samples), 1.0),
                            support=count / len(text_samples),
                            examples=[ngram],
                            metadata={
                                "n": n,
                                "extracted_at": datetime.utcnow().isoformat()
                            }
                        )
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"N-gram pattern extraction failed: {e}")
            return []
    
    async def _extract_regex_patterns(self, text_samples: List[str]) -> List[ExtractedPattern]:
        """Extract regex patterns"""
        try:
            patterns = []
            
            # Define common regex patterns
            regex_patterns = [
                (r'ignore\s+previous\s+instructions', 'ignore_instructions'),
                (r'system\s*:\s*you\s+are\s+now', 'system_roleplay'),
                (r'<!--\s*payload\s*-->', 'html_comment'),
                (r'```\s*\w+\s*\n.*\n```', 'code_block'),
                (r'user\s*:\s*.*\s*assistant\s*:\s*', 'conversation_format'),
                (r'let\'s\s+play\s+a\s+game', 'game_roleplay'),
                (r'imagine\s+you\s+are', 'imagination_roleplay'),
                (r'hypothetical\s+scenario', 'hypothetical_scenario'),
                (r'for\s+research\s+purposes', 'research_purposes'),
                (r'creative\s+writing\s+exercise', 'creative_writing')
            ]
            
            for regex, pattern_name in regex_patterns:
                matches = []
                for text in text_samples:
                    if re.search(regex, text, re.IGNORECASE):
                        matches.append(text)
                
                if matches:
                    pattern_id = f"regex_{pattern_name}_{hashlib.md5(regex.encode()).hexdigest()[:8]}"
                    
                    pattern = ExtractedPattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType.TEXT_PATTERN,
                        pattern_content=regex,
                        frequency=len(matches),
                        confidence=min(len(matches) / len(text_samples), 1.0),
                        support=len(matches) / len(text_samples),
                        examples=matches[:5],  # Limit examples
                        metadata={
                            "regex": regex,
                            "pattern_name": pattern_name,
                            "extracted_at": datetime.utcnow().isoformat()
                        }
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Regex pattern extraction failed: {e}")
            return []
    
    async def _extract_keyword_patterns(self, text_samples: List[str]) -> List[ExtractedPattern]:
        """Extract keyword patterns"""
        try:
            patterns = []
            
            # Define keyword categories
            keyword_categories = {
                "injection_keywords": ["inject", "bypass", "exploit", "hack", "jailbreak"],
                "roleplay_keywords": ["pretend", "act", "roleplay", "imagine", "game"],
                "technical_keywords": ["system", "admin", "root", "sudo", "command"],
                "social_keywords": ["urgent", "important", "help", "please", "need"],
                "coding_keywords": ["code", "script", "function", "program", "debug"]
            }
            
            for category, keywords in keyword_categories.items():
                matches = []
                for text in text_samples:
                    text_lower = text.lower()
                    if any(keyword in text_lower for keyword in keywords):
                        matches.append(text)
                
                if matches:
                    pattern_id = f"keyword_{category}_{hashlib.md5(category.encode()).hexdigest()[:8]}"
                    
                    pattern = ExtractedPattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType.TEXT_PATTERN,
                        pattern_content=f"Keywords: {', '.join(keywords)}",
                        frequency=len(matches),
                        confidence=min(len(matches) / len(text_samples), 1.0),
                        support=len(matches) / len(text_samples),
                        examples=matches[:5],
                        metadata={
                            "category": category,
                            "keywords": keywords,
                            "extracted_at": datetime.utcnow().isoformat()
                        }
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Keyword pattern extraction failed: {e}")
            return []
    
    async def _extract_behavioral_patterns(self, processed_incidents: List[ProcessedIncident]) -> List[ExtractedPattern]:
        """Extract behavioral patterns"""
        try:
            patterns = []
            
            # Group incidents by type
            incidents_by_type = defaultdict(list)
            for incident in processed_incidents:
                incidents_by_type[incident.incident.incident_type.value].append(incident)
            
            # Extract patterns for each type
            for incident_type, incidents in incidents_by_type.items():
                if len(incidents) < 2:
                    continue
                
                # Extract common features
                common_features = await self._find_common_features(incidents)
                
                for feature, value in common_features.items():
                    pattern_id = f"behavioral_{incident_type}_{feature}_{hashlib.md5(str(value).encode()).hexdigest()[:8]}"
                    
                    pattern = ExtractedPattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType.BEHAVIORAL_PATTERN,
                        pattern_content=f"{feature}: {value}",
                        frequency=len(incidents),
                        confidence=min(len(incidents) / len(processed_incidents), 1.0),
                        support=len(incidents) / len(processed_incidents),
                        examples=[f"Incident {inc.incident.incident_id}" for inc in incidents[:5]],
                        metadata={
                            "incident_type": incident_type,
                            "feature": feature,
                            "value": value,
                            "extracted_at": datetime.utcnow().isoformat()
                        }
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Behavioral pattern extraction failed: {e}")
            return []
    
    async def _find_common_features(self, incidents: List[ProcessedIncident]) -> Dict[str, Any]:
        """Find common features across incidents"""
        try:
            common_features = {}
            
            if not incidents:
                return common_features
            
            # Get all feature keys
            all_keys = set()
            for incident in incidents:
                all_keys.update(incident.features.keys())
            
            # Find common values
            for key in all_keys:
                values = [incident.features.get(key) for incident in incidents if key in incident.features]
                if values:
                    # Find most common value
                    value_counts = Counter(values)
                    most_common = value_counts.most_common(1)[0]
                    
                    if most_common[1] > len(incidents) * 0.5:  # More than 50% of incidents
                        common_features[key] = most_common[0]
            
            return common_features
            
        except Exception as e:
            logger.error(f"Common feature finding failed: {e}")
            return {}
    
    async def _extract_temporal_patterns(self, processed_incidents: List[ProcessedIncident]) -> List[ExtractedPattern]:
        """Extract temporal patterns"""
        try:
            patterns = []
            
            # Group incidents by time periods
            hourly_incidents = defaultdict(list)
            daily_incidents = defaultdict(list)
            
            for incident in processed_incidents:
                timestamp = datetime.fromisoformat(incident.incident.timestamp.replace('Z', '+00:00'))
                hour = timestamp.hour
                day = timestamp.weekday()
                
                hourly_incidents[hour].append(incident)
                daily_incidents[day].append(incident)
            
            # Extract hourly patterns
            for hour, incidents in hourly_incidents.items():
                if len(incidents) >= 2:
                    pattern_id = f"temporal_hourly_{hour}_{hashlib.md5(str(hour).encode()).hexdigest()[:8]}"
                    
                    pattern = ExtractedPattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType.TEMPORAL_PATTERN,
                        pattern_content=f"Hour {hour}:00",
                        frequency=len(incidents),
                        confidence=min(len(incidents) / len(processed_incidents), 1.0),
                        support=len(incidents) / len(processed_incidents),
                        examples=[f"Incident {inc.incident.incident_id}" for inc in incidents[:5]],
                        metadata={
                            "hour": hour,
                            "extracted_at": datetime.utcnow().isoformat()
                        }
                    )
                    patterns.append(pattern)
            
            # Extract daily patterns
            for day, incidents in daily_incidents.items():
                if len(incidents) >= 2:
                    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    pattern_id = f"temporal_daily_{day}_{hashlib.md5(str(day).encode()).hexdigest()[:8]}"
                    
                    pattern = ExtractedPattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType.TEMPORAL_PATTERN,
                        pattern_content=f"Day: {day_names[day]}",
                        frequency=len(incidents),
                        confidence=min(len(incidents) / len(processed_incidents), 1.0),
                        support=len(incidents) / len(processed_incidents),
                        examples=[f"Incident {inc.incident.incident_id}" for inc in incidents[:5]],
                        metadata={
                            "day": day,
                            "day_name": day_names[day],
                            "extracted_at": datetime.utcnow().isoformat()
                        }
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Temporal pattern extraction failed: {e}")
            return []
    
    async def _extract_structural_patterns(self, processed_incidents: List[ProcessedIncident]) -> List[ExtractedPattern]:
        """Extract structural patterns"""
        try:
            patterns = []
            
            # Extract payload length patterns
            payload_lengths = [len(inc.incident.payload) for inc in processed_incidents if inc.incident.payload]
            
            if payload_lengths:
                # Group by length ranges
                length_ranges = {
                    "short": (0, 100),
                    "medium": (100, 500),
                    "long": (500, 1000),
                    "very_long": (1000, float('inf'))
                }
                
                for range_name, (min_len, max_len) in length_ranges.items():
                    count = sum(1 for length in payload_lengths if min_len <= length < max_len)
                    
                    if count >= 2:
                        pattern_id = f"structural_length_{range_name}_{hashlib.md5(range_name.encode()).hexdigest()[:8]}"
                        
                        pattern = ExtractedPattern(
                            pattern_id=pattern_id,
                            pattern_type=PatternType.STRUCTURAL_PATTERN,
                            pattern_content=f"Length range: {range_name} ({min_len}-{max_len})",
                            frequency=count,
                            confidence=min(count / len(payload_lengths), 1.0),
                            support=count / len(payload_lengths),
                            examples=[f"Length: {length}" for length in payload_lengths if min_len <= length < max_len][:5],
                            metadata={
                                "range_name": range_name,
                                "min_length": min_len,
                                "max_length": max_len,
                                "extracted_at": datetime.utcnow().isoformat()
                            }
                        )
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Structural pattern extraction failed: {e}")
            return []
    
    async def _extract_semantic_patterns(self, processed_incidents: List[ProcessedIncident]) -> List[ExtractedPattern]:
        """Extract semantic patterns"""
        try:
            patterns = []
            
            # Extract attack vector patterns
            attack_vectors = [inc.incident.attack_vector for inc in processed_incidents if inc.incident.attack_vector]
            
            if attack_vectors:
                vector_counts = Counter(attack_vectors)
                
                for vector, count in vector_counts.most_common(10):
                    if count >= 2:
                        pattern_id = f"semantic_vector_{hashlib.md5(vector.encode()).hexdigest()[:8]}"
                        
                        pattern = ExtractedPattern(
                            pattern_id=pattern_id,
                            pattern_type=PatternType.SEMANTIC_PATTERN,
                            pattern_content=f"Attack vector: {vector}",
                            frequency=count,
                            confidence=min(count / len(attack_vectors), 1.0),
                            support=count / len(attack_vectors),
                            examples=[f"Vector: {vector}" for _ in range(min(count, 5))],
                            metadata={
                                "attack_vector": vector,
                                "extracted_at": datetime.utcnow().isoformat()
                            }
                        )
                        patterns.append(pattern)
            
            # Extract target model patterns
            target_models = [inc.incident.target_model for inc in processed_incidents if inc.incident.target_model != "unknown"]
            
            if target_models:
                model_counts = Counter(target_models)
                
                for model, count in model_counts.most_common(10):
                    if count >= 2:
                        pattern_id = f"semantic_model_{hashlib.md5(model.encode()).hexdigest()[:8]}"
                        
                        pattern = ExtractedPattern(
                            pattern_id=pattern_id,
                            pattern_type=PatternType.SEMANTIC_PATTERN,
                            pattern_content=f"Target model: {model}",
                            frequency=count,
                            confidence=min(count / len(target_models), 1.0),
                            support=count / len(target_models),
                            examples=[f"Model: {model}" for _ in range(min(count, 5))],
                            metadata={
                                "target_model": model,
                                "extracted_at": datetime.utcnow().isoformat()
                            }
                        )
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Semantic pattern extraction failed: {e}")
            return []
    
    async def _calculate_co_occurrence(self, processed_incidents: List[ProcessedIncident]):
        """Calculate pattern co-occurrence"""
        try:
            for incident in processed_incidents:
                patterns = incident.patterns
                for i, pattern1 in enumerate(patterns):
                    for j, pattern2 in enumerate(patterns[i+1:], i+1):
                        key = tuple(sorted([pattern1, pattern2]))
                        self.pattern_co_occurrence[key] += 1
            
        except Exception as e:
            logger.error(f"Co-occurrence calculation failed: {e}")
    
    async def cluster_patterns(self, patterns: List[ExtractedPattern]) -> List[PatternCluster]:
        """
        Cluster similar patterns
        """
        try:
            logger.info(f"Clustering {len(patterns)} patterns")
            
            clusters = []
            
            # Group patterns by type
            patterns_by_type = defaultdict(list)
            for pattern in patterns:
                patterns_by_type[pattern.pattern_type].append(pattern)
            
            # Cluster each type separately
            for pattern_type, type_patterns in patterns_by_type.items():
                if len(type_patterns) < 2:
                    continue
                
                type_clusters = await self._cluster_by_type(pattern_type, type_patterns)
                clusters.extend(type_clusters)
            
            # Store clusters
            for cluster in clusters:
                self.clusters[cluster.cluster_id] = cluster
            
            logger.info(f"Created {len(clusters)} pattern clusters")
            return clusters
            
        except Exception as e:
            logger.error(f"Pattern clustering failed: {e}")
            return []
    
    async def _cluster_by_type(self, pattern_type: PatternType, patterns: List[ExtractedPattern]) -> List[PatternCluster]:
        """Cluster patterns by type"""
        try:
            clusters = []
            
            if pattern_type == PatternType.TEXT_PATTERN:
                # Cluster by similarity
                clusters = await self._cluster_text_patterns(patterns)
            elif pattern_type == PatternType.BEHAVIORAL_PATTERN:
                # Cluster by feature similarity
                clusters = await self._cluster_behavioral_patterns(patterns)
            else:
                # Simple clustering by content similarity
                clusters = await self._cluster_by_content_similarity(patterns)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Type-based clustering failed for {pattern_type}: {e}")
            return []
    
    async def _cluster_text_patterns(self, patterns: List[ExtractedPattern]) -> List[PatternCluster]:
        """Cluster text patterns by similarity"""
        try:
            clusters = []
            
            # Simple clustering based on content similarity
            used_patterns = set()
            
            for i, pattern1 in enumerate(patterns):
                if pattern1.pattern_id in used_patterns:
                    continue
                
                cluster_patterns = [pattern1]
                used_patterns.add(pattern1.pattern_id)
                
                for j, pattern2 in enumerate(patterns[i+1:], i+1):
                    if pattern2.pattern_id in used_patterns:
                        continue
                    
                    # Calculate similarity
                    similarity = self._calculate_text_similarity(pattern1.pattern_content, pattern2.pattern_content)
                    
                    if similarity > 0.7:  # Similarity threshold
                        cluster_patterns.append(pattern2)
                        used_patterns.add(pattern2.pattern_id)
                
                if len(cluster_patterns) > 1:
                    cluster_id = f"text_cluster_{hashlib.md5(str(len(clusters)).encode()).hexdigest()[:8]}"
                    
                    cluster = PatternCluster(
                        cluster_id=cluster_id,
                        patterns=cluster_patterns,
                        centroid=pattern1.pattern_content,
                        size=len(cluster_patterns),
                        cohesion=self._calculate_cluster_cohesion(cluster_patterns),
                        metadata={
                            "pattern_type": pattern_type.value,
                            "created_at": datetime.utcnow().isoformat()
                        }
                    )
                    clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Text pattern clustering failed: {e}")
            return []
    
    async def _cluster_behavioral_patterns(self, patterns: List[ExtractedPattern]) -> List[PatternCluster]:
        """Cluster behavioral patterns"""
        try:
            clusters = []
            
            # Group by incident type
            patterns_by_incident_type = defaultdict(list)
            for pattern in patterns:
                incident_type = pattern.metadata.get("incident_type", "unknown")
                patterns_by_incident_type[incident_type].append(pattern)
            
            for incident_type, type_patterns in patterns_by_incident_type.items():
                if len(type_patterns) > 1:
                    cluster_id = f"behavioral_cluster_{incident_type}_{hashlib.md5(incident_type.encode()).hexdigest()[:8]}"
                    
                    cluster = PatternCluster(
                        cluster_id=cluster_id,
                        patterns=type_patterns,
                        centroid=incident_type,
                        size=len(type_patterns),
                        cohesion=self._calculate_cluster_cohesion(type_patterns),
                        metadata={
                            "pattern_type": pattern_type.value,
                            "incident_type": incident_type,
                            "created_at": datetime.utcnow().isoformat()
                        }
                    )
                    clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Behavioral pattern clustering failed: {e}")
            return []
    
    async def _cluster_by_content_similarity(self, patterns: List[ExtractedPattern]) -> List[PatternCluster]:
        """Cluster patterns by content similarity"""
        try:
            clusters = []
            
            # Simple clustering based on content similarity
            used_patterns = set()
            
            for i, pattern1 in enumerate(patterns):
                if pattern1.pattern_id in used_patterns:
                    continue
                
                cluster_patterns = [pattern1]
                used_patterns.add(pattern1.pattern_id)
                
                for j, pattern2 in enumerate(patterns[i+1:], i+1):
                    if pattern2.pattern_id in used_patterns:
                        continue
                    
                    # Calculate similarity
                    similarity = self._calculate_content_similarity(pattern1, pattern2)
                    
                    if similarity > 0.6:  # Similarity threshold
                        cluster_patterns.append(pattern2)
                        used_patterns.add(pattern2.pattern_id)
                
                if len(cluster_patterns) > 1:
                    cluster_id = f"content_cluster_{hashlib.md5(str(len(clusters)).encode()).hexdigest()[:8]}"
                    
                    cluster = PatternCluster(
                        cluster_id=cluster_id,
                        patterns=cluster_patterns,
                        centroid=pattern1.pattern_content,
                        size=len(cluster_patterns),
                        cohesion=self._calculate_cluster_cohesion(cluster_patterns),
                        metadata={
                            "pattern_type": pattern_type.value,
                            "created_at": datetime.utcnow().isoformat()
                        }
                    )
                    clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Content similarity clustering failed: {e}")
            return []
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity"""
        try:
            # Simple Jaccard similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_content_similarity(self, pattern1: ExtractedPattern, pattern2: ExtractedPattern) -> float:
        """Calculate content similarity between patterns"""
        try:
            # Combine multiple similarity measures
            text_sim = self._calculate_text_similarity(pattern1.pattern_content, pattern2.pattern_content)
            
            # Frequency similarity
            freq_sim = 1.0 - abs(pattern1.frequency - pattern2.frequency) / max(pattern1.frequency, pattern2.frequency)
            
            # Confidence similarity
            conf_sim = 1.0 - abs(pattern1.confidence - pattern2.confidence)
            
            # Weighted average
            return (text_sim * 0.5 + freq_sim * 0.3 + conf_sim * 0.2)
            
        except Exception:
            return 0.0
    
    def _calculate_cluster_cohesion(self, patterns: List[ExtractedPattern]) -> float:
        """Calculate cluster cohesion"""
        try:
            if len(patterns) < 2:
                return 1.0
            
            similarities = []
            for i in range(len(patterns)):
                for j in range(i + 1, len(patterns)):
                    sim = self._calculate_content_similarity(patterns[i], patterns[j])
                    similarities.append(sim)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception:
            return 0.0
    
    async def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get pattern extraction statistics"""
        try:
            return {
                "total_patterns": len(self.patterns),
                "total_clusters": len(self.clusters),
                "pattern_types": {pt.value: len([p for p in self.patterns.values() if p.pattern_type == pt]) for pt in PatternType},
                "pattern_frequency_distribution": dict(Counter(self.pattern_frequency.values())),
                "cluster_size_distribution": {cluster_id: cluster.size for cluster_id, cluster in self.clusters.items()},
                "co_occurrence_pairs": len(self.pattern_co_occurrence)
            }
            
        except Exception as e:
            logger.error(f"Pattern statistics retrieval failed: {e}")
            return {}
    
    async def export_patterns(self, format: str = "json") -> str:
        """Export extracted patterns"""
        try:
            if format.lower() == "json":
                data = {
                    "patterns": {k: v.__dict__ for k, v in self.patterns.items()},
                    "clusters": {k: v.__dict__ for k, v in self.clusters.items()},
                    "statistics": await self.get_pattern_statistics(),
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Pattern export failed: {e}")
            return ""
