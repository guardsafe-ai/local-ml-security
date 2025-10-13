"""
Continuous Learning Service
Implements automated pattern learning and threat intelligence
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import httpx
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import re
import hashlib

logger = logging.getLogger(__name__)


class LearningType(Enum):
    """Types of continuous learning"""
    PATTERN_LEARNING = "pattern_learning"
    THREAT_INTELLIGENCE = "threat_intelligence"
    ANOMALY_DETECTION = "anomaly_detection"
    ADAPTIVE_ATTACKS = "adaptive_attacks"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"


class ThreatLevel(Enum):
    """Threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AttackPattern:
    """Attack pattern data structure"""
    pattern_id: str
    pattern_text: str
    attack_type: str
    success_rate: float
    frequency: int
    last_seen: datetime
    threat_level: ThreatLevel
    features: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class ThreatIntelligence:
    """Threat intelligence data structure"""
    threat_id: str
    title: str
    description: str
    threat_type: str
    severity: ThreatLevel
    source: str
    confidence: float
    indicators: List[str]
    mitigation: List[str]
    first_seen: datetime
    last_updated: datetime


@dataclass
class Anomaly:
    """Anomaly detection data structure"""
    anomaly_id: str
    anomaly_type: str
    description: str
    severity: ThreatLevel
    confidence: float
    features: Dict[str, Any]
    timestamp: datetime
    context: Dict[str, Any]


class ContinuousLearningService:
    """
    Service for continuous learning and threat intelligence
    """
    
    def __init__(self):
        """Initialize continuous learning service"""
        self.attack_patterns: Dict[str, AttackPattern] = {}
        self.threat_intelligence: Dict[str, ThreatIntelligence] = {}
        self.anomalies: List[Anomaly] = []
        self.learning_models: Dict[str, Any] = {}
        self.pattern_clusters: Dict[str, List[str]] = {}
        self.behavioral_profiles: Dict[str, Dict[str, Any]] = {}
        self.http_client: Optional[httpx.AsyncClient] = None
        self._setup_learning_models()
        logger.info("✅ Continuous Learning Service initialized")
    
    def _setup_learning_models(self):
        """Setup machine learning models for continuous learning"""
        try:
            # Text vectorizer for pattern analysis
            self.learning_models['text_vectorizer'] = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 3)
            )
            
            # Clustering model for pattern grouping
            self.learning_models['pattern_clusterer'] = DBSCAN(
                eps=0.3,
                min_samples=2,
                metric='cosine'
            )
            
            # Anomaly detection model
            self.learning_models['anomaly_detector'] = KMeans(
                n_clusters=5,
                random_state=42
            )
            
            logger.info("✅ Learning models initialized")
            
        except Exception as e:
            logger.error(f"Learning models setup failed: {e}")
    
    async def initialize(self):
        """Initialize HTTP client and load existing data"""
        try:
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=50)
            )
            
            # Load existing patterns and intelligence
            await self._load_existing_data()
            
            logger.info("✅ Continuous Learning Service initialized")
            
        except Exception as e:
            logger.error(f"Continuous learning initialization failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.http_client:
                await self.http_client.aclose()
            logger.info("✅ Continuous Learning Service cleaned up")
        except Exception as e:
            logger.error(f"Continuous learning cleanup failed: {e}")
    
    async def _load_existing_data(self):
        """Load existing attack patterns and threat intelligence"""
        try:
            # This would typically load from a database or file
            # For now, we'll initialize with empty data
            logger.info("✅ Existing data loaded")
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
    
    async def learn_from_attack_data(self, attack_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Learn from new attack data
        
        Args:
            attack_data: New attack data to learn from
            
        Returns:
            Learning results
        """
        try:
            learning_results = {
                "timestamp": datetime.now().isoformat(),
                "patterns_learned": 0,
                "threats_identified": 0,
                "anomalies_detected": 0,
                "clusters_updated": 0
            }
            
            # Extract attack patterns
            patterns = await self._extract_attack_patterns(attack_data)
            for pattern in patterns:
                await self._update_attack_pattern(pattern)
                learning_results["patterns_learned"] += 1
            
            # Identify threats
            threats = await self._identify_threats(attack_data)
            for threat in threats:
                await self._update_threat_intelligence(threat)
                learning_results["threats_identified"] += 1
            
            # Detect anomalies
            anomalies = await self._detect_anomalies(attack_data)
            for anomaly in anomalies:
                self.anomalies.append(anomaly)
                learning_results["anomalies_detected"] += 1
            
            # Update pattern clusters
            await self._update_pattern_clusters()
            learning_results["clusters_updated"] = len(self.pattern_clusters)
            
            # Update behavioral profiles
            await self._update_behavioral_profiles(attack_data)
            
            logger.info(f"✅ Learned from attack data: {learning_results}")
            return learning_results
            
        except Exception as e:
            logger.error(f"Learning from attack data failed: {e}")
            return {"error": str(e)}
    
    async def _extract_attack_patterns(self, attack_data: Dict[str, Any]) -> List[AttackPattern]:
        """Extract attack patterns from attack data"""
        try:
            patterns = []
            
            # Extract text patterns
            attack_texts = attack_data.get('attack_texts', [])
            for i, text in enumerate(attack_texts):
                pattern_id = f"pattern_{hashlib.md5(text.encode()).hexdigest()[:8]}"
                
                # Extract features
                features = await self._extract_pattern_features(text)
                
                # Determine attack type
                attack_type = await self._classify_attack_type(text, features)
                
                # Calculate success rate (mock for now)
                success_rate = attack_data.get('success_rate', 0.5)
                
                pattern = AttackPattern(
                    pattern_id=pattern_id,
                    pattern_text=text,
                    attack_type=attack_type,
                    success_rate=success_rate,
                    frequency=1,
                    last_seen=datetime.now(),
                    threat_level=await self._assess_threat_level(features, success_rate),
                    features=features,
                    metadata={
                        'source': attack_data.get('source', 'unknown'),
                        'model_name': attack_data.get('model_name', 'unknown'),
                        'timestamp': datetime.now().isoformat()
                    }
                )
                
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
            return []
    
    async def _extract_pattern_features(self, text: str) -> Dict[str, Any]:
        """Extract features from attack pattern text"""
        try:
            features = {
                'length': len(text),
                'word_count': len(text.split()),
                'special_chars': len(re.findall(r'[^a-zA-Z0-9\s]', text)),
                'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
                'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
                'has_injection_keywords': any(keyword in text.lower() for keyword in [
                    'inject', 'payload', 'script', 'alert', 'prompt', 'jailbreak'
                ]),
                'has_encoding': any(encoding in text.lower() for encoding in [
                    'base64', 'url', 'hex', 'unicode'
                ]),
                'has_sql_keywords': any(keyword in text.lower() for keyword in [
                    'select', 'insert', 'update', 'delete', 'drop', 'union'
                ]),
                'has_xss_keywords': any(keyword in text.lower() for keyword in [
                    'script', 'onclick', 'onload', 'onerror', 'javascript'
                ])
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}
    
    async def _classify_attack_type(self, text: str, features: Dict[str, Any]) -> str:
        """Classify attack type based on text and features"""
        try:
            # Simple rule-based classification
            if features.get('has_injection_keywords', False):
                if features.get('has_sql_keywords', False):
                    return 'sql_injection'
                elif features.get('has_xss_keywords', False):
                    return 'xss'
                else:
                    return 'prompt_injection'
            elif 'jailbreak' in text.lower():
                return 'jailbreak'
            elif 'extract' in text.lower() or 'leak' in text.lower():
                return 'data_extraction'
            elif 'bypass' in text.lower() or 'evade' in text.lower():
                return 'evasion'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.error(f"Attack type classification failed: {e}")
            return 'unknown'
    
    async def _assess_threat_level(self, features: Dict[str, Any], success_rate: float) -> ThreatLevel:
        """Assess threat level based on features and success rate"""
        try:
            threat_score = 0
            
            # Base score from success rate
            threat_score += success_rate * 40
            
            # Feature-based scoring
            if features.get('has_injection_keywords', False):
                threat_score += 20
            if features.get('has_encoding', False):
                threat_score += 15
            if features.get('special_chars', 0) > 10:
                threat_score += 10
            if features.get('length', 0) > 100:
                threat_score += 5
            
            # Determine threat level
            if threat_score >= 80:
                return ThreatLevel.CRITICAL
            elif threat_score >= 60:
                return ThreatLevel.HIGH
            elif threat_score >= 40:
                return ThreatLevel.MEDIUM
            else:
                return ThreatLevel.LOW
                
        except Exception as e:
            logger.error(f"Threat level assessment failed: {e}")
            return ThreatLevel.LOW
    
    async def _update_attack_pattern(self, pattern: AttackPattern):
        """Update or create attack pattern"""
        try:
            if pattern.pattern_id in self.attack_patterns:
                # Update existing pattern
                existing = self.attack_patterns[pattern.pattern_id]
                existing.frequency += 1
                existing.last_seen = pattern.last_seen
                existing.success_rate = (existing.success_rate + pattern.success_rate) / 2
                existing.threat_level = pattern.threat_level
                existing.metadata.update(pattern.metadata)
            else:
                # Create new pattern
                self.attack_patterns[pattern.pattern_id] = pattern
            
            logger.debug(f"Updated attack pattern: {pattern.pattern_id}")
            
        except Exception as e:
            logger.error(f"Attack pattern update failed: {e}")
    
    async def _identify_threats(self, attack_data: Dict[str, Any]) -> List[ThreatIntelligence]:
        """Identify threats from attack data"""
        try:
            threats = []
            
            # Analyze attack patterns for new threats
            attack_texts = attack_data.get('attack_texts', [])
            for text in attack_texts:
                # Check for known threat indicators
                threat_indicators = await self._analyze_threat_indicators(text)
                
                if threat_indicators:
                    threat_id = f"threat_{hashlib.md5(text.encode()).hexdigest()[:8]}"
                    
                    threat = ThreatIntelligence(
                        threat_id=threat_id,
                        title=f"New threat pattern detected",
                        description=f"Threat pattern: {text[:100]}...",
                        threat_type=threat_indicators.get('type', 'unknown'),
                        severity=threat_indicators.get('severity', ThreatLevel.MEDIUM),
                        source='continuous_learning',
                        confidence=threat_indicators.get('confidence', 0.5),
                        indicators=threat_indicators.get('indicators', []),
                        mitigation=threat_indicators.get('mitigation', []),
                        first_seen=datetime.now(),
                        last_updated=datetime.now()
                    )
                    
                    threats.append(threat)
            
            return threats
            
        except Exception as e:
            logger.error(f"Threat identification failed: {e}")
            return []
    
    async def _analyze_threat_indicators(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyze text for threat indicators"""
        try:
            indicators = []
            threat_type = 'unknown'
            severity = ThreatLevel.LOW
            confidence = 0.0
            
            # Check for various threat indicators
            if re.search(r'(?i)(malware|virus|trojan|backdoor)', text):
                indicators.append('malware_reference')
                threat_type = 'malware'
                severity = ThreatLevel.HIGH
                confidence += 0.3
            
            if re.search(r'(?i)(exploit|vulnerability|cve)', text):
                indicators.append('exploit_reference')
                threat_type = 'exploit'
                severity = ThreatLevel.HIGH
                confidence += 0.4
            
            if re.search(r'(?i)(phishing|scam|fraud)', text):
                indicators.append('phishing_reference')
                threat_type = 'phishing'
                severity = ThreatLevel.MEDIUM
                confidence += 0.2
            
            if re.search(r'(?i)(ddos|flood|overload)', text):
                indicators.append('ddos_reference')
                threat_type = 'ddos'
                severity = ThreatLevel.MEDIUM
                confidence += 0.2
            
            if indicators:
                return {
                    'type': threat_type,
                    'severity': severity,
                    'confidence': min(confidence, 1.0),
                    'indicators': indicators,
                    'mitigation': await self._get_mitigation_strategies(threat_type)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Threat indicator analysis failed: {e}")
            return None
    
    async def _get_mitigation_strategies(self, threat_type: str) -> List[str]:
        """Get mitigation strategies for threat type"""
        try:
            mitigation_strategies = {
                'malware': [
                    'Implement endpoint detection and response (EDR)',
                    'Use sandboxing for suspicious content',
                    'Regular security awareness training'
                ],
                'exploit': [
                    'Keep systems and software updated',
                    'Implement vulnerability scanning',
                    'Use application security testing'
                ],
                'phishing': [
                    'Implement email security filters',
                    'Conduct phishing simulations',
                    'User awareness training'
                ],
                'ddos': [
                    'Implement DDoS protection services',
                    'Use rate limiting and traffic shaping',
                    'Distributed infrastructure design'
                ]
            }
            
            return mitigation_strategies.get(threat_type, [
                'Implement general security controls',
                'Monitor for suspicious activity',
                'Regular security assessments'
            ])
            
        except Exception as e:
            logger.error(f"Mitigation strategies retrieval failed: {e}")
            return ['Implement general security controls']
    
    async def _update_threat_intelligence(self, threat: ThreatIntelligence):
        """Update or create threat intelligence"""
        try:
            if threat.threat_id in self.threat_intelligence:
                # Update existing threat
                existing = self.threat_intelligence[threat.threat_id]
                existing.last_updated = threat.last_updated
                existing.confidence = max(existing.confidence, threat.confidence)
                existing.indicators.extend(threat.indicators)
                existing.mitigation.extend(threat.mitigation)
            else:
                # Create new threat
                self.threat_intelligence[threat.threat_id] = threat
            
            logger.debug(f"Updated threat intelligence: {threat.threat_id}")
            
        except Exception as e:
            logger.error(f"Threat intelligence update failed: {e}")
    
    async def _detect_anomalies(self, attack_data: Dict[str, Any]) -> List[Anomaly]:
        """Detect anomalies in attack data"""
        try:
            anomalies = []
            
            # Analyze attack frequency
            attack_count = attack_data.get('attack_count', 0)
            if attack_count > 100:  # Threshold for high attack frequency
                anomaly = Anomaly(
                    anomaly_id=f"anomaly_freq_{datetime.now().timestamp()}",
                    anomaly_type='high_frequency',
                    description=f'Unusually high attack frequency: {attack_count}',
                    severity=ThreatLevel.HIGH,
                    confidence=0.8,
                    features={'attack_count': attack_count},
                    timestamp=datetime.now(),
                    context={'source': attack_data.get('source', 'unknown')}
                )
                anomalies.append(anomaly)
            
            # Analyze success rate
            success_rate = attack_data.get('success_rate', 0)
            if success_rate > 0.8:  # Threshold for high success rate
                anomaly = Anomaly(
                    anomaly_id=f"anomaly_success_{datetime.now().timestamp()}",
                    anomaly_type='high_success_rate',
                    description=f'Unusually high success rate: {success_rate:.2f}',
                    severity=ThreatLevel.MEDIUM,
                    confidence=0.7,
                    features={'success_rate': success_rate},
                    timestamp=datetime.now(),
                    context={'source': attack_data.get('source', 'unknown')}
                )
                anomalies.append(anomaly)
            
            # Analyze attack patterns
            attack_texts = attack_data.get('attack_texts', [])
            if len(attack_texts) > 0:
                # Check for unusual patterns
                unusual_patterns = await self._detect_unusual_patterns(attack_texts)
                for pattern in unusual_patterns:
                    anomaly = Anomaly(
                        anomaly_id=f"anomaly_pattern_{datetime.now().timestamp()}",
                        anomaly_type='unusual_pattern',
                        description=f'Unusual attack pattern detected: {pattern}',
                        severity=ThreatLevel.MEDIUM,
                        confidence=0.6,
                        features={'pattern': pattern},
                        timestamp=datetime.now(),
                        context={'source': attack_data.get('source', 'unknown')}
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []
    
    async def _detect_unusual_patterns(self, attack_texts: List[str]) -> List[str]:
        """Detect unusual patterns in attack texts"""
        try:
            unusual_patterns = []
            
            for text in attack_texts:
                # Check for very long texts
                if len(text) > 1000:
                    unusual_patterns.append('very_long_text')
                
                # Check for high special character ratio
                special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
                if special_chars / len(text) > 0.3:
                    unusual_patterns.append('high_special_char_ratio')
                
                # Check for unusual encoding patterns
                if re.search(r'%[0-9A-Fa-f]{2}', text):
                    unusual_patterns.append('url_encoding')
                
                if re.search(r'\\x[0-9A-Fa-f]{2}', text):
                    unusual_patterns.append('hex_encoding')
            
            return list(set(unusual_patterns))
            
        except Exception as e:
            logger.error(f"Unusual pattern detection failed: {e}")
            return []
    
    async def _update_pattern_clusters(self):
        """Update pattern clusters using clustering algorithms"""
        try:
            if not self.attack_patterns:
                return
            
            # Prepare data for clustering
            texts = [pattern.pattern_text for pattern in self.attack_patterns.values()]
            pattern_ids = list(self.attack_patterns.keys())
            
            if len(texts) < 2:
                return
            
            # Vectorize texts
            try:
                vectors = self.learning_models['text_vectorizer'].fit_transform(texts)
            except:
                # If vectorizer not fitted yet, fit it
                self.learning_models['text_vectorizer'].fit(texts)
                vectors = self.learning_models['text_vectorizer'].transform(texts)
            
            # Perform clustering
            cluster_labels = self.learning_models['pattern_clusterer'].fit_predict(vectors)
            
            # Update clusters
            self.pattern_clusters = defaultdict(list)
            for pattern_id, cluster_label in zip(pattern_ids, cluster_labels):
                if cluster_label != -1:  # -1 means noise in DBSCAN
                    self.pattern_clusters[f"cluster_{cluster_label}"].append(pattern_id)
            
            logger.info(f"✅ Updated pattern clusters: {len(self.pattern_clusters)} clusters")
            
        except Exception as e:
            logger.error(f"Pattern clustering failed: {e}")
    
    async def _update_behavioral_profiles(self, attack_data: Dict[str, Any]):
        """Update behavioral profiles based on attack data"""
        try:
            source = attack_data.get('source', 'unknown')
            
            if source not in self.behavioral_profiles:
                self.behavioral_profiles[source] = {
                    'attack_count': 0,
                    'success_rate': 0.0,
                    'attack_types': Counter(),
                    'threat_levels': Counter(),
                    'first_seen': datetime.now(),
                    'last_seen': datetime.now()
                }
            
            profile = self.behavioral_profiles[source]
            profile['attack_count'] += 1
            profile['last_seen'] = datetime.now()
            
            # Update success rate
            current_success = attack_data.get('success_rate', 0)
            profile['success_rate'] = (profile['success_rate'] + current_success) / 2
            
            # Update attack types
            attack_types = attack_data.get('attack_types', [])
            for attack_type in attack_types:
                profile['attack_types'][attack_type] += 1
            
            # Update threat levels
            threat_levels = attack_data.get('threat_levels', [])
            for threat_level in threat_levels:
                profile['threat_levels'][threat_level] += 1
            
            logger.debug(f"Updated behavioral profile for {source}")
            
        except Exception as e:
            logger.error(f"Behavioral profile update failed: {e}")
    
    async def get_learned_patterns(self, attack_type: str = None, threat_level: ThreatLevel = None) -> List[Dict[str, Any]]:
        """Get learned attack patterns with optional filtering"""
        try:
            patterns = []
            
            for pattern in self.attack_patterns.values():
                if attack_type and pattern.attack_type != attack_type:
                    continue
                if threat_level and pattern.threat_level != threat_level:
                    continue
                
                patterns.append({
                    'pattern_id': pattern.pattern_id,
                    'pattern_text': pattern.pattern_text,
                    'attack_type': pattern.attack_type,
                    'success_rate': pattern.success_rate,
                    'frequency': pattern.frequency,
                    'last_seen': pattern.last_seen.isoformat(),
                    'threat_level': pattern.threat_level.value,
                    'features': pattern.features,
                    'metadata': pattern.metadata
                })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Learned patterns retrieval failed: {e}")
            return []
    
    async def get_threat_intelligence(self, threat_type: str = None, severity: ThreatLevel = None) -> List[Dict[str, Any]]:
        """Get threat intelligence with optional filtering"""
        try:
            threats = []
            
            for threat in self.threat_intelligence.values():
                if threat_type and threat.threat_type != threat_type:
                    continue
                if severity and threat.severity != severity:
                    continue
                
                threats.append({
                    'threat_id': threat.threat_id,
                    'title': threat.title,
                    'description': threat.description,
                    'threat_type': threat.threat_type,
                    'severity': threat.severity.value,
                    'source': threat.source,
                    'confidence': threat.confidence,
                    'indicators': threat.indicators,
                    'mitigation': threat.mitigation,
                    'first_seen': threat.first_seen.isoformat(),
                    'last_updated': threat.last_updated.isoformat()
                })
            
            return threats
            
        except Exception as e:
            logger.error(f"Threat intelligence retrieval failed: {e}")
            return []
    
    async def get_anomalies(self, anomaly_type: str = None, severity: ThreatLevel = None) -> List[Dict[str, Any]]:
        """Get detected anomalies with optional filtering"""
        try:
            anomalies = []
            
            for anomaly in self.anomalies:
                if anomaly_type and anomaly.anomaly_type != anomaly_type:
                    continue
                if severity and anomaly.severity != severity:
                    continue
                
                anomalies.append({
                    'anomaly_id': anomaly.anomaly_id,
                    'anomaly_type': anomaly.anomaly_type,
                    'description': anomaly.description,
                    'severity': anomaly.severity.value,
                    'confidence': anomaly.confidence,
                    'features': anomaly.features,
                    'timestamp': anomaly.timestamp.isoformat(),
                    'context': anomaly.context
                })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomalies retrieval failed: {e}")
            return []
    
    async def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning summary"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_patterns': len(self.attack_patterns),
                'total_threats': len(self.threat_intelligence),
                'total_anomalies': len(self.anomalies),
                'pattern_clusters': len(self.pattern_clusters),
                'behavioral_profiles': len(self.behavioral_profiles),
                'pattern_distribution': {},
                'threat_distribution': {},
                'anomaly_distribution': {},
                'top_attack_types': [],
                'top_threat_types': [],
                'recent_anomalies': []
            }
            
            # Pattern distribution
            attack_types = Counter(pattern.attack_type for pattern in self.attack_patterns.values())
            summary['pattern_distribution'] = dict(attack_types)
            summary['top_attack_types'] = attack_types.most_common(5)
            
            # Threat distribution
            threat_types = Counter(threat.threat_type for threat in self.threat_intelligence.values())
            summary['threat_distribution'] = dict(threat_types)
            summary['top_threat_types'] = threat_types.most_common(5)
            
            # Anomaly distribution
            anomaly_types = Counter(anomaly.anomaly_type for anomaly in self.anomalies)
            summary['anomaly_distribution'] = dict(anomaly_types)
            
            # Recent anomalies (last 24 hours)
            recent_time = datetime.now() - timedelta(hours=24)
            recent_anomalies = [a for a in self.anomalies if a.timestamp > recent_time]
            summary['recent_anomalies'] = [
                {
                    'anomaly_id': a.anomaly_id,
                    'anomaly_type': a.anomaly_type,
                    'severity': a.severity.value,
                    'timestamp': a.timestamp.isoformat()
                }
                for a in recent_anomalies
            ]
            
            return summary
            
        except Exception as e:
            logger.error(f"Learning summary generation failed: {e}")
            return {"error": str(e)}
    
    async def generate_adaptive_attacks(self, target_model: str, attack_count: int = 10) -> List[Dict[str, Any]]:
        """Generate adaptive attacks based on learned patterns"""
        try:
            adaptive_attacks = []
            
            # Get relevant patterns for the target model
            relevant_patterns = [
                pattern for pattern in self.attack_patterns.values()
                if pattern.metadata.get('model_name') == target_model or pattern.metadata.get('model_name') == 'unknown'
            ]
            
            if not relevant_patterns:
                # Use general patterns if no model-specific patterns
                relevant_patterns = list(self.attack_patterns.values())
            
            # Generate adaptive attacks
            for i in range(attack_count):
                if not relevant_patterns:
                    break
                
                # Select a base pattern
                base_pattern = relevant_patterns[i % len(relevant_patterns)]
                
                # Apply adaptations
                adapted_text = await self._adapt_attack_pattern(base_pattern.pattern_text)
                
                adaptive_attack = {
                    'attack_id': f"adaptive_{i}_{datetime.now().timestamp()}",
                    'base_pattern_id': base_pattern.pattern_id,
                    'adapted_text': adapted_text,
                    'attack_type': base_pattern.attack_type,
                    'adaptation_method': 'pattern_evolution',
                    'confidence': base_pattern.success_rate,
                    'target_model': target_model,
                    'generated_at': datetime.now().isoformat()
                }
                
                adaptive_attacks.append(adaptive_attack)
            
            return adaptive_attacks
            
        except Exception as e:
            logger.error(f"Adaptive attack generation failed: {e}")
            return []
    
    async def _adapt_attack_pattern(self, base_text: str) -> str:
        """Adapt an attack pattern to create variations"""
        try:
            adapted_text = base_text
            
            # Apply various adaptations
            adaptations = [
                self._add_noise,
                self._change_encoding,
                self._add_synonyms,
                self._modify_structure
            ]
            
            # Randomly apply 1-2 adaptations
            import random
            num_adaptations = random.randint(1, 2)
            selected_adaptations = random.sample(adaptations, num_adaptations)
            
            for adaptation in selected_adaptations:
                adapted_text = adaptation(adapted_text)
            
            return adapted_text
            
        except Exception as e:
            logger.error(f"Attack pattern adaptation failed: {e}")
            return base_text
    
    def _add_noise(self, text: str) -> str:
        """Add noise to text"""
        import random
        noise_chars = [' ', '\t', '\n', '\r']
        if len(text) > 10:
            pos = random.randint(0, len(text) - 1)
            text = text[:pos] + random.choice(noise_chars) + text[pos:]
        return text
    
    def _change_encoding(self, text: str) -> str:
        """Change encoding of text"""
        import random
        if random.random() < 0.3:  # 30% chance
            # URL encode some characters
            text = text.replace(' ', '%20')
            text = text.replace('=', '%3D')
        return text
    
    def _add_synonyms(self, text: str) -> str:
        """Add synonyms to text"""
        synonyms = {
            'inject': ['insert', 'inject', 'embed'],
            'script': ['code', 'script', 'program'],
            'alert': ['warning', 'alert', 'notification']
        }
        
        for word, syns in synonyms.items():
            if word in text.lower():
                import random
                text = text.replace(word, random.choice(syns))
        
        return text
    
    def _modify_structure(self, text: str) -> str:
        """Modify text structure"""
        import random
        if random.random() < 0.4:  # 40% chance
            # Add extra characters
            text = text + ';'
        return text
