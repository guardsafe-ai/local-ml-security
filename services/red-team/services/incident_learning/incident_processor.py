"""
Incident Processor
Processes security incidents and extracts relevant information for learning
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class IncidentType(Enum):
    """Types of security incidents"""
    PROMPT_INJECTION = "prompt_injection"
    JAILEBREAK = "jailbreak"
    ADVERSARIAL_ATTACK = "adversarial_attack"
    DATA_POISONING = "data_poisoning"
    MODEL_EXTRACTION = "model_extraction"
    PRIVACY_ATTACK = "privacy_attack"
    BIAS_ATTACK = "bias_attack"
    ROBUSTNESS_ATTACK = "robustness_attack"
    UNKNOWN = "unknown"


class IncidentSeverity(Enum):
    """Incident severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SecurityIncident:
    """Security incident representation"""
    incident_id: str
    incident_type: IncidentType
    severity: IncidentSeverity
    description: str
    attack_vector: str
    payload: str
    target_model: str
    success: bool
    detection_method: str
    mitigation_applied: str
    timestamp: str
    source_ip: str
    user_agent: str
    session_id: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProcessedIncident:
    """Processed incident with extracted features"""
    incident: SecurityIncident
    features: Dict[str, Any]
    patterns: List[str]
    attack_signature: str
    risk_score: float
    confidence: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class IncidentProcessor:
    """
    Incident Processor
    Processes security incidents and extracts relevant information for learning
    """
    
    def __init__(self):
        """Initialize incident processor"""
        self.incident_patterns = self._load_incident_patterns()
        self.attack_signatures = self._load_attack_signatures()
        self.feature_extractors = self._load_feature_extractors()
        
        logger.info("✅ Initialized Incident Processor")
    
    def _load_incident_patterns(self) -> Dict[str, List[str]]:
        """Load incident detection patterns"""
        return {
            "prompt_injection": [
                r"ignore\s+previous\s+instructions",
                r"system\s*:\s*you\s+are\s+now",
                r"<!--\s*payload\s*-->",
                r"```\s*\w+\s*\n.*\n```",
                r"user\s*:\s*.*\s*assistant\s*:\s*"
            ],
            "jailbreak": [
                r"let's\s+play\s+a\s+game",
                r"imagine\s+you\s+are",
                r"hypothetical\s+scenario",
                r"for\s+research\s+purposes",
                r"creative\s+writing\s+exercise"
            ],
            "adversarial_attack": [
                r"classify\s+this\s+text",
                r"analyze\s+this\s+input",
                r"process\s+this\s+data",
                r"evaluate\s+this\s+content",
                r"respond\s+to\s+this\s+query"
            ],
            "data_poisoning": [
                r"training\s+data",
                r"model\s+training",
                r"data\s+augmentation",
                r"backdoor\s+trigger",
                r"poisoned\s+sample"
            ],
            "model_extraction": [
                r"model\s+architecture",
                r"neural\s+network\s+structure",
                r"model\s+parameters",
                r"weight\s+values",
                r"model\s+weights"
            ],
            "privacy_attack": [
                r"personal\s+information",
                r"private\s+data",
                r"confidential\s+information",
                r"data\s+privacy",
                r"privacy\s+violation"
            ],
            "bias_attack": [
                r"discriminatory",
                r"biased\s+content",
                r"stereotypical",
                r"unfair\s+treatment",
                r"discrimination"
            ],
            "robustness_attack": [
                r"adversarial\s+example",
                r"perturbed\s+input",
                r"noise\s+injection",
                r"input\s+manipulation",
                r"robustness\s+test"
            ]
        }
    
    def _load_attack_signatures(self) -> Dict[str, str]:
        """Load attack signatures"""
        return {
            "prompt_injection": "PI",
            "jailbreak": "JB",
            "adversarial_attack": "AA",
            "data_poisoning": "DP",
            "model_extraction": "ME",
            "privacy_attack": "PA",
            "bias_attack": "BA",
            "robustness_attack": "RA"
        }
    
    def _load_feature_extractors(self) -> Dict[str, callable]:
        """Load feature extractors"""
        return {
            "text_length": self._extract_text_length,
            "special_chars": self._extract_special_chars,
            "keyword_density": self._extract_keyword_density,
            "entropy": self._extract_entropy,
            "n_gram_frequency": self._extract_n_gram_frequency,
            "syntax_patterns": self._extract_syntax_patterns,
            "semantic_features": self._extract_semantic_features,
            "temporal_features": self._extract_temporal_features
        }
    
    async def process_incident(self, incident_data: Dict[str, Any]) -> ProcessedIncident:
        """
        Process security incident and extract features
        """
        try:
            logger.info(f"Processing incident: {incident_data.get('incident_id', 'unknown')}")
            
            # Parse incident data
            incident = await self._parse_incident_data(incident_data)
            
            # Extract features
            features = await self._extract_features(incident)
            
            # Extract patterns
            patterns = await self._extract_patterns(incident)
            
            # Generate attack signature
            attack_signature = await self._generate_attack_signature(incident, patterns)
            
            # Calculate risk score
            risk_score = await self._calculate_risk_score(incident, features, patterns)
            
            # Calculate confidence
            confidence = await self._calculate_confidence(incident, features, patterns)
            
            return ProcessedIncident(
                incident=incident,
                features=features,
                patterns=patterns,
                attack_signature=attack_signature,
                risk_score=risk_score,
                confidence=confidence,
                metadata={
                    "processed_at": datetime.utcnow().isoformat(),
                    "processor_version": "1.0.0"
                }
            )
            
        except Exception as e:
            logger.error(f"Incident processing failed: {e}")
            raise
    
    async def _parse_incident_data(self, incident_data: Dict[str, Any]) -> SecurityIncident:
        """Parse incident data into SecurityIncident object"""
        try:
            # Extract required fields
            incident_id = incident_data.get("incident_id", self._generate_incident_id())
            incident_type = IncidentType(incident_data.get("incident_type", "unknown"))
            severity = IncidentSeverity(incident_data.get("severity", "info"))
            description = incident_data.get("description", "")
            attack_vector = incident_data.get("attack_vector", "")
            payload = incident_data.get("payload", "")
            target_model = incident_data.get("target_model", "unknown")
            success = incident_data.get("success", False)
            detection_method = incident_data.get("detection_method", "unknown")
            mitigation_applied = incident_data.get("mitigation_applied", "")
            timestamp = incident_data.get("timestamp", datetime.utcnow().isoformat())
            source_ip = incident_data.get("source_ip", "")
            user_agent = incident_data.get("user_agent", "")
            session_id = incident_data.get("session_id", "")
            
            return SecurityIncident(
                incident_id=incident_id,
                incident_type=incident_type,
                severity=severity,
                description=description,
                attack_vector=attack_vector,
                payload=payload,
                target_model=target_model,
                success=success,
                detection_method=detection_method,
                mitigation_applied=mitigation_applied,
                timestamp=timestamp,
                source_ip=source_ip,
                user_agent=user_agent,
                session_id=session_id,
                metadata=incident_data.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Incident data parsing failed: {e}")
            raise
    
    async def _extract_features(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Extract features from incident"""
        try:
            features = {}
            
            # Extract text features from payload
            text = incident.payload or incident.description
            
            for feature_name, extractor in self.feature_extractors.items():
                try:
                    features[feature_name] = extractor(text)
                except Exception as e:
                    logger.warning(f"Feature extraction failed for {feature_name}: {e}")
                    features[feature_name] = None
            
            # Extract incident-specific features
            features["incident_type"] = incident.incident_type.value
            features["severity"] = incident.severity.value
            features["success"] = incident.success
            features["target_model"] = incident.target_model
            features["detection_method"] = incident.detection_method
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}
    
    async def _extract_patterns(self, incident: SecurityIncident) -> List[str]:
        """Extract patterns from incident"""
        try:
            patterns = []
            text = incident.payload or incident.description
            
            # Check against incident patterns
            for pattern_type, pattern_list in self.incident_patterns.items():
                for pattern in pattern_list:
                    if re.search(pattern, text, re.IGNORECASE):
                        patterns.append(f"{pattern_type}:{pattern}")
            
            # Extract custom patterns
            custom_patterns = await self._extract_custom_patterns(text)
            patterns.extend(custom_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
            return []
    
    async def _extract_custom_patterns(self, text: str) -> List[str]:
        """Extract custom patterns from text"""
        try:
            patterns = []
            
            # Extract URL patterns
            url_patterns = re.findall(r'https?://[^\s]+', text)
            for url in url_patterns:
                patterns.append(f"url:{url}")
            
            # Extract email patterns
            email_patterns = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            for email in email_patterns:
                patterns.append(f"email:{email}")
            
            # Extract IP address patterns
            ip_patterns = re.findall(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', text)
            for ip in ip_patterns:
                patterns.append(f"ip:{ip}")
            
            # Extract code patterns
            code_patterns = re.findall(r'```[\s\S]*?```', text)
            for code in code_patterns:
                patterns.append(f"code_block:{code[:50]}...")
            
            # Extract command patterns
            command_patterns = re.findall(r'\b(?:ls|cat|grep|find|wget|curl|nc|netcat)\b', text)
            for cmd in command_patterns:
                patterns.append(f"command:{cmd}")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Custom pattern extraction failed: {e}")
            return []
    
    async def _generate_attack_signature(self, incident: SecurityIncident, patterns: List[str]) -> str:
        """Generate attack signature"""
        try:
            # Get base signature
            base_sig = self.attack_signatures.get(incident.incident_type.value, "UNK")
            
            # Add pattern-based signature
            pattern_sig = ""
            if patterns:
                pattern_hash = hashlib.md5("|".join(patterns).encode()).hexdigest()[:8]
                pattern_sig = f"_{pattern_hash}"
            
            # Add payload-based signature
            payload_sig = ""
            if incident.payload:
                payload_hash = hashlib.md5(incident.payload.encode()).hexdigest()[:8]
                payload_sig = f"_{payload_hash}"
            
            return f"{base_sig}{pattern_sig}{payload_sig}"
            
        except Exception as e:
            logger.error(f"Attack signature generation failed: {e}")
            return "UNK"
    
    async def _calculate_risk_score(self, incident: SecurityIncident, features: Dict[str, Any], patterns: List[str]) -> float:
        """Calculate risk score for incident"""
        try:
            risk_score = 0.0
            
            # Base risk score from severity
            severity_scores = {
                "critical": 1.0,
                "high": 0.8,
                "medium": 0.6,
                "low": 0.4,
                "info": 0.2
            }
            risk_score += severity_scores.get(incident.severity.value, 0.0) * 0.4
            
            # Pattern-based risk score
            pattern_risk = len(patterns) * 0.1
            risk_score += min(pattern_risk, 0.3)
            
            # Success-based risk score
            if incident.success:
                risk_score += 0.2
            
            # Feature-based risk score
            if features.get("special_chars", 0) > 0.1:
                risk_score += 0.1
            
            if features.get("entropy", 0) > 0.8:
                risk_score += 0.1
            
            # Cap risk score at 1.0
            return min(risk_score, 1.0)
            
        except Exception as e:
            logger.error(f"Risk score calculation failed: {e}")
            return 0.5
    
    async def _calculate_confidence(self, incident: SecurityIncident, features: Dict[str, Any], patterns: List[str]) -> float:
        """Calculate confidence score for incident classification"""
        try:
            confidence = 0.0
            
            # Pattern-based confidence
            if patterns:
                confidence += 0.4
            
            # Feature-based confidence
            if features.get("text_length", 0) > 10:
                confidence += 0.2
            
            if features.get("keyword_density", 0) > 0.1:
                confidence += 0.2
            
            # Detection method confidence
            if incident.detection_method != "unknown":
                confidence += 0.2
            
            # Cap confidence at 1.0
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _generate_incident_id(self) -> str:
        """Generate unique incident ID"""
        return f"inc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8]}"
    
    # Feature extractors
    def _extract_text_length(self, text: str) -> int:
        """Extract text length feature"""
        return len(text) if text else 0
    
    def _extract_special_chars(self, text: str) -> float:
        """Extract special character density"""
        if not text:
            return 0.0
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
        return special_chars / len(text)
    
    def _extract_keyword_density(self, text: str) -> float:
        """Extract keyword density"""
        if not text:
            return 0.0
        keywords = ["inject", "bypass", "exploit", "attack", "hack", "jailbreak", "prompt", "system"]
        keyword_count = sum(1 for keyword in keywords if keyword.lower() in text.lower())
        return keyword_count / len(text.split())
    
    def _extract_entropy(self, text: str) -> float:
        """Extract text entropy"""
        if not text:
            return 0.0
        from collections import Counter
        char_counts = Counter(text)
        total_chars = len(text)
        entropy = -sum((count / total_chars) * (count / total_chars).bit_length() for count in char_counts.values())
        return entropy / 8.0  # Normalize to 0-1
    
    def _extract_n_gram_frequency(self, text: str) -> Dict[str, float]:
        """Extract n-gram frequency"""
        if not text:
            return {}
        words = text.lower().split()
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words)-2)]
        
        from collections import Counter
        bigram_counts = Counter(bigrams)
        trigram_counts = Counter(trigrams)
        
        return {
            "bigram_freq": dict(bigram_counts.most_common(10)),
            "trigram_freq": dict(trigram_counts.most_common(10))
        }
    
    def _extract_syntax_patterns(self, text: str) -> Dict[str, Any]:
        """Extract syntax patterns"""
        if not text:
            return {}
        
        return {
            "sentence_count": len(re.findall(r'[.!?]+', text)),
            "question_count": len(re.findall(r'\?', text)),
            "exclamation_count": len(re.findall(r'!', text)),
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            "digit_ratio": sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        }
    
    def _extract_semantic_features(self, text: str) -> Dict[str, Any]:
        """Extract semantic features"""
        if not text:
            return {}
        
        # Simple semantic features
        return {
            "word_count": len(text.split()),
            "avg_word_length": sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0,
            "unique_word_ratio": len(set(text.lower().split())) / len(text.split()) if text.split() else 0
        }
    
    def _extract_temporal_features(self, text: str) -> Dict[str, Any]:
        """Extract temporal features"""
        if not text:
            return {}
        
        # Extract time-related patterns
        time_patterns = re.findall(r'\b(?:now|today|yesterday|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', text.lower())
        date_patterns = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
        
        return {
            "time_mentions": len(time_patterns),
            "date_mentions": len(date_patterns),
            "temporal_density": (len(time_patterns) + len(date_patterns)) / len(text.split()) if text.split() else 0
        }
    
    async def batch_process_incidents(self, incidents: List[Dict[str, Any]]) -> List[ProcessedIncident]:
        """Process multiple incidents in batch"""
        try:
            logger.info(f"Batch processing {len(incidents)} incidents")
            
            processed_incidents = []
            for incident_data in incidents:
                try:
                    processed = await self.process_incident(incident_data)
                    processed_incidents.append(processed)
                except Exception as e:
                    logger.error(f"Failed to process incident: {e}")
                    continue
            
            logger.info(f"Successfully processed {len(processed_incidents)} incidents")
            return processed_incidents
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return []
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "feature_extractors": len(self.feature_extractors),
            "incident_patterns": sum(len(patterns) for patterns in self.incident_patterns.values()),
            "attack_signatures": len(self.attack_signatures),
            "supported_incident_types": [t.value for t in IncidentType],
            "supported_severity_levels": [s.value for s in IncidentSeverity]
        }
