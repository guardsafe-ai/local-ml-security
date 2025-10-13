"""
Data Privacy Service - Data Anonymization
Advanced data anonymization and PII detection
"""

import logging
import re
import hashlib
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class DataAnonymizer:
    """Advanced data anonymization service"""
    
    def __init__(self):
        # PII patterns for detection
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'url': r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
            'name': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'  # Simple name pattern
        }
        
        # Initialize encryption key
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text"""
        detected_pii = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected_pii[pii_type] = matches
        
        return detected_pii
    
    def anonymize_text(self, text: str, anonymization_level: str = "medium", 
                      preserve_format: bool = True) -> Dict[str, Any]:
        """Anonymize text based on specified level"""
        try:
            # Detect PII
            detected_pii = self.detect_pii(text)
            anonymized_text = text
            anonymization_methods = []
            confidence_scores = {}
            
            for pii_type, matches in detected_pii.items():
                for match in matches:
                    if anonymization_level == "low":
                        # Simple masking
                        if pii_type == "email":
                            masked = self._mask_email(match)
                        elif pii_type == "phone":
                            masked = self._mask_phone(match)
                        else:
                            masked = self._mask_generic(match)
                        anonymized_text = anonymized_text.replace(match, masked)
                        anonymization_methods.append("masking")
                        confidence_scores[pii_type] = 0.7
                        
                    elif anonymization_level == "medium":
                        # Hash-based anonymization
                        hashed = self._hash_value(match)
                        anonymized_text = anonymized_text.replace(match, hashed)
                        anonymization_methods.append("hashing")
                        confidence_scores[pii_type] = 0.9
                        
                    elif anonymization_level == "high":
                        # Encryption-based anonymization
                        encrypted = self._encrypt_value(match)
                        anonymized_text = anonymized_text.replace(match, encrypted)
                        anonymization_methods.append("encryption")
                        confidence_scores[pii_type] = 1.0
            
            return {
                "original_text": text,
                "anonymized_text": anonymized_text,
                "anonymization_method": ", ".join(set(anonymization_methods)),
                "pii_detected": list(detected_pii.keys()),
                "confidence_scores": confidence_scores,
                "created_at": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error anonymizing text: {e}")
            return {
                "original_text": text,
                "anonymized_text": text,
                "anonymization_method": "error",
                "pii_detected": [],
                "confidence_scores": {},
                "created_at": datetime.now()
            }
    
    def _mask_email(self, email: str) -> str:
        """Mask email address"""
        if '@' in email:
            local, domain = email.split('@', 1)
            return f"{local[0]}***@{domain}"
        return "***@***"
    
    def _mask_phone(self, phone: str) -> str:
        """Mask phone number"""
        digits = re.sub(r'\D', '', phone)
        if len(digits) >= 10:
            return f"***-***-{digits[-4:]}"
        return "***-***-****"
    
    def _mask_generic(self, value: str) -> str:
        """Generic masking"""
        if len(value) <= 4:
            return "***"
        return f"{value[0]}***{value[-1]}"
    
    def _hash_value(self, value: str) -> str:
        """Hash value for anonymization"""
        return hashlib.sha256(value.encode()).hexdigest()[:8]
    
    def _encrypt_value(self, value: str) -> str:
        """Encrypt value for anonymization"""
        return self.cipher.encrypt(value.encode()).decode()
    
    def deidentify_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """De-identify structured data"""
        deidentified = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                # Check if value contains PII
                pii_detected = self.detect_pii(value)
                if pii_detected:
                    # Anonymize the value
                    anonymized = self.anonymize_text(value, "medium")
                    deidentified[key] = anonymized["anonymized_text"]
                else:
                    deidentified[key] = value
            elif isinstance(value, dict):
                deidentified[key] = self.deidentify_data(value)
            elif isinstance(value, list):
                deidentified[key] = [
                    self.deidentify_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                deidentified[key] = value
        
        return deidentified
