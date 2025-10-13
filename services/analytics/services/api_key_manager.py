"""
API Key Management Service
Handles API key generation, validation, and management
"""

import logging
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class APIKeyType(Enum):
    """Types of API keys"""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"
    SERVICE = "service"

@dataclass
class APIKey:
    """API Key model"""
    key_id: str
    key_hash: str
    name: str
    description: str
    key_type: APIKeyType
    permissions: List[str]
    created_by: str
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    usage_count: int = 0
    is_active: bool = True
    rate_limit: int = 1000  # requests per hour
    ip_whitelist: List[str] = None

class APIKeyManager:
    """Manages API keys for service authentication"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.api_keys = {}  # In-memory store (should be replaced with DB)
        self.key_cache = {}  # Cache for quick lookups
        
        # Initialize default service keys
        self._create_default_keys()
    
    def _create_default_keys(self):
        """Create default API keys for services"""
        try:
            # Service-to-service communication key
            service_key = self.generate_api_key(
                name="Service Communication",
                description="Default key for inter-service communication",
                key_type=APIKeyType.SERVICE,
                permissions=["*"],  # Full permissions for service keys
                created_by="system",
                expires_at=None  # Never expires
            )
            
            logger.info("✅ [API_KEY] Default service API key created")
            
        except Exception as e:
            logger.error(f"❌ [API_KEY] Failed to create default keys: {e}")
    
    def generate_api_key(self, name: str, description: str, key_type: APIKeyType,
                       permissions: List[str], created_by: str,
                       expires_at: Optional[datetime] = None,
                       rate_limit: int = 1000,
                       ip_whitelist: List[str] = None) -> str:
        """
        Generate a new API key
        
        Args:
            name: Name for the API key
            description: Description of the key's purpose
            key_type: Type of API key
            permissions: List of permissions
            created_by: User who created the key
            expires_at: Optional expiration date
            rate_limit: Rate limit per hour
            ip_whitelist: Optional IP whitelist
            
        Returns:
            The generated API key
        """
        try:
            # Generate unique key ID
            key_id = secrets.token_urlsafe(16)
            
            # Generate API key
            api_key = f"gsk_{secrets.token_urlsafe(32)}"
            
            # Hash the key for storage
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Create API key record
            api_key_record = APIKey(
                key_id=key_id,
                key_hash=key_hash,
                name=name,
                description=description,
                key_type=key_type,
                permissions=permissions,
                created_by=created_by,
                created_at=datetime.now(timezone.utc),
                expires_at=expires_at,
                last_used=None,
                usage_count=0,
                is_active=True,
                rate_limit=rate_limit,
                ip_whitelist=ip_whitelist or []
            )
            
            # Store the key
            self.api_keys[key_id] = api_key_record
            self.key_cache[key_hash] = api_key_record
            
            logger.info(f"✅ [API_KEY] Generated API key: {name} ({key_type.value})")
            
            return api_key
            
        except Exception as e:
            logger.error(f"❌ [API_KEY] Failed to generate API key: {e}")
            raise
    
    def validate_api_key(self, api_key: str, required_permission: str = None,
                        ip_address: str = None) -> Dict[str, Any]:
        """
        Validate an API key
        
        Args:
            api_key: The API key to validate
            required_permission: Required permission to check
            ip_address: Client IP address for whitelist check
            
        Returns:
            Validation result with user info
        """
        try:
            # Hash the provided key
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Look up the key
            api_key_record = self.key_cache.get(key_hash)
            if not api_key_record:
                logger.warning("🚨 [API_KEY] Invalid API key provided")
                raise ValueError("Invalid API key")
            
            # Check if key is active
            if not api_key_record.is_active:
                logger.warning(f"🚨 [API_KEY] Inactive API key used: {api_key_record.name}")
                raise ValueError("API key is inactive")
            
            # Check expiration
            if api_key_record.expires_at and api_key_record.expires_at < datetime.now(timezone.utc):
                logger.warning(f"🚨 [API_KEY] Expired API key used: {api_key_record.name}")
                raise ValueError("API key has expired")
            
            # Check IP whitelist
            if api_key_record.ip_whitelist and ip_address:
                if ip_address not in api_key_record.ip_whitelist:
                    logger.warning(f"🚨 [API_KEY] IP not whitelisted: {ip_address}")
                    raise ValueError("IP address not authorized")
            
            # Check permission
            if required_permission:
                if "*" not in api_key_record.permissions and required_permission not in api_key_record.permissions:
                    logger.warning(f"🚨 [API_KEY] Insufficient permissions: {required_permission}")
                    raise ValueError("Insufficient permissions")
            
            # Update usage statistics
            api_key_record.last_used = datetime.now(timezone.utc)
            api_key_record.usage_count += 1
            
            logger.debug(f"✅ [API_KEY] Valid API key: {api_key_record.name}")
            
            return {
                "key_id": api_key_record.key_id,
                "name": api_key_record.name,
                "key_type": api_key_record.key_type.value,
                "permissions": api_key_record.permissions,
                "created_by": api_key_record.created_by,
                "rate_limit": api_key_record.rate_limit,
                "usage_count": api_key_record.usage_count
            }
            
        except Exception as e:
            logger.error(f"❌ [API_KEY] API key validation failed: {e}")
            raise
    
    def revoke_api_key(self, key_id: str, revoked_by: str) -> bool:
        """
        Revoke an API key
        
        Args:
            key_id: ID of the key to revoke
            revoked_by: User who revoked the key
            
        Returns:
            True if successful
        """
        try:
            api_key_record = self.api_keys.get(key_id)
            if not api_key_record:
                raise ValueError("API key not found")
            
            # Deactivate the key
            api_key_record.is_active = False
            
            # Remove from cache
            if api_key_record.key_hash in self.key_cache:
                del self.key_cache[api_key_record.key_hash]
            
            logger.info(f"✅ [API_KEY] API key revoked: {api_key_record.name} by {revoked_by}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [API_KEY] Failed to revoke API key: {e}")
            return False
    
    def list_api_keys(self, created_by: str = None) -> List[Dict[str, Any]]:
        """
        List API keys
        
        Args:
            created_by: Filter by creator
            
        Returns:
            List of API key information
        """
        try:
            keys = []
            for key_record in self.api_keys.values():
                if created_by and key_record.created_by != created_by:
                    continue
                
                keys.append({
                    "key_id": key_record.key_id,
                    "name": key_record.name,
                    "description": key_record.description,
                    "key_type": key_record.key_type.value,
                    "permissions": key_record.permissions,
                    "created_by": key_record.created_by,
                    "created_at": key_record.created_at.isoformat(),
                    "expires_at": key_record.expires_at.isoformat() if key_record.expires_at else None,
                    "last_used": key_record.last_used.isoformat() if key_record.last_used else None,
                    "usage_count": key_record.usage_count,
                    "is_active": key_record.is_active,
                    "rate_limit": key_record.rate_limit
                })
            
            return keys
            
        except Exception as e:
            logger.error(f"❌ [API_KEY] Failed to list API keys: {e}")
            return []
    
    def get_api_key_stats(self) -> Dict[str, Any]:
        """Get API key statistics"""
        try:
            total_keys = len(self.api_keys)
            active_keys = sum(1 for k in self.api_keys.values() if k.is_active)
            expired_keys = sum(1 for k in self.api_keys.values() 
                             if k.expires_at and k.expires_at < datetime.now(timezone.utc))
            
            key_types = {}
            for key_record in self.api_keys.values():
                key_type = key_record.key_type.value
                key_types[key_type] = key_types.get(key_type, 0) + 1
            
            return {
                "total_keys": total_keys,
                "active_keys": active_keys,
                "expired_keys": expired_keys,
                "key_types": key_types,
                "total_usage": sum(k.usage_count for k in self.api_keys.values())
            }
            
        except Exception as e:
            logger.error(f"❌ [API_KEY] Failed to get API key stats: {e}")
            return {}
    
    def rotate_api_key(self, key_id: str, rotated_by: str) -> str:
        """
        Rotate an API key (generate new key, revoke old one)
        
        Args:
            key_id: ID of the key to rotate
            rotated_by: User who rotated the key
            
        Returns:
            New API key
        """
        try:
            old_key_record = self.api_keys.get(key_id)
            if not old_key_record:
                raise ValueError("API key not found")
            
            # Generate new key with same properties
            new_api_key = self.generate_api_key(
                name=f"{old_key_record.name} (Rotated)",
                description=old_key_record.description,
                key_type=old_key_record.key_type,
                permissions=old_key_record.permissions,
                created_by=rotated_by,
                expires_at=old_key_record.expires_at,
                rate_limit=old_key_record.rate_limit,
                ip_whitelist=old_key_record.ip_whitelist
            )
            
            # Revoke old key
            self.revoke_api_key(key_id, rotated_by)
            
            logger.info(f"✅ [API_KEY] API key rotated: {old_key_record.name}")
            return new_api_key
            
        except Exception as e:
            logger.error(f"❌ [API_KEY] Failed to rotate API key: {e}")
            raise

# Global API key manager
api_key_manager = APIKeyManager()
