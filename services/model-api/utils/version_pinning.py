"""
Enhanced Model Version Pinning for Production Safety
Ensures explicit version specification and maintains audit trails
"""

import os
import time
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class VersionPin:
    """Represents a pinned model version"""
    model_name: str
    version: str
    environment: Environment
    pinned_at: datetime
    pinned_by: str
    reason: str
    metadata: Dict[str, Any]
    checksum: str

@dataclass
class VersionAccess:
    """Represents a model version access event"""
    model_name: str
    version: str
    environment: Environment
    accessed_at: datetime
    accessed_by: str
    operation: str
    success: bool
    error_message: Optional[str] = None

class VersionPinningManager:
    """Manages model version pinning and access control"""
    
    def __init__(self):
        self.environment = self._detect_environment()
        self.pinned_versions: Dict[str, VersionPin] = {}
        self.access_log: List[VersionAccess] = []
        self.max_access_log_size = 10000
        
    def _detect_environment(self) -> Environment:
        """Detect current environment"""
        env_str = os.getenv("ENVIRONMENT", "development").lower()
        if env_str in ["prod", "production"]:
            return Environment.PRODUCTION
        elif env_str in ["staging", "stage"]:
            return Environment.STAGING
        else:
            return Environment.DEVELOPMENT
    
    def _generate_checksum(self, model_name: str, version: str) -> str:
        """Generate checksum for version pin"""
        content = f"{model_name}:{version}:{self.environment.value}:{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def pin_version(self, model_name: str, version: str, pinned_by: str, 
                   reason: str, metadata: Dict[str, Any] = None) -> VersionPin:
        """Pin a model version for production safety"""
        try:
            # Validate version format
            if not version or version.lower() in ["latest", "none", ""]:
                raise ValueError(f"Invalid version '{version}': Must be explicit version number")
            
            # Check if version is already pinned
            if model_name in self.pinned_versions:
                existing_pin = self.pinned_versions[model_name]
                if existing_pin.version == version:
                    logger.warning(f"⚠️ [VERSION_PINNING] Version {model_name}:{version} already pinned")
                    return existing_pin
                else:
                    logger.info(f"🔄 [VERSION_PINNING] Updating pin for {model_name}: {existing_pin.version} → {version}")
            
            # Create new version pin
            version_pin = VersionPin(
                model_name=model_name,
                version=version,
                environment=self.environment,
                pinned_at=datetime.now(timezone.utc),
                pinned_by=pinned_by,
                reason=reason,
                metadata=metadata or {},
                checksum=self._generate_checksum(model_name, version)
            )
            
            self.pinned_versions[model_name] = version_pin
            
            logger.info(f"✅ [VERSION_PINNING] Pinned {model_name}:{version} in {self.environment.value}")
            logger.info(f"   Pinned by: {pinned_by}")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   Checksum: {version_pin.checksum}")
            
            return version_pin
            
        except Exception as e:
            logger.error(f"❌ [VERSION_PINNING] Failed to pin version {model_name}:{version}: {e}")
            raise
    
    def validate_version_access(self, model_name: str, version: str, 
                              accessed_by: str, operation: str) -> bool:
        """Validate if version access is allowed"""
        try:
            # In development, allow any version
            if self.environment == Environment.DEVELOPMENT:
                self._log_access(model_name, version, accessed_by, operation, True)
                return True
            
            # In staging, warn about unpinned versions but allow
            if self.environment == Environment.STAGING:
                if model_name not in self.pinned_versions:
                    logger.warning(f"⚠️ [VERSION_PINNING] Accessing unpinned model {model_name} in staging")
                self._log_access(model_name, version, accessed_by, operation, True)
                return True
            
            # In production, enforce strict version pinning
            if self.environment == Environment.PRODUCTION:
                if model_name not in self.pinned_versions:
                    error_msg = f"Model {model_name} not pinned for production access"
                    logger.error(f"❌ [VERSION_PINNING] {error_msg}")
                    self._log_access(model_name, version, accessed_by, operation, False, error_msg)
                    return False
                
                pinned_version = self.pinned_versions[model_name]
                if pinned_version.version != version:
                    error_msg = f"Version mismatch: pinned={pinned_version.version}, requested={version}"
                    logger.error(f"❌ [VERSION_PINNING] {error_msg}")
                    self._log_access(model_name, version, accessed_by, operation, False, error_msg)
                    return False
                
                # Log successful access
                self._log_access(model_name, version, accessed_by, operation, True)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ [VERSION_PINNING] Version validation failed: {e}")
            self._log_access(model_name, version, accessed_by, operation, False, str(e))
            return False
    
    def _log_access(self, model_name: str, version: str, accessed_by: str, 
                   operation: str, success: bool, error_message: str = None):
        """Log model version access"""
        try:
            access = VersionAccess(
                model_name=model_name,
                version=version,
                environment=self.environment,
                accessed_at=datetime.now(timezone.utc),
                accessed_by=accessed_by,
                operation=operation,
                success=success,
                error_message=error_message
            )
            
            self.access_log.append(access)
            
            # Trim log if too large
            if len(self.access_log) > self.max_access_log_size:
                self.access_log = self.access_log[-self.max_access_log_size:]
            
            # Log to console
            status = "✅" if success else "❌"
            logger.info(f"{status} [VERSION_ACCESS] {model_name}:{version} by {accessed_by} ({operation})")
            if error_message:
                logger.error(f"   Error: {error_message}")
                
        except Exception as e:
            logger.error(f"❌ [VERSION_PINNING] Failed to log access: {e}")
    
    def get_pinned_version(self, model_name: str) -> Optional[VersionPin]:
        """Get pinned version for a model"""
        return self.pinned_versions.get(model_name)
    
    def get_access_log(self, model_name: str = None, limit: int = 100) -> List[VersionAccess]:
        """Get access log, optionally filtered by model"""
        log = self.access_log
        if model_name:
            log = [access for access in log if access.model_name == model_name]
        return log[-limit:] if limit else log
    
    def get_pinning_summary(self) -> Dict[str, Any]:
        """Get summary of version pinning status"""
        return {
            "environment": self.environment.value,
            "total_pinned_models": len(self.pinned_versions),
            "pinned_models": {
                model_name: {
                    "version": pin.version,
                    "pinned_at": pin.pinned_at.isoformat(),
                    "pinned_by": pin.pinned_by,
                    "reason": pin.reason,
                    "checksum": pin.checksum
                }
                for model_name, pin in self.pinned_versions.items()
            },
            "total_access_events": len(self.access_log),
            "recent_access_events": len([a for a in self.access_log if (datetime.now(timezone.utc) - a.accessed_at).total_seconds() < 3600])
        }
    
    def unpin_version(self, model_name: str, unpinned_by: str) -> bool:
        """Unpin a model version (only in development/staging)"""
        try:
            if self.environment == Environment.PRODUCTION:
                logger.error(f"❌ [VERSION_PINNING] Cannot unpin {model_name} in production")
                return False
            
            if model_name not in self.pinned_versions:
                logger.warning(f"⚠️ [VERSION_PINNING] Model {model_name} not pinned")
                return False
            
            del self.pinned_versions[model_name]
            logger.info(f"✅ [VERSION_PINNING] Unpinned {model_name} by {unpinned_by}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [VERSION_PINNING] Failed to unpin {model_name}: {e}")
            return False

# Global instance
_version_pinning_manager = None

def get_version_pinning_manager() -> VersionPinningManager:
    """Get global version pinning manager instance"""
    global _version_pinning_manager
    if _version_pinning_manager is None:
        _version_pinning_manager = VersionPinningManager()
    return _version_pinning_manager

def validate_model_version(model_name: str, version: str, operation: str = "load") -> bool:
    """Validate model version access (convenience function)"""
    manager = get_version_pinning_manager()
    return manager.validate_version_access(model_name, version, "system", operation)

def pin_model_version(model_name: str, version: str, reason: str, 
                     pinned_by: str = "system", metadata: Dict[str, Any] = None) -> VersionPin:
    """Pin a model version (convenience function)"""
    manager = get_version_pinning_manager()
    return manager.pin_version(model_name, version, pinned_by, reason, metadata)