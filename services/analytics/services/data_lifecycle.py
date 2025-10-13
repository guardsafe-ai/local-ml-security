"""
Data Lifecycle Management Service
Implements data lifecycle policies and archival
"""

import logging
import os
import shutil
import gzip
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class DataClass(Enum):
    """Data classification levels"""
    HOT = "hot"          # Frequently accessed, keep on fast storage
    WARM = "warm"        # Occasionally accessed, can be compressed
    COLD = "cold"        # Rarely accessed, archive to cheap storage
    ARCHIVED = "archived" # Very rarely accessed, compressed archive

class LifecycleAction(Enum):
    """Lifecycle actions"""
    COMPRESS = "compress"
    ARCHIVE = "archive"
    DELETE = "delete"
    MOVE_TO_COLD = "move_to_cold"

@dataclass
class LifecyclePolicy:
    """Data lifecycle policy"""
    name: str
    data_class: DataClass
    retention_days: int
    actions: List[LifecycleAction]
    conditions: Dict[str, Any]
    enabled: bool = True

@dataclass
class DataFile:
    """Data file metadata"""
    file_path: str
    file_name: str
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int
    data_class: DataClass
    checksum: str
    compressed: bool = False
    archived: bool = False

class DataLifecycleManager:
    """Manages data lifecycle policies and archival"""
    
    def __init__(self, base_path: str = "/data", archive_path: str = "/archive"):
        self.base_path = base_path
        self.archive_path = archive_path
        self.policies = self._initialize_default_policies()
        self.data_files = {}
        
        # Ensure directories exist
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.archive_path, exist_ok=True)
        
        # Create subdirectories
        for data_class in DataClass:
            os.makedirs(os.path.join(self.base_path, data_class.value), exist_ok=True)
    
    def _initialize_default_policies(self) -> Dict[str, LifecyclePolicy]:
        """Initialize default lifecycle policies"""
        return {
            "training_data": LifecyclePolicy(
                name="Training Data",
                data_class=DataClass.HOT,
                retention_days=90,
                actions=[LifecycleAction.COMPRESS, LifecycleAction.MOVE_TO_COLD],
                conditions={"min_size_mb": 100, "access_threshold": 5}
            ),
            "prediction_logs": LifecyclePolicy(
                name="Prediction Logs",
                data_class=DataClass.WARM,
                retention_days=30,
                actions=[LifecycleAction.COMPRESS, LifecycleAction.ARCHIVE],
                conditions={"min_size_mb": 50, "access_threshold": 2}
            ),
            "model_artifacts": LifecyclePolicy(
                name="Model Artifacts",
                data_class=DataClass.HOT,
                retention_days=365,
                actions=[LifecycleAction.COMPRESS],
                conditions={"min_size_mb": 200}
            ),
            "audit_logs": LifecyclePolicy(
                name="Audit Logs",
                data_class=DataClass.COLD,
                retention_days=2555,  # 7 years for compliance
                actions=[LifecycleAction.COMPRESS, LifecycleAction.ARCHIVE],
                conditions={"min_size_mb": 10}
            ),
            "temp_files": LifecyclePolicy(
                name="Temporary Files",
                data_class=DataClass.HOT,
                retention_days=7,
                actions=[LifecycleAction.DELETE],
                conditions={"min_age_hours": 24}
            )
        }
    
    async def scan_data_files(self) -> List[DataFile]:
        """Scan and catalog all data files"""
        try:
            data_files = []
            
            for root, dirs, files in os.walk(self.base_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Get file stats
                    stat = os.stat(file_path)
                    size_bytes = stat.st_size
                    created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
                    last_accessed = datetime.fromtimestamp(stat.st_atime, tz=timezone.utc)
                    
                    # Determine data class from path
                    data_class = self._determine_data_class(file_path)
                    
                    # Calculate checksum
                    checksum = self._calculate_checksum(file_path)
                    
                    data_file = DataFile(
                        file_path=file_path,
                        file_name=file,
                        size_bytes=size_bytes,
                        created_at=created_at,
                        last_accessed=last_accessed,
                        access_count=0,  # Would need to track this separately
                        data_class=data_class,
                        checksum=checksum,
                        compressed=file.endswith('.gz'),
                        archived=False
                    )
                    
                    data_files.append(data_file)
                    self.data_files[file_path] = data_file
            
            logger.info(f"✅ [LIFECYCLE] Scanned {len(data_files)} data files")
            return data_files
            
        except Exception as e:
            logger.error(f"❌ [LIFECYCLE] Failed to scan data files: {e}")
            return []
    
    def _determine_data_class(self, file_path: str) -> DataClass:
        """Determine data class from file path"""
        path_lower = file_path.lower()
        
        if "training" in path_lower or "model" in path_lower:
            return DataClass.HOT
        elif "prediction" in path_lower or "inference" in path_lower:
            return DataClass.WARM
        elif "audit" in path_lower or "log" in path_lower:
            return DataClass.COLD
        elif "temp" in path_lower or "tmp" in path_lower:
            return DataClass.HOT
        else:
            return DataClass.WARM
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        try:
            import hashlib
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"❌ [LIFECYCLE] Failed to calculate checksum: {e}")
            return ""
    
    async def apply_lifecycle_policies(self) -> Dict[str, Any]:
        """Apply lifecycle policies to all data files"""
        try:
            results = {
                "compressed": 0,
                "archived": 0,
                "deleted": 0,
                "moved": 0,
                "errors": 0,
                "space_saved_mb": 0
            }
            
            data_files = await self.scan_data_files()
            current_time = datetime.now(timezone.utc)
            
            for data_file in data_files:
                try:
                    # Find applicable policy
                    policy = self._find_applicable_policy(data_file)
                    if not policy or not policy.enabled:
                        continue
                    
                    # Check if policy conditions are met
                    if not self._check_policy_conditions(data_file, policy, current_time):
                        continue
                    
                    # Apply policy actions
                    for action in policy.actions:
                        if action == LifecycleAction.COMPRESS:
                            if await self._compress_file(data_file):
                                results["compressed"] += 1
                                results["space_saved_mb"] += data_file.size_bytes / (1024 * 1024) * 0.7  # Assume 70% compression
                        
                        elif action == LifecycleAction.ARCHIVE:
                            if await self._archive_file(data_file):
                                results["archived"] += 1
                        
                        elif action == LifecycleAction.DELETE:
                            if await self._delete_file(data_file):
                                results["deleted"] += 1
                                results["space_saved_mb"] += data_file.size_bytes / (1024 * 1024)
                        
                        elif action == LifecycleAction.MOVE_TO_COLD:
                            if await self._move_to_cold_storage(data_file):
                                results["moved"] += 1
                
                except Exception as e:
                    logger.error(f"❌ [LIFECYCLE] Error processing file {data_file.file_path}: {e}")
                    results["errors"] += 1
            
            logger.info(f"✅ [LIFECYCLE] Applied policies: {results}")
            return results
            
        except Exception as e:
            logger.error(f"❌ [LIFECYCLE] Failed to apply lifecycle policies: {e}")
            return {"error": str(e)}
    
    def _find_applicable_policy(self, data_file: DataFile) -> Optional[LifecyclePolicy]:
        """Find applicable policy for data file"""
        file_name_lower = data_file.file_name.lower()
        
        for policy_name, policy in self.policies.items():
            if policy_name in file_name_lower or policy.data_class == data_file.data_class:
                return policy
        
        return None
    
    def _check_policy_conditions(self, data_file: DataFile, policy: LifecyclePolicy, current_time: datetime) -> bool:
        """Check if policy conditions are met"""
        try:
            conditions = policy.conditions
            
            # Check age
            age_days = (current_time - data_file.created_at).days
            if age_days < policy.retention_days:
                return False
            
            # Check size
            min_size_mb = conditions.get("min_size_mb", 0)
            if data_file.size_bytes < min_size_mb * 1024 * 1024:
                return False
            
            # Check access threshold
            access_threshold = conditions.get("access_threshold", 0)
            if data_file.access_count > access_threshold:
                return False
            
            # Check minimum age in hours
            min_age_hours = conditions.get("min_age_hours", 0)
            age_hours = (current_time - data_file.created_at).total_seconds() / 3600
            if age_hours < min_age_hours:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ [LIFECYCLE] Failed to check policy conditions: {e}")
            return False
    
    async def _compress_file(self, data_file: DataFile) -> bool:
        """Compress a data file"""
        try:
            if data_file.compressed:
                return False
            
            compressed_path = data_file.file_path + ".gz"
            
            with open(data_file.file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original file
            os.remove(data_file.file_path)
            
            # Update metadata
            data_file.file_path = compressed_path
            data_file.compressed = True
            
            logger.info(f"✅ [LIFECYCLE] Compressed file: {data_file.file_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [LIFECYCLE] Failed to compress file {data_file.file_path}: {e}")
            return False
    
    async def _archive_file(self, data_file: DataFile) -> bool:
        """Archive a data file"""
        try:
            if data_file.archived:
                return False
            
            # Create archive path
            archive_file_path = os.path.join(self.archive_path, data_file.file_name)
            
            # Move file to archive
            shutil.move(data_file.file_path, archive_file_path)
            
            # Update metadata
            data_file.file_path = archive_file_path
            data_file.archived = True
            
            logger.info(f"✅ [LIFECYCLE] Archived file: {data_file.file_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [LIFECYCLE] Failed to archive file {data_file.file_path}: {e}")
            return False
    
    async def _delete_file(self, data_file: DataFile) -> bool:
        """Delete a data file"""
        try:
            os.remove(data_file.file_path)
            
            # Remove from catalog
            if data_file.file_path in self.data_files:
                del self.data_files[data_file.file_path]
            
            logger.info(f"✅ [LIFECYCLE] Deleted file: {data_file.file_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [LIFECYCLE] Failed to delete file {data_file.file_path}: {e}")
            return False
    
    async def _move_to_cold_storage(self, data_file: DataFile) -> bool:
        """Move file to cold storage"""
        try:
            cold_path = os.path.join(self.base_path, DataClass.COLD.value, data_file.file_name)
            
            # Move file
            shutil.move(data_file.file_path, cold_path)
            
            # Update metadata
            data_file.file_path = cold_path
            data_file.data_class = DataClass.COLD
            
            logger.info(f"✅ [LIFECYCLE] Moved to cold storage: {data_file.file_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [LIFECYCLE] Failed to move file to cold storage: {e}")
            return False
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            stats = {
                "total_files": 0,
                "total_size_mb": 0,
                "by_class": {},
                "compressed_files": 0,
                "archived_files": 0
            }
            
            data_files = await self.scan_data_files()
            
            for data_file in data_files:
                stats["total_files"] += 1
                stats["total_size_mb"] += data_file.size_bytes / (1024 * 1024)
                
                # Count by class
                class_name = data_file.data_class.value
                if class_name not in stats["by_class"]:
                    stats["by_class"][class_name] = {"count": 0, "size_mb": 0}
                
                stats["by_class"][class_name]["count"] += 1
                stats["by_class"][class_name]["size_mb"] += data_file.size_bytes / (1024 * 1024)
                
                # Count compressed/archived
                if data_file.compressed:
                    stats["compressed_files"] += 1
                if data_file.archived:
                    stats["archived_files"] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ [LIFECYCLE] Failed to get storage stats: {e}")
            return {"error": str(e)}
    
    def add_lifecycle_policy(self, policy: LifecyclePolicy):
        """Add a new lifecycle policy"""
        self.policies[policy.name.lower().replace(" ", "_")] = policy
        logger.info(f"✅ [LIFECYCLE] Added policy: {policy.name}")
    
    def update_lifecycle_policy(self, policy_name: str, **kwargs):
        """Update an existing lifecycle policy"""
        if policy_name in self.policies:
            policy = self.policies[policy_name]
            for key, value in kwargs.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
            logger.info(f"✅ [LIFECYCLE] Updated policy: {policy_name}")
    
    def list_lifecycle_policies(self) -> List[Dict[str, Any]]:
        """List all lifecycle policies"""
        policies = []
        for name, policy in self.policies.items():
            policies.append({
                "name": policy.name,
                "data_class": policy.data_class.value,
                "retention_days": policy.retention_days,
                "actions": [action.value for action in policy.actions],
                "conditions": policy.conditions,
                "enabled": policy.enabled
            })
        return policies

# Global data lifecycle manager
data_lifecycle_manager = DataLifecycleManager()
