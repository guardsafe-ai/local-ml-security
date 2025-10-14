"""
Training Resource Cleanup Utilities
Handles cleanup of training resources on timeout or failure
"""

import logging
import asyncio
import torch
import gc
import os
import tempfile
from typing import Optional, Dict, Any
import psutil

logger = logging.getLogger(__name__)

class TrainingResourceManager:
    """Manages training resources and cleanup"""
    
    def __init__(self):
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.cleanup_callbacks: Dict[str, list] = {}
    
    def register_job(self, job_id: str, resources: Dict[str, Any] = None):
        """Register a training job and its resources"""
        self.active_jobs[job_id] = {
            "resources": resources or {},
            "start_time": asyncio.get_event_loop().time(),
            "cleanup_callbacks": []
        }
        logger.info(f"📝 [CLEANUP] Registered job {job_id} for resource tracking")
    
    def add_cleanup_callback(self, job_id: str, callback, *args, **kwargs):
        """Add a cleanup callback for a specific job"""
        if job_id not in self.active_jobs:
            logger.warning(f"⚠️ [CLEANUP] Job {job_id} not registered, cannot add callback")
            return
        
        self.active_jobs[job_id]["cleanup_callbacks"].append((callback, args, kwargs))
        logger.debug(f"🔧 [CLEANUP] Added cleanup callback for job {job_id}")
    
    async def cleanup_job_resources(self, job_id: str, reason: str = "unknown") -> bool:
        """Clean up all resources for a specific job"""
        if job_id not in self.active_jobs:
            logger.warning(f"⚠️ [CLEANUP] Job {job_id} not found for cleanup")
            return False
        
        try:
            job_info = self.active_jobs[job_id]
            logger.info(f"🧹 [CLEANUP] Starting cleanup for job {job_id} (reason: {reason})")
            
            # Execute all cleanup callbacks
            for callback, args, kwargs in job_info["cleanup_callbacks"]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(*args, **kwargs)
                    else:
                        callback(*args, **kwargs)
                    logger.debug(f"✅ [CLEANUP] Executed callback for job {job_id}")
                except Exception as e:
                    logger.error(f"❌ [CLEANUP] Error in cleanup callback for job {job_id}: {e}")
            
            # Standard cleanup operations
            await self._standard_cleanup(job_id)
            
            # Remove from active jobs
            del self.active_jobs[job_id]
            
            logger.info(f"✅ [CLEANUP] Completed cleanup for job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [CLEANUP] Error during cleanup for job {job_id}: {e}")
            return False
    
    async def _standard_cleanup(self, job_id: str):
        """Perform standard cleanup operations"""
        try:
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug(f"🧹 [CLEANUP] Cleared CUDA cache for job {job_id}")
            
            # Force garbage collection
            gc.collect()
            logger.debug(f"🧹 [CLEANUP] Ran garbage collection for job {job_id}")
            
            # Log memory usage
            memory_info = self._get_memory_info()
            logger.info(f"📊 [CLEANUP] Memory after cleanup: {memory_info}")
            
        except Exception as e:
            logger.error(f"❌ [CLEANUP] Error in standard cleanup for job {job_id}: {e}")
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            info = {
                "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
                "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            }
            
            # Add CUDA memory if available
            if torch.cuda.is_available():
                info.update({
                    "cuda_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                    "cuda_cached_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                })
            
            return info
            
        except Exception as e:
            logger.error(f"❌ [CLEANUP] Error getting memory info: {e}")
            return {"error": str(e)}
    
    async def cleanup_all_jobs(self):
        """Clean up all active jobs (emergency cleanup)"""
        logger.warning("🚨 [CLEANUP] Emergency cleanup of all active jobs")
        
        for job_id in list(self.active_jobs.keys()):
            await self.cleanup_job_resources(job_id, "emergency_cleanup")
        
        logger.info("✅ [CLEANUP] Emergency cleanup completed")
    
    def get_active_jobs(self) -> Dict[str, Any]:
        """Get information about active jobs"""
        return {
            job_id: {
                "start_time": info["start_time"],
                "duration_seconds": asyncio.get_event_loop().time() - info["start_time"],
                "callback_count": len(info["cleanup_callbacks"])
            }
            for job_id, info in self.active_jobs.items()
        }

# Global resource manager instance
resource_manager = TrainingResourceManager()

async def cleanup_training_resources(job_id: str, reason: str = "timeout") -> bool:
    """
    Clean up training resources for a specific job
    
    Args:
        job_id: The training job ID
        reason: Reason for cleanup (timeout, failure, success, etc.)
    
    Returns:
        bool: True if cleanup was successful
    """
    return await resource_manager.cleanup_job_resources(job_id, reason)

def register_training_job(job_id: str, resources: Dict[str, Any] = None):
    """Register a training job for resource tracking"""
    resource_manager.register_job(job_id, resources)

def add_cleanup_callback(job_id: str, callback, *args, **kwargs):
    """Add a cleanup callback for a training job"""
    resource_manager.add_cleanup_callback(job_id, callback, *args, **kwargs)
