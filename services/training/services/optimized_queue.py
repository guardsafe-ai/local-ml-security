"""
Training Queue Optimization Service
Implements job batching and priority-based resource allocation
"""

import logging
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class JobPriority(Enum):
    """Job priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

class JobStatus(Enum):
    """Job status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TrainingJob:
    """Training job model"""
    job_id: str
    user_id: str
    model_name: str
    dataset_path: str
    priority: JobPriority
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: int = 0  # seconds
    actual_duration: int = 0
    resource_requirements: Dict[str, Any] = None
    batch_group: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None

@dataclass
class ResourceAllocation:
    """Resource allocation for jobs"""
    cpu_cores: int
    memory_gb: int
    gpu_count: int
    gpu_memory_gb: int
    estimated_cost: float

class OptimizedTrainingQueue:
    """Optimized training queue with batching and priority management"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://redis:6379/4")
        self.redis_client = None
        self.jobs = {}
        self.batch_groups = {}
        self.resource_pools = {
            "cpu": {"total": 16, "available": 16},
            "memory": {"total": 64, "available": 64},  # GB
            "gpu": {"total": 2, "available": 2},
            "gpu_memory": {"total": 24, "available": 24}  # GB
        }
        self.batch_config = {
            "max_batch_size": 4,
            "max_wait_time": 300,  # 5 minutes
            "similarity_threshold": 0.8
        }
        
    async def get_redis_client(self) -> redis.Redis:
        """Get or create Redis client"""
        if self.redis_client is None:
            try:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                await self.redis_client.ping()
                logger.info("✅ [TRAINING_QUEUE] Redis client connected")
            except Exception as e:
                logger.error(f"❌ [TRAINING_QUEUE] Redis connection failed: {e}")
                self.redis_client = None
        return self.redis_client
    
    async def submit_job(self, user_id: str, model_name: str, dataset_path: str,
                        priority: JobPriority = JobPriority.NORMAL,
                        resource_requirements: Dict[str, Any] = None) -> str:
        """
        Submit a training job to the queue
        
        Args:
            user_id: User submitting the job
            model_name: Name of the model to train
            dataset_path: Path to training dataset
            priority: Job priority
            resource_requirements: Resource requirements
            
        Returns:
            Job ID
        """
        try:
            job_id = f"job_{int(time.time() * 1000)}_{user_id}"
            
            job = TrainingJob(
                job_id=job_id,
                user_id=user_id,
                model_name=model_name,
                dataset_path=dataset_path,
                priority=priority,
                status=JobStatus.PENDING,
                created_at=datetime.now(timezone.utc),
                resource_requirements=resource_requirements or {}
            )
            
            # Estimate duration based on model and dataset
            job.estimated_duration = self._estimate_job_duration(model_name, dataset_path)
            
            # Store job
            self.jobs[job_id] = job
            
            # Add to Redis queue
            redis_client = await self.get_redis_client()
            if redis_client:
                job_data = {
                    "job_id": job.job_id,
                    "user_id": job.user_id,
                    "model_name": job.model_name,
                    "dataset_path": job.dataset_path,
                    "priority": job.priority.value,
                    "status": job.status.value,
                    "created_at": job.created_at.isoformat(),
                    "estimated_duration": job.estimated_duration,
                    "resource_requirements": json.dumps(job.resource_requirements)
                }
                await redis_client.hset(f"job:{job_id}", mapping=job_data)
                await redis_client.zadd("job_queue", {job_id: priority.value})
            
            # Try to batch with similar jobs
            await self._try_batch_job(job)
            
            logger.info(f"✅ [TRAINING_QUEUE] Submitted job: {job_id} (priority: {priority.value})")
            return job_id
            
        except Exception as e:
            logger.error(f"❌ [TRAINING_QUEUE] Failed to submit job: {e}")
            raise
    
    def _estimate_job_duration(self, model_name: str, dataset_path: str) -> int:
        """Estimate job duration based on model and dataset"""
        # Simple estimation logic
        base_duration = 1800  # 30 minutes base
        
        # Adjust based on model complexity
        if "bert" in model_name.lower():
            base_duration *= 2
        elif "gpt" in model_name.lower():
            base_duration *= 3
        
        # Adjust based on dataset size (simplified)
        try:
            if os.path.exists(dataset_path):
                size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
                if size_mb > 1000:  # > 1GB
                    base_duration *= 2
                elif size_mb > 100:  # > 100MB
                    base_duration *= 1.5
        except:
            pass
        
        return int(base_duration)
    
    async def _try_batch_job(self, job: TrainingJob):
        """Try to batch job with similar jobs"""
        try:
            # Find similar jobs that can be batched
            similar_jobs = []
            
            for existing_job in self.jobs.values():
                if (existing_job.status == JobStatus.PENDING and
                    existing_job.model_name == job.model_name and
                    existing_job.job_id != job.job_id):
                    
                    similarity = self._calculate_job_similarity(job, existing_job)
                    if similarity >= self.batch_config["similarity_threshold"]:
                        similar_jobs.append((existing_job, similarity))
            
            # Sort by similarity
            similar_jobs.sort(key=lambda x: x[1], reverse=True)
            
            # Create batch if we have enough similar jobs
            if len(similar_jobs) >= 2:
                batch_id = f"batch_{int(time.time() * 1000)}"
                
                # Add jobs to batch
                batch_jobs = [job] + [j[0] for j in similar_jobs[:self.batch_config["max_batch_size"] - 1]]
                
                for batch_job in batch_jobs:
                    batch_job.batch_group = batch_id
                    batch_job.status = JobStatus.QUEUED
                
                self.batch_groups[batch_id] = {
                    "jobs": batch_jobs,
                    "created_at": datetime.now(timezone.utc),
                    "status": "pending"
                }
                
                logger.info(f"✅ [TRAINING_QUEUE] Created batch {batch_id} with {len(batch_jobs)} jobs")
            
            else:
                # No batching possible, queue individually
                job.status = JobStatus.QUEUED
                
        except Exception as e:
            logger.error(f"❌ [TRAINING_QUEUE] Failed to batch job: {e}")
    
    def _calculate_job_similarity(self, job1: TrainingJob, job2: TrainingJob) -> float:
        """Calculate similarity between two jobs"""
        similarity = 0.0
        
        # Model similarity
        if job1.model_name == job2.model_name:
            similarity += 0.4
        
        # Dataset similarity (simplified)
        if job1.dataset_path == job2.dataset_path:
            similarity += 0.3
        
        # Resource requirements similarity
        if job1.resource_requirements == job2.resource_requirements:
            similarity += 0.2
        
        # Priority similarity
        if job1.priority == job2.priority:
            similarity += 0.1
        
        return similarity
    
    async def process_queue(self) -> List[str]:
        """Process the training queue and return jobs to execute"""
        try:
            jobs_to_execute = []
            
            # Process batches first
            for batch_id, batch_info in self.batch_groups.items():
                if batch_info["status"] == "pending":
                    # Check if batch is ready to execute
                    if self._is_batch_ready(batch_info):
                        jobs_to_execute.extend([job.job_id for job in batch_info["jobs"]])
                        batch_info["status"] = "running"
            
            # Process individual jobs
            pending_jobs = [job for job in self.jobs.values() 
                           if job.status == JobStatus.QUEUED and not job.batch_group]
            
            # Sort by priority and creation time
            pending_jobs.sort(key=lambda x: (x.priority.value, x.created_at), reverse=True)
            
            # Allocate resources
            for job in pending_jobs:
                if self._can_allocate_resources(job):
                    jobs_to_execute.append(job.job_id)
                    job.status = JobStatus.RUNNING
                    job.started_at = datetime.now(timezone.utc)
            
            logger.info(f"✅ [TRAINING_QUEUE] Processing {len(jobs_to_execute)} jobs")
            return jobs_to_execute
            
        except Exception as e:
            logger.error(f"❌ [TRAINING_QUEUE] Failed to process queue: {e}")
            return []
    
    def _is_batch_ready(self, batch_info: Dict[str, Any]) -> bool:
        """Check if batch is ready to execute"""
        try:
            # Check if all jobs in batch are ready
            for job in batch_info["jobs"]:
                if job.status != JobStatus.QUEUED:
                    return False
            
            # Check resource availability for entire batch
            total_requirements = self._calculate_batch_requirements(batch_info["jobs"])
            return self._can_allocate_resources_batch(total_requirements)
            
        except Exception as e:
            logger.error(f"❌ [TRAINING_QUEUE] Failed to check batch readiness: {e}")
            return False
    
    def _calculate_batch_requirements(self, jobs: List[TrainingJob]) -> Dict[str, Any]:
        """Calculate total resource requirements for a batch"""
        total_requirements = {
            "cpu_cores": 0,
            "memory_gb": 0,
            "gpu_count": 0,
            "gpu_memory_gb": 0
        }
        
        for job in jobs:
            req = job.resource_requirements
            total_requirements["cpu_cores"] += req.get("cpu_cores", 2)
            total_requirements["memory_gb"] += req.get("memory_gb", 4)
            total_requirements["gpu_count"] += req.get("gpu_count", 0)
            total_requirements["gpu_memory_gb"] += req.get("gpu_memory_gb", 0)
        
        return total_requirements
    
    def _can_allocate_resources(self, job: TrainingJob) -> bool:
        """Check if resources can be allocated for a job"""
        req = job.resource_requirements
        
        cpu_needed = req.get("cpu_cores", 2)
        memory_needed = req.get("memory_gb", 4)
        gpu_needed = req.get("gpu_count", 0)
        gpu_memory_needed = req.get("gpu_memory_gb", 0)
        
        return (self.resource_pools["cpu"]["available"] >= cpu_needed and
                self.resource_pools["memory"]["available"] >= memory_needed and
                self.resource_pools["gpu"]["available"] >= gpu_needed and
                self.resource_pools["gpu_memory"]["available"] >= gpu_memory_needed)
    
    def _can_allocate_resources_batch(self, requirements: Dict[str, Any]) -> bool:
        """Check if resources can be allocated for a batch"""
        return (self.resource_pools["cpu"]["available"] >= requirements["cpu_cores"] and
                self.resource_pools["memory"]["available"] >= requirements["memory_gb"] and
                self.resource_pools["gpu"]["available"] >= requirements["gpu_count"] and
                self.resource_pools["gpu_memory"]["available"] >= requirements["gpu_memory_gb"])
    
    async def complete_job(self, job_id: str, success: bool, error_message: str = None):
        """Mark job as completed"""
        try:
            job = self.jobs.get(job_id)
            if not job:
                logger.warning(f"⚠️ [TRAINING_QUEUE] Job not found: {job_id}")
                return
            
            job.completed_at = datetime.now(timezone.utc)
            job.actual_duration = int((job.completed_at - job.started_at).total_seconds()) if job.started_at else 0
            
            if success:
                job.status = JobStatus.COMPLETED
            else:
                job.status = JobStatus.FAILED
                job.error_message = error_message
                
                # Retry logic
                if job.retry_count < job.max_retries:
                    job.retry_count += 1
                    job.status = JobStatus.PENDING
                    job.started_at = None
                    job.completed_at = None
                    logger.info(f"🔄 [TRAINING_QUEUE] Retrying job {job_id} (attempt {job.retry_count})")
            
            # Release resources
            self._release_resources(job)
            
            # Update Redis
            redis_client = await self.get_redis_client()
            if redis_client:
                await redis_client.hset(f"job:{job_id}", mapping={
                    "status": job.status.value,
                    "completed_at": job.completed_at.isoformat(),
                    "actual_duration": job.actual_duration,
                    "retry_count": job.retry_count,
                    "error_message": job.error_message or ""
                })
                await redis_client.zrem("job_queue", job_id)
            
            logger.info(f"✅ [TRAINING_QUEUE] Completed job: {job_id} (success: {success})")
            
        except Exception as e:
            logger.error(f"❌ [TRAINING_QUEUE] Failed to complete job: {e}")
    
    def _release_resources(self, job: TrainingJob):
        """Release resources allocated to a job"""
        req = job.resource_requirements
        
        self.resource_pools["cpu"]["available"] += req.get("cpu_cores", 2)
        self.resource_pools["memory"]["available"] += req.get("memory_gb", 4)
        self.resource_pools["gpu"]["available"] += req.get("gpu_count", 0)
        self.resource_pools["gpu_memory"]["available"] += req.get("gpu_memory_gb", 0)
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        try:
            status = {
                "total_jobs": len(self.jobs),
                "pending_jobs": 0,
                "queued_jobs": 0,
                "running_jobs": 0,
                "completed_jobs": 0,
                "failed_jobs": 0,
                "batches": len(self.batch_groups),
                "resource_utilization": {
                    "cpu": (self.resource_pools["cpu"]["total"] - self.resource_pools["cpu"]["available"]) / self.resource_pools["cpu"]["total"] * 100,
                    "memory": (self.resource_pools["memory"]["total"] - self.resource_pools["memory"]["available"]) / self.resource_pools["memory"]["total"] * 100,
                    "gpu": (self.resource_pools["gpu"]["total"] - self.resource_pools["gpu"]["available"]) / self.resource_pools["gpu"]["total"] * 100
                },
                "average_wait_time": 0,
                "average_duration": 0
            }
            
            # Count jobs by status
            for job in self.jobs.values():
                if job.status == JobStatus.PENDING:
                    status["pending_jobs"] += 1
                elif job.status == JobStatus.QUEUED:
                    status["queued_jobs"] += 1
                elif job.status == JobStatus.RUNNING:
                    status["running_jobs"] += 1
                elif job.status == JobStatus.COMPLETED:
                    status["completed_jobs"] += 1
                elif job.status == JobStatus.FAILED:
                    status["failed_jobs"] += 1
            
            # Calculate averages
            completed_jobs = [job for job in self.jobs.values() if job.status == JobStatus.COMPLETED]
            if completed_jobs:
                wait_times = [(job.started_at - job.created_at).total_seconds() for job in completed_jobs if job.started_at]
                durations = [job.actual_duration for job in completed_jobs if job.actual_duration > 0]
                
                if wait_times:
                    status["average_wait_time"] = sum(wait_times) / len(wait_times)
                if durations:
                    status["average_duration"] = sum(durations) / len(durations)
            
            return status
            
        except Exception as e:
            logger.error(f"❌ [TRAINING_QUEUE] Failed to get queue status: {e}")
            return {"error": str(e)}
    
    async def cancel_job(self, job_id: str, user_id: str) -> bool:
        """Cancel a job"""
        try:
            job = self.jobs.get(job_id)
            if not job:
                return False
            
            if job.user_id != user_id:
                logger.warning(f"🚨 [TRAINING_QUEUE] Unauthorized cancel attempt: {job_id}")
                return False
            
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return False
            
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now(timezone.utc)
            
            # Release resources
            self._release_resources(job)
            
            logger.info(f"✅ [TRAINING_QUEUE] Cancelled job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [TRAINING_QUEUE] Failed to cancel job: {e}")
            return False

# Global training queue
optimized_training_queue = OptimizedTrainingQueue()
