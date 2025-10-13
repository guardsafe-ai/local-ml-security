"""
Training Queue Service
Redis-based job queue for concurrent training jobs with priority and resource management
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    """Training job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class JobPriority(Enum):
    """Job priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class TrainingJob:
    """Training job definition"""
    job_id: str
    model_name: str
    training_data_path: str
    config: Dict[str, Any]
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 3600  # 1 hour default
    resource_requirements: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.resource_requirements is None:
            self.resource_requirements = {
                "cpu_cores": 2,
                "memory_gb": 4,
                "gpu_required": False
            }

class TrainingQueue:
    """Redis-based training job queue with priority and resource management"""
    
    def __init__(self, redis_url: str = "redis://redis:6379/0"):
        self.redis_url = redis_url
        self.redis_client = None
        self.workers = []
        self.max_workers = 2  # Maximum concurrent training jobs
        self.running_jobs = {}  # job_id -> TrainingJob
        self.job_callbacks = {}  # job_id -> callback function
        self.is_running = False
        
        # Queue names
        self.pending_queue = "training:pending"
        self.running_queue = "training:running"
        self.completed_queue = "training:completed"
        self.failed_queue = "training:failed"
        self.job_data_key = "training:job:{}"
        
    async def initialize(self):
        """Initialize Redis connection and start workers"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("✅ Training queue connected to Redis")
            
            # Start worker processes
            self.is_running = True
            for i in range(self.max_workers):
                worker = asyncio.create_task(self._worker(f"worker-{i}"))
                self.workers.append(worker)
            
            logger.info(f"✅ Started {self.max_workers} training queue workers")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize training queue: {e}")
            raise
    
    async def close(self):
        """Close Redis connection and stop workers"""
        self.is_running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("✅ Training queue closed")
    
    async def submit_job(
        self,
        model_name: str,
        training_data_path: str,
        config: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
        timeout_seconds: int = 3600,
        resource_requirements: Dict[str, Any] = None,
        callback: Optional[Callable] = None
    ) -> str:
        """Submit a training job to the queue"""
        try:
            job_id = f"train_{model_name}_{int(time.time())}"
            
            job = TrainingJob(
                job_id=job_id,
                model_name=model_name,
                training_data_path=training_data_path,
                config=config,
                priority=priority,
                timeout_seconds=timeout_seconds,
                resource_requirements=resource_requirements or {}
            )
            
            # Store job data
            job_data = asdict(job)
            job_data['created_at'] = job.created_at.isoformat()
            job_data['priority'] = job.priority.value
            job_data['status'] = job.status.value
            
            # Convert complex objects to JSON strings and handle None values
            job_data['config'] = json.dumps(job_data['config'])
            job_data['resource_requirements'] = json.dumps(job_data['resource_requirements'])
            
            # Handle None values by converting to empty strings
            for key, value in job_data.items():
                if value is None:
                    job_data[key] = ""
            
            await self.redis_client.hset(
                self.job_data_key.format(job_id),
                mapping=job_data
            )
            
            # Add to pending queue with priority score
            priority_score = job.priority.value
            await self.redis_client.zadd(
                self.pending_queue,
                {job_id: priority_score}
            )
            
            # Store callback if provided
            if callback:
                self.job_callbacks[job_id] = callback
            
            logger.info(f"📝 Submitted training job: {job_id} (priority: {priority.name})")
            return job_id
            
        except Exception as e:
            logger.error(f"❌ Failed to submit training job: {e}")
            raise
    
    async def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get job status and details"""
        try:
            job_data = await self.redis_client.hgetall(self.job_data_key.format(job_id))
            if not job_data:
                return None
            
            # Convert back to TrainingJob
            job = TrainingJob(
                job_id=job_data['job_id'],
                model_name=job_data['model_name'],
                training_data_path=job_data['training_data_path'],
                config=json.loads(job_data['config']),
                priority=JobPriority(int(job_data['priority'])),
                status=JobStatus(job_data['status']),
                created_at=datetime.fromisoformat(job_data['created_at']),
                progress=float(job_data.get('progress', 0.0)),
                error_message=job_data.get('error_message'),
                retry_count=int(job_data.get('retry_count', 0)),
                max_retries=int(job_data.get('max_retries', 3)),
                timeout_seconds=int(job_data.get('timeout_seconds', 3600)),
                resource_requirements=json.loads(job_data.get('resource_requirements', '{}'))
            )
            
            if job_data.get('started_at'):
                job.started_at = datetime.fromisoformat(job_data['started_at'])
            if job_data.get('completed_at'):
                job.completed_at = datetime.fromisoformat(job_data['completed_at'])
            if job_data.get('result'):
                job.result = json.loads(job_data['result'])
            
            return job
            
        except Exception as e:
            logger.error(f"❌ Failed to get job status for {job_id}: {e}")
            return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job"""
        try:
            job = await self.get_job_status(job_id)
            if not job:
                return False
            
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return False
            
            # Update job status
            await self._update_job_status(job_id, JobStatus.CANCELLED)
            
            # Remove from queues
            await self.redis_client.zrem(self.pending_queue, job_id)
            await self.redis_client.zrem(self.running_queue, job_id)
            
            # Remove from running jobs
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
            
            logger.info(f"🚫 Cancelled training job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to cancel job {job_id}: {e}")
            return False
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        try:
            pending_count = await self.redis_client.zcard(self.pending_queue)
            running_count = await self.redis_client.zcard(self.running_queue)
            completed_count = await self.redis_client.zcard(self.completed_queue)
            failed_count = await self.redis_client.zcard(self.failed_queue)
            
            # Get recent jobs
            recent_jobs = []
            for queue_name in [self.completed_queue, self.failed_queue, self.running_queue]:
                jobs = await self.redis_client.zrevrange(queue_name, 0, 4, withscores=True)
                for job_id, score in jobs:
                    job = await self.get_job_status(job_id)
                    if job:
                        recent_jobs.append(job)
            
            return {
                "pending_jobs": pending_count,
                "running_jobs": running_count,
                "completed_jobs": completed_count,
                "failed_jobs": failed_count,
                "total_jobs": pending_count + running_count + completed_count + failed_count,
                "max_workers": self.max_workers,
                "active_workers": len([w for w in self.workers if not w.done()]),
                "recent_jobs": [asdict(job) for job in recent_jobs[:10]]
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get queue stats: {e}")
            return {}
    
    async def _worker(self, worker_name: str):
        """Worker process that processes jobs from the queue"""
        logger.info(f"🔄 Starting training queue worker: {worker_name}")
        
        while self.is_running:
            try:
                # Get next job from pending queue (highest priority first)
                job_data = await self.redis_client.bzpopmin(self.pending_queue, timeout=1)
                if not job_data:
                    continue
                
                job_id = job_data[1]
                job = await self.get_job_status(job_id)
                if not job:
                    continue
                
                # Check if we have capacity
                if len(self.running_jobs) >= self.max_workers:
                    # Put job back in queue
                    await self.redis_client.zadd(
                        self.pending_queue,
                        {job_id: job.priority.value}
                    )
                    await asyncio.sleep(1)
                    continue
                
                # Start processing job
                await self._process_job(job, worker_name)
                
            except asyncio.CancelledError:
                logger.info(f"🛑 Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Worker {worker_name} error: {e}")
                await asyncio.sleep(5)
        
        logger.info(f"✅ Worker {worker_name} stopped")
    
    async def _process_job(self, job: TrainingJob, worker_name: str):
        """Process a single training job"""
        try:
            logger.info(f"🚀 [{worker_name}] Starting job: {job.job_id}")
            
            # Update job status to running
            await self._update_job_status(job.job_id, JobStatus.RUNNING)
            job.started_at = datetime.utcnow()
            
            # Add to running jobs
            self.running_jobs[job.job_id] = job
            
            # Add to running queue
            await self.redis_client.zadd(
                self.running_queue,
                {job.job_id: time.time()}
            )
            
            # Import training function here to avoid circular imports
            from .model_trainer import ModelTrainer
            from models.requests import TrainingRequest, TrainingConfig
            trainer = ModelTrainer()
            
            # Create TrainingConfig object
            training_config = TrainingConfig(
                model_name=job.model_name,
                max_length=job.config.get('max_length', 256),
                batch_size=job.config.get('batch_size', 8),
                learning_rate=job.config.get('learning_rate', 2e-5),
                num_epochs=job.config.get('epochs', 2),
                warmup_steps=job.config.get('warmup_steps', 100),
                weight_decay=job.config.get('weight_decay', 0.01),
                evaluation_strategy=job.config.get('evaluation_strategy', 'steps'),
                eval_steps=job.config.get('eval_steps', 500),
                save_steps=job.config.get('save_steps', 500),
                load_best_model_at_end=job.config.get('load_best_model_at_end', True),
                metric_for_best_model=job.config.get('metric_for_best_model', 'eval_f1'),
                greater_is_better=job.config.get('greater_is_better', True)
            )
            
            # Create TrainingRequest object
            training_request = TrainingRequest(
                model_name=job.model_name,
                training_data_path=job.training_data_path,
                config=training_config
            )
            
            # Start training with timeout
            training_task = asyncio.create_task(
                trainer.train_model(training_request)
            )
            
            # Wait for completion or timeout
            try:
                result = await asyncio.wait_for(
                    training_task,
                    timeout=job.timeout_seconds
                )
                
                # Job completed successfully
                await self._complete_job(job.job_id, result)
                logger.info(f"✅ [{worker_name}] Completed job: {job.job_id}")
                
            except asyncio.TimeoutError:
                # Job timed out
                await self._fail_job(job.job_id, "Job timed out")
                logger.warning(f"⏰ [{worker_name}] Job timed out: {job.job_id}")
                
            except Exception as e:
                # Job failed
                await self._fail_job(job.job_id, str(e))
                logger.error(f"❌ [{worker_name}] Job failed: {job.job_id} - {e}")
            
        except Exception as e:
            logger.error(f"❌ [{worker_name}] Error processing job {job.job_id}: {e}")
            await self._fail_job(job.job_id, str(e))
        
        finally:
            # Clean up
            if job.job_id in self.running_jobs:
                del self.running_jobs[job.job_id]
            await self.redis_client.zrem(self.running_queue, job.job_id)
    
    async def _update_job_status(self, job_id: str, status: JobStatus, **updates):
        """Update job status and other fields"""
        try:
            update_data = {"status": status.value}
            update_data.update(updates)
            
            # Convert datetime objects to ISO strings and handle None values
            for key, value in update_data.items():
                if isinstance(value, datetime):
                    update_data[key] = value.isoformat()
                elif isinstance(value, dict):
                    update_data[key] = json.dumps(value)
                elif value is None:
                    update_data[key] = ""
            
            await self.redis_client.hset(
                self.job_data_key.format(job_id),
                mapping=update_data
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to update job status for {job_id}: {e}")
    
    async def _complete_job(self, job_id: str, result: Dict[str, Any]):
        """Mark job as completed"""
        await self._update_job_status(
            job_id,
            JobStatus.COMPLETED,
            completed_at=datetime.utcnow(),
            progress=100.0,
            result=result
        )
        
        # Move to completed queue
        await self.redis_client.zadd(
            self.completed_queue,
            {job_id: time.time()}
        )
        
        # Call callback if exists
        if job_id in self.job_callbacks:
            try:
                await self.job_callbacks[job_id](job_id, JobStatus.COMPLETED, result)
            except Exception as e:
                logger.error(f"❌ Callback error for job {job_id}: {e}")
            finally:
                del self.job_callbacks[job_id]
    
    async def _fail_job(self, job_id: str, error_message: str):
        """Mark job as failed"""
        await self._update_job_status(
            job_id,
            JobStatus.FAILED,
            completed_at=datetime.utcnow(),
            error_message=error_message
        )
        
        # Move to failed queue
        await self.redis_client.zadd(
            self.failed_queue,
            {job_id: time.time()}
        )
        
        # Call callback if exists
        if job_id in self.job_callbacks:
            try:
                await self.job_callbacks[job_id](job_id, JobStatus.FAILED, {"error": error_message})
            except Exception as e:
                logger.error(f"❌ Callback error for job {job_id}: {e}")
            finally:
                del self.job_callbacks[job_id]

# Global instance
training_queue = TrainingQueue()
