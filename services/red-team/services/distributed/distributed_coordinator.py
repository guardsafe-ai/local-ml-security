"""
Distributed Coordinator
Coordinates distributed task execution across multiple workers
"""

import asyncio
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json
from collections import defaultdict
import uuid

from .task_queue import TaskQueueManager, TaskPriority, TaskStatus
from .worker_manager import WorkerManager, WorkerType, WorkerConfig
from .load_balancer import LoadBalancer, LoadBalancingStrategy, TaskType, LoadBalancingConfig

logger = logging.getLogger(__name__)


class CoordinationStrategy(Enum):
    """Coordination strategies"""
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
    HYBRID = "hybrid"


class ScalingPolicy(Enum):
    """Scaling policies"""
    MANUAL = "manual"
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    QUEUE_BASED = "queue_based"
    HYBRID = "hybrid"


@dataclass
class DistributedConfig:
    """Distributed coordination configuration"""
    coordination_strategy: CoordinationStrategy = CoordinationStrategy.CENTRALIZED
    scaling_policy: ScalingPolicy = ScalingPolicy.QUEUE_BASED
    auto_scaling: bool = True
    min_workers: int = 1
    max_workers: int = 10
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.2
    scaling_cooldown: int = 300  # seconds
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS
    task_timeout: int = 3600  # seconds
    retry_policy: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.retry_policy is None:
            self.retry_policy = {
                "max_retries": 3,
                "retry_delay": 1.0,
                "exponential_backoff": True
            }
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DistributedTask:
    """Distributed task representation"""
    task_id: str
    task_name: str
    task_type: TaskType
    priority: TaskPriority
    args: List[Any]
    kwargs: Dict[str, Any]
    assigned_worker: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DistributedResult:
    """Distributed task result"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    worker_id: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DistributedCoordinator:
    """
    Distributed Coordinator
    Coordinates distributed task execution across multiple workers
    """
    
    def __init__(self, config: DistributedConfig = None):
        """Initialize distributed coordinator"""
        self.config = config or DistributedConfig()
        
        # Initialize components
        self.task_queue = TaskQueueManager()
        self.worker_manager = WorkerManager()
        self.load_balancer = LoadBalancer(LoadBalancingConfig(
            strategy=self.config.load_balancing_strategy
        ))
        
        # Task tracking
        self.distributed_tasks: Dict[str, DistributedTask] = {}
        self.task_results: Dict[str, DistributedResult] = {}
        
        # Scaling
        self.last_scaling_time = datetime.utcnow()
        self.scaling_history: List[Dict[str, Any]] = []
        
        # Monitoring
        self.monitoring_task = None
        self.coordination_active = False
        
        logger.info("✅ Initialized Distributed Coordinator")
    
    async def start_coordination(self):
        """Start distributed coordination"""
        try:
            if self.coordination_active:
                return
            
            self.coordination_active = True
            
            # Start monitoring
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Start worker manager monitoring
            await self.worker_manager.start_monitoring()
            
            logger.info("Distributed coordination started")
            
        except Exception as e:
            logger.error(f"Failed to start coordination: {e}")
            raise
    
    async def stop_coordination(self):
        """Stop distributed coordination"""
        try:
            self.coordination_active = False
            
            # Stop monitoring
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Stop worker manager monitoring
            await self.worker_manager.stop_monitoring()
            
            logger.info("Distributed coordination stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop coordination: {e}")
    
    async def submit_distributed_task(self, 
                                    task_name: str,
                                    task_type: TaskType,
                                    args: List[Any] = None,
                                    kwargs: Dict[str, Any] = None,
                                    priority: TaskPriority = TaskPriority.NORMAL,
                                    max_retries: int = None) -> str:
        """Submit a distributed task"""
        try:
            if args is None:
                args = []
            if kwargs is None:
                kwargs = {}
            if max_retries is None:
                max_retries = self.config.retry_policy["max_retries"]
            
            # Generate task ID
            task_id = str(uuid.uuid4())
            
            # Create distributed task
            distributed_task = DistributedTask(
                task_id=task_id,
                task_name=task_name,
                task_type=task_type,
                priority=priority,
                args=args,
                kwargs=kwargs,
                max_retries=max_retries
            )
            
            # Store task
            self.distributed_tasks[task_id] = distributed_task
            
            # Submit to task queue
            await self.task_queue.submit_task(
                task_name=task_name,
                args=args,
                kwargs=kwargs,
                priority=priority,
                max_retries=max_retries
            )
            
            logger.info(f"Distributed task submitted: {task_id} ({task_name})")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to submit distributed task: {e}")
            raise
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status"""
        try:
            if task_id in self.distributed_tasks:
                return self.distributed_tasks[task_id].status
            
            # Check task queue
            return await self.task_queue.get_task_status(task_id)
            
        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {e}")
            return None
    
    async def get_task_result(self, task_id: str) -> Optional[DistributedResult]:
        """Get task result"""
        try:
            if task_id in self.task_results:
                return self.task_results[task_id]
            
            # Check task queue
            queue_result = await self.task_queue.get_task_result(task_id)
            if queue_result:
                # Convert to distributed result
                distributed_result = DistributedResult(
                    task_id=task_id,
                    success=queue_result.status == TaskStatus.SUCCESS,
                    result=queue_result.result,
                    error=queue_result.error,
                    worker_id=queue_result.metadata.get("worker_id") if queue_result.metadata else None,
                    execution_time=0.0,  # Not available from queue result
                    retry_count=queue_result.retry_count
                )
                
                # Store result
                self.task_results[task_id] = distributed_result
                
                return distributed_result
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get task result for {task_id}: {e}")
            return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        try:
            # Cancel in task queue
            success = await self.task_queue.cancel_task(task_id)
            
            # Update distributed task status
            if task_id in self.distributed_tasks:
                self.distributed_tasks[task_id].status = TaskStatus.REVOKED
                self.distributed_tasks[task_id].completed_at = datetime.utcnow()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    async def scale_workers(self, 
                          worker_type: WorkerType, 
                          target_count: int,
                          config_template: WorkerConfig = None) -> int:
        """Scale workers"""
        try:
            # Check scaling cooldown
            if (datetime.utcnow() - self.last_scaling_time).seconds < self.config.scaling_cooldown:
                logger.info("Scaling cooldown active, skipping scaling")
                return len(await self.worker_manager.get_workers_by_type(worker_type))
            
            # Perform scaling
            actual_count = await self.worker_manager.scale_workers(worker_type, target_count, config_template)
            
            # Update load balancer
            workers = await self.worker_manager.get_workers_by_type(worker_type)
            for worker in workers:
                if worker.status.value in ["RUNNING", "IDLE", "BUSY"]:
                    await self.load_balancer.register_worker(
                        worker.worker_id, 
                        worker_type.value,
                        worker.config.weight if worker.config else 1.0
                    )
            
            # Record scaling event
            self.scaling_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "worker_type": worker_type.value,
                "target_count": target_count,
                "actual_count": actual_count,
                "reason": "manual_scaling"
            })
            
            self.last_scaling_time = datetime.utcnow()
            
            logger.info(f"Scaled {worker_type.value} workers to {actual_count}")
            return actual_count
            
        except Exception as e:
            logger.error(f"Failed to scale workers: {e}")
            return 0
    
    async def auto_scale(self):
        """Perform automatic scaling based on policy"""
        try:
            if not self.config.auto_scaling:
                return
            
            # Get current metrics
            worker_stats = await self.worker_manager.get_worker_stats()
            queue_stats = await self.task_queue.get_queue_stats()
            
            # Scale based on policy
            if self.config.scaling_policy == ScalingPolicy.CPU_BASED:
                await self._scale_based_on_cpu(worker_stats)
            elif self.config.scaling_policy == ScalingPolicy.MEMORY_BASED:
                await self._scale_based_on_memory(worker_stats)
            elif self.config.scaling_policy == ScalingPolicy.QUEUE_BASED:
                await self._scale_based_on_queue(queue_stats)
            elif self.config.scaling_policy == ScalingPolicy.HYBRID:
                await self._scale_based_on_hybrid(worker_stats, queue_stats)
            
        except Exception as e:
            logger.error(f"Auto scaling failed: {e}")
    
    async def _scale_based_on_cpu(self, worker_stats: Dict[str, Any]):
        """Scale based on CPU usage"""
        try:
            avg_cpu = worker_stats.get("total_cpu_usage_percent", 0) / max(worker_stats.get("total_workers", 1), 1)
            
            if avg_cpu > self.config.scale_up_threshold * 100:
                # Scale up
                current_workers = worker_stats.get("running_workers", 0)
                if current_workers < self.config.max_workers:
                    target_count = min(current_workers + 2, self.config.max_workers)
                    await self.scale_workers(WorkerType.GENERAL_WORKER, target_count)
            
            elif avg_cpu < self.config.scale_down_threshold * 100:
                # Scale down
                current_workers = worker_stats.get("running_workers", 0)
                if current_workers > self.config.min_workers:
                    target_count = max(current_workers - 1, self.config.min_workers)
                    await self.scale_workers(WorkerType.GENERAL_WORKER, target_count)
            
        except Exception as e:
            logger.error(f"CPU-based scaling failed: {e}")
    
    async def _scale_based_on_memory(self, worker_stats: Dict[str, Any]):
        """Scale based on memory usage"""
        try:
            avg_memory = worker_stats.get("total_memory_usage_mb", 0) / max(worker_stats.get("total_workers", 1), 1)
            memory_threshold = 1024  # MB per worker
            
            if avg_memory > memory_threshold * self.config.scale_up_threshold:
                # Scale up
                current_workers = worker_stats.get("running_workers", 0)
                if current_workers < self.config.max_workers:
                    target_count = min(current_workers + 2, self.config.max_workers)
                    await self.scale_workers(WorkerType.GENERAL_WORKER, target_count)
            
            elif avg_memory < memory_threshold * self.config.scale_down_threshold:
                # Scale down
                current_workers = worker_stats.get("running_workers", 0)
                if current_workers > self.config.min_workers:
                    target_count = max(current_workers - 1, self.config.min_workers)
                    await self.scale_workers(WorkerType.GENERAL_WORKER, target_count)
            
        except Exception as e:
            logger.error(f"Memory-based scaling failed: {e}")
    
    async def _scale_based_on_queue(self, queue_stats: Dict[str, Any]):
        """Scale based on queue length"""
        try:
            pending_tasks = queue_stats.get("task_counts", {}).get("pending", 0)
            running_workers = queue_stats.get("worker_stats", {}).get("total_workers", 0)
            
            if pending_tasks > running_workers * 2:  # More than 2 tasks per worker
                # Scale up
                if running_workers < self.config.max_workers:
                    target_count = min(running_workers + 2, self.config.max_workers)
                    await self.scale_workers(WorkerType.GENERAL_WORKER, target_count)
            
            elif pending_tasks < running_workers * 0.5:  # Less than 0.5 tasks per worker
                # Scale down
                if running_workers > self.config.min_workers:
                    target_count = max(running_workers - 1, self.config.min_workers)
                    await self.scale_workers(WorkerType.GENERAL_WORKER, target_count)
            
        except Exception as e:
            logger.error(f"Queue-based scaling failed: {e}")
    
    async def _scale_based_on_hybrid(self, worker_stats: Dict[str, Any], queue_stats: Dict[str, Any]):
        """Scale based on hybrid metrics"""
        try:
            # Combine CPU, memory, and queue metrics
            avg_cpu = worker_stats.get("total_cpu_usage_percent", 0) / max(worker_stats.get("total_workers", 1), 1)
            avg_memory = worker_stats.get("total_memory_usage_mb", 0) / max(worker_stats.get("total_workers", 1), 1)
            pending_tasks = queue_stats.get("task_counts", {}).get("pending", 0)
            running_workers = worker_stats.get("running_workers", 0)
            
            # Calculate composite score
            cpu_score = min(avg_cpu / 100, 1.0)
            memory_score = min(avg_memory / 1024, 1.0)  # Normalize to 1GB
            queue_score = min(pending_tasks / max(running_workers * 2, 1), 1.0)
            
            composite_score = (cpu_score + memory_score + queue_score) / 3
            
            if composite_score > self.config.scale_up_threshold:
                # Scale up
                if running_workers < self.config.max_workers:
                    target_count = min(running_workers + 2, self.config.max_workers)
                    await self.scale_workers(WorkerType.GENERAL_WORKER, target_count)
            
            elif composite_score < self.config.scale_down_threshold:
                # Scale down
                if running_workers > self.config.min_workers:
                    target_count = max(running_workers - 1, self.config.min_workers)
                    await self.scale_workers(WorkerType.GENERAL_WORKER, target_count)
            
        except Exception as e:
            logger.error(f"Hybrid scaling failed: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.coordination_active:
            try:
                # Update worker metrics in load balancer
                workers = await self.worker_manager.get_all_workers()
                for worker in workers:
                    await self.load_balancer.update_worker_metrics(worker.worker_id, {
                        "active_tasks": worker.active_tasks,
                        "completed_tasks": worker.completed_tasks,
                        "failed_tasks": worker.tasks_failed,
                        "cpu_usage": worker.cpu_usage,
                        "memory_usage": worker.memory_usage,
                        "gpu_usage": worker.gpu_usage
                    })
                
                # Perform auto scaling
                await self.auto_scale()
                
                # Clean up completed tasks
                await self.task_queue.cleanup_completed_tasks()
                
                # Sleep before next iteration
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30)
    
    async def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination statistics"""
        try:
            # Get component stats
            task_queue_stats = await self.task_queue.get_queue_stats()
            worker_stats = await self.worker_manager.get_worker_stats()
            load_balancer_stats = await self.load_balancer.get_load_balancer_stats()
            
            # Calculate distributed task stats
            total_tasks = len(self.distributed_tasks)
            completed_tasks = len([t for t in self.distributed_tasks.values() if t.status == TaskStatus.SUCCESS])
            failed_tasks = len([t for t in self.distributed_tasks.values() if t.status == TaskStatus.FAILURE])
            pending_tasks = len([t for t in self.distributed_tasks.values() if t.status == TaskStatus.PENDING])
            
            return {
                "coordination_active": self.coordination_active,
                "total_distributed_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "pending_tasks": pending_tasks,
                "task_queue_stats": task_queue_stats,
                "worker_stats": worker_stats,
                "load_balancer_stats": load_balancer_stats,
                "scaling_history": self.scaling_history[-10:],  # Last 10 scaling events
                "config": {
                    "coordination_strategy": self.config.coordination_strategy.value,
                    "scaling_policy": self.config.scaling_policy.value,
                    "auto_scaling": self.config.auto_scaling,
                    "min_workers": self.config.min_workers,
                    "max_workers": self.config.max_workers
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get coordination stats: {e}")
            return {}
    
    async def export_coordination_data(self, format: str = "json") -> str:
        """Export coordination data"""
        try:
            if format.lower() == "json":
                data = {
                    "distributed_tasks": [asdict(t) for t in self.distributed_tasks.values()],
                    "task_results": [asdict(r) for r in self.task_results.values()],
                    "stats": await self.get_coordination_stats(),
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Coordination data export failed: {e}")
            return ""
