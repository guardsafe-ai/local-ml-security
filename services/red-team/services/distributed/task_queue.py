"""
Task Queue Manager
Implements distributed task queue using Celery and Redis
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import redis
from celery import Celery
from celery.result import AsyncResult
from celery.exceptions import Retry, WorkerLostError
import pickle
import base64

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"


class TaskPriority(Enum):
    """Task priority enumeration"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class TaskRequest:
    """Task request structure"""
    task_id: str
    task_name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    eta: Optional[datetime] = None
    expires: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TaskResult:
    """Task result structure"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TaskQueueManager:
    """
    Task Queue Manager
    Manages distributed task execution using Celery and Redis
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379/0",
                 celery_broker: str = "redis://localhost:6379/0",
                 celery_backend: str = "redis://localhost:6379/0"):
        """Initialize task queue manager"""
        self.redis_url = redis_url
        self.celery_broker = celery_broker
        self.celery_backend = celery_backend
        
        # Initialize Redis connection
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        
        # Initialize Celery app
        self.celery_app = Celery(
            'red_team_tasks',
            broker=celery_broker,
            backend=celery_backend
        )
        
        # Configure Celery
        self._configure_celery()
        
        # Task tracking
        self.active_tasks: Dict[str, AsyncResult] = {}
        self.task_results: Dict[str, TaskResult] = {}
        
        logger.info("✅ Initialized Task Queue Manager")
    
    def _configure_celery(self):
        """Configure Celery settings"""
        self.celery_app.conf.update(
            task_serializer='pickle',
            accept_content=['pickle'],
            result_serializer='pickle',
            timezone='UTC',
            enable_utc=True,
            task_track_started=True,
            task_time_limit=3600,  # 1 hour
            task_soft_time_limit=3300,  # 55 minutes
            worker_prefetch_multiplier=1,
            task_acks_late=True,
            worker_disable_rate_limits=True,
            task_routes={
                'red_team_tasks.execute_attack': {'queue': 'attack_queue'},
                'red_team_tasks.execute_analysis': {'queue': 'analysis_queue'},
                'red_team_tasks.execute_certification': {'queue': 'certification_queue'},
                'red_team_tasks.execute_privacy_attack': {'queue': 'privacy_queue'},
            },
            task_annotations={
                'red_team_tasks.execute_attack': {'rate_limit': '10/m'},
                'red_team_tasks.execute_analysis': {'rate_limit': '5/m'},
                'red_team_tasks.execute_certification': {'rate_limit': '2/m'},
                'red_team_tasks.execute_privacy_attack': {'rate_limit': '3/m'},
            }
        )
    
    async def submit_task(self, 
                         task_name: str,
                         args: List[Any] = None,
                         kwargs: Dict[str, Any] = None,
                         priority: TaskPriority = TaskPriority.NORMAL,
                         eta: Optional[datetime] = None,
                         expires: Optional[datetime] = None,
                         max_retries: int = 3) -> str:
        """Submit a task to the queue"""
        try:
            if args is None:
                args = []
            if kwargs is None:
                kwargs = {}
            
            # Generate unique task ID
            task_id = str(uuid.uuid4())
            
            # Create task request
            task_request = TaskRequest(
                task_id=task_id,
                task_name=task_name,
                args=args,
                kwargs=kwargs,
                priority=priority,
                eta=eta,
                expires=expires,
                max_retries=max_retries
            )
            
            # Submit to Celery
            celery_task = self.celery_app.send_task(
                task_name,
                args=args,
                kwargs=kwargs,
                task_id=task_id,
                eta=eta,
                expires=expires,
                retry=True,
                retry_policy={
                    'max_retries': max_retries,
                    'interval_start': 0,
                    'interval_step': 0.2,
                    'interval_max': 0.2,
                }
            )
            
            # Track task
            self.active_tasks[task_id] = celery_task
            
            # Store task request in Redis
            await self._store_task_request(task_request)
            
            logger.info(f"Task submitted: {task_id} ({task_name})")
            return task_id
            
        except Exception as e:
            logger.error(f"Task submission failed: {e}")
            raise
    
    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Get task status"""
        try:
            if task_id in self.active_tasks:
                celery_task = self.active_tasks[task_id]
                celery_status = celery_task.status
                
                # Map Celery status to our status
                status_mapping = {
                    'PENDING': TaskStatus.PENDING,
                    'STARTED': TaskStatus.STARTED,
                    'SUCCESS': TaskStatus.SUCCESS,
                    'FAILURE': TaskStatus.FAILURE,
                    'RETRY': TaskStatus.RETRY,
                    'REVOKED': TaskStatus.REVOKED
                }
                
                return status_mapping.get(celery_status, TaskStatus.PENDING)
            else:
                # Check if task is in results
                if task_id in self.task_results:
                    return self.task_results[task_id].status
                else:
                    return TaskStatus.PENDING
                    
        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {e}")
            return TaskStatus.PENDING
    
    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get task result"""
        try:
            if task_id in self.task_results:
                return self.task_results[task_id]
            
            if task_id in self.active_tasks:
                celery_task = self.active_tasks[task_id]
                
                if celery_task.ready():
                    # Task completed
                    status = await self.get_task_status(task_id)
                    
                    if status == TaskStatus.SUCCESS:
                        result = celery_task.result
                        error = None
                    else:
                        result = None
                        error = str(celery_task.result) if celery_task.result else "Unknown error"
                    
                    # Create task result
                    task_result = TaskResult(
                        task_id=task_id,
                        status=status,
                        result=result,
                        error=error,
                        completed_at=datetime.utcnow()
                    )
                    
                    # Store result
                    self.task_results[task_id] = task_result
                    
                    # Remove from active tasks
                    del self.active_tasks[task_id]
                    
                    return task_result
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get task result for {task_id}: {e}")
            return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        try:
            if task_id in self.active_tasks:
                celery_task = self.active_tasks[task_id]
                celery_task.revoke(terminate=True)
                
                # Update status
                if task_id in self.task_results:
                    self.task_results[task_id].status = TaskStatus.REVOKED
                else:
                    self.task_results[task_id] = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.REVOKED,
                        completed_at=datetime.utcnow()
                    )
                
                # Remove from active tasks
                del self.active_tasks[task_id]
                
                logger.info(f"Task cancelled: {task_id}")
                return True
            else:
                logger.warning(f"Task not found for cancellation: {task_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        try:
            # Get Celery stats
            inspect = self.celery_app.control.inspect()
            
            # Get active tasks
            active_tasks = inspect.active()
            scheduled_tasks = inspect.scheduled()
            reserved_tasks = inspect.reserved()
            
            # Get worker stats
            worker_stats = inspect.stats()
            
            # Count tasks by status
            task_counts = {
                'pending': 0,
                'started': 0,
                'success': 0,
                'failure': 0,
                'retry': 0,
                'revoked': 0
            }
            
            for task_id, task_result in self.task_results.items():
                status = task_result.status.value.lower()
                if status in task_counts:
                    task_counts[status] += 1
            
            # Count active tasks
            task_counts['pending'] += len(self.active_tasks)
            
            return {
                'active_tasks': active_tasks or {},
                'scheduled_tasks': scheduled_tasks or {},
                'reserved_tasks': reserved_tasks or {},
                'worker_stats': worker_stats or {},
                'task_counts': task_counts,
                'total_tasks': sum(task_counts.values()),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}
    
    async def cleanup_completed_tasks(self, older_than_hours: int = 24):
        """Clean up completed tasks older than specified hours"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
            
            tasks_to_remove = []
            for task_id, task_result in self.task_results.items():
                if (task_result.completed_at and 
                    task_result.completed_at < cutoff_time and
                    task_result.status in [TaskStatus.SUCCESS, TaskStatus.FAILURE, TaskStatus.REVOKED]):
                    tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.task_results[task_id]
            
            logger.info(f"Cleaned up {len(tasks_to_remove)} completed tasks")
            return len(tasks_to_remove)
            
        except Exception as e:
            logger.error(f"Failed to cleanup completed tasks: {e}")
            return 0
    
    async def _store_task_request(self, task_request: TaskRequest):
        """Store task request in Redis"""
        try:
            # Serialize task request
            task_data = json.dumps(asdict(task_request), default=str)
            
            # Store in Redis with expiration
            self.redis_client.setex(
                f"task_request:{task_request.task_id}",
                3600,  # 1 hour expiration
                task_data
            )
            
        except Exception as e:
            logger.error(f"Failed to store task request: {e}")
    
    async def _get_task_request(self, task_id: str) -> Optional[TaskRequest]:
        """Get task request from Redis"""
        try:
            task_data = self.redis_client.get(f"task_request:{task_id}")
            if task_data:
                task_dict = json.loads(task_data)
                return TaskRequest(**task_dict)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get task request: {e}")
            return None
    
    async def get_task_history(self, limit: int = 100) -> List[TaskResult]:
        """Get task history"""
        try:
            # Get all task results sorted by completion time
            all_results = list(self.task_results.values())
            all_results.sort(key=lambda x: x.completed_at or datetime.min, reverse=True)
            
            return all_results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get task history: {e}")
            return []
    
    async def export_task_data(self, format: str = "json") -> str:
        """Export task data"""
        try:
            if format.lower() == "json":
                data = {
                    "active_tasks": list(self.active_tasks.keys()),
                    "task_results": [asdict(r) for r in self.task_results.values()],
                    "queue_stats": await self.get_queue_stats(),
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Task data export failed: {e}")
            return ""


# Celery task definitions
@celery_app.task(bind=True, name='red_team_tasks.execute_attack')
def execute_attack_task(self, attack_type: str, model_data: bytes, input_data: bytes, config: dict):
    """Execute attack task"""
    try:
        # Deserialize data
        model = pickle.loads(model_data)
        inputs = pickle.loads(input_data)
        
        # Import attack modules dynamically
        if attack_type == "gradient_attack":
            from ..adversarial_ml.gradient_attacks import GradientBasedAttacks
            attacker = GradientBasedAttacks()
            result = attacker.fgsm_attack(inputs, **config)
        
        elif attack_type == "word_attack":
            from ..adversarial_ml.word_level_attacks import TextFoolerAttack
            attacker = TextFoolerAttack()
            result = attacker.attack(inputs, **config)
        
        elif attack_type == "multi_turn_attack":
            from ..adversarial_ml.multi_turn_attacks import MultiTurnAttacks
            attacker = MultiTurnAttacks()
            result = attacker.tap_attack(inputs, **config)
        
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        return {
            "success": True,
            "result": result,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(f"Attack task failed: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(bind=True, name='red_team_tasks.execute_analysis')
def execute_analysis_task(self, analysis_type: str, model_data: bytes, input_data: bytes, config: dict):
    """Execute analysis task"""
    try:
        # Deserialize data
        model = pickle.loads(model_data)
        inputs = pickle.loads(input_data)
        
        # Import analysis modules dynamically
        if analysis_type == "behavior_analysis":
            from ..behavior_analysis.behavior_analyzer import BehaviorAnalyzer
            analyzer = BehaviorAnalyzer()
            result = await analyzer.analyze_behavior(inputs, config)
        
        elif analysis_type == "privacy_analysis":
            from ..privacy_attacks.privacy_attack_manager import PrivacyAttackManager
            analyzer = PrivacyAttackManager()
            result = await analyzer.perform_privacy_analysis(inputs, config)
        
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        return {
            "success": True,
            "result": result,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(f"Analysis task failed: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(bind=True, name='red_team_tasks.execute_certification')
def execute_certification_task(self, cert_type: str, model_data: bytes, input_data: bytes, config: dict):
    """Execute certification task"""
    try:
        # Deserialize data
        model = pickle.loads(model_data)
        inputs = pickle.loads(input_data)
        
        # Import certification modules dynamically
        if cert_type == "randomized_smoothing":
            from ..certification.randomized_smoothing import RandomizedSmoothingCertifier
            certifier = RandomizedSmoothingCertifier()
            result = await certifier.certify_robustness(model, inputs, **config)
        
        elif cert_type == "ibp":
            from ..certification.interval_bound_propagation import IBPCertifier
            certifier = IBPCertifier()
            result = await certifier.certify_robustness(model, inputs, **config)
        
        else:
            raise ValueError(f"Unknown certification type: {cert_type}")
        
        return {
            "success": True,
            "result": result,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(f"Certification task failed: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(bind=True, name='red_team_tasks.execute_privacy_attack')
def execute_privacy_attack_task(self, attack_type: str, model_data: bytes, target_data: bytes, config: dict):
    """Execute privacy attack task"""
    try:
        # Deserialize data
        model = pickle.loads(model_data)
        targets = pickle.loads(target_data)
        
        # Import privacy attack modules dynamically
        if attack_type == "membership_inference":
            from ..privacy_attacks.membership_inference import MembershipInferenceAttacker
            attacker = MembershipInferenceAttacker()
            result = await attacker.perform_attack(model, targets, **config)
        
        elif attack_type == "model_inversion":
            from ..privacy_attacks.model_inversion import ModelInversionAttacker
            attacker = ModelInversionAttacker()
            result = await attacker.perform_inversion(model, **config)
        
        elif attack_type == "data_extraction":
            from ..privacy_attacks.data_extraction import DataExtractionAttacker
            attacker = DataExtractionAttacker()
            result = await attacker.perform_extraction(model, **config)
        
        else:
            raise ValueError(f"Unknown privacy attack type: {attack_type}")
        
        return {
            "success": True,
            "result": result,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(f"Privacy attack task failed: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=3)
