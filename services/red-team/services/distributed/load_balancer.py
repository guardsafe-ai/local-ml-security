"""
Load Balancer
Implements load balancing for distributed task execution
"""

import asyncio
import logging
import time
import random
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"
    LEAST_CPU = "least_cpu"
    LEAST_MEMORY = "least_memory"


class TaskType(Enum):
    """Task types for routing"""
    ATTACK = "attack"
    ANALYSIS = "analysis"
    CERTIFICATION = "certification"
    PRIVACY_ATTACK = "privacy_attack"
    GENERAL = "general"


@dataclass
class WorkerMetrics:
    """Worker performance metrics"""
    worker_id: str
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_response_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    last_heartbeat: Optional[datetime] = None
    weight: float = 1.0
    health_score: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LoadBalancingConfig:
    """Load balancing configuration"""
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS
    health_check_interval: int = 30  # seconds
    response_time_window: int = 300  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    enable_sticky_sessions: bool = False
    sticky_session_timeout: int = 3600  # seconds
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: float = 0.5  # failure rate
    circuit_breaker_timeout: int = 60  # seconds
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RoutingDecision:
    """Routing decision result"""
    worker_id: str
    confidence: float
    strategy_used: LoadBalancingStrategy
    reasoning: str
    fallback_workers: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.fallback_workers is None:
            self.fallback_workers = []
        if self.metadata is None:
            self.metadata = {}


class LoadBalancer:
    """
    Load Balancer
    Implements load balancing for distributed task execution
    """
    
    def __init__(self, config: LoadBalancingConfig = None):
        """Initialize load balancer"""
        self.config = config or LoadBalancingConfig()
        
        # Worker tracking
        self.workers: Dict[str, WorkerMetrics] = {}
        self.worker_queues: Dict[str, deque] = defaultdict(deque)
        self.round_robin_index: Dict[str, int] = defaultdict(int)
        
        # Performance tracking
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.task_counts: Dict[str, int] = defaultdict(int)
        
        # Circuit breaker
        self.circuit_breakers: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "failures": 0,
            "last_failure": None,
            "state": "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        })
        
        # Sticky sessions
        self.sticky_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Health monitoring
        self.health_check_task = None
        self.monitoring_active = False
        
        logger.info("✅ Initialized Load Balancer")
    
    async def register_worker(self, worker_id: str, worker_type: str = "general", weight: float = 1.0):
        """Register a worker"""
        try:
            self.workers[worker_id] = WorkerMetrics(
                worker_id=worker_id,
                weight=weight,
                last_heartbeat=datetime.utcnow(),
                metadata={"worker_type": worker_type}
            )
            
            # Initialize round robin index
            self.round_robin_index[worker_type] = 0
            
            logger.info(f"Worker registered: {worker_id} (type: {worker_type})")
            
        except Exception as e:
            logger.error(f"Failed to register worker {worker_id}: {e}")
    
    async def unregister_worker(self, worker_id: str):
        """Unregister a worker"""
        try:
            if worker_id in self.workers:
                del self.workers[worker_id]
                
                # Clean up related data
                if worker_id in self.response_times:
                    del self.response_times[worker_id]
                if worker_id in self.task_counts:
                    del self.task_counts[worker_id]
                if worker_id in self.circuit_breakers:
                    del self.circuit_breakers[worker_id]
                
                logger.info(f"Worker unregistered: {worker_id}")
            
        except Exception as e:
            logger.error(f"Failed to unregister worker {worker_id}: {e}")
    
    async def update_worker_metrics(self, worker_id: str, metrics: Dict[str, Any]):
        """Update worker metrics"""
        try:
            if worker_id not in self.workers:
                return
            
            worker = self.workers[worker_id]
            
            # Update basic metrics
            if "active_tasks" in metrics:
                worker.active_tasks = metrics["active_tasks"]
            if "completed_tasks" in metrics:
                worker.completed_tasks = metrics["completed_tasks"]
            if "failed_tasks" in metrics:
                worker.failed_tasks = metrics["failed_tasks"]
            if "cpu_usage" in metrics:
                worker.cpu_usage = metrics["cpu_usage"]
            if "memory_usage" in metrics:
                worker.memory_usage = metrics["memory_usage"]
            if "gpu_usage" in metrics:
                worker.gpu_usage = metrics["gpu_usage"]
            
            worker.last_heartbeat = datetime.utcnow()
            
            # Update health score
            worker.health_score = self._calculate_health_score(worker)
            
        except Exception as e:
            logger.error(f"Failed to update worker metrics for {worker_id}: {e}")
    
    async def record_task_completion(self, worker_id: str, response_time: float, success: bool = True):
        """Record task completion"""
        try:
            if worker_id not in self.workers:
                return
            
            worker = self.workers[worker_id]
            
            # Update response times
            self.response_times[worker_id].append(response_time)
            worker.avg_response_time = statistics.mean(self.response_times[worker_id])
            
            # Update task counts
            if success:
                worker.completed_tasks += 1
            else:
                worker.failed_tasks += 1
            
            # Update circuit breaker
            if not success:
                self._update_circuit_breaker(worker_id, False)
            else:
                self._update_circuit_breaker(worker_id, True)
            
        except Exception as e:
            logger.error(f"Failed to record task completion for {worker_id}: {e}")
    
    async def select_worker(self, 
                          task_type: TaskType, 
                          task_priority: int = 1,
                          session_id: Optional[str] = None) -> RoutingDecision:
        """Select worker for task execution"""
        try:
            # Check for sticky session
            if (self.config.enable_sticky_sessions and 
                session_id and 
                session_id in self.sticky_sessions):
                
                sticky_info = self.sticky_sessions[session_id]
                if (datetime.utcnow() - sticky_info["timestamp"]).seconds < self.config.sticky_session_timeout:
                    worker_id = sticky_info["worker_id"]
                    if worker_id in self.workers and self._is_worker_healthy(worker_id):
                        return RoutingDecision(
                            worker_id=worker_id,
                            confidence=0.9,
                            strategy_used=LoadBalancingStrategy.CONSISTENT_HASH,
                            reasoning="Sticky session routing"
                        )
            
            # Filter available workers
            available_workers = self._get_available_workers(task_type)
            
            if not available_workers:
                raise Exception("No available workers for task type: " + task_type.value)
            
            # Apply circuit breaker filtering
            healthy_workers = [w for w in available_workers if self._is_worker_healthy(w.worker_id)]
            
            if not healthy_workers:
                # All workers are unhealthy, use any available worker
                healthy_workers = available_workers
            
            # Select worker based on strategy
            selected_worker = await self._select_worker_by_strategy(healthy_workers, task_type, task_priority)
            
            # Update sticky session if enabled
            if self.config.enable_sticky_sessions and session_id:
                self.sticky_sessions[session_id] = {
                    "worker_id": selected_worker.worker_id,
                    "timestamp": datetime.utcnow()
                }
            
            # Create routing decision
            decision = RoutingDecision(
                worker_id=selected_worker.worker_id,
                confidence=self._calculate_selection_confidence(selected_worker, healthy_workers),
                strategy_used=self.config.strategy,
                reasoning=self._generate_selection_reasoning(selected_worker, healthy_workers),
                fallback_workers=[w.worker_id for w in healthy_workers[:3] if w.worker_id != selected_worker.worker_id]
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Failed to select worker: {e}")
            raise
    
    def _get_available_workers(self, task_type: TaskType) -> List[WorkerMetrics]:
        """Get available workers for task type"""
        try:
            available_workers = []
            
            for worker in self.workers.values():
                # Check if worker supports this task type
                worker_type = worker.metadata.get("worker_type", "general")
                
                if (task_type == TaskType.GENERAL or 
                    worker_type == task_type.value or 
                    worker_type == "general"):
                    
                    # Check if worker is not overloaded
                    if worker.active_tasks < 10:  # Configurable threshold
                        available_workers.append(worker)
            
            return available_workers
            
        except Exception as e:
            logger.error(f"Failed to get available workers: {e}")
            return []
    
    async def _select_worker_by_strategy(self, 
                                       workers: List[WorkerMetrics], 
                                       task_type: TaskType, 
                                       task_priority: int) -> WorkerMetrics:
        """Select worker based on configured strategy"""
        try:
            if not workers:
                raise Exception("No workers available")
            
            if len(workers) == 1:
                return workers[0]
            
            if self.config.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_selection(workers, task_type)
            
            elif self.config.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_selection(workers)
            
            elif self.config.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return self._least_response_time_selection(workers)
            
            elif self.config.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection(workers, task_type)
            
            elif self.config.strategy == LoadBalancingStrategy.RANDOM:
                return self._random_selection(workers)
            
            elif self.config.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                return self._consistent_hash_selection(workers, task_type)
            
            elif self.config.strategy == LoadBalancingStrategy.LEAST_CPU:
                return self._least_cpu_selection(workers)
            
            elif self.config.strategy == LoadBalancingStrategy.LEAST_MEMORY:
                return self._least_memory_selection(workers)
            
            else:
                # Default to least connections
                return self._least_connections_selection(workers)
                
        except Exception as e:
            logger.error(f"Failed to select worker by strategy: {e}")
            return workers[0]
    
    def _round_robin_selection(self, workers: List[WorkerMetrics], task_type: TaskType) -> WorkerMetrics:
        """Round robin selection"""
        worker_type = task_type.value
        index = self.round_robin_index[worker_type]
        selected_worker = workers[index % len(workers)]
        self.round_robin_index[worker_type] = (index + 1) % len(workers)
        return selected_worker
    
    def _least_connections_selection(self, workers: List[WorkerMetrics]) -> WorkerMetrics:
        """Least connections selection"""
        return min(workers, key=lambda w: w.active_tasks)
    
    def _least_response_time_selection(self, workers: List[WorkerMetrics]) -> WorkerMetrics:
        """Least response time selection"""
        return min(workers, key=lambda w: w.avg_response_time if w.avg_response_time > 0 else float('inf'))
    
    def _weighted_round_robin_selection(self, workers: List[WorkerMetrics], task_type: TaskType) -> WorkerMetrics:
        """Weighted round robin selection"""
        # Calculate total weight
        total_weight = sum(w.weight for w in workers)
        
        # Select based on weight
        random_value = random.uniform(0, total_weight)
        current_weight = 0
        
        for worker in workers:
            current_weight += worker.weight
            if random_value <= current_weight:
                return worker
        
        return workers[-1]  # Fallback
    
    def _random_selection(self, workers: List[WorkerMetrics]) -> WorkerMetrics:
        """Random selection"""
        return random.choice(workers)
    
    def _consistent_hash_selection(self, workers: List[WorkerMetrics], task_type: TaskType) -> WorkerMetrics:
        """Consistent hash selection"""
        # Simple hash-based selection
        hash_value = hash(task_type.value) % len(workers)
        return workers[hash_value]
    
    def _least_cpu_selection(self, workers: List[WorkerMetrics]) -> WorkerMetrics:
        """Least CPU usage selection"""
        return min(workers, key=lambda w: w.cpu_usage)
    
    def _least_memory_selection(self, workers: List[WorkerMetrics]) -> WorkerMetrics:
        """Least memory usage selection"""
        return min(workers, key=lambda w: w.memory_usage)
    
    def _is_worker_healthy(self, worker_id: str) -> bool:
        """Check if worker is healthy"""
        try:
            if worker_id not in self.workers:
                return False
            
            worker = self.workers[worker_id]
            
            # Check if worker is responsive
            if (worker.last_heartbeat and 
                datetime.utcnow() - worker.last_heartbeat > timedelta(minutes=5)):
                return False
            
            # Check circuit breaker
            if self.config.enable_circuit_breaker:
                circuit_breaker = self.circuit_breakers[worker_id]
                if circuit_breaker["state"] == "OPEN":
                    # Check if timeout has passed
                    if (circuit_breaker["last_failure"] and 
                        datetime.utcnow() - circuit_breaker["last_failure"] > timedelta(seconds=self.config.circuit_breaker_timeout)):
                        circuit_breaker["state"] = "HALF_OPEN"
                        return True
                    return False
            
            # Check health score
            return worker.health_score > 0.5
            
        except Exception as e:
            logger.error(f"Failed to check worker health for {worker_id}: {e}")
            return False
    
    def _calculate_health_score(self, worker: WorkerMetrics) -> float:
        """Calculate worker health score"""
        try:
            score = 1.0
            
            # Penalize high CPU usage
            if worker.cpu_usage > 80:
                score -= 0.3
            elif worker.cpu_usage > 60:
                score -= 0.1
            
            # Penalize high memory usage
            if worker.memory_usage > 80:
                score -= 0.3
            elif worker.memory_usage > 60:
                score -= 0.1
            
            # Penalize high active task count
            if worker.active_tasks > 8:
                score -= 0.2
            elif worker.active_tasks > 5:
                score -= 0.1
            
            # Penalize high failure rate
            total_tasks = worker.completed_tasks + worker.failed_tasks
            if total_tasks > 0:
                failure_rate = worker.failed_tasks / total_tasks
                if failure_rate > 0.5:
                    score -= 0.4
                elif failure_rate > 0.2:
                    score -= 0.2
            
            # Penalize old heartbeat
            if worker.last_heartbeat:
                age = datetime.utcnow() - worker.last_heartbeat
                if age > timedelta(minutes=2):
                    score -= 0.3
                elif age > timedelta(minutes=1):
                    score -= 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Failed to calculate health score: {e}")
            return 0.0
    
    def _update_circuit_breaker(self, worker_id: str, success: bool):
        """Update circuit breaker state"""
        try:
            circuit_breaker = self.circuit_breakers[worker_id]
            
            if success:
                if circuit_breaker["state"] == "HALF_OPEN":
                    circuit_breaker["state"] = "CLOSED"
                circuit_breaker["failures"] = 0
            else:
                circuit_breaker["failures"] += 1
                circuit_breaker["last_failure"] = datetime.utcnow()
                
                # Check if threshold exceeded
                if circuit_breaker["failures"] >= 5:  # Configurable threshold
                    circuit_breaker["state"] = "OPEN"
            
        except Exception as e:
            logger.error(f"Failed to update circuit breaker for {worker_id}: {e}")
    
    def _calculate_selection_confidence(self, selected_worker: WorkerMetrics, available_workers: List[WorkerMetrics]) -> float:
        """Calculate selection confidence"""
        try:
            # Base confidence on health score and relative performance
            base_confidence = selected_worker.health_score
            
            # Adjust based on relative performance
            if len(available_workers) > 1:
                avg_health = statistics.mean(w.health_score for w in available_workers)
                if selected_worker.health_score > avg_health:
                    base_confidence += 0.1
                else:
                    base_confidence -= 0.1
            
            return max(0.0, min(1.0, base_confidence))
            
        except Exception as e:
            logger.error(f"Failed to calculate selection confidence: {e}")
            return 0.5
    
    def _generate_selection_reasoning(self, selected_worker: WorkerMetrics, available_workers: List[WorkerMetrics]) -> str:
        """Generate selection reasoning"""
        try:
            reasoning_parts = []
            
            # Strategy
            reasoning_parts.append(f"Strategy: {self.config.strategy.value}")
            
            # Health score
            reasoning_parts.append(f"Health score: {selected_worker.health_score:.2f}")
            
            # Active tasks
            reasoning_parts.append(f"Active tasks: {selected_worker.active_tasks}")
            
            # Resource usage
            if selected_worker.cpu_usage > 0:
                reasoning_parts.append(f"CPU usage: {selected_worker.cpu_usage:.1f}%")
            
            if selected_worker.memory_usage > 0:
                reasoning_parts.append(f"Memory usage: {selected_worker.memory_usage:.1f}%")
            
            # Response time
            if selected_worker.avg_response_time > 0:
                reasoning_parts.append(f"Avg response time: {selected_worker.avg_response_time:.2f}s")
            
            return "; ".join(reasoning_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate selection reasoning: {e}")
            return "Unknown"
    
    async def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        try:
            total_workers = len(self.workers)
            healthy_workers = len([w for w in self.workers.values() if self._is_worker_healthy(w.worker_id)])
            
            # Calculate average metrics
            avg_cpu = statistics.mean([w.cpu_usage for w in self.workers.values()]) if self.workers else 0
            avg_memory = statistics.mean([w.memory_usage for w in self.workers.values()]) if self.workers else 0
            avg_response_time = statistics.mean([w.avg_response_time for w in self.workers.values() if w.avg_response_time > 0]) if self.workers else 0
            
            # Circuit breaker stats
            circuit_breaker_stats = {}
            for worker_id, cb in self.circuit_breakers.items():
                circuit_breaker_stats[worker_id] = {
                    "state": cb["state"],
                    "failures": cb["failures"],
                    "last_failure": cb["last_failure"].isoformat() if cb["last_failure"] else None
                }
            
            return {
                "total_workers": total_workers,
                "healthy_workers": healthy_workers,
                "unhealthy_workers": total_workers - healthy_workers,
                "avg_cpu_usage": avg_cpu,
                "avg_memory_usage": avg_memory,
                "avg_response_time": avg_response_time,
                "strategy": self.config.strategy.value,
                "circuit_breakers": circuit_breaker_stats,
                "sticky_sessions": len(self.sticky_sessions),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get load balancer stats: {e}")
            return {}
    
    async def export_load_balancer_data(self, format: str = "json") -> str:
        """Export load balancer data"""
        try:
            if format.lower() == "json":
                data = {
                    "workers": [asdict(w) for w in self.workers.values()],
                    "stats": await self.get_load_balancer_stats(),
                    "config": asdict(self.config),
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Load balancer data export failed: {e}")
            return ""
