"""
Batch Optimizer
Implements batch inference optimization for improved performance
"""

import asyncio
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class BatchStrategy(Enum):
    """Batch processing strategies"""
    FIXED_SIZE = "fixed_size"
    DYNAMIC_SIZE = "dynamic_size"
    TIME_BASED = "time_based"
    ADAPTIVE = "adaptive"


class OptimizationLevel(Enum):
    """Optimization levels"""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    MAXIMUM = "maximum"


@dataclass
class BatchConfig:
    """Batch processing configuration"""
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    max_batch_size: int = 32
    min_batch_size: int = 1
    batch_timeout: float = 0.1  # seconds
    max_wait_time: float = 1.0  # seconds
    optimization_level: OptimizationLevel = OptimizationLevel.ADVANCED
    enable_gpu_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    max_parallel_workers: int = 4
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BatchRequest:
    """Batch processing request"""
    request_id: str
    data: Any
    task_type: str
    priority: int = 1
    created_at: datetime = None
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BatchResult:
    """Batch processing result"""
    request_id: str
    result: Any
    processing_time: float
    batch_size: int
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BatchStats:
    """Batch processing statistics"""
    total_requests: int = 0
    total_batches: int = 0
    avg_batch_size: float = 0.0
    avg_processing_time: float = 0.0
    throughput_per_second: float = 0.0
    gpu_utilization: float = 0.0
    memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BatchOptimizer:
    """
    Batch Optimizer
    Implements batch inference optimization for improved performance
    """
    
    def __init__(self, config: BatchConfig = None):
        """Initialize batch optimizer"""
        self.config = config or BatchConfig()
        
        # Request queues by task type
        self.request_queues: Dict[str, deque] = defaultdict(deque)
        self.queue_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
        # Processing state
        self.processing_active = False
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        
        # Statistics
        self.stats = BatchStats()
        self.processing_times: deque = deque(maxlen=1000)
        self.batch_sizes: deque = deque(maxlen=1000)
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_parallel_workers)
        
        # Model optimization cache
        self.optimized_models: Dict[str, Any] = {}
        
        logger.info("✅ Initialized Batch Optimizer")
    
    async def start_processing(self):
        """Start batch processing"""
        try:
            if self.processing_active:
                return
            
            self.processing_active = True
            
            # Start processing tasks for each queue
            for task_type in self.request_queues.keys():
                task = asyncio.create_task(self._process_batches(task_type))
                self.processing_tasks[task_type] = task
            
            logger.info("Batch processing started")
            
        except Exception as e:
            logger.error(f"Failed to start batch processing: {e}")
            raise
    
    async def stop_processing(self):
        """Stop batch processing"""
        try:
            self.processing_active = False
            
            # Cancel all processing tasks
            for task in self.processing_tasks.values():
                task.cancel()
            
            # Wait for tasks to complete
            if self.processing_tasks:
                await asyncio.gather(*self.processing_tasks.values(), return_exceptions=True)
            
            self.processing_tasks.clear()
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            logger.info("Batch processing stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop batch processing: {e}")
    
    async def submit_request(self, 
                           data: Any, 
                           task_type: str, 
                           priority: int = 1,
                           timeout: Optional[float] = None) -> str:
        """Submit a request for batch processing"""
        try:
            request_id = f"{task_type}_{int(time.time() * 1000)}_{id(data)}"
            
            request = BatchRequest(
                request_id=request_id,
                data=data,
                task_type=task_type,
                priority=priority,
                timeout=timeout
            )
            
            # Add to appropriate queue
            with self.queue_locks[task_type]:
                self.request_queues[task_type].append(request)
            
            # Start processing task if not already running
            if task_type not in self.processing_tasks:
                task = asyncio.create_task(self._process_batches(task_type))
                self.processing_tasks[task_type] = task
            
            self.stats.total_requests += 1
            
            logger.debug(f"Request submitted: {request_id} ({task_type})")
            return request_id
            
        except Exception as e:
            logger.error(f"Failed to submit request: {e}")
            raise
    
    async def process_batch(self, 
                          requests: List[BatchRequest], 
                          model: nn.Module,
                          processor: Callable) -> List[BatchResult]:
        """Process a batch of requests"""
        try:
            if not requests:
                return []
            
            start_time = time.time()
            batch_size = len(requests)
            
            # Prepare batch data
            batch_data = await self._prepare_batch_data(requests)
            
            # Optimize model if needed
            optimized_model = await self._get_optimized_model(model, batch_data)
            
            # Process batch
            if asyncio.iscoroutinefunction(processor):
                batch_results = await processor(optimized_model, batch_data)
            else:
                # Run in thread pool for CPU-bound operations
                loop = asyncio.get_event_loop()
                batch_results = await loop.run_in_executor(
                    self.thread_pool, 
                    processor, 
                    optimized_model, 
                    batch_data
                )
            
            # Create individual results
            results = []
            processing_time = time.time() - start_time
            
            for i, request in enumerate(requests):
                result = BatchResult(
                    request_id=request.request_id,
                    result=batch_results[i] if i < len(batch_results) else None,
                    processing_time=processing_time,
                    batch_size=batch_size,
                    success=True
                )
                results.append(result)
            
            # Update statistics
            self._update_stats(processing_time, batch_size)
            
            logger.debug(f"Batch processed: {batch_size} requests in {processing_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Return error results
            results = []
            for request in requests:
                result = BatchResult(
                    request_id=request.request_id,
                    result=None,
                    processing_time=0.0,
                    batch_size=len(requests),
                    success=False,
                    error=str(e)
                )
                results.append(result)
            return results
    
    async def _process_batches(self, task_type: str):
        """Process batches for a specific task type"""
        try:
            while self.processing_active:
                # Get requests for batching
                requests = await self._get_batch_requests(task_type)
                
                if not requests:
                    await asyncio.sleep(0.01)  # Short sleep if no requests
                    continue
                
                # Process batch (this would need to be implemented based on specific use case)
                # For now, we'll just simulate processing
                await self._simulate_batch_processing(requests)
                
        except asyncio.CancelledError:
            logger.info(f"Batch processing cancelled for {task_type}")
        except Exception as e:
            logger.error(f"Batch processing error for {task_type}: {e}")
    
    async def _get_batch_requests(self, task_type: str) -> List[BatchRequest]:
        """Get requests for batching based on strategy"""
        try:
            with self.queue_locks[task_type]:
                if not self.request_queues[task_type]:
                    return []
                
                if self.config.strategy == BatchStrategy.FIXED_SIZE:
                    return self._get_fixed_size_batch(task_type)
                elif self.config.strategy == BatchStrategy.DYNAMIC_SIZE:
                    return self._get_dynamic_size_batch(task_type)
                elif self.config.strategy == BatchStrategy.TIME_BASED:
                    return self._get_time_based_batch(task_type)
                elif self.config.strategy == BatchStrategy.ADAPTIVE:
                    return self._get_adaptive_batch(task_type)
                else:
                    return self._get_fixed_size_batch(task_type)
                    
        except Exception as e:
            logger.error(f"Failed to get batch requests for {task_type}: {e}")
            return []
    
    def _get_fixed_size_batch(self, task_type: str) -> List[BatchRequest]:
        """Get fixed size batch"""
        requests = []
        queue = self.request_queues[task_type]
        
        while len(requests) < self.config.max_batch_size and queue:
            requests.append(queue.popleft())
        
        return requests
    
    def _get_dynamic_size_batch(self, task_type: str) -> List[BatchRequest]:
        """Get dynamic size batch based on queue length"""
        queue = self.request_queues[task_type]
        queue_size = len(queue)
        
        # Adjust batch size based on queue length
        if queue_size >= 20:
            batch_size = min(self.config.max_batch_size, queue_size)
        elif queue_size >= 10:
            batch_size = min(self.config.max_batch_size // 2, queue_size)
        else:
            batch_size = min(self.config.min_batch_size, queue_size)
        
        requests = []
        for _ in range(min(batch_size, queue_size)):
            if queue:
                requests.append(queue.popleft())
        
        return requests
    
    def _get_time_based_batch(self, task_type: str) -> List[BatchRequest]:
        """Get time-based batch"""
        requests = []
        queue = self.request_queues[task_type]
        start_time = time.time()
        
        while (len(requests) < self.config.max_batch_size and 
               queue and 
               time.time() - start_time < self.config.batch_timeout):
            requests.append(queue.popleft())
        
        return requests
    
    def _get_adaptive_batch(self, task_type: str) -> List[BatchRequest]:
        """Get adaptive batch based on performance metrics"""
        queue = self.request_queues[task_type]
        queue_size = len(queue)
        
        if not queue_size:
            return []
        
        # Calculate optimal batch size based on recent performance
        if self.processing_times:
            avg_processing_time = np.mean(list(self.processing_times)[-10:])  # Last 10 batches
            avg_batch_size = np.mean(list(self.batch_sizes)[-10:]) if self.batch_sizes else 1
            
            # Estimate optimal batch size
            if avg_processing_time > 0 and avg_batch_size > 0:
                throughput = avg_batch_size / avg_processing_time
                optimal_batch_size = min(int(throughput * self.config.batch_timeout), self.config.max_batch_size)
            else:
                optimal_batch_size = self.config.max_batch_size
        else:
            optimal_batch_size = self.config.max_batch_size
        
        # Ensure minimum batch size
        optimal_batch_size = max(optimal_batch_size, self.config.min_batch_size)
        
        # Get batch
        requests = []
        for _ in range(min(optimal_batch_size, queue_size)):
            requests.append(queue.popleft())
        
        return requests
    
    async def _prepare_batch_data(self, requests: List[BatchRequest]) -> Any:
        """Prepare batch data from requests"""
        try:
            # Extract data from requests
            data_list = [req.data for req in requests]
            
            # Convert to batch format
            if isinstance(data_list[0], torch.Tensor):
                return torch.stack(data_list)
            elif isinstance(data_list[0], np.ndarray):
                return np.stack(data_list)
            elif isinstance(data_list[0], (list, tuple)):
                return data_list
            else:
                return data_list
                
        except Exception as e:
            logger.error(f"Failed to prepare batch data: {e}")
            return data_list
    
    async def _get_optimized_model(self, model: nn.Module, batch_data: Any) -> nn.Module:
        """Get optimized model for batch processing"""
        try:
            model_key = f"{type(model).__name__}_{hash(str(batch_data.shape) if hasattr(batch_data, 'shape') else str(batch_data))}"
            
            if model_key in self.optimized_models:
                return self.optimized_models[model_key]
            
            # Create optimized model
            optimized_model = await self._optimize_model(model, batch_data)
            self.optimized_models[model_key] = optimized_model
            
            return optimized_model
            
        except Exception as e:
            logger.error(f"Failed to get optimized model: {e}")
            return model
    
    async def _optimize_model(self, model: nn.Module, batch_data: Any) -> nn.Module:
        """Optimize model for batch processing"""
        try:
            optimized_model = model
            
            if self.config.optimization_level == OptimizationLevel.BASIC:
                # Basic optimizations
                optimized_model = optimized_model.eval()
                
            elif self.config.optimization_level == OptimizationLevel.ADVANCED:
                # Advanced optimizations
                optimized_model = optimized_model.eval()
                
                # Enable optimizations
                if hasattr(torch, 'jit'):
                    try:
                        optimized_model = torch.jit.optimize_for_inference(optimized_model)
                    except Exception:
                        pass
                
            elif self.config.optimization_level == OptimizationLevel.MAXIMUM:
                # Maximum optimizations
                optimized_model = optimized_model.eval()
                
                # JIT compilation
                if hasattr(torch, 'jit'):
                    try:
                        optimized_model = torch.jit.script(optimized_model)
                    except Exception:
                        try:
                            optimized_model = torch.jit.trace(optimized_model, batch_data[:1])
                        except Exception:
                            pass
                
                # Additional optimizations
                if hasattr(torch, 'compile'):
                    try:
                        optimized_model = torch.compile(optimized_model)
                    except Exception:
                        pass
            
            return optimized_model
            
        except Exception as e:
            logger.error(f"Failed to optimize model: {e}")
            return model
    
    async def _simulate_batch_processing(self, requests: List[BatchRequest]):
        """Simulate batch processing (placeholder)"""
        try:
            # This is a placeholder - in real implementation, this would call the actual processor
            processing_time = np.random.uniform(0.01, 0.1)  # Simulate processing time
            await asyncio.sleep(processing_time)
            
            # Update statistics
            self._update_stats(processing_time, len(requests))
            
        except Exception as e:
            logger.error(f"Simulated batch processing failed: {e}")
    
    def _update_stats(self, processing_time: float, batch_size: int):
        """Update processing statistics"""
        try:
            self.processing_times.append(processing_time)
            self.batch_sizes.append(batch_size)
            
            # Update stats
            self.stats.total_batches += 1
            self.stats.avg_batch_size = np.mean(list(self.batch_sizes))
            self.stats.avg_processing_time = np.mean(list(self.processing_times))
            
            if self.stats.avg_processing_time > 0:
                self.stats.throughput_per_second = self.stats.avg_batch_size / self.stats.avg_processing_time
            
            # Update memory usage
            try:
                import psutil
                process = psutil.Process()
                self.stats.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"Failed to update stats: {e}")
    
    async def get_batch_stats(self) -> BatchStats:
        """Get batch processing statistics"""
        try:
            # Update GPU utilization if available
            if torch.cuda.is_available():
                try:
                    self.stats.gpu_utilization = torch.cuda.utilization()
                except Exception:
                    pass
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Failed to get batch stats: {e}")
            return self.stats
    
    async def clear_queues(self):
        """Clear all request queues"""
        try:
            for task_type in self.request_queues.keys():
                with self.queue_locks[task_type]:
                    self.request_queues[task_type].clear()
            
            logger.info("All request queues cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear queues: {e}")
    
    async def export_batch_data(self, format: str = "json") -> str:
        """Export batch processing data"""
        try:
            if format.lower() == "json":
                data = {
                    "config": asdict(self.config),
                    "stats": asdict(await self.get_batch_stats()),
                    "queue_sizes": {task_type: len(queue) for task_type, queue in self.request_queues.items()},
                    "processing_active": self.processing_active,
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Batch data export failed: {e}")
            return ""
