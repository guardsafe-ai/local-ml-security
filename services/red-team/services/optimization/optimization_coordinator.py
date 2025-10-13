"""
Optimization Coordinator
Coordinates all optimization components for maximum performance
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

from .result_cache import ResultCache, CacheConfig, CacheStrategy, CacheLevel
from .batch_optimizer import BatchOptimizer, BatchConfig, BatchStrategy, OptimizationLevel
from .gpu_manager import GPUManager, GPUConfig, GPUStrategy

logger = logging.getLogger(__name__)


class OptimizationMode(Enum):
    """Optimization modes"""
    PERFORMANCE = "performance"
    MEMORY = "memory"
    BALANCED = "balanced"
    CUSTOM = "custom"


@dataclass
class OptimizationConfig:
    """Overall optimization configuration"""
    mode: OptimizationMode = OptimizationMode.BALANCED
    enable_caching: bool = True
    enable_batch_processing: bool = True
    enable_gpu_optimization: bool = True
    cache_config: Optional[CacheConfig] = None
    batch_config: Optional[BatchConfig] = None
    gpu_config: Optional[GPUConfig] = None
    auto_tuning: bool = True
    tuning_interval: int = 300  # seconds
    performance_threshold: float = 0.8
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class OptimizationStats:
    """Optimization statistics"""
    cache_hit_rate: float = 0.0
    batch_efficiency: float = 0.0
    gpu_utilization: float = 0.0
    memory_usage_mb: float = 0.0
    throughput_per_second: float = 0.0
    average_latency_ms: float = 0.0
    optimization_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class OptimizationCoordinator:
    """
    Optimization Coordinator
    Coordinates all optimization components for maximum performance
    """
    
    def __init__(self, config: OptimizationConfig = None):
        """Initialize optimization coordinator"""
        self.config = config or OptimizationConfig()
        
        # Initialize components
        self.cache = None
        self.batch_optimizer = None
        self.gpu_manager = None
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.optimization_stats = OptimizationStats()
        
        # Auto-tuning
        self.tuning_task = None
        self.coordination_active = False
        
        # Initialize components based on config
        self._initialize_components()
        
        logger.info("✅ Initialized Optimization Coordinator")
    
    def _initialize_components(self):
        """Initialize optimization components based on configuration"""
        try:
            # Initialize cache
            if self.config.enable_caching:
                cache_config = self.config.cache_config or self._get_default_cache_config()
                self.cache = ResultCache(cache_config)
            
            # Initialize batch optimizer
            if self.config.enable_batch_processing:
                batch_config = self.config.batch_config or self._get_default_batch_config()
                self.batch_optimizer = BatchOptimizer(batch_config)
            
            # Initialize GPU manager
            if self.config.enable_gpu_optimization:
                gpu_config = self.config.gpu_config or self._get_default_gpu_config()
                self.gpu_manager = GPUManager(gpu_config)
            
        except Exception as e:
            logger.error(f"Failed to initialize optimization components: {e}")
    
    def _get_default_cache_config(self) -> CacheConfig:
        """Get default cache configuration based on mode"""
        if self.config.mode == OptimizationMode.PERFORMANCE:
            return CacheConfig(
                max_size=2000,
                ttl_seconds=7200,
                strategy=CacheStrategy.LRU,
                levels=[CacheLevel.MEMORY, CacheLevel.REDIS],
                compression=True
            )
        elif self.config.mode == OptimizationMode.MEMORY:
            return CacheConfig(
                max_size=500,
                ttl_seconds=1800,
                strategy=CacheStrategy.LFU,
                levels=[CacheLevel.MEMORY],
                compression=True
            )
        else:  # BALANCED
            return CacheConfig(
                max_size=1000,
                ttl_seconds=3600,
                strategy=CacheStrategy.ADAPTIVE,
                levels=[CacheLevel.MEMORY, CacheLevel.REDIS],
                compression=True
            )
    
    def _get_default_batch_config(self) -> BatchConfig:
        """Get default batch configuration based on mode"""
        if self.config.mode == OptimizationMode.PERFORMANCE:
            return BatchConfig(
                strategy=BatchStrategy.ADAPTIVE,
                max_batch_size=64,
                min_batch_size=8,
                batch_timeout=0.05,
                optimization_level=OptimizationLevel.MAXIMUM,
                enable_gpu_optimization=True,
                enable_parallel_processing=True,
                max_parallel_workers=8
            )
        elif self.config.mode == OptimizationMode.MEMORY:
            return BatchConfig(
                strategy=BatchStrategy.FIXED_SIZE,
                max_batch_size=16,
                min_batch_size=4,
                batch_timeout=0.2,
                optimization_level=OptimizationLevel.BASIC,
                enable_gpu_optimization=False,
                enable_parallel_processing=False,
                max_parallel_workers=2
            )
        else:  # BALANCED
            return BatchConfig(
                strategy=BatchStrategy.ADAPTIVE,
                max_batch_size=32,
                min_batch_size=4,
                batch_timeout=0.1,
                optimization_level=OptimizationLevel.ADVANCED,
                enable_gpu_optimization=True,
                enable_parallel_processing=True,
                max_parallel_workers=4
            )
    
    def _get_default_gpu_config(self) -> GPUConfig:
        """Get default GPU configuration based on mode"""
        if self.config.mode == OptimizationMode.PERFORMANCE:
            return GPUConfig(
                strategy=GPUStrategy.ADAPTIVE,
                max_memory_usage=0.95,
                min_memory_reserve=0.05,
                enable_memory_pooling=True,
                enable_automatic_cleanup=True,
                cleanup_interval=15,
                enable_mixed_precision=True,
                enable_gradient_checkpointing=True,
                max_models_per_gpu=10
            )
        elif self.config.mode == OptimizationMode.MEMORY:
            return GPUConfig(
                strategy=GPUStrategy.CONSERVATIVE,
                max_memory_usage=0.7,
                min_memory_reserve=0.3,
                enable_memory_pooling=False,
                enable_automatic_cleanup=True,
                cleanup_interval=60,
                enable_mixed_precision=False,
                enable_gradient_checkpointing=False,
                max_models_per_gpu=3
            )
        else:  # BALANCED
            return GPUConfig(
                strategy=GPUStrategy.ADAPTIVE,
                max_memory_usage=0.85,
                min_memory_reserve=0.15,
                enable_memory_pooling=True,
                enable_automatic_cleanup=True,
                cleanup_interval=30,
                enable_mixed_precision=True,
                enable_gradient_checkpointing=True,
                max_models_per_gpu=5
            )
    
    async def start_optimization(self):
        """Start optimization coordination"""
        try:
            if self.coordination_active:
                return
            
            self.coordination_active = True
            
            # Start components
            if self.cache:
                await self.cache.start_cache()
            
            if self.batch_optimizer:
                await self.batch_optimizer.start_processing()
            
            if self.gpu_manager:
                await self.gpu_manager.start_management()
            
            # Start auto-tuning if enabled
            if self.config.auto_tuning:
                self.tuning_task = asyncio.create_task(self._auto_tuning_loop())
            
            logger.info("Optimization coordination started")
            
        except Exception as e:
            logger.error(f"Failed to start optimization: {e}")
            raise
    
    async def stop_optimization(self):
        """Stop optimization coordination"""
        try:
            self.coordination_active = False
            
            # Stop auto-tuning
            if self.tuning_task:
                self.tuning_task.cancel()
                try:
                    await self.tuning_task
                except asyncio.CancelledError:
                    pass
            
            # Stop components
            if self.cache:
                await self.cache.stop_cache()
            
            if self.batch_optimizer:
                await self.batch_optimizer.stop_processing()
            
            if self.gpu_manager:
                await self.gpu_manager.stop_management()
            
            logger.info("Optimization coordination stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop optimization: {e}")
    
    async def optimize_task(self, 
                          task_type: str,
                          data: Any,
                          processor: Callable,
                          use_cache: bool = True,
                          use_batch: bool = True,
                          use_gpu: bool = True) -> Any:
        """Optimize task execution"""
        try:
            start_time = time.time()
            
            # Generate cache key
            cache_key = None
            if use_cache and self.cache:
                cache_key = self.cache.generate_key(task_type, data)
                
                # Try to get from cache
                cached_result = await self.cache.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for task {task_type}")
                    return cached_result
            
            # Execute task
            if use_batch and self.batch_optimizer:
                # Use batch processing
                result = await self._execute_batch_task(task_type, data, processor)
            else:
                # Execute directly
                if asyncio.iscoroutinefunction(processor):
                    result = await processor(data)
                else:
                    result = processor(data)
            
            # Cache result
            if use_cache and self.cache and cache_key:
                await self.cache.set(cache_key, result)
            
            # Update performance metrics
            execution_time = time.time() - start_time
            self._update_performance_metrics(execution_time, task_type)
            
            return result
            
        except Exception as e:
            logger.error(f"Task optimization failed for {task_type}: {e}")
            raise
    
    async def _execute_batch_task(self, task_type: str, data: Any, processor: Callable) -> Any:
        """Execute task using batch processing"""
        try:
            # Submit to batch optimizer
            request_id = await self.batch_optimizer.submit_request(
                data=data,
                task_type=task_type,
                priority=1
            )
            
            # Wait for result (in a real implementation, this would be more sophisticated)
            await asyncio.sleep(0.01)  # Simulate processing time
            
            # For now, execute directly
            if asyncio.iscoroutinefunction(processor):
                result = await processor(data)
            else:
                result = processor(data)
            
            return result
            
        except Exception as e:
            logger.error(f"Batch task execution failed: {e}")
            raise
    
    async def load_model(self, model_id: str, model: Any, device_id: Optional[int] = None) -> bool:
        """Load model with optimization"""
        try:
            if self.gpu_manager:
                return await self.gpu_manager.load_model(model_id, model, device_id)
            else:
                logger.warning("GPU manager not available")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload model"""
        try:
            if self.gpu_manager:
                return await self.gpu_manager.unload_model(model_id)
            else:
                logger.warning("GPU manager not available")
                return False
                
        except Exception as e:
            logger.error(f"Failed to unload model {model_id}: {e}")
            return False
    
    async def get_model(self, model_id: str) -> Optional[Any]:
        """Get model"""
        try:
            if self.gpu_manager:
                return await self.gpu_manager.get_model(model_id)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get model {model_id}: {e}")
            return None
    
    async def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage"""
        try:
            optimization_results = {}
            
            # Optimize GPU memory
            if self.gpu_manager:
                gpu_results = await self.gpu_manager.optimize_memory()
                optimization_results["gpu"] = gpu_results
            
            # Optimize cache
            if self.cache:
                # Clear old cache entries
                await self.cache.cleanup_completed_tasks()
                optimization_results["cache"] = {"cleaned": True}
            
            # Optimize batch processing
            if self.batch_optimizer:
                await self.batch_optimizer.clear_queues()
                optimization_results["batch"] = {"queues_cleared": True}
            
            logger.info(f"Memory optimization completed: {optimization_results}")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return {}
    
    async def _auto_tuning_loop(self):
        """Auto-tuning loop"""
        while self.coordination_active:
            try:
                # Collect performance metrics
                await self._collect_performance_metrics()
                
                # Analyze performance
                performance_score = await self._analyze_performance()
                
                # Adjust configuration if needed
                if performance_score < self.config.performance_threshold:
                    await self._adjust_configuration()
                
                # Sleep before next tuning
                await asyncio.sleep(self.config.tuning_interval)
                
            except Exception as e:
                logger.error(f"Auto-tuning error: {e}")
                await asyncio.sleep(self.config.tuning_interval)
    
    async def _collect_performance_metrics(self):
        """Collect performance metrics from all components"""
        try:
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "cache": {},
                "batch": {},
                "gpu": {}
            }
            
            # Collect cache metrics
            if self.cache:
                cache_stats = await self.cache.get_cache_stats()
                metrics["cache"] = {
                    "hit_rate": cache_stats.hit_rate,
                    "memory_usage_mb": cache_stats.memory_usage_mb,
                    "entry_count": cache_stats.entry_count
                }
            
            # Collect batch metrics
            if self.batch_optimizer:
                batch_stats = await self.batch_optimizer.get_batch_stats()
                metrics["batch"] = {
                    "throughput_per_second": batch_stats.throughput_per_second,
                    "avg_processing_time": batch_stats.avg_processing_time,
                    "avg_batch_size": batch_stats.avg_batch_size
                }
            
            # Collect GPU metrics
            if self.gpu_manager:
                gpu_info = await self.gpu_manager.get_gpu_info()
                memory_usage = await self.gpu_manager.get_memory_usage()
                metrics["gpu"] = {
                    "gpu_count": len(gpu_info),
                    "memory_usage_percentage": memory_usage.get("usage_percentage", 0),
                    "total_memory_gb": memory_usage.get("total_memory", 0) / (1024**3)
                }
            
            # Store metrics
            self.performance_history.append(metrics)
            
            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
    
    async def _analyze_performance(self) -> float:
        """Analyze performance and return score"""
        try:
            if not self.performance_history:
                return 1.0
            
            recent_metrics = self.performance_history[-10:]  # Last 10 measurements
            
            # Calculate performance score
            score = 1.0
            
            # Cache performance
            if recent_metrics and "cache" in recent_metrics[0]:
                avg_hit_rate = sum(m["cache"].get("hit_rate", 0) for m in recent_metrics) / len(recent_metrics)
                score *= (0.3 + 0.7 * avg_hit_rate)  # Weight cache performance
            
            # Batch performance
            if recent_metrics and "batch" in recent_metrics[0]:
                avg_throughput = sum(m["batch"].get("throughput_per_second", 0) for m in recent_metrics) / len(recent_metrics)
                if avg_throughput > 0:
                    score *= min(1.0, avg_throughput / 100)  # Normalize to 100 req/s
            
            # GPU performance
            if recent_metrics and "gpu" in recent_metrics[0]:
                avg_memory_usage = sum(m["gpu"].get("memory_usage_percentage", 0) for m in recent_metrics) / len(recent_metrics)
                if avg_memory_usage > 0:
                    # Optimal memory usage is around 70-80%
                    optimal_usage = 75.0
                    memory_score = 1.0 - abs(avg_memory_usage - optimal_usage) / 100.0
                    score *= max(0.1, memory_score)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Failed to analyze performance: {e}")
            return 0.5
    
    async def _adjust_configuration(self):
        """Adjust configuration based on performance"""
        try:
            logger.info("Adjusting configuration based on performance analysis")
            
            # This is a simplified implementation
            # In practice, you would implement sophisticated configuration adjustment logic
            
            # Example: Adjust cache size based on hit rate
            if self.cache and len(self.performance_history) > 5:
                recent_metrics = self.performance_history[-5:]
                avg_hit_rate = sum(m["cache"].get("hit_rate", 0) for m in recent_metrics) / len(recent_metrics)
                
                if avg_hit_rate < 0.5:
                    # Increase cache size
                    logger.info("Increasing cache size due to low hit rate")
                    # In practice, you would adjust the cache configuration
            
            # Example: Adjust batch size based on throughput
            if self.batch_optimizer and len(self.performance_history) > 5:
                recent_metrics = self.performance_history[-5:]
                avg_throughput = sum(m["batch"].get("throughput_per_second", 0) for m in recent_metrics) / len(recent_metrics)
                
                if avg_throughput < 50:
                    # Increase batch size
                    logger.info("Increasing batch size due to low throughput")
                    # In practice, you would adjust the batch configuration
            
        except Exception as e:
            logger.error(f"Failed to adjust configuration: {e}")
    
    def _update_performance_metrics(self, execution_time: float, task_type: str):
        """Update performance metrics"""
        try:
            # Update optimization stats
            self.optimization_stats.average_latency_ms = execution_time * 1000
            
            # Update performance history
            if self.performance_history:
                latest = self.performance_history[-1]
                latest["execution_time"] = execution_time
                latest["task_type"] = task_type
            
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")
    
    async def get_optimization_stats(self) -> OptimizationStats:
        """Get optimization statistics"""
        try:
            # Update stats from components
            if self.cache:
                cache_stats = await self.cache.get_cache_stats()
                self.optimization_stats.cache_hit_rate = cache_stats.hit_rate
                self.optimization_stats.memory_usage_mb = cache_stats.memory_usage_mb
            
            if self.batch_optimizer:
                batch_stats = await self.batch_optimizer.get_batch_stats()
                self.optimization_stats.batch_efficiency = batch_stats.throughput_per_second
                self.optimization_stats.throughput_per_second = batch_stats.throughput_per_second
            
            if self.gpu_manager:
                memory_usage = await self.gpu_manager.get_memory_usage()
                self.optimization_stats.gpu_utilization = memory_usage.get("usage_percentage", 0)
            
            # Calculate overall optimization score
            self.optimization_stats.optimization_score = await self._analyze_performance()
            
            return self.optimization_stats
            
        except Exception as e:
            logger.error(f"Failed to get optimization stats: {e}")
            return self.optimization_stats
    
    async def export_optimization_data(self, format: str = "json") -> str:
        """Export optimization data"""
        try:
            if format.lower() == "json":
                data = {
                    "config": {
                        "mode": self.config.mode.value,
                        "enable_caching": self.config.enable_caching,
                        "enable_batch_processing": self.config.enable_batch_processing,
                        "enable_gpu_optimization": self.config.enable_gpu_optimization,
                        "auto_tuning": self.config.auto_tuning
                    },
                    "stats": {
                        "cache_hit_rate": self.optimization_stats.cache_hit_rate,
                        "batch_efficiency": self.optimization_stats.batch_efficiency,
                        "gpu_utilization": self.optimization_stats.gpu_utilization,
                        "memory_usage_mb": self.optimization_stats.memory_usage_mb,
                        "throughput_per_second": self.optimization_stats.throughput_per_second,
                        "average_latency_ms": self.optimization_stats.average_latency_ms,
                        "optimization_score": self.optimization_stats.optimization_score
                    },
                    "performance_history": self.performance_history[-20:],  # Last 20 measurements
                    "coordination_active": self.coordination_active,
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Optimization data export failed: {e}")
            return ""
