"""
GPU Manager
Implements GPU memory management and optimization
"""

import asyncio
import logging
import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json
from collections import defaultdict
import threading
import gc
import psutil

logger = logging.getLogger(__name__)


class GPUStrategy(Enum):
    """GPU management strategies"""
    STATIC = "static"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"
    CONSERVATIVE = "conservative"


class MemoryPolicy(Enum):
    """Memory management policies"""
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    WORST_FIT = "worst_fit"
    BUDDY = "buddy"


@dataclass
class GPUConfig:
    """GPU configuration"""
    strategy: GPUStrategy = GPUStrategy.ADAPTIVE
    max_memory_usage: float = 0.9  # 90% of total GPU memory
    min_memory_reserve: float = 0.1  # 10% reserved
    memory_policy: MemoryPolicy = MemoryPolicy.BEST_FIT
    enable_memory_pooling: bool = True
    enable_automatic_cleanup: bool = True
    cleanup_interval: int = 30  # seconds
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True
    max_models_per_gpu: int = 5
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GPUInfo:
    """GPU information"""
    device_id: int
    name: str
    total_memory: int  # bytes
    allocated_memory: int  # bytes
    free_memory: int  # bytes
    utilization: float  # percentage
    temperature: float  # celsius
    power_usage: float  # watts
    is_available: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ModelInfo:
    """Model information"""
    model_id: str
    model: nn.Module
    device: torch.device
    memory_usage: int  # bytes
    last_used: datetime
    access_count: int = 0
    priority: int = 1
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MemoryAllocation:
    """Memory allocation information"""
    allocation_id: str
    size: int  # bytes
    device: torch.device
    allocated_at: datetime
    is_active: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class GPUManager:
    """
    GPU Manager
    Implements GPU memory management and optimization
    """
    
    def __init__(self, config: GPUConfig = None):
        """Initialize GPU manager"""
        self.config = config or GPUConfig()
        
        # GPU information
        self.gpu_info: Dict[int, GPUInfo] = {}
        self.available_gpus: List[int] = []
        
        # Model management
        self.models: Dict[str, ModelInfo] = {}
        self.model_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
        # Memory management
        self.memory_allocations: Dict[str, MemoryAllocation] = {}
        self.memory_pool: Dict[torch.device, List[torch.Tensor]] = defaultdict(list)
        
        # Monitoring
        self.monitoring_task = None
        self.cleanup_task = None
        self.management_active = False
        
        # Statistics
        self.stats = {
            "total_allocations": 0,
            "total_deallocations": 0,
            "memory_fragmentation": 0.0,
            "gpu_utilization": 0.0,
            "models_loaded": 0,
            "models_unloaded": 0
        }
        
        logger.info("✅ Initialized GPU Manager")
    
    async def start_management(self):
        """Start GPU management"""
        try:
            if self.management_active:
                return
            
            # Initialize GPU information
            await self._initialize_gpu_info()
            
            self.management_active = True
            
            # Start monitoring
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Start cleanup
            if self.config.enable_automatic_cleanup:
                self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("GPU management started")
            
        except Exception as e:
            logger.error(f"Failed to start GPU management: {e}")
            raise
    
    async def stop_management(self):
        """Stop GPU management"""
        try:
            self.management_active = False
            
            # Stop monitoring
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Stop cleanup
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Clean up all models
            await self._cleanup_all_models()
            
            logger.info("GPU management stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop GPU management: {e}")
    
    async def load_model(self, 
                        model_id: str, 
                        model: nn.Module, 
                        device_id: Optional[int] = None,
                        priority: int = 1) -> bool:
        """Load model to GPU"""
        try:
            if model_id in self.models:
                logger.warning(f"Model {model_id} already loaded")
                return True
            
            # Select GPU device
            if device_id is None:
                device_id = await self._select_optimal_gpu()
            
            if device_id is None:
                logger.error("No available GPU for model loading")
                return False
            
            device = torch.device(f"cuda:{device_id}")
            
            # Check memory availability
            if not await self._check_memory_availability(device, model):
                # Try to free up memory
                await self._free_memory_for_model(device, model)
                
                if not await self._check_memory_availability(device, model):
                    logger.error(f"Insufficient GPU memory for model {model_id}")
                    return False
            
            # Move model to GPU
            model = model.to(device)
            
            # Calculate memory usage
            memory_usage = await self._calculate_model_memory_usage(model)
            
            # Create model info
            model_info = ModelInfo(
                model_id=model_id,
                model=model,
                device=device,
                memory_usage=memory_usage,
                last_used=datetime.utcnow(),
                priority=priority
            )
            
            # Store model info
            with self.model_locks[model_id]:
                self.models[model_id] = model_info
            
            # Update statistics
            self.stats["models_loaded"] += 1
            
            logger.info(f"Model {model_id} loaded to GPU {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload model from GPU"""
        try:
            if model_id not in self.models:
                logger.warning(f"Model {model_id} not found")
                return False
            
            model_info = self.models[model_id]
            
            # Move model to CPU
            model_info.model.cpu()
            
            # Clear GPU cache
            if model_info.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Remove model info
            with self.model_locks[model_id]:
                del self.models[model_id]
            
            # Update statistics
            self.stats["models_unloaded"] += 1
            
            logger.info(f"Model {model_id} unloaded from GPU")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model {model_id}: {e}")
            return False
    
    async def get_model(self, model_id: str) -> Optional[nn.Module]:
        """Get model by ID"""
        try:
            if model_id not in self.models:
                return None
            
            model_info = self.models[model_id]
            
            # Update access info
            model_info.last_used = datetime.utcnow()
            model_info.access_count += 1
            
            return model_info.model
            
        except Exception as e:
            logger.error(f"Failed to get model {model_id}: {e}")
            return None
    
    async def allocate_memory(self, 
                            size: int, 
                            device: torch.device,
                            persistent: bool = False) -> Optional[str]:
        """Allocate GPU memory"""
        try:
            allocation_id = f"alloc_{int(time.time() * 1000)}_{id(size)}"
            
            # Check memory availability
            if not await self._check_memory_availability(device, size):
                logger.warning(f"Insufficient memory for allocation {allocation_id}")
                return None
            
            # Allocate memory
            tensor = torch.empty(size, dtype=torch.uint8, device=device)
            
            # Store allocation info
            allocation = MemoryAllocation(
                allocation_id=allocation_id,
                size=size,
                device=device,
                allocated_at=datetime.utcnow()
            )
            
            self.memory_allocations[allocation_id] = allocation
            
            # Add to memory pool if enabled
            if self.config.enable_memory_pooling and not persistent:
                self.memory_pool[device].append(tensor)
            
            # Update statistics
            self.stats["total_allocations"] += 1
            
            logger.debug(f"Memory allocated: {allocation_id} ({size} bytes)")
            return allocation_id
            
        except Exception as e:
            logger.error(f"Failed to allocate memory: {e}")
            return None
    
    async def deallocate_memory(self, allocation_id: str) -> bool:
        """Deallocate GPU memory"""
        try:
            if allocation_id not in self.memory_allocations:
                logger.warning(f"Allocation {allocation_id} not found")
                return False
            
            allocation = self.memory_allocations[allocation_id]
            
            # Remove from memory pool
            if allocation.device in self.memory_pool:
                self.memory_pool[allocation.device] = [
                    tensor for tensor in self.memory_pool[allocation.device]
                    if tensor.numel() != allocation.size
                ]
            
            # Clear GPU cache
            if allocation.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Remove allocation info
            del self.memory_allocations[allocation_id]
            
            # Update statistics
            self.stats["total_deallocations"] += 1
            
            logger.debug(f"Memory deallocated: {allocation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deallocate memory {allocation_id}: {e}")
            return False
    
    async def optimize_memory(self) -> Dict[str, Any]:
        """Optimize GPU memory usage"""
        try:
            optimization_results = {
                "freed_memory": 0,
                "defragmented_memory": 0,
                "models_unloaded": 0,
                "allocations_cleaned": 0
            }
            
            # Clean up inactive allocations
            inactive_allocations = [
                alloc_id for alloc_id, alloc in self.memory_allocations.items()
                if not alloc.is_active and (datetime.utcnow() - alloc.allocated_at).seconds > 300
            ]
            
            for alloc_id in inactive_allocations:
                if await self.deallocate_memory(alloc_id):
                    optimization_results["allocations_cleaned"] += 1
            
            # Unload least recently used models
            if len(self.models) > self.config.max_models_per_gpu:
                lru_models = sorted(
                    self.models.items(),
                    key=lambda x: x[1].last_used
                )
                
                models_to_unload = lru_models[:len(self.models) - self.config.max_models_per_gpu]
                
                for model_id, _ in models_to_unload:
                    if await self.unload_model(model_id):
                        optimization_results["models_unloaded"] += 1
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                optimization_results["freed_memory"] = await self._get_freed_memory()
            
            # Force garbage collection
            gc.collect()
            
            logger.info(f"Memory optimization completed: {optimization_results}")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return {}
    
    async def get_gpu_info(self, device_id: Optional[int] = None) -> Union[GPUInfo, Dict[int, GPUInfo]]:
        """Get GPU information"""
        try:
            if device_id is not None:
                return self.gpu_info.get(device_id)
            else:
                return self.gpu_info.copy()
                
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
            return {}
    
    async def get_memory_usage(self, device_id: Optional[int] = None) -> Dict[str, Any]:
        """Get memory usage statistics"""
        try:
            if device_id is not None:
                gpu_info = self.gpu_info.get(device_id)
                if gpu_info:
                    return {
                        "total_memory": gpu_info.total_memory,
                        "allocated_memory": gpu_info.allocated_memory,
                        "free_memory": gpu_info.free_memory,
                        "usage_percentage": gpu_info.allocated_memory / gpu_info.total_memory * 100
                    }
                return {}
            else:
                total_memory = sum(gpu.total_memory for gpu in self.gpu_info.values())
                total_allocated = sum(gpu.allocated_memory for gpu in self.gpu_info.values())
                
                return {
                    "total_memory": total_memory,
                    "allocated_memory": total_allocated,
                    "free_memory": total_memory - total_allocated,
                    "usage_percentage": total_allocated / total_memory * 100 if total_memory > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {}
    
    async def _initialize_gpu_info(self):
        """Initialize GPU information"""
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA not available")
                return
            
            device_count = torch.cuda.device_count()
            
            for device_id in range(device_count):
                device = torch.device(f"cuda:{device_id}")
                
                # Get GPU properties
                props = torch.cuda.get_device_properties(device_id)
                total_memory = props.total_memory
                
                # Get current memory usage
                allocated_memory = torch.cuda.memory_allocated(device_id)
                free_memory = total_memory - allocated_memory
                
                # Get utilization (if available)
                utilization = 0.0
                try:
                    utilization = torch.cuda.utilization(device_id)
                except Exception:
                    pass
                
                # Create GPU info
                gpu_info = GPUInfo(
                    device_id=device_id,
                    name=props.name,
                    total_memory=total_memory,
                    allocated_memory=allocated_memory,
                    free_memory=free_memory,
                    utilization=utilization,
                    temperature=0.0,  # Would need additional libraries
                    power_usage=0.0   # Would need additional libraries
                )
                
                self.gpu_info[device_id] = gpu_info
                self.available_gpus.append(device_id)
            
            logger.info(f"Initialized {len(self.gpu_info)} GPUs")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU info: {e}")
    
    async def _select_optimal_gpu(self) -> Optional[int]:
        """Select optimal GPU for model loading"""
        try:
            if not self.available_gpus:
                return None
            
            if self.config.strategy == GPUStrategy.STATIC:
                return self.available_gpus[0]
            
            elif self.config.strategy == GPUStrategy.DYNAMIC:
                # Select GPU with most free memory
                best_gpu = None
                max_free_memory = 0
                
                for device_id in self.available_gpus:
                    gpu_info = self.gpu_info[device_id]
                    if gpu_info.free_memory > max_free_memory:
                        max_free_memory = gpu_info.free_memory
                        best_gpu = device_id
                
                return best_gpu
            
            elif self.config.strategy == GPUStrategy.ADAPTIVE:
                # Select GPU based on multiple factors
                best_gpu = None
                best_score = -1
                
                for device_id in self.available_gpus:
                    gpu_info = self.gpu_info[device_id]
                    
                    # Calculate score based on free memory and utilization
                    memory_score = gpu_info.free_memory / gpu_info.total_memory
                    utilization_score = 1.0 - (gpu_info.utilization / 100.0)
                    model_count_score = 1.0 - (len([m for m in self.models.values() if m.device.index == device_id]) / self.config.max_models_per_gpu)
                    
                    total_score = memory_score * 0.5 + utilization_score * 0.3 + model_count_score * 0.2
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_gpu = device_id
                
                return best_gpu
            
            elif self.config.strategy == GPUStrategy.CONSERVATIVE:
                # Select GPU with least utilization
                best_gpu = None
                min_utilization = 100.0
                
                for device_id in self.available_gpus:
                    gpu_info = self.gpu_info[device_id]
                    if gpu_info.utilization < min_utilization:
                        min_utilization = gpu_info.utilization
                        best_gpu = device_id
                
                return best_gpu
            
            else:
                return self.available_gpus[0]
                
        except Exception as e:
            logger.error(f"Failed to select optimal GPU: {e}")
            return None
    
    async def _check_memory_availability(self, device: torch.device, model_or_size) -> bool:
        """Check if there's enough memory available"""
        try:
            if device.type != 'cuda':
                return True
            
            device_id = device.index
            gpu_info = self.gpu_info.get(device_id)
            
            if not gpu_info:
                return False
            
            # Calculate required memory
            if isinstance(model_or_size, nn.Module):
                required_memory = await self._calculate_model_memory_usage(model_or_size)
            else:
                required_memory = model_or_size
            
            # Check if enough memory is available
            available_memory = gpu_info.free_memory
            max_usable_memory = gpu_info.total_memory * self.config.max_memory_usage
            
            return (available_memory >= required_memory and 
                   available_memory + required_memory <= max_usable_memory)
            
        except Exception as e:
            logger.error(f"Failed to check memory availability: {e}")
            return False
    
    async def _calculate_model_memory_usage(self, model: nn.Module) -> int:
        """Calculate model memory usage"""
        try:
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            
            # Estimate memory usage (parameters + gradients + optimizer states)
            param_memory = param_count * 4  # 4 bytes per float32
            gradient_memory = param_count * 4  # Gradients
            optimizer_memory = param_count * 8  # Adam optimizer states
            
            total_memory = param_memory + gradient_memory + optimizer_memory
            
            return total_memory
            
        except Exception as e:
            logger.error(f"Failed to calculate model memory usage: {e}")
            return 1024 * 1024  # Default 1MB
    
    async def _free_memory_for_model(self, device: torch.device, model: nn.Module):
        """Free memory for model loading"""
        try:
            # Unload least recently used models on the same device
            device_models = [
                (model_id, model_info) for model_id, model_info in self.models.items()
                if model_info.device == device
            ]
            
            if device_models:
                # Sort by last used time
                device_models.sort(key=lambda x: x[1].last_used)
                
                # Unload oldest models
                for model_id, _ in device_models:
                    await self.unload_model(model_id)
                    
                    # Check if we have enough memory now
                    if await self._check_memory_availability(device, model):
                        break
            
            # Clear GPU cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Failed to free memory for model: {e}")
    
    async def _get_freed_memory(self) -> int:
        """Get amount of freed memory"""
        try:
            if not torch.cuda.is_available():
                return 0
            
            # This is a simplified implementation
            # In practice, you'd track memory before and after operations
            return 0
            
        except Exception as e:
            logger.error(f"Failed to get freed memory: {e}")
            return 0
    
    async def _monitoring_loop(self):
        """GPU monitoring loop"""
        while self.management_active:
            try:
                # Update GPU information
                await self._update_gpu_info()
                
                # Check for memory pressure
                await self._check_memory_pressure()
                
                # Sleep before next check
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_loop(self):
        """Memory cleanup loop"""
        while self.management_active:
            try:
                # Perform memory optimization
                await self.optimize_memory()
                
                # Sleep before next cleanup
                await asyncio.sleep(self.config.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(self.config.cleanup_interval)
    
    async def _update_gpu_info(self):
        """Update GPU information"""
        try:
            for device_id, gpu_info in self.gpu_info.items():
                # Update memory usage
                allocated_memory = torch.cuda.memory_allocated(device_id)
                gpu_info.allocated_memory = allocated_memory
                gpu_info.free_memory = gpu_info.total_memory - allocated_memory
                
                # Update utilization
                try:
                    gpu_info.utilization = torch.cuda.utilization(device_id)
                except Exception:
                    pass
                
        except Exception as e:
            logger.error(f"Failed to update GPU info: {e}")
    
    async def _check_memory_pressure(self):
        """Check for memory pressure and take action"""
        try:
            for device_id, gpu_info in self.gpu_info.items():
                memory_usage = gpu_info.allocated_memory / gpu_info.total_memory
                
                if memory_usage > self.config.max_memory_usage:
                    logger.warning(f"High memory usage on GPU {device_id}: {memory_usage:.2%}")
                    
                    # Trigger memory optimization
                    await self.optimize_memory()
                
        except Exception as e:
            logger.error(f"Failed to check memory pressure: {e}")
    
    async def _cleanup_all_models(self):
        """Clean up all loaded models"""
        try:
            model_ids = list(self.models.keys())
            
            for model_id in model_ids:
                await self.unload_model(model_id)
            
            logger.info(f"Cleaned up {len(model_ids)} models")
            
        except Exception as e:
            logger.error(f"Failed to cleanup all models: {e}")
    
    async def export_gpu_data(self, format: str = "json") -> str:
        """Export GPU management data"""
        try:
            if format.lower() == "json":
                data = {
                    "gpu_info": {str(k): asdict(v) for k, v in self.gpu_info.items()},
                    "models": {k: {
                        "model_id": v.model_id,
                        "device": str(v.device),
                        "memory_usage": v.memory_usage,
                        "last_used": v.last_used.isoformat(),
                        "access_count": v.access_count,
                        "priority": v.priority
                    } for k, v in self.models.items()},
                    "stats": self.stats,
                    "config": asdict(self.config),
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"GPU data export failed: {e}")
            return ""
