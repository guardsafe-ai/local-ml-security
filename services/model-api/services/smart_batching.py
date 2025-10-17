"""
Smart Dynamic Batching System
Implements intelligent batching strategies based on request patterns and system load
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from collections import deque
from enum import Enum
import threading

logger = logging.getLogger(__name__)

class BatchingStrategy(Enum):
    """Batching strategies based on use case"""
    SINGLE_REQUEST = "single_request"      # Direct inference for low latency
    BATCH_OPTIMIZED = "batch_optimized"    # Dynamic batching for throughput
    ADAPTIVE = "adaptive"                  # Automatically choose based on load

@dataclass
class SmartBatchConfig:
    """Smart batching configuration"""
    # Basic batching parameters
    max_batch_size: int = 16
    max_wait_time_ms: int = 100
    min_batch_size: int = 2
    
    # Adaptive parameters
    enable_adaptive: bool = True
    load_threshold: float = 0.7  # CPU/GPU utilization threshold
    latency_threshold_ms: int = 200  # Max acceptable latency
    
    # Strategy-specific parameters
    single_request_timeout_ms: int = 50
    batch_optimization_threshold: int = 4  # Min requests for batching
    
    # Queue management
    max_queue_size: int = 200
    queue_overflow_strategy: str = "reject"  # "reject" or "wait"

class SmartBatcher:
    """Intelligent dynamic batching system"""
    
    def __init__(self, config: SmartBatchConfig = None, inference_function: Callable = None):
        self.config = config or SmartBatchConfig()
        self.inference_function = inference_function
        
        # Request queues for different strategies
        self.single_queue = deque()
        self.batch_queue = deque()
        
        # System state
        self.current_load = 0.0
        self.avg_latency = 0.0
        self.active_requests = 0
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "single_requests": 0,
            "batched_requests": 0,
            "avg_batch_size": 0.0,
            "avg_latency_ms": 0.0,
            "throughput_rps": 0.0,
            "queue_overflows": 0
        }
        
        # Background processing
        self.processing = False
        self.batch_lock = asyncio.Lock()
        
    async def add_request(self, 
                         text: str, 
                         request_id: str = None,
                         strategy: BatchingStrategy = None,
                         metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Add a request with intelligent routing based on strategy and system state
        """
        if not request_id:
            request_id = f"req_{int(time.time() * 1000)}_{id(text)}"
            
        if not strategy:
            strategy = self._determine_strategy()
            
        self.stats["total_requests"] += 1
        
        if strategy == BatchingStrategy.SINGLE_REQUEST:
            return await self._process_single_request(text, request_id, metadata)
        elif strategy == BatchingStrategy.BATCH_OPTIMIZED:
            return await self._process_batch_request(text, request_id, metadata)
        elif strategy == BatchingStrategy.ADAPTIVE:
            return await self._process_adaptive_request(text, request_id, metadata)
        else:
            raise ValueError(f"Unknown batching strategy: {strategy}")
    
    def _determine_strategy(self) -> BatchingStrategy:
        """Determine the best batching strategy based on current system state"""
        if not self.config.enable_adaptive:
            return BatchingStrategy.SINGLE_REQUEST
            
        # Check if we should use batch optimization
        if (self.active_requests >= self.config.batch_optimization_threshold and 
            self.current_load < self.config.load_threshold):
            return BatchingStrategy.BATCH_OPTIMIZED
        
        # Check if latency is too high for batching
        if self.avg_latency > self.config.latency_threshold_ms:
            return BatchingStrategy.SINGLE_REQUEST
            
        # Default to single request for low latency
        return BatchingStrategy.SINGLE_REQUEST
    
    async def _process_single_request(self, text: str, request_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process single request with direct inference for optimal latency"""
        start_time = time.time()
        self.stats["single_requests"] += 1
        self.active_requests += 1
        
        try:
            logger.info(f"🔄 [SINGLE] Processing request {request_id} directly")
            
            # Execute inference directly
            if self.inference_function:
                result = await self.inference_function([text], [{"request_id": request_id, "metadata": metadata}])
                result = result[0] if result else {}
            else:
                # Fallback to direct processing
                result = {"text": text, "prediction": "unknown", "confidence": 0.0}
            
            processing_time = (time.time() - start_time) * 1000
            result["processing_time_ms"] = processing_time
            result["strategy"] = "single_request"
            
            # Update latency tracking
            self._update_latency_tracking(processing_time)
            
            logger.info(f"✅ [SINGLE] Completed request {request_id} in {processing_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"❌ [SINGLE] Error processing request {request_id}: {e}")
            return {
                "text": text,
                "prediction": "error",
                "confidence": 0.0,
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "strategy": "single_request"
            }
        finally:
            self.active_requests -= 1
    
    async def _process_batch_request(self, text: str, request_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process request using dynamic batching for optimal throughput"""
        start_time = time.time()
        
        # Create future for async result
        future = asyncio.Future()
        request_data = {
            "text": text,
            "request_id": request_id,
            "metadata": metadata,
            "future": future,
            "timestamp": start_time
        }
        
        async with self.batch_lock:
            # Add to batch queue
            if len(self.batch_queue) >= self.config.max_queue_size:
                if self.config.queue_overflow_strategy == "reject":
                    self.stats["queue_overflows"] += 1
                    raise Exception("Queue overflow - too many pending requests")
                else:
                    # Wait for queue space
                    await asyncio.sleep(0.001)
            
            self.batch_queue.append(request_data)
            
            # Check if we should process the batch
            should_process = (
                len(self.batch_queue) >= self.config.max_batch_size or
                (len(self.batch_queue) >= self.config.min_batch_size and 
                 (time.time() - self.batch_queue[0]["timestamp"]) * 1000 >= self.config.max_wait_time_ms)
            )
            
            if should_process:
                await self._process_batch()
        
        # Wait for result
        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            self.stats["batched_requests"] += 1
            return result
        except asyncio.TimeoutError:
            logger.error(f"❌ [BATCH] Timeout waiting for request {request_id}")
            return {
                "text": text,
                "prediction": "timeout",
                "confidence": 0.0,
                "error": "Request timeout",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "strategy": "batch_optimized"
            }
    
    async def _process_adaptive_request(self, text: str, request_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process request using adaptive strategy selection"""
        # For now, use single request strategy
        # In a full implementation, this would dynamically choose based on real-time metrics
        return await self._process_single_request(text, request_id, metadata)
    
    async def _process_batch(self):
        """Process the current batch of requests"""
        if not self.batch_queue:
            return
            
        batch_requests = []
        while self.batch_queue and len(batch_requests) < self.config.max_batch_size:
            batch_requests.append(self.batch_queue.popleft())
        
        if not batch_requests:
            return
            
        start_time = time.time()
        logger.info(f"🔄 [BATCH] Processing batch of {len(batch_requests)} requests")
        
        try:
            # Extract texts and metadata
            texts = [req["text"] for req in batch_requests]
            metadata_list = [req["metadata"] for req in batch_requests]
            
            # Execute batch inference
            if self.inference_function:
                results = await self.inference_function(texts, metadata_list)
            else:
                # Fallback to individual processing
                results = []
                for text in texts:
                    results.append({"text": text, "prediction": "unknown", "confidence": 0.0})
            
            # Complete futures with results
            for i, (request, result) in enumerate(zip(batch_requests, results)):
                processing_time = (time.time() - start_time) * 1000
                result["processing_time_ms"] = processing_time
                result["strategy"] = "batch_optimized"
                result["batch_size"] = len(batch_requests)
                
                if not request["future"].done():
                    request["future"].set_result(result)
            
            # Update statistics
            batch_time = (time.time() - start_time) * 1000
            self.stats["avg_batch_size"] = (
                (self.stats["avg_batch_size"] * (self.stats["total_requests"] - len(batch_requests)) + 
                 len(batch_requests)) / self.stats["total_requests"]
            )
            
            logger.info(f"✅ [BATCH] Completed batch in {batch_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"❌ [BATCH] Error processing batch: {e}")
            # Complete all futures with error results
            for request in batch_requests:
                error_result = {
                    "text": request["text"],
                    "prediction": "error",
                    "confidence": 0.0,
                    "error": str(e),
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "strategy": "batch_optimized"
                }
                if not request["future"].done():
                    request["future"].set_result(error_result)
    
    def _update_latency_tracking(self, latency_ms: float):
        """Update latency tracking for adaptive decisions"""
        if self.avg_latency == 0:
            self.avg_latency = latency_ms
        else:
            # Exponential moving average
            self.avg_latency = 0.9 * self.avg_latency + 0.1 * latency_ms
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current batching statistics"""
        return {
            **self.stats,
            "current_load": self.current_load,
            "avg_latency_ms": self.avg_latency,
            "active_requests": self.active_requests,
            "queue_sizes": {
                "single_queue": len(self.single_queue),
                "batch_queue": len(self.batch_queue)
            }
        }
    
    def update_system_load(self, load: float):
        """Update system load for adaptive decisions"""
        self.current_load = load
