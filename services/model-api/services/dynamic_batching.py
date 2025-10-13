"""
Dynamic Batching Service
Implements dynamic batching for efficient inference
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import threading

logger = logging.getLogger(__name__)

@dataclass
class BatchRequest:
    """Represents a single request in a batch"""
    text: str
    request_id: str
    timestamp: float
    future: asyncio.Future
    metadata: Dict[str, Any] = None

@dataclass
class BatchConfig:
    """Configuration for dynamic batching"""
    max_batch_size: int = 8
    max_wait_time_ms: int = 50  # Maximum time to wait for batch
    min_batch_size: int = 1     # Minimum batch size before processing
    max_queue_size: int = 100   # Maximum requests in queue

class DynamicBatcher:
    """Handles dynamic batching of inference requests"""
    
    def __init__(self, config: BatchConfig = None, inference_function=None):
        self.config = config or BatchConfig()
        self.request_queue = deque()
        self.batch_lock = asyncio.Lock()
        self.processing = False
        self.inference_function = inference_function
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0.0,
            "avg_wait_time": 0.0,
            "queue_overflows": 0
        }
        
    async def add_request(self, text: str, request_id: str = None, 
                         metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Add a request to the batch queue
        
        Args:
            text: Input text for prediction
            request_id: Optional request ID
            metadata: Optional metadata
            
        Returns:
            Prediction result
        """
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000)}"
        
        # Create batch request
        future = asyncio.Future()
        batch_request = BatchRequest(
            text=text,
            request_id=request_id,
            timestamp=time.time(),
            future=future,
            metadata=metadata or {}
        )
        
        # Add to queue with atomic check
        async with self.batch_lock:
            # Check queue size INSIDE lock to prevent race condition
            if len(self.request_queue) >= self.config.max_queue_size:
                self.stats["queue_overflows"] += 1
                logger.warning(f"⚠️ [BATCH] Queue overflow, rejecting request {request_id}")
                raise Exception("Request queue is full")
            
            self.request_queue.append(batch_request)
            self.stats["total_requests"] += 1
            
            # Atomic check-and-set for processing flag INSIDE lock
            if not self.processing:
                self.processing = True
                asyncio.create_task(self._process_batches())
        
        # Wait for result
        try:
            result = await future
            return result
        except Exception as e:
            logger.error(f"❌ [BATCH] Request {request_id} failed: {e}")
            raise
    
    async def _process_batches(self):
        """Process batches from the queue"""
        try:
            while True:
                # Get batch from queue
                batch = await self._get_next_batch()
                if not batch:
                    break
                
                # Process batch
                await self._process_batch(batch)
                
        except Exception as e:
            logger.error(f"❌ [BATCH] Error in batch processing: {e}")
        finally:
            self.processing = False
    
    async def _get_next_batch(self) -> List[BatchRequest]:
        """Get the next batch of requests to process"""
        batch = []
        start_time = time.time()
        
        async with self.batch_lock:
            # Wait for minimum batch size or timeout
            while len(batch) < self.config.min_batch_size:
                if not self.request_queue:
                    return batch
                
                # Check if we should wait for more requests
                elapsed_ms = (time.time() - start_time) * 1000
                if (len(batch) >= self.config.min_batch_size and 
                    elapsed_ms >= self.config.max_wait_time_ms):
                    break
                
                if len(batch) >= self.config.max_batch_size:
                    break
                
                # Add request to batch
                request = self.request_queue.popleft()
                batch.append(request)
                
                # If we have max batch size, process immediately
                if len(batch) >= self.config.max_batch_size:
                    break
                
                # If queue is empty, process what we have
                if not self.request_queue:
                    break
                
                # Wait a bit for more requests
                if len(batch) < self.config.min_batch_size:
                    await asyncio.sleep(0.001)  # 1ms wait
        
        return batch
    
    async def _process_batch(self, batch: List[BatchRequest]):
        """Process a batch of requests"""
        if not batch:
            return
        
        try:
            start_time = time.time()
            
            # Extract texts for batch processing
            texts = [req.text for req in batch]
            request_ids = [req.request_id for req in batch]
            
            logger.info(f"🔄 [BATCH] Processing batch of {len(batch)} requests")
            
            # Call the actual inference function
            if self.inference_function:
                results = await self.inference_function(texts, batch)
            else:
                # Fallback: process individually
                results = []
                for req in batch:
                    # This is a placeholder - in practice, this should call the model
                    result = {"prediction": "unknown", "confidence": 0.0, "error": "No inference function provided"}
                    results.append(result)
            
            # Distribute results back to futures
            for i, result in enumerate(results):
                if i < len(batch):
                    batch[i].future.set_result(result)
            
            # Update statistics
            processing_time = time.time() - start_time
            avg_wait_time = sum(time.time() - req.timestamp for req in batch) / len(batch)
            
            self.stats["total_batches"] += 1
            self.stats["avg_batch_size"] = (
                (self.stats["avg_batch_size"] * (self.stats["total_batches"] - 1) + len(batch)) / 
                self.stats["total_batches"]
            )
            self.stats["avg_wait_time"] = (
                (self.stats["avg_wait_time"] * (self.stats["total_batches"] - 1) + avg_wait_time) / 
                self.stats["total_batches"]
            )
            
            logger.info(f"✅ [BATCH] Processed batch in {processing_time:.3f}s, avg wait: {avg_wait_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ [BATCH] Error processing batch: {e}")
            # Set error for all requests in batch
            for req in batch:
                req.future.set_exception(e)
    
    async def _inference_function(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Placeholder for actual inference function
        This should be replaced with the actual model inference
        """
        # This is a placeholder - in real implementation, this would call the model
        results = []
        for text in texts:
            # Simulate inference
            await asyncio.sleep(0.01)  # 10ms simulation
            results.append({
                "prediction": "benign",
                "confidence": 0.95,
                "probabilities": {"benign": 0.95, "malicious": 0.05},
                "text": text,
                "batch_processed": True
            })
        return results
    
    def set_inference_function(self, func):
        """Set the actual inference function"""
        self._inference_function = func
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics"""
        return {
            "stats": self.stats.copy(),
            "queue_size": len(self.request_queue),
            "processing": self.processing,
            "config": {
                "max_batch_size": self.config.max_batch_size,
                "max_wait_time_ms": self.config.max_wait_time_ms,
                "min_batch_size": self.config.min_batch_size,
                "max_queue_size": self.config.max_queue_size
            }
        }
    
    def reset_stats(self):
        """Reset batching statistics"""
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0.0,
            "avg_wait_time": 0.0,
            "queue_overflows": 0
        }

# Global batcher instance
dynamic_batcher = DynamicBatcher()
