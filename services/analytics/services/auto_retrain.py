"""
Auto-Retrain Service
Automatically triggers model retraining when drift is detected above threshold
"""

import logging
import asyncio
import httpx
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import time

# Import shared HTTP client
from utils.http_client import get_http_client, close_http_client

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Simple circuit breaker implementation"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        """Check if the circuit breaker allows execution"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record a successful operation"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Record a failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class RetrainTrigger(Enum):
    """Types of retrain triggers"""
    DATA_DRIFT = "data_drift"
    MODEL_DRIFT = "model_drift"
    PERFORMANCE_DROP = "performance_drop"
    MANUAL = "manual"

@dataclass
class RetrainConfig:
    """Configuration for auto-retrain service"""
    # Drift thresholds
    data_drift_threshold: float = 0.2  # PSI threshold for data drift
    model_drift_threshold: float = 0.15  # Performance drop threshold
    performance_drop_threshold: float = 0.05  # Accuracy/F1 drop threshold
    
    # Retrain settings
    min_samples_for_retrain: int = 100  # Minimum samples needed for retrain
    retrain_cooldown_hours: int = 24  # Hours to wait between retrains
    max_retrain_attempts: int = 3  # Maximum retrain attempts per day
    
    # Model settings
    target_models: List[str] = None  # Models to monitor (None = all)
    retrain_priority: str = "high"  # Priority for retrain jobs
    
    def __post_init__(self):
        if self.target_models is None:
            self.target_models = ["distilbert", "bert-base", "roberta-base"]

@dataclass
class RetrainEvent:
    """Represents a retrain event"""
    event_id: str
    model_name: str
    trigger: RetrainTrigger
    drift_score: float
    threshold: float
    timestamp: datetime
    status: str = "pending"  # pending, in_progress, completed, failed
    retrain_job_id: Optional[str] = None
    error_message: Optional[str] = None

class AutoRetrainService:
    """Service for automatically triggering model retraining on drift detection"""
    
    def __init__(self, config: Optional[RetrainConfig] = None):
        self.config = config or RetrainConfig()
        self.retrain_events: Dict[str, RetrainEvent] = {}
        self.last_retrain_times: Dict[str, datetime] = {}
        self.daily_retrain_counts: Dict[str, int] = {}
        
        # Service URLs
        self.training_service_url = "http://training:8002"
        self.analytics_service_url = "http://analytics:8006"
        self.model_api_url = "http://model-api:8000"
        
        # Circuit breakers for each service
        self.circuit_breakers = {
            "training": CircuitBreaker(failure_threshold=3, timeout=120),
            "analytics": CircuitBreaker(failure_threshold=3, timeout=120),
            "model_api": CircuitBreaker(failure_threshold=3, timeout=120)
        }
        
        logger.info("🔄 Auto-Retrain Service initialized")
    
    async def check_drift_and_retrain(self, model_name: str) -> Optional[RetrainEvent]:
        """
        Check for drift and trigger retraining if needed
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            RetrainEvent if retraining was triggered, None otherwise
        """
        try:
            logger.info(f"🔍 [AUTO-RETRAIN] Checking drift for model: {model_name}")
            
            # Check if model is in target list
            if model_name not in self.config.target_models:
                logger.info(f"ℹ️ [AUTO-RETRAIN] Model {model_name} not in target list, skipping")
                return None
            
            # Check cooldown period
            if not self._can_retrain(model_name):
                logger.info(f"⏰ [AUTO-RETRAIN] Model {model_name} in cooldown period, skipping")
                return None
            
            # Check daily retrain limit
            if not self._within_daily_limit(model_name):
                logger.warning(f"⚠️ [AUTO-RETRAIN] Daily retrain limit reached for {model_name}")
                return None
            
            # Get recent drift detection results
            drift_results = await self._get_recent_drift_results(model_name)
            if not drift_results:
                logger.info(f"ℹ️ [AUTO-RETRAIN] No recent drift results for {model_name}")
                return None
            
            # Evaluate drift severity
            retrain_trigger = self._evaluate_drift_severity(model_name, drift_results)
            if not retrain_trigger:
                logger.info(f"✅ [AUTO-RETRAIN] Drift within acceptable limits for {model_name}")
                return None
            
            # Trigger retraining
            retrain_event = await self._trigger_retraining(model_name, retrain_trigger)
            if retrain_event:
                logger.info(f"🚀 [AUTO-RETRAIN] Retraining triggered for {model_name}: {retrain_event.event_id}")
                return retrain_event
            
        except Exception as e:
            logger.error(f"❌ [AUTO-RETRAIN] Error checking drift for {model_name}: {e}")
        
        return None
    
    def _can_retrain(self, model_name: str) -> bool:
        """Check if model can be retrained (cooldown period)"""
        if model_name not in self.last_retrain_times:
            return True
        
        last_retrain = self.last_retrain_times[model_name]
        cooldown_end = last_retrain + timedelta(hours=self.config.retrain_cooldown_hours)
        
        return datetime.now() > cooldown_end
    
    def _within_daily_limit(self, model_name: str) -> bool:
        """Check if within daily retrain limit"""
        today = datetime.now().date()
        daily_key = f"{model_name}_{today}"
        
        current_count = self.daily_retrain_counts.get(daily_key, 0)
        return current_count < self.config.max_retrain_attempts
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def _get_recent_drift_results(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get recent drift detection results for a model"""
        try:
            # Check circuit breaker
            if not self.circuit_breakers["analytics"].can_execute():
                logger.warning("⚠️ [AUTO-RETRAIN] Analytics service circuit breaker is OPEN")
                return None
            
            client = await get_http_client()
            # Get recent drift detection results
            response = await client.get(
                f"{self.analytics_service_url}/drift/recent-results",
                params={"model_name": model_name, "hours": 24}
            )
                
                if response.status_code == 200:
                    self.circuit_breakers["analytics"].record_success()
                    return response.json()
                else:
                    self.circuit_breakers["analytics"].record_failure()
                    logger.warning(f"⚠️ [AUTO-RETRAIN] Failed to get drift results: {response.status_code}")
                    return None
                    
        except Exception as e:
            self.circuit_breakers["analytics"].record_failure()
            logger.error(f"❌ [AUTO-RETRAIN] Error getting drift results: {e}")
            raise  # Re-raise for retry logic
    
    def _evaluate_drift_severity(self, model_name: str, drift_results: Dict[str, Any]) -> Optional[RetrainTrigger]:
        """
        Evaluate drift severity and determine if retraining is needed
        
        Args:
            model_name: Name of the model
            drift_results: Recent drift detection results
            
        Returns:
            RetrainTrigger if retraining needed, None otherwise
        """
        try:
            # Check data drift
            data_drift_score = drift_results.get("data_drift_score", 0.0)
            if data_drift_score > self.config.data_drift_threshold:
                logger.warning(f"🚨 [AUTO-RETRAIN] Data drift detected: {data_drift_score:.3f} > {self.config.data_drift_threshold}")
                return RetrainTrigger.DATA_DRIFT
            
            # Check model drift
            model_drift_score = drift_results.get("model_drift_score", 0.0)
            if model_drift_score > self.config.model_drift_threshold:
                logger.warning(f"🚨 [AUTO-RETRAIN] Model drift detected: {model_drift_score:.3f} > {self.config.model_drift_threshold}")
                return RetrainTrigger.MODEL_DRIFT
            
            # Check performance drop
            performance_drop = drift_results.get("performance_drop", 0.0)
            if performance_drop > self.config.performance_drop_threshold:
                logger.warning(f"🚨 [AUTO-RETRAIN] Performance drop detected: {performance_drop:.3f} > {self.config.performance_drop_threshold}")
                return RetrainTrigger.PERFORMANCE_DROP
            
            return None
            
        except Exception as e:
            logger.error(f"❌ [AUTO-RETRAIN] Error evaluating drift severity: {e}")
            return None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def _trigger_retraining(self, model_name: str, trigger: RetrainTrigger) -> Optional[RetrainEvent]:
        """
        Trigger retraining for a model
        
        Args:
            model_name: Name of the model to retrain
            trigger: What triggered the retraining
            
        Returns:
            RetrainEvent if successful, None otherwise
        """
        try:
            # Check circuit breaker
            if not self.circuit_breakers["training"].can_execute():
                logger.warning("⚠️ [AUTO-RETRAIN] Training service circuit breaker is OPEN")
                return None
            
            # Create retrain event
            event_id = f"retrain_{model_name}_{int(datetime.now().timestamp())}"
            retrain_event = RetrainEvent(
                event_id=event_id,
                model_name=model_name,
                trigger=trigger,
                drift_score=0.0,  # Will be updated with actual score
                threshold=0.0,    # Will be updated with actual threshold
                timestamp=datetime.now()
            )
            
            # Get latest training data
            training_data_path = await self._get_latest_training_data()
            if not training_data_path:
                logger.error(f"❌ [AUTO-RETRAIN] No training data available for {model_name}")
                retrain_event.status = "failed"
                retrain_event.error_message = "No training data available"
                return retrain_event
            
            # Submit retrain job
            client = await get_http_client()
            retrain_request = {
                    "model_name": model_name,
                    "training_data_path": training_data_path,
                    "config": {
                        "model_name": model_name,
                        "epochs": 5,  # Reduced for faster retraining
                        "batch_size": 32,
                        "learning_rate": 2e-5,
                        "max_length": 512,
                        "validation_split": 0.2,
                        "early_stopping": True,
                        "patience": 3
                    },
                    "priority": self.config.retrain_priority,
                    "auto_retrain": True,
                    "trigger_reason": trigger.value
                }
                
                response = await client.post(
                    f"{self.training_service_url}/train",
                    json=retrain_request
                )
                
                if response.status_code == 200:
                    self.circuit_breakers["training"].record_success()
                    result = response.json()
                    retrain_event.retrain_job_id = result.get("job_id")
                    retrain_event.status = "in_progress"
                    
                    # Update tracking
                    self.retrain_events[event_id] = retrain_event
                    self.last_retrain_times[model_name] = datetime.now()
                    
                    # Update daily count
                    today = datetime.now().date()
                    daily_key = f"{model_name}_{today}"
                    self.daily_retrain_counts[daily_key] = self.daily_retrain_counts.get(daily_key, 0) + 1
                    
                    logger.info(f"✅ [AUTO-RETRAIN] Retrain job submitted: {retrain_event.retrain_job_id}")
                    return retrain_event
                else:
                    self.circuit_breakers["training"].record_failure()
                    logger.error(f"❌ [AUTO-RETRAIN] Failed to submit retrain job: {response.status_code}")
                    retrain_event.status = "failed"
                    retrain_event.error_message = f"HTTP {response.status_code}: {response.text}"
                    return retrain_event
                    
        except Exception as e:
            self.circuit_breakers["training"].record_failure()
            logger.error(f"❌ [AUTO-RETRAIN] Error triggering retraining: {e}")
            if 'retrain_event' in locals():
                retrain_event.status = "failed"
                retrain_event.error_message = str(e)
            raise  # Re-raise for retry logic
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def _get_latest_training_data(self) -> Optional[str]:
        """Get the latest training data path from production inference logs"""
        try:
            # Check circuit breaker
            if not self.circuit_breakers["training"].can_execute():
                logger.warning("⚠️ [AUTO-RETRAIN] Training service circuit breaker is OPEN")
                return None
            
            client = await get_http_client()
            # First try to get production inference data for retraining
            response = await client.get(f"{self.analytics_service_url}/data/production-inference")
                
                if response.status_code == 200:
                    self.circuit_breakers["training"].record_success()
                    data_info = response.json()
                    if data_info and data_info.get("s3_path"):
                        logger.info(f"✅ [AUTO-RETRAIN] Using production inference data: {data_info['s3_path']}")
                        return data_info["s3_path"]
                
                # Fallback to fresh training data if production data not available
                logger.info("ℹ️ [AUTO-RETRAIN] Production inference data not available, trying fresh training data")
                response = await client.get(f"{self.training_service_url}/data/fresh-data")
                
                if response.status_code == 200:
                    self.circuit_breakers["training"].record_success()
                    data_info = response.json()
                    if data_info and len(data_info) > 0:
                        # Get the most recent data file
                        latest_data = max(data_info, key=lambda x: x.get("timestamp", ""))
                        logger.info(f"✅ [AUTO-RETRAIN] Using fresh training data: {latest_data.get('s3_path')}")
                        return latest_data.get("s3_path")
                
                self.circuit_breakers["training"].record_failure()
                logger.warning("⚠️ [AUTO-RETRAIN] No training data available")
                return None
                
        except Exception as e:
            self.circuit_breakers["training"].record_failure()
            logger.error(f"❌ [AUTO-RETRAIN] Error getting training data: {e}")
            raise  # Re-raise for retry logic
    
    async def monitor_retrain_jobs(self):
        """Monitor retrain jobs and update their status"""
        try:
            for event_id, event in self.retrain_events.items():
                if event.status == "in_progress" and event.retrain_job_id:
                    await self._update_retrain_job_status(event)
                    
        except Exception as e:
            logger.error(f"❌ [AUTO-RETRAIN] Error monitoring retrain jobs: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def _update_retrain_job_status(self, event: RetrainEvent):
        """Update the status of a retrain job"""
        try:
            # Check circuit breaker
            if not self.circuit_breakers["training"].can_execute():
                logger.warning("⚠️ [AUTO-RETRAIN] Training service circuit breaker is OPEN")
                return
            
            client = await get_http_client()
            response = await client.get(
                f"{self.training_service_url}/jobs/{event.retrain_job_id}"
            )
                
                if response.status_code == 200:
                    self.circuit_breakers["training"].record_success()
                    job_info = response.json()
                    status = job_info.get("status", "unknown")
                    
                    if status == "completed":
                        event.status = "completed"
                        logger.info(f"✅ [AUTO-RETRAIN] Retrain job completed: {event.retrain_job_id}")
                        
                        # Trigger model promotion
                        await self._promote_retrained_model(event)
                        
                    elif status == "failed":
                        event.status = "failed"
                        event.error_message = job_info.get("error_message", "Unknown error")
                        logger.error(f"❌ [AUTO-RETRAIN] Retrain job failed: {event.retrain_job_id}")
                else:
                    self.circuit_breakers["training"].record_failure()
                    logger.warning(f"⚠️ [AUTO-RETRAIN] Failed to get job status: {response.status_code}")
                        
        except Exception as e:
            self.circuit_breakers["training"].record_failure()
            logger.error(f"❌ [AUTO-RETRAIN] Error updating job status: {e}")
            raise  # Re-raise for retry logic
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def _promote_retrained_model(self, event: RetrainEvent):
        """Promote the retrained model to production"""
        try:
            # Check circuit breaker
            if not self.circuit_breakers["training"].can_execute():
                logger.warning("⚠️ [AUTO-RETRAIN] Training service circuit breaker is OPEN")
                return
            
            logger.info(f"🔄 [AUTO-RETRAIN] Promoting retrained model: {event.model_name}")
            
            client = await get_http_client()
            # Promote model to production
            response = await client.post(
                f"{self.training_service_url}/models/{event.model_name}/promote",
                json={
                    "target_stage": "Production",
                    "reason": f"Auto-retrain triggered by {event.trigger.value}",
                    "drift_score": event.drift_score,
                    "threshold": event.threshold
                }
            )
                
                if response.status_code == 200:
                    self.circuit_breakers["training"].record_success()
                    logger.info(f"✅ [AUTO-RETRAIN] Model promoted successfully: {event.model_name}")
                else:
                    self.circuit_breakers["training"].record_failure()
                    logger.warning(f"⚠️ [AUTO-RETRAIN] Model promotion failed: {response.status_code}")
                    
        except Exception as e:
            self.circuit_breakers["training"].record_failure()
            logger.error(f"❌ [AUTO-RETRAIN] Error promoting model: {e}")
            raise  # Re-raise for retry logic
    
    async def get_retrain_status(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get status of retrain events"""
        try:
            if model_name:
                events = {k: v for k, v in self.retrain_events.items() 
                         if v.model_name == model_name}
            else:
                events = self.retrain_events
            
            return {
                "total_events": len(events),
                "events": [
                    {
                        "event_id": event.event_id,
                        "model_name": event.model_name,
                        "trigger": event.trigger.value,
                        "status": event.status,
                        "timestamp": event.timestamp.isoformat(),
                        "retrain_job_id": event.retrain_job_id,
                        "error_message": event.error_message
                    }
                    for event in events.values()
                ],
                "last_retrain_times": {
                    model: time.isoformat() 
                    for model, time in self.last_retrain_times.items()
                },
                "daily_counts": self.daily_retrain_counts
            }
            
        except Exception as e:
            logger.error(f"❌ [AUTO-RETRAIN] Error getting retrain status: {e}")
            return {"error": str(e)}
    
    async def start_monitoring(self):
        """Start the auto-retrain monitoring loop"""
        logger.info("🚀 [AUTO-RETRAIN] Starting monitoring loop")
        
        while True:
            try:
                # Check each target model
                for model_name in self.config.target_models:
                    await self.check_drift_and_retrain(model_name)
                
                # Monitor retrain jobs
                await self.monitor_retrain_jobs()
                
                # Wait before next check (every 30 minutes)
                await asyncio.sleep(1800)
                
            except Exception as e:
                logger.error(f"❌ [AUTO-RETRAIN] Error in monitoring loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

# Global instance
auto_retrain_service = AutoRetrainService()
