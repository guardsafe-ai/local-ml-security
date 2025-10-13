"""
Auto-Retrain Routes
API endpoints for automatic model retraining on drift detection
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime

from services.auto_retrain import auto_retrain_service, RetrainConfig, RetrainTrigger

logger = logging.getLogger(__name__)
router = APIRouter()

class RetrainConfigRequest(BaseModel):
    """Request model for updating retrain configuration"""
    data_drift_threshold: Optional[float] = None
    model_drift_threshold: Optional[float] = None
    performance_drop_threshold: Optional[float] = None
    min_samples_for_retrain: Optional[int] = None
    retrain_cooldown_hours: Optional[int] = None
    max_retrain_attempts: Optional[int] = None
    target_models: Optional[List[str]] = None
    retrain_priority: Optional[str] = None

class ManualRetrainRequest(BaseModel):
    """Request model for manual retrain trigger"""
    model_name: str
    reason: str = "Manual trigger"
    priority: str = "normal"

class RetrainStatusResponse(BaseModel):
    """Response model for retrain status"""
    total_events: int
    events: List[Dict[str, Any]]
    last_retrain_times: Dict[str, str]
    daily_counts: Dict[str, int]

@router.get("/config", response_model=Dict[str, Any])
async def get_retrain_config():
    """Get current auto-retrain configuration"""
    try:
        config = auto_retrain_service.config
        return {
            "data_drift_threshold": config.data_drift_threshold,
            "model_drift_threshold": config.model_drift_threshold,
            "performance_drop_threshold": config.performance_drop_threshold,
            "min_samples_for_retrain": config.min_samples_for_retrain,
            "retrain_cooldown_hours": config.retrain_cooldown_hours,
            "max_retrain_attempts": config.max_retrain_attempts,
            "target_models": config.target_models,
            "retrain_priority": config.retrain_priority
        }
    except Exception as e:
        logger.error(f"Error getting retrain config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/config")
async def update_retrain_config(request: RetrainConfigRequest):
    """Update auto-retrain configuration"""
    try:
        config = auto_retrain_service.config
        
        # Update configuration fields
        if request.data_drift_threshold is not None:
            config.data_drift_threshold = request.data_drift_threshold
        if request.model_drift_threshold is not None:
            config.model_drift_threshold = request.model_drift_threshold
        if request.performance_drop_threshold is not None:
            config.performance_drop_threshold = request.performance_drop_threshold
        if request.min_samples_for_retrain is not None:
            config.min_samples_for_retrain = request.min_samples_for_retrain
        if request.retrain_cooldown_hours is not None:
            config.retrain_cooldown_hours = request.retrain_cooldown_hours
        if request.max_retrain_attempts is not None:
            config.max_retrain_attempts = request.max_retrain_attempts
        if request.target_models is not None:
            config.target_models = request.target_models
        if request.retrain_priority is not None:
            config.retrain_priority = request.retrain_priority
        
        logger.info(f"✅ [AUTO-RETRAIN] Configuration updated")
        return {"message": "Configuration updated successfully", "config": config.__dict__}
        
    except Exception as e:
        logger.error(f"Error updating retrain config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trigger/{model_name}")
async def trigger_manual_retrain(model_name: str, request: ManualRetrainRequest):
    """Manually trigger retraining for a specific model"""
    try:
        logger.info(f"🔄 [AUTO-RETRAIN] Manual retrain triggered for {model_name}")
        
        # Check if model is in target list
        if model_name not in auto_retrain_service.config.target_models:
            raise HTTPException(
                status_code=400, 
                detail=f"Model {model_name} not in target models list"
            )
        
        # Check cooldown period
        if not auto_retrain_service._can_retrain(model_name):
            raise HTTPException(
                status_code=429,
                detail=f"Model {model_name} is in cooldown period"
            )
        
        # Check daily limit
        if not auto_retrain_service._within_daily_limit(model_name):
            raise HTTPException(
                status_code=429,
                detail=f"Daily retrain limit reached for {model_name}"
            )
        
        # Create manual retrain event
        event_id = f"manual_retrain_{model_name}_{int(datetime.now().timestamp())}"
        retrain_event = auto_retrain_service.RetrainEvent(
            event_id=event_id,
            model_name=model_name,
            trigger=RetrainTrigger.MANUAL,
            drift_score=0.0,
            threshold=0.0,
            timestamp=datetime.now()
        )
        
        # Trigger retraining
        result = await auto_retrain_service._trigger_retraining(model_name, RetrainTrigger.MANUAL)
        
        if result:
            return {
                "message": f"Manual retrain triggered for {model_name}",
                "event_id": result.event_id,
                "job_id": result.retrain_job_id,
                "status": result.status
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to trigger retraining")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering manual retrain: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=RetrainStatusResponse)
async def get_retrain_status(model_name: Optional[str] = None):
    """Get status of retrain events"""
    try:
        status = await auto_retrain_service.get_retrain_status(model_name)
        return RetrainStatusResponse(**status)
    except Exception as e:
        logger.error(f"Error getting retrain status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{model_name}", response_model=RetrainStatusResponse)
async def get_model_retrain_status(model_name: str):
    """Get retrain status for a specific model"""
    try:
        status = await auto_retrain_service.get_retrain_status(model_name)
        return RetrainStatusResponse(**status)
    except Exception as e:
        logger.error(f"Error getting model retrain status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/check-drift/{model_name}")
async def check_drift_and_retrain(model_name: str):
    """Manually check drift and trigger retraining if needed"""
    try:
        logger.info(f"🔍 [AUTO-RETRAIN] Manual drift check for {model_name}")
        
        result = await auto_retrain_service.check_drift_and_retrain(model_name)
        
        if result:
            return {
                "message": f"Drift detected and retraining triggered for {model_name}",
                "event_id": result.event_id,
                "trigger": result.trigger.value,
                "job_id": result.retrain_job_id,
                "status": result.status
            }
        else:
            return {
                "message": f"No drift detected or retraining not needed for {model_name}",
                "drift_within_limits": True
            }
            
    except Exception as e:
        logger.error(f"Error checking drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/events")
async def get_retrain_events(limit: int = 50, status: Optional[str] = None):
    """Get retrain events with optional filtering"""
    try:
        events = list(auto_retrain_service.retrain_events.values())
        
        # Filter by status if provided
        if status:
            events = [e for e in events if e.status == status]
        
        # Sort by timestamp (newest first)
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Limit results
        events = events[:limit]
        
        return {
            "events": [
                {
                    "event_id": event.event_id,
                    "model_name": event.model_name,
                    "trigger": event.trigger.value,
                    "drift_score": event.drift_score,
                    "threshold": event.threshold,
                    "status": event.status,
                    "timestamp": event.timestamp.isoformat(),
                    "retrain_job_id": event.retrain_job_id,
                    "error_message": event.error_message
                }
                for event in events
            ],
            "total_count": len(events)
        }
        
    except Exception as e:
        logger.error(f"Error getting retrain events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/events/{event_id}")
async def cancel_retrain_event(event_id: str):
    """Cancel a retrain event (if still pending)"""
    try:
        if event_id not in auto_retrain_service.retrain_events:
            raise HTTPException(status_code=404, detail="Event not found")
        
        event = auto_retrain_service.retrain_events[event_id]
        
        if event.status == "completed":
            raise HTTPException(status_code=400, detail="Cannot cancel completed event")
        
        if event.status == "in_progress":
            # Try to cancel the training job
            try:
                import httpx
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.delete(
                        f"http://training:8002/jobs/{event.retrain_job_id}"
                    )
                    if response.status_code == 200:
                        event.status = "cancelled"
                        logger.info(f"✅ [AUTO-RETRAIN] Event cancelled: {event_id}")
                    else:
                        logger.warning(f"⚠️ [AUTO-RETRAIN] Failed to cancel job: {response.status_code}")
            except Exception as e:
                logger.error(f"Error cancelling job: {e}")
        
        event.status = "cancelled"
        
        return {"message": f"Event {event_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling retrain event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start-monitoring")
async def start_monitoring(background_tasks: BackgroundTasks):
    """Start the auto-retrain monitoring loop"""
    try:
        # Start monitoring in background
        background_tasks.add_task(auto_retrain_service.start_monitoring)
        
        logger.info("🚀 [AUTO-RETRAIN] Monitoring started")
        return {"message": "Auto-retrain monitoring started"}
        
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check for auto-retrain service"""
    try:
        return {
            "status": "healthy",
            "service": "auto-retrain",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "target_models": auto_retrain_service.config.target_models,
                "data_drift_threshold": auto_retrain_service.config.data_drift_threshold,
                "model_drift_threshold": auto_retrain_service.config.model_drift_threshold,
                "performance_drop_threshold": auto_retrain_service.config.performance_drop_threshold
            },
            "stats": {
                "total_events": len(auto_retrain_service.retrain_events),
                "active_events": len([e for e in auto_retrain_service.retrain_events.values() 
                                    if e.status in ["pending", "in_progress"]]),
                "completed_events": len([e for e in auto_retrain_service.retrain_events.values() 
                                       if e.status == "completed"]),
                "failed_events": len([e for e in auto_retrain_service.retrain_events.values() 
                                    if e.status == "failed"])
            }
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))
