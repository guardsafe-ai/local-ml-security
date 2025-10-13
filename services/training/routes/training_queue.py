"""
Training Queue API Routes
API endpoints for managing training job queue
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from services.training_queue import training_queue, JobPriority, JobStatus

logger = logging.getLogger(__name__)
router = APIRouter()

class TrainingJobRequest(BaseModel):
    """Training job request model"""
    model_name: str
    training_data_path: str
    config: Dict[str, Any]
    priority: str = "NORMAL"
    timeout_seconds: int = 3600
    resource_requirements: Optional[Dict[str, Any]] = None

class TrainingJobResponse(BaseModel):
    """Training job response model"""
    job_id: str
    status: str
    message: str
    created_at: str

class JobStatusResponse(BaseModel):
    """Job status response model"""
    job_id: str
    model_name: str
    status: str
    progress: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

class QueueStatsResponse(BaseModel):
    """Queue statistics response model"""
    pending_jobs: int
    running_jobs: int
    completed_jobs: int
    failed_jobs: int
    total_jobs: int
    max_workers: int
    active_workers: int
    recent_jobs: list

@router.post("/queue/submit", response_model=TrainingJobResponse)
async def submit_training_job(request: TrainingJobRequest):
    """Submit a training job to the queue"""
    try:
        # Convert priority string to enum
        priority_map = {
            "LOW": JobPriority.LOW,
            "NORMAL": JobPriority.NORMAL,
            "HIGH": JobPriority.HIGH,
            "URGENT": JobPriority.URGENT
        }
        
        priority = priority_map.get(request.priority.upper(), JobPriority.NORMAL)
        
        # Submit job to queue
        job_id = await training_queue.submit_job(
            model_name=request.model_name,
            training_data_path=request.training_data_path,
            config=request.config,
            priority=priority,
            timeout_seconds=request.timeout_seconds,
            resource_requirements=request.resource_requirements
        )
        
        return TrainingJobResponse(
            job_id=job_id,
            status="submitted",
            message=f"Training job {job_id} submitted to queue",
            created_at=training_queue.running_jobs.get(job_id, {}).get('created_at', 'unknown')
        )
        
    except Exception as e:
        logger.error(f"Failed to submit training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/queue/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of a specific training job"""
    try:
        job = await training_queue.get_job_status(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        return JobStatusResponse(
            job_id=job.job_id,
            model_name=job.model_name,
            status=job.status.value,
            progress=job.progress,
            created_at=job.created_at.isoformat(),
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            error_message=job.error_message,
            result=job.result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/queue/job/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a training job"""
    try:
        success = await training_queue.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or cannot be cancelled")
        
        return {
            "status": "success",
            "message": f"Job {job_id} cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/queue/stats", response_model=QueueStatsResponse)
async def get_queue_stats():
    """Get training queue statistics"""
    try:
        stats = await training_queue.get_queue_stats()
        
        return QueueStatsResponse(
            pending_jobs=stats.get("pending_jobs", 0),
            running_jobs=stats.get("running_jobs", 0),
            completed_jobs=stats.get("completed_jobs", 0),
            failed_jobs=stats.get("failed_jobs", 0),
            total_jobs=stats.get("total_jobs", 0),
            max_workers=stats.get("max_workers", 0),
            active_workers=stats.get("active_workers", 0),
            recent_jobs=stats.get("recent_jobs", [])
        )
        
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/queue/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 50):
    """List training jobs with optional status filter"""
    try:
        stats = await training_queue.get_queue_stats()
        recent_jobs = stats.get("recent_jobs", [])
        
        # Filter by status if provided
        if status:
            recent_jobs = [job for job in recent_jobs if job.get("status", "").lower() == status.lower()]
        
        # Limit results
        recent_jobs = recent_jobs[:limit]
        
        return {
            "jobs": recent_jobs,
            "count": len(recent_jobs),
            "filters": {
                "status": status,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/queue/retry/{job_id}")
async def retry_job(job_id: str):
    """Retry a failed training job"""
    try:
        job = await training_queue.get_job_status(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        if job.status != JobStatus.FAILED:
            raise HTTPException(status_code=400, detail=f"Job {job_id} is not in failed status")
        
        if job.retry_count >= job.max_retries:
            raise HTTPException(status_code=400, detail=f"Job {job_id} has exceeded maximum retries")
        
        # Resubmit job with incremented retry count
        new_job_id = await training_queue.submit_job(
            model_name=job.model_name,
            training_data_path=job.training_data_path,
            config=job.config,
            priority=job.priority,
            timeout_seconds=job.timeout_seconds,
            resource_requirements=job.resource_requirements
        )
        
        return {
            "status": "success",
            "message": f"Job {job_id} retried as {new_job_id}",
            "new_job_id": new_job_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
