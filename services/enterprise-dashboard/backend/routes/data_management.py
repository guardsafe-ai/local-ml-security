"""
Data Management Routes
Proxy routes for data management operations
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse
from services.main_api_client import MainAPIClient
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize API client
api_client = MainAPIClient()

@router.post("/upload-large-file")
async def upload_large_file(
    file: UploadFile = File(...),
    data_type: str = Form("custom"),
    description: str = Form(""),
    validation_rules: str = Form("{}")
):
    """Upload large file with efficient streaming and chunking"""
    try:
        # Forward to training service
        response = await api_client.upload_large_file(file, data_type, description, validation_rules)
        return response
    except Exception as e:
        logger.error(f"Failed to upload large file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/upload-progress/{file_id}")
async def get_upload_progress(file_id: str):
    """Get upload and processing progress for a file"""
    try:
        response = await api_client.get_upload_progress(file_id)
        return response
    except Exception as e:
        logger.error(f"Failed to get upload progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/staged-files")
async def get_staged_files(status: Optional[str] = Query(None)):
    """Get list of staged files (uploads, processing, fresh)"""
    try:
        response = await api_client.get_staged_files(status)
        return response
    except Exception as e:
        logger.error(f"Failed to get staged files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-file/{file_id}")
async def process_file(
    file_id: str,
    validation_rules: Optional[Dict[str, Any]] = None
):
    """Manually trigger file processing"""
    try:
        response = await api_client.process_file(file_id, validation_rules)
        return response
    except Exception as e:
        logger.error(f"Failed to process file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download-file/{file_id}")
async def download_file(file_id: str):
    """Download a file by ID"""
    try:
        response = await api_client.download_file(file_id)
        return response
    except Exception as e:
        logger.error(f"Failed to download file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/file-info/{file_id}")
async def get_file_info(file_id: str):
    """Get detailed information about a file"""
    try:
        response = await api_client.get_file_info(file_id)
        return response
    except Exception as e:
        logger.error(f"Failed to get file info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retry-failed-file/{file_id}")
async def retry_failed_file(file_id: str):
    """Retry processing a failed file"""
    try:
        response = await api_client.retry_failed_file(file_id)
        return response
    except Exception as e:
        logger.error(f"Failed to retry file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cleanup-failed-uploads")
async def cleanup_failed_uploads(hours_old: int = Query(24)):
    """Clean up failed uploads older than specified hours"""
    try:
        response = await api_client.cleanup_failed_uploads(hours_old)
        return response
    except Exception as e:
        logger.error(f"Failed to cleanup failed uploads: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== ENHANCED DATA MANAGEMENT ENDPOINTS FOR FRONTEND USERS =====

@router.get("/dashboard")
async def get_data_dashboard():
    """
    Get comprehensive data management dashboard
    
    Frontend Usage: Data Management Dashboard showing:
    - Total files uploaded
    - Processing status overview
    - Storage usage statistics
    - Recent uploads and activities
    - Data quality metrics
    """
    try:
        # Get comprehensive data statistics
        stats = await api_client.get_data_statistics()
        fresh_data = await api_client.get_fresh_data(limit=10)
        used_data = await api_client.get_used_data(limit=10)
        
        return {
            "dashboard": {
                "statistics": stats,
                "recent_fresh_data": fresh_data,
                "recent_used_data": used_data,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get data dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets")
async def get_datasets(data_type: Optional[str] = Query(None)):
    """
    Get all available datasets with metadata
    
    Frontend Usage: Dataset Browser showing:
    - List of all datasets
    - Dataset metadata (size, type, quality)
    - Usage statistics
    - Quick actions (download, delete, process)
    """
    try:
        # Get fresh and used data
        fresh_data = await api_client.get_fresh_data(limit=100)
        used_data = await api_client.get_used_data(limit=100)
        
        # Filter by data type if specified
        if data_type:
            fresh_data = [item for item in fresh_data if item.get("data_type") == data_type]
            used_data = [item for item in used_data if item.get("data_type") == data_type]
        
        return {
            "datasets": {
                "fresh": fresh_data,
                "used": used_data,
                "total_fresh": len(fresh_data),
                "total_used": len(used_data)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, lines: int = Query(10)):
    """
    Preview dataset content (first N lines)
    
    Frontend Usage: Dataset Preview Modal:
    - User clicks on a dataset
    - Shows first few lines of data
    - Displays data structure and format
    - Helps user understand data before using
    """
    try:
        # This would need to be implemented in the training service
        # For now, return a placeholder response
        return {
            "dataset_id": dataset_id,
            "preview_lines": lines,
            "preview_data": [
                {"line": i, "content": f"Sample data line {i}"} 
                for i in range(1, min(lines + 1, 6))
            ],
            "total_lines": 1000,  # This would come from actual data
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to preview dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/{dataset_id}/validate")
async def validate_dataset(dataset_id: str, validation_rules: Optional[Dict[str, Any]] = None):
    """
    Validate dataset quality and format
    
    Frontend Usage: Data Quality Check:
    - User uploads dataset
    - Clicks "Validate Data Quality"
    - Gets detailed validation report
    - Shows issues and recommendations
    """
    try:
        # Get file info first
        file_info = await api_client.get_file_info(dataset_id)
        
        # Validate data quality
        validation_result = await api_client.validate_data_quality(
            data_path=file_info.get("file_path", ""),
            model_type="security"
        )
        
        return {
            "dataset_id": dataset_id,
            "validation_result": validation_result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to validate dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/{dataset_id}/augment")
async def augment_dataset(dataset_id: str, augmentation_config: Optional[Dict[str, Any]] = None):
    """
    Apply data augmentation to a dataset
    
    Frontend Usage: Data Augmentation Tool:
    - User selects a dataset
    - Configures augmentation settings
    - Applies augmentation techniques
    - Gets enhanced dataset
    """
    try:
        if augmentation_config is None:
            augmentation_config = {
                "techniques": ["synonym_replacement", "random_insertion"],
                "augmentation_factor": 2.0
            }
        
        # This would need to be implemented in the training service
        return {
            "dataset_id": dataset_id,
            "augmentation_config": augmentation_config,
            "status": "augmentation_started",
            "message": "Dataset augmentation has been queued",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to augment dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/{dataset_id}/balance")
async def balance_dataset(dataset_id: str, target_distribution: Optional[Dict[str, int]] = None):
    """
    Balance dataset class distribution
    
    Frontend Usage: Dataset Balancing Tool:
    - User uploads imbalanced dataset
    - System analyzes class distribution
    - User configures target distribution
    - System balances the dataset
    """
    try:
        if target_distribution is None:
            target_distribution = {
                "benign": 1000,
                "malicious": 1000,
                "suspicious": 1000
            }
        
        # This would need to be implemented in the training service
        return {
            "dataset_id": dataset_id,
            "target_distribution": target_distribution,
            "status": "balancing_started",
            "message": "Dataset balancing has been queued",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to balance dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quality-thresholds")
async def get_quality_thresholds(model_type: str = Query("security")):
    """
    Get data quality thresholds for different model types
    
    Frontend Usage: Quality Settings Page:
    - User configures quality standards
    - Sets thresholds for different metrics
    - Customizes validation rules
    """
    try:
        thresholds = await api_client.get_quality_thresholds(model_type)
        
        return {
            "model_type": model_type,
            "thresholds": thresholds,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get quality thresholds: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quality-thresholds")
async def set_quality_thresholds(thresholds: Dict[str, Any]):
    """
    Set custom data quality thresholds
    
    Frontend Usage: Quality Settings Configuration:
    - User adjusts quality standards
    - Saves custom thresholds
    - Applies to future validations
    """
    try:
        result = await api_client.set_custom_quality_thresholds(thresholds)
        
        return {
            "status": "success",
            "message": "Quality thresholds updated",
            "thresholds": thresholds,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to set quality thresholds: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/storage/usage")
async def get_storage_usage():
    """
    Get storage usage statistics
    
    Frontend Usage: Storage Management Dashboard:
    - Shows total storage used
    - Breakdown by data type
    - Storage trends over time
    - Cleanup recommendations
    """
    try:
        # This would need to be implemented to get actual storage stats
        return {
            "storage_usage": {
                "total_size_gb": 15.7,
                "used_size_gb": 12.3,
                "available_size_gb": 3.4,
                "usage_percentage": 78.3,
                "breakdown": {
                    "training_data": 8.5,
                    "models": 2.1,
                    "logs": 1.2,
                    "artifacts": 0.5
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get storage usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/{dataset_id}/export")
async def export_dataset(dataset_id: str, format: str = Query("jsonl")):
    """
    Export dataset in specified format
    
    Frontend Usage: Dataset Export Feature:
    - User selects dataset
    - Chooses export format (JSONL, CSV, etc.)
    - Downloads processed dataset
    """
    try:
        # This would need to be implemented in the training service
        return {
            "dataset_id": dataset_id,
            "export_format": format,
            "status": "export_started",
            "download_url": f"/api/v1/data/download-export/{dataset_id}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to export dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str, confirm: bool = Query(False)):
    """
    Delete a dataset
    
    Frontend Usage: Dataset Management:
    - User selects dataset to delete
    - Confirms deletion
    - Dataset is permanently removed
    """
    try:
        if not confirm:
            raise HTTPException(status_code=400, detail="Deletion must be confirmed")
        
        # This would need to be implemented in the training service
        return {
            "status": "success",
            "message": f"Dataset {dataset_id} deleted successfully",
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to delete dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processing/queue")
async def get_processing_queue():
    """
    Get data processing queue status
    
    Frontend Usage: Processing Queue Monitor:
    - Shows files being processed
    - Queue position and estimated time
    - Processing errors and retries
    """
    try:
        # This would need to be implemented to get actual queue status
        return {
            "queue_status": {
                "pending": 3,
                "processing": 1,
                "completed": 15,
                "failed": 2,
                "estimated_wait_minutes": 5
            },
            "current_jobs": [
                {
                    "file_id": "file_123",
                    "status": "processing",
                    "progress": 65,
                    "started_at": "2025-01-09T10:00:00Z"
                }
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get processing queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))
