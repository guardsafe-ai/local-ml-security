"""
Data Management Routes
Proxy routes for data management operations
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse
from services.api_client import APIClient

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize API client
api_client = APIClient()

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
