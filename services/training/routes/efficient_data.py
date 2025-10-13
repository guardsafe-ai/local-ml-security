"""
Efficient Data Management API Routes
Full implementation of the efficient data management system
"""

import os
import asyncio
import tempfile
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from efficient_data_manager import DataType, DataStatus
from shared_data_manager import shared_data_manager
import json
import logging

logger = logging.getLogger(__name__)

# Use shared data manager instance
data_manager = shared_data_manager

# Pydantic models for API responses
class UploadProgressResponse(BaseModel):
    file_id: str
    status: str
    progress: float
    file_size: int
    chunk_count: int
    error: Optional[str] = None

class FileInfoResponse(BaseModel):
    file_id: str
    original_name: str
    minio_path: str
    s3_url: str  # Full S3 URL for easy copying
    data_type: str
    status: str
    upload_time: str
    file_size: int
    file_hash: str
    used_count: int
    last_used: Optional[str] = None
    training_jobs: List[str]
    metadata: Dict[str, Any]
    processing_progress: float
    processing_error: Optional[str] = None
    chunk_count: int

class StagedFilesResponse(BaseModel):
    files: List[FileInfoResponse]
    total_count: int
    by_status: Dict[str, int]

class ValidationRules(BaseModel):
    max_file_size: Optional[int] = None
    allowed_formats: Optional[List[str]] = None
    min_records: Optional[int] = None
    custom_validation: Optional[Dict[str, Any]] = None

router = APIRouter()

@router.post("/upload-large-file")
async def upload_large_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    data_type: str = Form("custom"),
    description: str = Form(""),
    metadata: str = Form("{}")
):
    """Upload large file with chunked upload support"""
    try:
        # Input validation and sanitization
        try:
            from utils.input_sanitizer import input_sanitizer
            sanitized_description = input_sanitizer.sanitize_text(description, max_length=1000) if description else ""
            sanitized_metadata = input_sanitizer.sanitize_text(metadata, max_length=5000) if metadata else "{}"
            description = sanitized_description
            metadata = sanitized_metadata
        except Exception as e:
            logger.error(f"Input sanitization failed: {e}")
            raise HTTPException(status_code=400, detail="Input validation failed")
        
        # Parse metadata
        try:
            metadata_dict = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            metadata_dict = {}
        
        # Add description to metadata
        if description:
            metadata_dict["description"] = description
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Upload using efficient data manager
            upload_result = await data_manager.upload_large_file(
                local_path=temp_path,
                data_type=data_type,
                metadata=metadata_dict,
                progress_callback=None  # We'll handle progress via polling
            )
            
            # Handle both string (old format) and dict (new format) responses
            if isinstance(upload_result, str):
                file_id = upload_result
                warning = None
                is_duplicate = False
            else:
                file_id = upload_result["file_id"]
                warning = upload_result.get("warning")
                is_duplicate = upload_result.get("is_duplicate", False)
            
            # Start background processing only for new uploads
            if not is_duplicate:
                background_tasks.add_task(process_uploaded_file_background, file_id)
            
            # Create appropriate response message
            if is_duplicate:
                message = f"Duplicate file detected: {file.filename} (using existing file)"
            else:
                message = f"File upload started: {file.filename}"
            
            response = {
                "status": "success",
                "message": message,
                "file_id": file_id,
                "file_size": file_size,
                "data_type": data_type
            }
            
            # Add warning if present
            if warning:
                response["warning"] = warning
                response["is_duplicate"] = is_duplicate
            
            return response
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_uploaded_file_background(file_id: str):
    """Background task to process uploaded file"""
    try:
        # First, check for PII in the uploaded data
        from main import classify_training_data
        
        # Get file content for PII detection
        file_info = data_manager.data_files.get(file_id)
        if file_info:
            # Download file content for PII detection
            try:
                file_content = data_manager.minio_client.get_object(
                    data_manager.bucket_name, 
                    file_info.minio_path
                )
                content_data = file_content.read().decode('utf-8')
                
                # Classify data for PII
                classification = await classify_training_data(
                    {"text": content_data[:1000]},  # Sample first 1000 chars
                    file_id
                )
                
                if classification and classification.get("contains_pii"):
                    logger.warning(f"⚠️ PII detected in {file_id}: {classification.get('pii_fields', [])}")
                    logger.warning(f"⚠️ Sensitivity score: {classification.get('sensitivity_score', 0.0)}")
                    
                    # For now, just log the warning. In production, you might want to:
                    # 1. Block the upload if sensitivity_score > 0.7
                    # 2. Auto-anonymize the data
                    # 3. Send alerts to data privacy team
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to check PII for {file_id}: {e}")
        
        # Process the file normally
        await data_manager.process_uploaded_file(file_id)
        logger.info(f"Background processing completed for file: {file_id}")
    except Exception as e:
        logger.error(f"Background processing failed for file {file_id}: {e}")

@router.get("/upload-progress/{file_id}")
async def get_upload_progress(file_id: str):
    """Get upload progress for a specific file"""
    try:
        progress = await data_manager.get_upload_progress(file_id)
        return UploadProgressResponse(**progress)
    except Exception as e:
        logger.error(f"Failed to get upload progress for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/staged-files")
async def get_staged_files(status: Optional[str] = None):
    """Get all staged files with optional status filter"""
    try:
        files = []
        for file_id, file_info in data_manager.data_files.items():
            file_status = file_info.status.value if hasattr(file_info.status, 'value') else str(file_info.status)
            if status and file_status != status:
                continue
                
            # Construct full S3 URL
            s3_url = f"s3://{data_manager.bucket_name}/{file_info.minio_path}"
            
            file_response = FileInfoResponse(
                file_id=file_info.file_id,
                original_name=file_info.original_name,
                minio_path=file_info.minio_path,
                s3_url=s3_url,
                data_type=file_info.data_type.value if hasattr(file_info.data_type, 'value') else str(file_info.data_type),
                status=file_info.status.value if hasattr(file_info.status, 'value') else str(file_info.status),
                upload_time=file_info.upload_time.isoformat() if hasattr(file_info.upload_time, 'isoformat') else str(file_info.upload_time),
                file_size=file_info.file_size,
                file_hash=file_info.file_hash,
                used_count=file_info.used_count,
                last_used=file_info.last_used.isoformat() if file_info.last_used and hasattr(file_info.last_used, 'isoformat') else (str(file_info.last_used) if file_info.last_used else None),
                training_jobs=file_info.training_jobs,
                metadata=file_info.metadata,
                processing_progress=file_info.processing_progress,
                processing_error=file_info.processing_error,
                chunk_count=file_info.chunk_count
            )
            files.append(file_response)
        
        # Count by status
        by_status = {}
        for file_info in data_manager.data_files.values():
            status_key = file_info.status.value if hasattr(file_info.status, 'value') else str(file_info.status)
            by_status[status_key] = by_status.get(status_key, 0) + 1
        
        return StagedFilesResponse(
            files=files,
            total_count=len(files),
            by_status=by_status
        )
        
    except Exception as e:
        logger.error(f"Failed to get staged files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-file/{file_id}")
async def process_file(file_id: str, validation_rules: Optional[ValidationRules] = None):
    """Process a staged file with validation"""
    try:
        rules = validation_rules.dict() if validation_rules else {}
        success = await data_manager.process_uploaded_file(file_id, rules)
        
        if success:
            return {
                "status": "success",
                "message": f"File {file_id} processed successfully"
            }
        else:
            return {
                "status": "error",
                "message": f"File {file_id} processing failed"
            }
            
    except Exception as e:
        logger.error(f"Failed to process file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download-file/{file_id}")
async def download_file(file_id: str):
    """Download a file by file_id"""
    try:
        if file_id not in data_manager.data_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_info = data_manager.data_files[file_id]
        
        # Get file from MinIO
        response = data_manager.s3_client.get_object(
            Bucket=data_manager.bucket_name,
            Key=file_info.minio_path
        )
        
        def generate():
            for chunk in response['Body'].iter_chunks(chunk_size=8192):
                yield chunk
        
        # Determine correct MIME type based on file extension
        file_extension = file_info.original_name.lower().split('.')[-1] if '.' in file_info.original_name else ''
        mime_type = "application/octet-stream"  # default
        
        if file_extension == 'jsonl':
            mime_type = "application/jsonl"
        elif file_extension == 'json':
            mime_type = "application/json"
        elif file_extension == 'txt':
            mime_type = "text/plain"
        elif file_extension == 'csv':
            mime_type = "text/csv"
        elif file_extension == 'xml':
            mime_type = "application/xml"
        
        return StreamingResponse(
            generate(),
            media_type=mime_type,
            headers={
                "Content-Disposition": f"attachment; filename=\"{file_info.original_name}\"",
                "Content-Type": mime_type,
                "Access-Control-Expose-Headers": "Content-Disposition, Content-Type, Content-Length",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to download file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/file-info/{file_id}")
async def get_file_info(file_id: str):
    """Get detailed information about a file"""
    try:
        if file_id not in data_manager.data_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_info = data_manager.data_files[file_id]
        
        # Construct full S3 URL
        s3_url = f"s3://{data_manager.bucket_name}/{file_info.minio_path}"
        
        return FileInfoResponse(
            file_id=file_info.file_id,
            original_name=file_info.original_name,
            minio_path=file_info.minio_path,
            s3_url=s3_url,
            data_type=file_info.data_type.value if hasattr(file_info.data_type, 'value') else str(file_info.data_type),
            status=file_info.status.value if hasattr(file_info.status, 'value') else str(file_info.status),
            upload_time=file_info.upload_time.isoformat() if hasattr(file_info.upload_time, 'isoformat') else str(file_info.upload_time),
            file_size=file_info.file_size,
            file_hash=file_info.file_hash,
            used_count=file_info.used_count,
            last_used=file_info.last_used.isoformat() if file_info.last_used and hasattr(file_info.last_used, 'isoformat') else (str(file_info.last_used) if file_info.last_used else None),
            training_jobs=file_info.training_jobs,
            metadata=file_info.metadata,
            processing_progress=file_info.processing_progress,
            processing_error=file_info.processing_error,
            chunk_count=file_info.chunk_count
        )
        
    except Exception as e:
        logger.error(f"Failed to get file info for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retry-failed-file/{file_id}")
async def retry_failed_file(file_id: str):
    """Retry processing a failed file"""
    try:
        if file_id not in data_manager.data_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_info = data_manager.data_files[file_id]
        
        if file_info.status != DataStatus.FAILED:
            raise HTTPException(status_code=400, detail="File is not in failed status")
        
        # Reset status and retry processing
        file_info.status = DataStatus.UPLOADED
        file_info.processing_error = None
        file_info.processing_progress = 0.0
        data_manager._save_data_tracking()
        
        # Retry processing
        success = await data_manager.process_uploaded_file(file_id)
        
        if success:
            return {
                "status": "success",
                "message": f"File {file_id} retry successful"
            }
        else:
            return {
                "status": "error",
                "message": f"File {file_id} retry failed"
            }
            
    except Exception as e:
        logger.error(f"Failed to retry file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cleanup-failed-uploads")
async def cleanup_failed_uploads(hours_old: int = 24):
    """Clean up failed uploads older than specified hours"""
    try:
        cleaned_count = await data_manager.cleanup_failed_uploads(hours_old)
        
        return {
            "status": "success",
            "message": f"Cleaned up {cleaned_count} failed uploads",
            "cleaned_count": cleaned_count
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup failed uploads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check for the efficient data management system"""
    try:
        active_uploads = len([f for f in data_manager.data_files.values() 
                             if (f.status.value if hasattr(f.status, 'value') else str(f.status)) == 'uploading'])
        failed_uploads = len([f for f in data_manager.data_files.values() 
                             if (f.status.value if hasattr(f.status, 'value') else str(f.status)) == 'failed'])
        fresh_files = len([f for f in data_manager.data_files.values() 
                          if (f.status.value if hasattr(f.status, 'value') else str(f.status)) == 'fresh'])
        
        return {
            "status": "healthy",
            "active_uploads": active_uploads,
            "failed_uploads": failed_uploads,
            "fresh_files": fresh_files,
            "total_files": len(data_manager.data_files),
            "max_workers": data_manager.max_workers,
            "chunk_size": data_manager.chunk_size
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.get("/metrics")
async def get_metrics():
    """Get comprehensive metrics for the data management system"""
    try:
        total_size = sum(f.file_size for f in data_manager.data_files.values())
        avg_file_size = total_size / len(data_manager.data_files) if data_manager.data_files else 0
        
        by_status = {}
        by_type = {}
        
        for file_info in data_manager.data_files.values():
            # Count by status
            status = file_info.status.value if hasattr(file_info.status, 'value') else str(file_info.status)
            by_status[status] = by_status.get(status, 0) + 1
            
            # Count by type
            data_type = file_info.data_type.value if hasattr(file_info.data_type, 'value') else str(file_info.data_type)
            by_type[data_type] = by_type.get(data_type, 0) + 1
        
        return {
            "total_files": len(data_manager.data_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "average_file_size_bytes": avg_file_size,
            "by_status": by_status,
            "by_type": by_type,
            "chunked_files": len([f for f in data_manager.data_files.values() if f.chunk_count > 0]),
            "system_config": {
                "max_workers": data_manager.max_workers,
                "chunk_size_mb": data_manager.chunk_size / (1024 * 1024),
                "max_file_size_gb": data_manager.max_file_size / (1024 * 1024 * 1024)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))