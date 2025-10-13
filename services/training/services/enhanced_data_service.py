"""
Enhanced Data Service
Full integration with EfficientDataManager for advanced data operations
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from efficient_data_manager import EfficientDataManager, DataType, DataStatus
from services.training_logs_service import TrainingLogsService

logger = logging.getLogger(__name__)

class EnhancedDataService:
    """Enhanced data service with full EfficientDataManager integration"""
    
    def __init__(self):
        self.data_manager = EfficientDataManager()
        self.upload_progress_callbacks: Dict[str, Callable] = {}
    
    async def upload_file_with_progress(
        self, 
        local_path: str, 
        data_type: str = "custom",
        description: str = "",
        metadata: Dict[str, Any] = None,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """Upload file with real-time progress tracking"""
        try:
            # Add description to metadata
            if metadata is None:
                metadata = {}
            if description:
                metadata["description"] = description
            metadata["upload_timestamp"] = datetime.now().isoformat()
            
            # Store progress callback
            if progress_callback:
                self.upload_progress_callbacks[local_path] = progress_callback
            
            # Upload using efficient data manager
            file_id = await self.data_manager.upload_large_file(
                local_path=local_path,
                data_type=data_type,
                metadata=metadata,
                progress_callback=self._progress_callback_wrapper
            )
            
            # Start background processing
            asyncio.create_task(self._process_uploaded_file_background(file_id))
            
            logger.info(f"✅ File upload started: {local_path} -> {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"❌ File upload failed: {e}")
            raise
    
    async def _progress_callback_wrapper(self, file_id: str, progress: float):
        """Wrapper for progress callbacks"""
        try:
            # Log progress to training logs if available
            await TrainingLogsService.log_training_event(
                file_id, "INFO", "data_upload", 
                f"Upload progress: {progress:.1f}%"
            )
            
            # Call custom progress callback if available
            for local_path, callback in self.upload_progress_callbacks.items():
                try:
                    await callback(file_id, progress)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")
                    
        except Exception as e:
            logger.error(f"Progress callback wrapper failed: {e}")
    
    async def _process_uploaded_file_background(self, file_id: str):
        """Background processing for uploaded files"""
        try:
            await TrainingLogsService.log_training_event(
                file_id, "INFO", "data_processing", 
                "Starting background file processing"
            )
            
            # Process the uploaded file
            success = await self.data_manager.process_uploaded_file(file_id)
            
            if success:
                await TrainingLogsService.log_training_event(
                    file_id, "INFO", "data_processing", 
                    "File processing completed successfully"
                )
            else:
                await TrainingLogsService.log_training_event(
                    file_id, "ERROR", "data_processing", 
                    "File processing failed"
                )
                
        except Exception as e:
            await TrainingLogsService.log_training_event(
                file_id, "ERROR", "data_processing", 
                f"Background processing error: {e}"
            )
            logger.error(f"Background processing failed for {file_id}: {e}")
    
    async def get_upload_progress(self, file_id: str) -> Dict[str, Any]:
        """Get upload progress for a file"""
        try:
            return await self.data_manager.get_upload_progress(file_id)
        except Exception as e:
            logger.error(f"Failed to get upload progress for {file_id}: {e}")
            return {"error": str(e)}
    
    async def get_fresh_data_files(self, data_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all fresh data files ready for training"""
        try:
            fresh_files = []
            
            for file_id, file_info in self.data_manager.data_files.items():
                if file_info.status == DataStatus.FRESH:
                    if data_type is None or file_info.data_type.value == data_type:
                        fresh_files.append({
                            "file_id": file_info.file_id,
                            "original_name": file_info.original_name,
                            "minio_path": file_info.minio_path,
                            "data_type": file_info.data_type.value,
                            "upload_time": file_info.upload_time.isoformat(),
                            "file_size": file_info.file_size,
                            "file_hash": file_info.file_hash,
                            "metadata": file_info.metadata
                        })
            
            return sorted(fresh_files, key=lambda x: x["upload_time"], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to get fresh data files: {e}")
            return []
    
    async def get_data_statistics(self) -> Dict[str, Any]:
        """Get comprehensive data statistics"""
        try:
            stats = {
                "total_files": len(self.data_manager.data_files),
                "by_status": {},
                "by_type": {},
                "total_size_bytes": 0,
                "total_size_mb": 0,
                "chunked_files": 0,
                "failed_uploads": 0
            }
            
            for file_info in self.data_manager.data_files.values():
                # Count by status
                status = file_info.status.value
                stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
                
                # Count by type
                data_type = file_info.data_type.value
                stats["by_type"][data_type] = stats["by_type"].get(data_type, 0) + 1
                
                # Total size
                stats["total_size_bytes"] += file_info.file_size
                
                # Count chunked files
                if file_info.chunk_count > 0:
                    stats["chunked_files"] += 1
                
                # Count failed uploads
                if file_info.status == DataStatus.FAILED:
                    stats["failed_uploads"] += 1
            
            # Convert to MB
            stats["total_size_mb"] = stats["total_size_bytes"] / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get data statistics: {e}")
            return {}
    
    async def cleanup_failed_uploads(self, hours_old: int = 24) -> int:
        """Clean up failed uploads"""
        try:
            cleaned_count = await self.data_manager.cleanup_failed_uploads(hours_old)
            
            await TrainingLogsService.log_training_event(
                "cleanup", "INFO", "data_cleanup", 
                f"Cleaned up {cleaned_count} failed uploads older than {hours_old} hours"
            )
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup failed uploads: {e}")
            return 0
    
    async def retry_failed_file(self, file_id: str) -> bool:
        """Retry processing a failed file"""
        try:
            if file_id not in self.data_manager.data_files:
                logger.error(f"File {file_id} not found")
                return False
            
            file_info = self.data_manager.data_files[file_id]
            
            if file_info.status != DataStatus.FAILED:
                logger.warning(f"File {file_id} is not in failed status")
                return False
            
            await TrainingLogsService.log_training_event(
                file_id, "INFO", "data_retry", 
                "Retrying failed file processing"
            )
            
            # Reset status
            file_info.status = DataStatus.UPLOADED
            file_info.processing_error = None
            file_info.processing_progress = 0.0
            self.data_manager._save_data_tracking()
            
            # Retry processing
            success = await self.data_manager.process_uploaded_file(file_id)
            
            if success:
                await TrainingLogsService.log_training_event(
                    file_id, "INFO", "data_retry", 
                    "File retry successful"
                )
            else:
                await TrainingLogsService.log_training_event(
                    file_id, "ERROR", "data_retry", 
                    "File retry failed"
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to retry file {file_id}: {e}")
            return False
    
    async def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed file information"""
        try:
            if file_id not in self.data_manager.data_files:
                return None
            
            file_info = self.data_manager.data_files[file_id]
            
            return {
                "file_id": file_info.file_id,
                "original_name": file_info.original_name,
                "minio_path": file_info.minio_path,
                "data_type": file_info.data_type.value,
                "status": file_info.status.value,
                "upload_time": file_info.upload_time.isoformat(),
                "file_size": file_info.file_size,
                "file_hash": file_info.file_hash,
                "used_count": file_info.used_count,
                "last_used": file_info.last_used.isoformat() if file_info.last_used else None,
                "training_jobs": file_info.training_jobs,
                "metadata": file_info.metadata,
                "processing_progress": file_info.processing_progress,
                "processing_error": file_info.processing_error,
                "chunk_count": file_info.chunk_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get file info for {file_id}: {e}")
            return None
    
    async def download_file(self, file_id: str) -> Optional[bytes]:
        """Download file content"""
        try:
            if file_id not in self.data_manager.data_files:
                return None
            
            file_info = self.data_manager.data_files[file_id]
            
            # Get file from MinIO
            response = self.data_manager.s3_client.get_object(
                Bucket=self.data_manager.bucket_name,
                Key=file_info.minio_path
            )
            
            return response['Body'].read()
            
        except Exception as e:
            logger.error(f"Failed to download file {file_id}: {e}")
            return None

# Global instance
enhanced_data_service = EnhancedDataService()
