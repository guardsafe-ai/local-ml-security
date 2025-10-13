"""
Efficient Data Management System for Large Files (GB-scale)
Handles user uploads, staging, validation, and processing efficiently
"""

import os
import json
import hashlib
import asyncio
import aiofiles
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import boto3
from botocore.exceptions import ClientError
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import tempfile
import shutil

logger = logging.getLogger(__name__)

class DataStatus(Enum):
    """Data file status in the lifecycle"""
    UPLOADING = "uploading"    # Currently being uploaded
    UPLOADED = "uploaded"      # Just uploaded, not processed
    PROCESSING = "processing"  # Being validated/processed
    FRESH = "fresh"           # Ready for training
    USED = "used"             # Already used in training
    ARCHIVED = "archived"     # Archived after multiple uses
    DEPRECATED = "deprecated" # Marked for deletion
    FAILED = "failed"         # Processing failed

class DataType(Enum):
    """Type of training data"""
    SAMPLE = "sample"
    RED_TEAM = "red_team"
    COMBINED = "combined"
    CUSTOM = "custom"
    MODEL_SPECIFIC = "model_specific"

@dataclass
class DataFileInfo:
    """Information about a data file in MinIO"""
    file_id: str
    original_name: str
    minio_path: str
    data_type: str
    status: str
    upload_time: datetime
    file_size: int
    file_hash: str
    used_count: int = 0
    last_used: Optional[datetime] = None
    training_jobs: List[str] = None
    metadata: Dict[str, Any] = None
    processing_progress: float = 0.0
    processing_error: Optional[str] = None
    chunk_count: int = 0  # For large files split into chunks
    
    def __post_init__(self):
        if self.training_jobs is None:
            self.training_jobs = []
        if self.metadata is None:
            self.metadata = {}

class EfficientDataManager:
    """Efficient data management system for large files"""
    
    def __init__(self, max_workers: int = None, chunk_size: int = 64 * 1024 * 1024):  # 64MB chunks
        # MinIO configuration
        self.s3_client = boto3.client(
            's3',
            endpoint_url='http://minio:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin'
        )
        self.bucket_name = 'ml-security'
        
        # Processing configuration
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.chunk_size = chunk_size
        self.max_file_size = 10 * 1024 * 1024 * 1024  # 10GB max
        self.temp_dir = Path("/tmp/ml_security_uploads")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Data tracking
        self.data_tracking_file = Path("/app/data/data_tracking.json")
        self.data_files = self._load_data_tracking()
        
        # Folder structure
        self.folders = {
            "uploads": "training-data/uploads/",
            "fresh": "training-data/fresh/",
            "used": "training-data/used/",
            "archived": "training-data/archived/",
            "backups": "training-data/backups/",
            "chunks": "training-data/chunks/"  # For large file chunks
        }
        
        # Ensure folder structure exists
        self._ensure_folders_exist()
        
        # Thread pool for I/O operations
        self.io_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
    def _ensure_folders_exist(self):
        """Ensure all required folders exist in MinIO"""
        try:
            for folder_name, folder_path in self.folders.items():
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=f"{folder_path}.gitkeep",
                    Body=b""
                )
            logger.info("✅ Data management folders created in MinIO")
        except Exception as e:
            logger.error(f"❌ Error creating data folders: {e}")
    
    async def upload_large_file(
        self, 
        local_path: str, 
        data_type: str = "custom",
        metadata: Dict[str, Any] = None,
        progress_callback: callable = None
    ) -> str:
        """Upload large file efficiently with streaming and chunking"""
        try:
            file_size = os.path.getsize(local_path)
            if file_size > self.max_file_size:
                raise ValueError(f"File too large: {file_size} bytes (max: {self.max_file_size})")
            
            # Generate file info
            file_id = f"file_{int(datetime.now().timestamp())}"
            original_name = Path(local_path).name
            
            # Calculate hash efficiently for large files
            file_hash = await self._calculate_file_hash_async(local_path)
            
            # Check if file already exists
            existing_file = self._find_file_by_hash(file_hash)
            if existing_file:
                warning_msg = f"File with same hash already exists: {existing_file.file_id}"
                logger.warning(warning_msg)
                return {
                    "file_id": existing_file.file_id,
                    "warning": warning_msg,
                    "is_duplicate": True
                }
            
            # Determine if file needs chunking
            needs_chunking = file_size > self.chunk_size
            
            if needs_chunking:
                return await self._upload_chunked_file(
                    local_path, file_id, original_name, file_hash, 
                    file_size, data_type, metadata, progress_callback
                )
            else:
                return await self._upload_single_file(
                    local_path, file_id, original_name, file_hash,
                    file_size, data_type, metadata, progress_callback
                )
                
        except Exception as e:
            logger.error(f"❌ Error uploading large file: {e}")
            raise
    
    async def _upload_single_file(
        self, local_path: str, file_id: str, original_name: str, 
        file_hash: str, file_size: int, data_type: str,
        metadata: Dict[str, Any], progress_callback: callable
    ) -> str:
        """Upload single file (not chunked)"""
        minio_path = f"{self.folders['uploads']}{file_id}_{original_name}"
        
        # Create file info
        file_info = DataFileInfo(
            file_id=file_id,
            original_name=original_name,
            minio_path=minio_path,
            data_type=data_type,
            status="uploading",
            upload_time=datetime.now(),
            file_size=file_size,
            file_hash=file_hash,
            metadata=metadata or {}
        )
        
        # Upload with progress tracking
        await self._upload_with_progress(
            local_path, minio_path, file_info, progress_callback
        )
        
        # Update status and save tracking
        file_info.status = "uploaded"
        self.data_files[file_id] = file_info
        self._save_data_tracking()
        
        logger.info(f"✅ Uploaded single file: {file_id}")
        return {
            "file_id": file_id,
            "warning": None,
            "is_duplicate": False
        }
    
    async def _upload_chunked_file(
        self, local_path: str, file_id: str, original_name: str,
        file_hash: str, file_size: int, data_type: str,
        metadata: Dict[str, Any], progress_callback: callable
    ) -> str:
        """Upload large file in chunks"""
        chunk_count = (file_size + self.chunk_size - 1) // self.chunk_size
        
        # Create file info
        file_info = DataFileInfo(
            file_id=file_id,
            original_name=original_name,
            minio_path=f"{self.folders['chunks']}{file_id}/",
            data_type=data_type,
            status="uploading",
            upload_time=datetime.now(),
            file_size=file_size,
            file_hash=file_hash,
            chunk_count=chunk_count,
            metadata=metadata or {}
        )
        
        # Upload chunks in parallel
        await self._upload_chunks_parallel(
            local_path, file_id, chunk_count, file_info, progress_callback
        )
        
        # Update status and save tracking
        file_info.status = "uploaded"
        self.data_files[file_id] = file_info
        self._save_data_tracking()
        
        logger.info(f"✅ Uploaded chunked file: {file_id} ({chunk_count} chunks)")
        return {
            "file_id": file_id,
            "warning": None,
            "is_duplicate": False
        }
    
    async def _upload_chunks_parallel(
        self, local_path: str, file_id: str, chunk_count: int,
        file_info: DataFileInfo, progress_callback: callable
    ):
        """Upload file chunks in parallel"""
        tasks = []
        
        for chunk_idx in range(chunk_count):
            chunk_path = f"{self.folders['chunks']}{file_id}/chunk_{chunk_idx:06d}"
            task = self._upload_chunk(
                local_path, chunk_path, chunk_idx, chunk_count, 
                file_info, progress_callback
            )
            tasks.append(task)
        
        # Execute all chunk uploads in parallel
        await asyncio.gather(*tasks)
    
    async def _upload_chunk(
        self, local_path: str, chunk_path: str, chunk_idx: int,
        total_chunks: int, file_info: DataFileInfo, progress_callback: callable
    ):
        """Upload a single chunk"""
        start_byte = chunk_idx * self.chunk_size
        end_byte = min(start_byte + self.chunk_size, file_info.file_size)
        chunk_size = end_byte - start_byte
        
        # Read chunk from file
        with open(local_path, 'rb') as f:
            f.seek(start_byte)
            chunk_data = f.read(chunk_size)
        
        # Upload chunk to MinIO
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.io_executor,
            self.s3_client.put_object,
            self.bucket_name,
            chunk_path,
            chunk_data
        )
        
        # Update progress
        if progress_callback:
            progress = ((chunk_idx + 1) / total_chunks) * 100
            await progress_callback(file_info.file_id, progress)
    
    async def _upload_with_progress(
        self, local_path: str, minio_path: str, 
        file_info: DataFileInfo, progress_callback: callable
    ):
        """Upload file with progress tracking"""
        file_size = file_info.file_size
        uploaded = 0
        
        def progress_cb(bytes_transferred):
            nonlocal uploaded
            uploaded += bytes_transferred
            if progress_callback:
                progress = (uploaded / file_size) * 100
                asyncio.create_task(progress_callback(file_info.file_id, progress))
        
        # Upload with progress tracking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.io_executor,
            self._upload_file_simple,
            local_path,
            minio_path
        )
    
    def _upload_file_simple(self, local_path: str, minio_path: str):
        """Upload file without progress callback"""
        with open(local_path, 'rb') as f:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=minio_path,
                Body=f.read()
            )
    
    def _upload_file_with_callback(self, local_path: str, minio_path: str, progress_cb: callable):
        """Upload file with progress callback"""
        with open(local_path, 'rb') as f:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=minio_path,
                Body=f.read(),
                Callback=progress_cb
            )
    
    async def _calculate_file_hash_async(self, file_path: str) -> str:
        """Calculate file hash efficiently for large files"""
        hash_md5 = hashlib.md5()
        
        def calculate_hash():
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.io_executor, calculate_hash)
    
    async def process_uploaded_file(
        self, 
        file_id: str, 
        validation_rules: Dict[str, Any] = None
    ) -> bool:
        """Process uploaded file (validate, clean, move to fresh)"""
        try:
            if file_id not in self.data_files:
                raise ValueError(f"File {file_id} not found")
            
            file_info = self.data_files[file_id]
            file_info.status = "processing"
            self._save_data_tracking()
            
            # Validate file
            if not await self._validate_file(file_info, validation_rules):
                file_info.status = "failed"
                file_info.processing_error = "Validation failed"
                self._save_data_tracking()
                return False
            
            # Process file (clean, transform, etc.)
            processed_path = await self._process_file(file_info)
            
            # Move to fresh folder
            fresh_path = f"{self.folders['fresh']}{file_info.file_id}_{file_info.original_name}"
            
            # Check if file is already in fresh folder
            if file_info.minio_path.startswith(self.folders['fresh']):
                logger.info(f"File {file_id} is already in fresh folder, skipping copy")
                # Just update the status and clear any previous errors
                file_info.status = "fresh"
                file_info.processing_progress = 100.0
                file_info.processing_error = None
            else:
                if file_info.chunk_count > 0:
                    # Reassemble chunks
                    await self._reassemble_chunks(file_info, fresh_path)
                else:
                    # Copy single file
                    await self._copy_file(file_info.minio_path, fresh_path)
                
                # Update file info
                file_info.minio_path = fresh_path
                file_info.status = "fresh"
                file_info.processing_progress = 100.0
                file_info.processing_error = None
            
            self._save_data_tracking()
            
            logger.info(f"✅ Processed file: {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing file {file_id}: {e}")
            if file_id in self.data_files:
                self.data_files[file_id].status = DataStatus.FAILED
                self.data_files[file_id].processing_error = str(e)
                self._save_data_tracking()
            return False
    
    async def _validate_file(self, file_info: DataFileInfo, rules: Dict[str, Any] = None) -> bool:
        """Validate uploaded file"""
        try:
            # Basic validation
            if file_info.file_size == 0:
                return False
            
            # Check file format
            if not file_info.original_name.endswith(('.jsonl', '.csv', '.txt')):
                logger.warning(f"Unsupported file format: {file_info.original_name}")
                return False
            
            # Sample validation for JSONL
            if file_info.original_name.endswith('.jsonl'):
                sample_data = await self._get_file_sample(file_info, 10)
                for line in sample_data:
                    # Skip empty lines
                    if not line.strip():
                        continue
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in file: {file_info.file_id}, line: '{line[:50]}...'")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    async def _get_file_sample(self, file_info: DataFileInfo, sample_size: int) -> List[str]:
        """Get sample lines from file for validation"""
        sample_lines = []
        
        if file_info.chunk_count > 0:
            # Read from first chunk
            chunk_path = f"{self.folders['chunks']}{file_info.file_id}/chunk_000000"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=chunk_path)
            content = response['Body'].read().decode('utf-8')
            sample_lines = content.split('\n')[:sample_size]
        else:
            # Read from single file
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_info.minio_path)
            content = response['Body'].read().decode('utf-8')
            sample_lines = content.split('\n')[:sample_size]
        
        return sample_lines
    
    async def _process_file(self, file_info: DataFileInfo) -> str:
        """Process file (clean, transform, etc.)"""
        # For now, just return the original path
        # In a real implementation, you might:
        # - Clean the data
        # - Transform formats
        # - Remove duplicates
        # - Normalize text
        return file_info.minio_path
    
    async def _reassemble_chunks(self, file_info: DataFileInfo, output_path: str):
        """Reassemble chunks into single file"""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Download and reassemble chunks
            with open(temp_path, 'wb') as output_file:
                for chunk_idx in range(file_info.chunk_count):
                    chunk_path = f"{self.folders['chunks']}{file_info.file_id}/chunk_{chunk_idx:06d}"
                    response = self.s3_client.get_object(Bucket=self.bucket_name, Key=chunk_path)
                    output_file.write(response['Body'].read())
            
            # Upload reassembled file
            with open(temp_path, 'rb') as f:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=output_path,
                    Body=f.read()
                )
            
            # Clean up chunks
            await self._cleanup_chunks(file_info.file_id)
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    async def _copy_file(self, source_path: str, dest_path: str):
        """Copy file from source to destination"""
        self.s3_client.copy_object(
            Bucket=self.bucket_name,
            CopySource={'Bucket': self.bucket_name, 'Key': source_path},
            Key=dest_path
        )
    
    async def _cleanup_chunks(self, file_id: str):
        """Clean up chunk files after reassembly"""
        try:
            # List all chunks for this file
            prefix = f"{self.folders['chunks']}{file_id}/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            # Delete all chunks
            for obj in response.get('Contents', []):
                self.s3_client.delete_object(
                    Bucket=self.bucket_name,
                    Key=obj['Key']
                )
            
            logger.info(f"Cleaned up chunks for file: {file_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up chunks: {e}")
    
    def _find_file_by_hash(self, file_hash: str) -> Optional[DataFileInfo]:
        """Find file by hash"""
        for file_info in self.data_files.values():
            if file_info.file_hash == file_hash:
                return file_info
        return None
    
    def _load_data_tracking(self) -> Dict[str, DataFileInfo]:
        """Load data file tracking information"""
        if self.data_tracking_file.exists():
            try:
                with open(self.data_tracking_file, 'r') as f:
                    data = json.load(f)
                    return {k: DataFileInfo(**v) for k, v in data.items()}
            except Exception as e:
                logger.error(f"Error loading data tracking: {e}")
        return {}
    
    def _save_data_tracking(self):
        """Save data file tracking information"""
        try:
            with open(self.data_tracking_file, 'w') as f:
                data = {k: asdict(v) for k, v in self.data_files.items()}
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving data tracking: {e}")
    
    async def get_upload_progress(self, file_id: str) -> Dict[str, Any]:
        """Get upload progress for a file"""
        if file_id not in self.data_files:
            return {"error": "File not found"}
        
        file_info = self.data_files[file_id]
        status = file_info.status.value if hasattr(file_info.status, 'value') else str(file_info.status)
        
        return {
            "file_id": file_id,
            "status": status,
            "progress": file_info.processing_progress,
            "file_size": file_info.file_size,
            "chunk_count": file_info.chunk_count,
            "error": file_info.processing_error
        }
    
    def mark_data_as_used(self, s3_key: str) -> bool:
        """Mark data as used by S3 key (compatibility method for model_trainer)"""
        try:
            # Find file by S3 key in tracking
            for file_id, file_info in self.data_files.items():
                if file_info.minio_path == s3_key:
                    # Update status to used
                    file_info.status = DataStatus.USED
                    file_info.used_count += 1
                    file_info.last_used = datetime.now()
                    self._save_data_tracking()
                    logger.info(f"✅ Marked data as used: {s3_key}")
                    return True
            
            # If not found in tracking, log and continue
            logger.info(f"File {s3_key} not in tracking, marking as used")
            return True
            
        except Exception as e:
            logger.error(f"Error marking data as used: {e}")
            return False
    
    def get_training_data_path(self, data_type: Optional[DataType] = None) -> str:
        """Get the best training data path (compatibility method for model_trainer)"""
        try:
            # Look for fresh data files
            fresh_files = []
            
            # List objects in fresh folder
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='training-data/fresh/'
            )
            
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.jsonl') and obj['Key'] != 'training-data/fresh/.gitkeep':
                    fresh_files.append(obj['Key'])
            
            if fresh_files:
                # Return the first fresh file
                return f"s3://{self.bucket_name}/{fresh_files[0]}"
            
            # Fallback to sample data
            return f"s3://{self.bucket_name}/training-data/sample_training_data_latest.jsonl"
            
        except Exception as e:
            logger.error(f"Error getting training data path: {e}")
            return f"s3://{self.bucket_name}/training-data/sample_training_data_latest.jsonl"
    
    async def cleanup_failed_uploads(self, hours_old: int = 24) -> int:
        """Clean up failed uploads older than specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours_old)
        cleaned_count = 0
        
        for file_id, file_info in list(self.data_files.items()):
            if (file_info.status == DataStatus.FAILED and 
                file_info.upload_time < cutoff_time):
                
                # Delete from MinIO
                try:
                    if file_info.chunk_count > 0:
                        await self._cleanup_chunks(file_id)
                    else:
                        self.s3_client.delete_object(
                            Bucket=self.bucket_name,
                            Key=file_info.minio_path
                        )
                    
                    # Remove from tracking
                    del self.data_files[file_id]
                    cleaned_count += 1
                    
                except Exception as e:
                    logger.error(f"Error cleaning up file {file_id}: {e}")
        
        if cleaned_count > 0:
            self._save_data_tracking()
        
        return cleaned_count
