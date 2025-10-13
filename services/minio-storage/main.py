"""
MinIO Storage Service - Modular Main
Modularized MinIO storage service with clean architecture
"""

import os
import json
import boto3
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MinIOStorageService:
    """Centralized storage service using MinIO"""
    
    def __init__(self):
        # MinIO configuration
        self.endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.bucket_name = os.getenv("MINIO_BUCKET", "ml-security")
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Ensure the bucket exists"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket {self.bucket_name} exists")
        except:
            try:
                self.s3_client.create_bucket(Bucket=self.bucket_name)
                logger.info(f"Created bucket {self.bucket_name}")
            except Exception as e:
                logger.error(f"Failed to create bucket: {e}")
                raise
    
    def upload_file(self, file_path: str, object_key: str) -> Dict[str, Any]:
        """Upload file to MinIO"""
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, object_key)
            return {
                "status": "success",
                "message": f"File uploaded successfully",
                "object_key": object_key,
                "bucket": self.bucket_name
            }
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def download_file(self, object_key: str, local_path: str) -> Dict[str, Any]:
        """Download file from MinIO"""
        try:
            self.s3_client.download_file(self.bucket_name, object_key, local_path)
            return {
                "status": "success",
                "message": f"File downloaded successfully",
                "object_key": object_key,
                "local_path": local_path
            }
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def list_files(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List files in bucket"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            files = []
            for obj in response.get('Contents', []):
                files.append({
                    "key": obj['Key'],
                    "size": obj['Size'],
                    "last_modified": obj['LastModified'].isoformat(),
                    "etag": obj['ETag']
                })
            
            return files
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def delete_file(self, object_key: str) -> Dict[str, Any]:
        """Delete file from MinIO"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_key)
            return {
                "status": "success",
                "message": f"File deleted successfully",
                "object_key": object_key
            }
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_file_info(self, object_key: str) -> Dict[str, Any]:
        """Get file information"""
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=object_key)
            return {
                "key": object_key,
                "size": response['ContentLength'],
                "last_modified": response['LastModified'].isoformat(),
                "content_type": response.get('ContentType', 'application/octet-stream'),
                "etag": response['ETag']
            }
        except Exception as e:
            logger.error(f"Failed to get file info: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# Initialize storage service
storage_service = MinIOStorageService()

# Create FastAPI application
app = FastAPI(
    title="ML Security MinIO Storage Service (Modular)",
    version="1.0.0",
    description="Centralized storage service using MinIO"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "minio-storage",
        "version": "1.0.0",
        "status": "running",
        "description": "Centralized storage service using MinIO"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test MinIO connection
        storage_service.s3_client.head_bucket(Bucket=storage_service.bucket_name)
        
        return {
            "status": "healthy",
            "service": "minio-storage",
            "timestamp": datetime.now(),
            "bucket": storage_service.bucket_name,
            "endpoint": storage_service.endpoint
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(file_path: str, object_key: str):
    """Upload file to MinIO"""
    return storage_service.upload_file(file_path, object_key)


@app.post("/download")
async def download_file(object_key: str, local_path: str):
    """Download file from MinIO"""
    return storage_service.download_file(object_key, local_path)


@app.get("/files")
async def list_files(prefix: str = ""):
    """List files in bucket"""
    return {"files": storage_service.list_files(prefix)}


@app.delete("/files/{object_key}")
async def delete_file(object_key: str):
    """Delete file from MinIO"""
    return storage_service.delete_file(object_key)


@app.get("/files/{object_key}/info")
async def get_file_info(object_key: str):
    """Get file information"""
    return storage_service.get_file_info(object_key)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8009)
