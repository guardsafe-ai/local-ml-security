# Efficient Data Management Implementation Guide

## Overview

This implementation provides an efficient system for handling large data files (up to GB) with proper staging, validation, and processing. It addresses the limitations of the current system and provides enterprise-grade data management.

## Key Features

### 1. **Efficient Upload System**
- **Streaming Uploads**: Files are streamed directly to MinIO without loading into memory
- **Chunked Uploads**: Large files (>64MB) are split into chunks and uploaded in parallel
- **Progress Tracking**: Real-time upload progress with callbacks
- **Resume Capability**: Failed uploads can be resumed from the last chunk

### 2. **Staging System**
- **Uploads Folder**: Raw user uploads go to `training-data/uploads/`
- **Processing Pipeline**: Files are validated and processed before moving to `fresh/`
- **Status Tracking**: Complete lifecycle tracking with status updates
- **Error Handling**: Failed uploads are tracked and can be retried

### 3. **Memory Efficiency**
- **Streaming Processing**: Files are processed in chunks, not loaded entirely into memory
- **Parallel Processing**: Multiple chunks are processed simultaneously
- **Temporary Files**: Large files use temporary storage during processing
- **Cleanup**: Automatic cleanup of temporary files and failed uploads

### 4. **Validation & Processing**
- **Format Validation**: JSONL, CSV, and text file validation
- **Data Quality Checks**: Sample validation for data integrity
- **Custom Rules**: Configurable validation rules
- **Background Processing**: Non-blocking file processing

## Implementation Steps

### Step 1: Install Dependencies

```bash
# Add to requirements.txt
aiofiles>=0.8.0
boto3>=1.26.0
botocore>=1.29.0
```

### Step 2: Update Main Training Service

```python
# In main.py, add the new routes
from routes.efficient_data import router as efficient_data_router

app.include_router(efficient_data_router, prefix="/data/efficient", tags=["efficient-data"])
```

### Step 3: Update Frontend API Service

```typescript
// Add to apiService.ts
export const efficientDataService = {
  uploadLargeFile: (file: File, dataType: string, description?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('data_type', dataType);
    formData.append('description', description || '');
    
    return api.post('/data/efficient/upload-large-file', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300000 // 5 minutes
    });
  },
  
  getUploadProgress: (fileId: string) => 
    api.get(`/data/efficient/upload-progress/${fileId}`),
  
  getStagedFiles: (status?: string) => 
    api.get('/data/efficient/staged-files', { params: { status } }),
  
  processFile: (fileId: string, validationRules?: any) =>
    api.post(`/data/efficient/process-file/${fileId}`, validationRules),
  
  downloadFile: (fileId: string) =>
    api.get(`/data/efficient/download-file/${fileId}`, { responseType: 'blob' }),
  
  getFileInfo: (fileId: string) =>
    api.get(`/data/efficient/file-info/${fileId}`),
  
  retryFailedFile: (fileId: string) =>
    api.post(`/data/efficient/retry-failed-file/${fileId}`),
  
  cleanupFailedUploads: (hoursOld: number = 24) =>
    api.delete('/data/efficient/cleanup-failed-uploads', { params: { hours_old: hoursOld } })
};
```

### Step 4: Create Frontend Components

```typescript
// LargeFileUpload.tsx
import React, { useState, useCallback } from 'react';
import { efficientDataService } from '../services/apiService';

interface UploadProgress {
  fileId: string;
  status: string;
  progress: number;
  fileSize: number;
  chunkCount: number;
  error?: string;
}

export const LargeFileUpload: React.FC = () => {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [files, setFiles] = useState<any[]>([]);

  const handleFileUpload = useCallback(async (file: File) => {
    setUploading(true);
    setProgress(null);
    
    try {
      const response = await efficientDataService.uploadLargeFile(
        file, 
        'custom', 
        `Uploaded ${file.name}`
      );
      
      const fileId = response.data.file_id;
      setProgress({
        fileId,
        status: 'uploading',
        progress: 0,
        fileSize: file.size,
        chunkCount: 0
      });
      
      // Poll for progress
      const progressInterval = setInterval(async () => {
        try {
          const progressResponse = await efficientDataService.getUploadProgress(fileId);
          const progressData = progressResponse.data;
          setProgress(progressData);
          
          if (progressData.status === 'fresh' || progressData.status === 'failed') {
            clearInterval(progressInterval);
            setUploading(false);
            if (progressData.status === 'fresh') {
              // Refresh file list
              loadStagedFiles();
            }
          }
        } catch (error) {
          console.error('Error getting progress:', error);
        }
      }, 1000);
      
    } catch (error) {
      console.error('Upload failed:', error);
      setUploading(false);
    }
  };

  const loadStagedFiles = useCallback(async () => {
    try {
      const response = await efficientDataService.getStagedFiles();
      setFiles(response.data.files);
    } catch (error) {
      console.error('Error loading files:', error);
    }
  };

  return (
    <div>
      <input
        type="file"
        onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
        disabled={uploading}
      />
      
      {progress && (
        <div>
          <h3>Upload Progress</h3>
          <p>Status: {progress.status}</p>
          <p>Progress: {progress.progress.toFixed(1)}%</p>
          <p>File Size: {(progress.fileSize / 1024 / 1024).toFixed(2)} MB</p>
          {progress.chunkCount > 0 && <p>Chunks: {progress.chunkCount}</p>}
          {progress.error && <p>Error: {progress.error}</p>}
        </div>
      )}
      
      <div>
        <h3>Staged Files</h3>
        {files.map(file => (
          <div key={file.file_id}>
            <p>{file.original_name} - {file.status} - {(file.file_size / 1024 / 1024).toFixed(2)} MB</p>
            {file.status === 'failed' && (
              <button onClick={() => retryFile(file.file_id)}>Retry</button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
```

## Usage Examples

### 1. Upload Large File

```bash
# Upload a large JSONL file
curl -X POST "http://localhost:8002/data/efficient/upload-large-file" \
  -F "file=@large_dataset.jsonl" \
  -F "data_type=custom" \
  -F "description=Large training dataset"
```

### 2. Check Upload Progress

```bash
# Check progress for a specific file
curl "http://localhost:8002/data/efficient/upload-progress/file_1234567890"
```

### 3. Get Staged Files

```bash
# Get all staged files
curl "http://localhost:8002/data/efficient/staged-files"

# Get only failed files
curl "http://localhost:8002/data/efficient/staged-files?status=failed"
```

### 4. Process File Manually

```bash
# Process a file with custom validation rules
curl -X POST "http://localhost:8002/data/efficient/process-file/file_1234567890" \
  -H "Content-Type: application/json" \
  -d '{
    "max_file_size": 1000000000,
    "allowed_formats": ["jsonl", "csv"],
    "min_records": 100
  }'
```

## Performance Optimizations

### 1. **Chunk Size Tuning**
```python
# Adjust chunk size based on file size
if file_size < 100 * 1024 * 1024:  # < 100MB
    chunk_size = 16 * 1024 * 1024   # 16MB chunks
elif file_size < 1024 * 1024 * 1024:  # < 1GB
    chunk_size = 64 * 1024 * 1024   # 64MB chunks
else:  # >= 1GB
    chunk_size = 128 * 1024 * 1024  # 128MB chunks
```

### 2. **Parallel Processing**
```python
# Adjust worker count based on system resources
max_workers = min(32, (os.cpu_count() or 1) * 2)
```

### 3. **Memory Management**
```python
# Use streaming for large files
def process_large_file(file_path: str):
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            # Process chunk
            yield process_chunk(chunk)
```

## Monitoring and Maintenance

### 1. **Health Checks**
```python
# Add health check endpoint
@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_uploads": len([f for f in data_manager.data_files.values() 
                              if f.status == DataStatus.UPLOADING]),
        "failed_uploads": len([f for f in data_manager.data_files.values() 
                              if f.status == DataStatus.FAILED]),
        "total_files": len(data_manager.data_files)
    }
```

### 2. **Cleanup Jobs**
```python
# Schedule cleanup job
import asyncio
from datetime import timedelta

async def cleanup_job():
    while True:
        await asyncio.sleep(3600)  # Run every hour
        cleaned = await data_manager.cleanup_failed_uploads(hours_old=24)
        logger.info(f"Cleaned up {cleaned} failed uploads")
```

### 3. **Metrics Collection**
```python
# Add metrics collection
def get_metrics():
    return {
        "total_uploads": len(data_manager.data_files),
        "successful_uploads": len([f for f in data_manager.data_files.values() 
                                 if f.status == DataStatus.FRESH]),
        "failed_uploads": len([f for f in data_manager.data_files.values() 
                             if f.status == DataStatus.FAILED]),
        "total_size": sum(f.file_size for f in data_manager.data_files.values()),
        "average_file_size": sum(f.file_size for f in data_manager.data_files.values()) / 
                           len(data_manager.data_files) if data_manager.data_files else 0
    }
```

## Security Considerations

### 1. **File Validation**
- Validate file types and sizes
- Scan for malicious content
- Implement rate limiting for uploads

### 2. **Access Control**
- Implement authentication for upload endpoints
- Add authorization checks
- Log all upload activities

### 3. **Data Protection**
- Encrypt sensitive data
- Implement data retention policies
- Add audit trails

## Error Handling

### 1. **Upload Failures**
- Automatic retry with exponential backoff
- Partial upload recovery
- Clear error messages

### 2. **Processing Failures**
- Detailed error logging
- Manual retry capability
- Fallback processing options

### 3. **System Failures**
- Graceful degradation
- Data consistency checks
- Recovery procedures

This implementation provides a robust, scalable solution for handling large data files efficiently while maintaining data integrity and providing excellent user experience.
