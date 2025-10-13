/**
 * Enhanced Data Service
 * Full integration with EfficientDataManager for advanced data operations
 */

import api from './apiService';

export interface UploadProgress {
  fileId: string;
  status: string;
  progress: number;
  fileSize: number;
  chunkCount: number;
  error?: string;
}

export interface FileInfo {
  fileId: string;
  originalName: string;
  minioPath: string;
  dataType: string;
  status: string;
  uploadTime: string;
  fileSize: number;
  fileHash: string;
  usedCount: number;
  lastUsed?: string;
  trainingJobs: string[];
  metadata: Record<string, any>;
  processingProgress: number;
  processingError?: string;
  chunkCount: number;
}

export interface StagedFilesResponse {
  files: FileInfo[];
  totalCount: number;
  byStatus: Record<string, number>;
}

export interface DataStatistics {
  totalFiles: number;
  byStatus: Record<string, number>;
  byType: Record<string, number>;
  totalSizeBytes: number;
  totalSizeMb: number;
  chunkedFiles: number;
  failedUploads: number;
}

export interface ValidationRules {
  maxFileSize?: number;
  allowedFormats?: string[];
  minRecords?: number;
  customValidation?: Record<string, any>;
}

class EnhancedDataService {
  /**
   * Upload large file with chunked upload support
   */
  async uploadLargeFile(
    file: File,
    dataType: string = 'custom',
    description: string = '',
    metadata: Record<string, any> = {},
    onProgress?: (progress: UploadProgress) => void
  ): Promise<{ fileId: string; fileSize: number }> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('data_type', dataType);
    formData.append('description', description);
    formData.append('metadata', JSON.stringify(metadata));

    const response = await api.post('/data/efficient/upload-large-file', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300000, // 5 minutes
    });

    const { file_id: fileId, file_size: fileSize } = response.data;

    // Start polling for progress if callback provided
    if (onProgress) {
      this.pollUploadProgress(fileId, onProgress);
    }

    return { fileId, fileSize };
  }

  /**
   * Poll upload progress
   */
  private async pollUploadProgress(
    fileId: string,
    onProgress: (progress: UploadProgress) => void
  ): Promise<void> {
    const pollInterval = 1000; // 1 second
    const maxPollTime = 300000; // 5 minutes
    const startTime = Date.now();

    const poll = async () => {
      try {
        const progress = await this.getUploadProgress(fileId);
        onProgress(progress);

        // Continue polling if still uploading and within time limit
        if (
          progress.status === 'uploading' &&
          Date.now() - startTime < maxPollTime
        ) {
          setTimeout(poll, pollInterval);
        }
      } catch (error) {
        console.error('Error polling upload progress:', error);
      }
    };

    poll();
  }

  /**
   * Get upload progress for a specific file
   */
  async getUploadProgress(fileId: string): Promise<UploadProgress> {
    const response = await api.get(`/data/efficient/upload-progress/${fileId}`);
    return response.data;
  }

  /**
   * Get all staged files with optional status filter
   */
  async getStagedFiles(status?: string): Promise<StagedFilesResponse> {
    const params = status ? { status } : {};
    const response = await api.get('/data/efficient/staged-files', { params });
    return response.data;
  }

  /**
   * Process a staged file with validation
   */
  async processFile(fileId: string, validationRules?: ValidationRules): Promise<{ status: string; message: string }> {
    const response = await api.post(`/data/efficient/process-file/${fileId}`, validationRules);
    return response.data;
  }

  /**
   * Download a file by file ID
   */
  async downloadFile(fileId: string): Promise<Blob> {
    const response = await api.get(`/data/efficient/download-file/${fileId}`, {
      responseType: 'blob',
    });
    return response.data;
  }

  /**
   * Get detailed information about a file
   */
  async getFileInfo(fileId: string): Promise<FileInfo> {
    const response = await api.get(`/data/efficient/file-info/${fileId}`);
    return response.data;
  }

  /**
   * Retry processing a failed file
   */
  async retryFailedFile(fileId: string): Promise<{ status: string; message: string }> {
    const response = await api.post(`/data/efficient/retry-failed-file/${fileId}`);
    return response.data;
  }

  /**
   * Clean up failed uploads
   */
  async cleanupFailedUploads(hoursOld: number = 24): Promise<{ status: string; message: string; cleanedCount: number }> {
    const response = await api.delete('/data/efficient/cleanup-failed-uploads', {
      params: { hours_old: hoursOld },
    });
    return response.data;
  }

  /**
   * Get health status of the data management system
   */
  async getHealthStatus(): Promise<{
    status: string;
    activeUploads: number;
    failedUploads: number;
    freshFiles: number;
    totalFiles: number;
    maxWorkers: number;
    chunkSize: number;
  }> {
    const response = await api.get('/data/efficient/health');
    return response.data;
  }

  /**
   * Get comprehensive metrics
   */
  async getMetrics(): Promise<{
    totalFiles: number;
    totalSizeBytes: number;
    totalSizeMb: number;
    averageFileSizeBytes: number;
    byStatus: Record<string, number>;
    byType: Record<string, number>;
    chunkedFiles: number;
    systemConfig: {
      maxWorkers: number;
      chunkSizeMb: number;
      maxFileSizeGb: number;
    };
  }> {
    const response = await api.get('/data/efficient/metrics');
    return response.data;
  }

  /**
   * Get fresh data files ready for training
   */
  async getFreshDataFiles(dataType?: string): Promise<FileInfo[]> {
    const params = dataType ? { data_type: dataType } : {};
    const response = await api.get('/data/efficient/staged-files', { params });
    return response.data.files.filter((file: FileInfo) => file.status === 'fresh');
  }

  /**
   * Get data statistics
   */
  async getDataStatistics(): Promise<DataStatistics> {
    const response = await api.get('/data/efficient/metrics');
    return {
      totalFiles: response.data.totalFiles,
      byStatus: response.data.byStatus,
      byType: response.data.byType,
      totalSizeBytes: response.data.totalSizeBytes,
      totalSizeMb: response.data.totalSizeMb,
      chunkedFiles: response.data.chunkedFiles,
      failedUploads: response.data.byStatus.failed || 0,
    };
  }

  /**
   * Format file size for display
   */
  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  /**
   * Format upload progress for display
   */
  formatProgress(progress: UploadProgress): string {
    return `${progress.progress.toFixed(1)}% (${this.formatFileSize(progress.fileSize)})`;
  }

  /**
   * Get status color for UI
   */
  getStatusColor(status: string): string {
    switch (status) {
      case 'fresh':
        return 'success';
      case 'uploading':
        return 'primary';
      case 'processing':
        return 'warning';
      case 'failed':
        return 'error';
      case 'used':
        return 'info';
      default:
        return 'default';
    }
  }

  /**
   * Get status icon for UI
   */
  getStatusIcon(status: string): string {
    switch (status) {
      case 'fresh':
        return '✅';
      case 'uploading':
        return '📤';
      case 'processing':
        return '⚙️';
      case 'failed':
        return '❌';
      case 'used':
        return '📋';
      default:
        return '📄';
    }
  }
}

export const enhancedDataService = new EnhancedDataService();
export default enhancedDataService;
