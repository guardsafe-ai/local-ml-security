import { useState, useCallback, useRef, useEffect } from 'react';
import { apiService } from '../services/apiService';
import { LoadingState, ErrorResponse } from '../types';

// Request throttling to prevent resource exhaustion
class RequestThrottler {
  private static instance: RequestThrottler;
  private requestQueue: Array<() => Promise<any>> = [];
  private isProcessing = false;
  private maxConcurrent = 3; // Limit concurrent requests
  private requestDelay = 100; // 100ms delay between requests

  static getInstance(): RequestThrottler {
    if (!RequestThrottler.instance) {
      RequestThrottler.instance = new RequestThrottler();
    }
    return RequestThrottler.instance;
  }

  async throttleRequest<T>(request: () => Promise<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      this.requestQueue.push(async () => {
        try {
          const result = await request();
          resolve(result);
        } catch (error) {
          reject(error);
        }
      });
      this.processQueue();
    });
  }

  private async processQueue() {
    if (this.isProcessing || this.requestQueue.length === 0) return;
    
    this.isProcessing = true;
    
    while (this.requestQueue.length > 0) {
      const batch = this.requestQueue.splice(0, this.maxConcurrent);
      await Promise.all(batch.map(request => request()));
      
      if (this.requestQueue.length > 0) {
        await new Promise(resolve => setTimeout(resolve, this.requestDelay));
      }
    }
    
    this.isProcessing = false;
  }
}

const requestThrottler = RequestThrottler.getInstance();

// =============================================================================
// COMPREHENSIVE API HOOKS FOR ML SECURITY DASHBOARD
// =============================================================================

// =============================================================================
// BASE API HOOK
// =============================================================================

interface UseApiOptions {
  onSuccess?: (data: any) => void;
  onError?: (error: ErrorResponse) => void;
}

export function useApi<T = any>(
  apiCall: () => Promise<T>,
  options: UseApiOptions = {}
) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<LoadingState>('idle');
  const [error, setError] = useState<ErrorResponse | null>(null);


  const {
    onSuccess,
    onError
  } = options;

  // Use ref to store the latest API call to avoid dependency issues
  const apiCallRef = useRef(apiCall);
  apiCallRef.current = apiCall;

  const execute = useCallback(async () => {
    try {
      setLoading('loading');
      setError(null);
      
      // Use throttled request to prevent resource exhaustion
      const result = await requestThrottler.throttleRequest(apiCallRef.current);
      setData(result);
      setLoading('success');
      
      if (onSuccess) {
        onSuccess(result);
      }
    } catch (err: any) {
      const errorResponse: ErrorResponse = {
        error: err.response?.data?.error || err.message || 'Unknown error',
        details: err.response?.data?.details,
        status_code: err.response?.status || 500,
        timestamp: new Date().toISOString(),
        request_id: err.response?.data?.request_id
      };
      
      setError(errorResponse);
      setLoading('error');
      
      if (onError) {
        onError(errorResponse);
      }
    }
  }, [onSuccess, onError]);

  const refetch = useCallback(() => {
    execute();
  }, [execute]);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading('idle');
  }, []);

  // NO AUTOMATIC LOADING - Only load when explicitly called
  // Removed all useEffect automatic loading

  return {
    data,
    loading,
    error,
    execute,
    refetch,
    reset
  };
}

// =============================================================================
// SYSTEM HEALTH HOOKS
// =============================================================================

export function useSystemHealth() {
  return useApi(
    () => apiService.getServicesHealth(),
    {} // Manual loading only
  );
}

export function useDashboardMetrics() {
  return useApi(
    () => apiService.getDashboardMetrics(),
    {} // Manual loading only
  );
}

export function useSystemStatus() {
  return useApi(
    () => apiService.getSystemStatus(),
    {} // Manual loading only
  );
}

// =============================================================================
// MODEL MANAGEMENT HOOKS
// =============================================================================

export function useModels() {
  return useApi(
    () => apiService.getAvailableModels(),
    {} // Manual loading only
  );
}

export function useModelRegistry() {
  return useApi(
    () => apiService.getModelRegistry(),
    {} // Manual loading only
  );
}

export function useModelInfo(modelName: string) {
  return useApi(
    () => apiService.getModelInfo(modelName),
    {}
  );
}

export function useModelVersions(modelName: string) {
  return useApi(
    () => apiService.getModelVersions(modelName),
    {}
  );
}

export function useModelPrediction() {
  const [loading, setLoading] = useState<LoadingState>('idle');
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<ErrorResponse | null>(null);

  const predict = useCallback(async (
    text: string,
    modelName?: string,
    ensemble?: boolean
  ) => {
    try {
      setLoading('loading');
      setError(null);
      
      const result = await apiService.predictModel(text, modelName, ensemble);
      setData(result);
      setLoading('success');
    } catch (err: any) {
      const errorResponse: ErrorResponse = {
        error: err.response?.data?.error || err.message || 'Prediction failed',
        details: err.response?.data?.details,
        status_code: err.response?.status || 500,
        timestamp: new Date().toISOString()
      };
      
      setError(errorResponse);
      setLoading('error');
    }
  }, []);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading('idle');
  }, []);

  return {
    data,
    loading,
    error,
    predict,
    reset
  };
}

// =============================================================================
// TRAINING HOOKS
// =============================================================================

export function useTrainingJobs() {
  return useApi(
    () => apiService.getTrainingJobs(),
    {} // Manual loading only
  );
}

export function useTrainingJob(jobId: string) {
  return useApi(
    () => apiService.getTrainingJob(jobId),
    {}
  );
}

export function useTrainingLogs() {
  return useApi(
    () => apiService.getTrainingLogs(),
  );
}

export function useTrainingModels() {
  return useApi(
    () => apiService.getTrainingModels(),
  );
}

export function useTrainingOperations() {
  const [loading, setLoading] = useState<LoadingState>('idle');
  const [error, setError] = useState<ErrorResponse | null>(null);

  const startTraining = useCallback(async (config: any) => {
    try {
      setLoading('loading');
      setError(null);
      
      const result = await apiService.startTraining(config);
      setLoading('success');
      return result;
    } catch (err: any) {
      const errorResponse: ErrorResponse = {
        error: err.response?.data?.error || err.message || 'Training start failed',
        details: err.response?.data?.details,
        status_code: err.response?.status || 500,
        timestamp: new Date().toISOString()
      };
      
      setError(errorResponse);
      setLoading('error');
      throw errorResponse;
    }
  }, []);

  const trainModel = useCallback(async (config: any) => {
    try {
      setLoading('loading');
      setError(null);
      
      const result = await apiService.trainModel(config);
      setLoading('success');
      return result;
    } catch (err: any) {
      const errorResponse: ErrorResponse = {
        error: err.response?.data?.error || err.message || 'Model training failed',
        details: err.response?.data?.details,
        status_code: err.response?.status || 500,
        timestamp: new Date().toISOString()
      };
      
      setError(errorResponse);
      setLoading('error');
      throw errorResponse;
    }
  }, []);

  const trainLoadedModel = useCallback(async (config: any) => {
    try {
      setLoading('loading');
      setError(null);
      
      const result = await apiService.trainLoadedModel(config);
      setLoading('success');
      return result;
    } catch (err: any) {
      const errorResponse: ErrorResponse = {
        error: err.response?.data?.error || err.message || 'Loaded model training failed',
        details: err.response?.data?.details,
        status_code: err.response?.status || 500,
        timestamp: new Date().toISOString()
      };
      
      setError(errorResponse);
      setLoading('error');
      throw errorResponse;
    }
  }, []);

  const cancelJob = useCallback(async (jobId: string) => {
    try {
      setLoading('loading');
      setError(null);
      
      const result = await apiService.cancelTrainingJob(jobId);
      setLoading('success');
      return result;
    } catch (err: any) {
      const errorResponse: ErrorResponse = {
        error: err.response?.data?.error || err.message || 'Job cancellation failed',
        details: err.response?.data?.details,
        status_code: err.response?.status || 500,
        timestamp: new Date().toISOString()
      };
      
      setError(errorResponse);
      setLoading('error');
      throw errorResponse;
    }
  }, []);

  return {
    loading,
    error,
    startTraining,
    trainModel,
    trainLoadedModel,
    cancelJob
  };
}

// =============================================================================
// RED TEAM HOOKS
// =============================================================================

export function useRedTeamStatus() {
  return useApi(
    () => apiService.getRedTeamStatus(),
  );
}

export function useRedTeamResults() {
  return useApi(
    () => apiService.getRedTeamResults(),
    {} // Manual loading only
  );
}

export function useRedTeamMetrics() {
  return useApi(
    () => apiService.getRedTeamMetrics(),
  );
}

export function useRedTeamOperations() {
  const [loading, setLoading] = useState<LoadingState>('idle');
  const [error, setError] = useState<ErrorResponse | null>(null);

  const startTesting = useCallback(async () => {
    try {
      setLoading('loading');
      setError(null);
      
      const result = await apiService.startRedTeamTesting();
      setLoading('success');
      return result;
    } catch (err: any) {
      const errorResponse: ErrorResponse = {
        error: err.response?.data?.error || err.message || 'Red team testing start failed',
        details: err.response?.data?.details,
        status_code: err.response?.status || 500,
        timestamp: new Date().toISOString()
      };
      
      setError(errorResponse);
      setLoading('error');
      throw errorResponse;
    }
  }, []);

  const stopTesting = useCallback(async () => {
    try {
      setLoading('loading');
      setError(null);
      
      const result = await apiService.stopRedTeamTesting();
      setLoading('success');
      return result;
    } catch (err: any) {
      const errorResponse: ErrorResponse = {
        error: err.response?.data?.error || err.message || 'Red team testing stop failed',
        details: err.response?.data?.details,
        status_code: err.response?.status || 500,
        timestamp: new Date().toISOString()
      };
      
      setError(errorResponse);
      setLoading('error');
      throw errorResponse;
    }
  }, []);

  const runTest = useCallback(async (config: any) => {
    try {
      setLoading('loading');
      setError(null);
      
      const result = await apiService.runRedTeamTest(config);
      setLoading('success');
      return result;
    } catch (err: any) {
      const errorResponse: ErrorResponse = {
        error: err.response?.data?.error || err.message || 'Red team test failed',
        details: err.response?.data?.details,
        status_code: err.response?.status || 500,
        timestamp: new Date().toISOString()
      };
      
      setError(errorResponse);
      setLoading('error');
      throw errorResponse;
    }
  }, []);

  const runAdvancedTest = useCallback(async (config: any) => {
    try {
      setLoading('loading');
      setError(null);
      
      const result = await apiService.runAdvancedTest(config);
      setLoading('success');
      return result;
    } catch (err: any) {
      const errorResponse: ErrorResponse = {
        error: err.response?.data?.error || err.message || 'Advanced red team test failed',
        details: err.response?.data?.details,
        status_code: err.response?.status || 500,
        timestamp: new Date().toISOString()
      };
      
      setError(errorResponse);
      setLoading('error');
      throw errorResponse;
    }
  }, []);

  return {
    loading,
    error,
    startTesting,
    stopTesting,
    runTest,
    runAdvancedTest
  };
}

// =============================================================================
// ANALYTICS HOOKS
// =============================================================================

export function useAnalyticsSummary() {
  return useApi(
    () => apiService.getAnalyticsSummary(),
    {} // Manual loading only
  );
}

export function useAnalyticsTrends(days?: number) {
  return useApi(
    () => apiService.getAnalyticsTrends(days),
  );
}

export function useRedTeamAnalytics(days?: number) {
  return useApi(
    () => apiService.getRedTeamAnalytics(days),
    {} // Manual loading only
  );
}

export function useModelPerformance(modelName: string) {
  return useApi(
    () => apiService.getModelPerformance(modelName),
    {}
  );
}

export function useModelComparison(modelName: string, days?: number) {
  return useApi(
    () => apiService.getModelComparison(modelName, days),
    {}
  );
}

// =============================================================================
// MLFLOW HOOKS
// =============================================================================

export function useMLflowExperiments() {
  return useApi(
    () => apiService.getMLflowExperiments(),
  );
}

export function useMLflowModels() {
  return useApi(
    () => apiService.getMLflowModels(),
  );
}

export function useMLflowRuns(experimentId?: string) {
  return useApi(
    () => apiService.getMLflowRuns(experimentId),
    {}
  );
}

// =============================================================================
// BUSINESS METRICS HOOKS
// =============================================================================

export function useBusinessMetrics() {
  return useApi(
    () => apiService.getBusinessMetrics(),
 // 5 minutes
  );
}

export function useDataPrivacyStatus() {
  return useApi(
    () => apiService.getDataPrivacyStatus(),
 // 5 minutes
  );
}

// =============================================================================
// MODEL CACHE HOOKS
// =============================================================================

export function useModelCacheStatus() {
  return useApi(
    () => apiService.getModelCacheStatus(),
  );
}

export function useModelCacheStats() {
  return useApi(
    () => apiService.getModelCacheStats(),
  );
}

// =============================================================================
// MONITORING HOOKS
// =============================================================================

export function usePrometheusMetrics() {
  return useApi(
    () => apiService.getPrometheusMetrics(),
  );
}

export function useGrafanaDashboards() {
  return useApi(
    () => apiService.getGrafanaDashboards(),
 // 5 minutes
  );
}

export function useJaegerTraces() {
  return useApi(
    () => apiService.getJaegerTraces(),
  );
}

// =============================================================================
// SYSTEM ADMINISTRATION HOOKS
// =============================================================================

export function useSystemConfig() {
  return useApi(
    () => apiService.getSystemConfig(),
 // 5 minutes
  );
}

export function useDatabaseStatus() {
  return useApi(
    () => apiService.getDatabaseStatus(),
  );
}

// =============================================================================
// UTILITY HOOKS
// =============================================================================

export function useModelOperations() {
  const [loading, setLoading] = useState<LoadingState>('idle');
  const [error, setError] = useState<ErrorResponse | null>(null);
  const [progress, setProgress] = useState<{ [modelName: string]: any }>({});

  const getProgress = useCallback(async (modelName: string) => {
    try {
      const response = await fetch(`http://localhost:8000/models/${modelName}/progress`);
      const progressData = await response.json();
      setProgress(prev => ({ ...prev, [modelName]: progressData }));
      return progressData;
    } catch (err) {
      console.error('Error fetching progress:', err);
      return null;
    }
  }, []);

  const loadModel = useCallback(async (modelName: string, version?: string) => {
    try {
      setLoading('loading');
      setError(null);
      
      // Start progress tracking immediately
      let progressInterval: NodeJS.Timeout;
      
      // Start the model loading in the background
      const loadPromise = apiService.loadModel(modelName, version);
      
      // Start progress polling
      progressInterval = setInterval(async () => {
        try {
          await getProgress(modelName);
        } catch (err) {
          console.error('Error fetching progress:', err);
        }
      }, 500); // Check progress every 500ms for more responsive updates
      
      // Wait for the model loading to complete
      const result = await loadPromise;
      
      // Clear progress tracking
      if (progressInterval) {
        clearInterval(progressInterval);
      }
      
      // Final progress update
      await getProgress(modelName);
      
      setLoading('success');
      return result;
    } catch (err: any) {
      // Clear progress tracking on error
      if (progressInterval) {
        clearInterval(progressInterval);
      }
      
      const errorResponse: ErrorResponse = {
        error: err.response?.data?.error || err.response?.data?.detail || err.message || 'Model loading failed',
        details: err.response?.data?.details || (err.response?.data?.detail ? `Server returned: ${err.response.data.detail}` : undefined),
        status_code: err.response?.status || 500,
        timestamp: new Date().toISOString()
      };
      
      setError(errorResponse);
      setLoading('error');
      throw errorResponse;
    }
  }, [getProgress]);

  const unloadModel = useCallback(async (modelName: string) => {
    try {
      setLoading('loading');
      setError(null);
      
      const result = await apiService.unloadModel(modelName);
      setLoading('success');
      return result;
    } catch (err: any) {
      const errorResponse: ErrorResponse = {
        error: err.response?.data?.error || err.message || 'Model unloading failed',
        details: err.response?.data?.details,
        status_code: err.response?.status || 500,
        timestamp: new Date().toISOString()
      };
      
      setError(errorResponse);
      setLoading('error');
      throw errorResponse;
    }
  }, []);

  const reloadModel = useCallback(async (modelName: string) => {
    try {
      setLoading('loading');
      setError(null);
      
      const result = await apiService.reloadModel(modelName);
      setLoading('success');
      return result;
    } catch (err: any) {
      const errorResponse: ErrorResponse = {
        error: err.response?.data?.error || err.message || 'Model reload failed',
        details: err.response?.data?.details,
        status_code: err.response?.status || 500,
        timestamp: new Date().toISOString()
      };
      
      setError(errorResponse);
      setLoading('error');
      throw errorResponse;
    }
  }, []);

  return {
    loading,
    error,
    progress,
    loadModel,
    unloadModel,
    reloadModel,
    getProgress
  };
}

