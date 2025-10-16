import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/apiService';

// Types for Model API Service
export interface PredictionRequest {
  text: string;
  models?: string[];
  ensemble?: boolean;
  return_probabilities?: boolean;
  return_embeddings?: boolean;
}

export interface PredictionResponse {
  text: string;
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  model_predictions: Record<string, any>;
  ensemble_used: boolean;
  processing_time_ms: number;
  timestamp: string;
}

export interface ModelInfo {
  name: string;
  type: string;
  loaded: boolean;
  path?: string;
  labels: string[];
  performance?: Record<string, number>;
  model_source?: string;
  model_version?: string;
  description?: string;
}

export interface ModelHealth {
  status: string;
  available_models: string[];
  total_models: number;
  timestamp: string;
}

export interface CacheStats {
  redis_connected: boolean;
  memory_used: string;
  connected_clients: number;
  total_commands_processed: number;
  keyspace_hits: number;
  keyspace_misses: number;
}

// Custom hooks for Model API Service
export const useModelAPI = () => {
  const [models, setModels] = useState<Record<string, ModelInfo>>({});
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null);
  const [health, setHealth] = useState<ModelHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Fetch all model API data in parallel
      const [
        modelsData,
        cacheStatsData,
        healthData
      ] = await Promise.all([
        apiService.getAvailableModels(),
        apiService.getModelCacheStats(),
        apiService.getHealthCheck()
      ]);

      setModels(modelsData.models || {});
      setCacheStats(cacheStatsData);
      setHealth(healthData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch model API data');
      console.error('Model API error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, []); // Empty dependency array - only run once on mount

  return {
    models,
    modelInfo,
    cacheStats,
    health,
    loading,
    error,
    refetch: fetchData
  };
};

// Hook for model predictions
export const useModelPrediction = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const predict = useCallback(async (request: PredictionRequest) => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.predictModel(
        request.text,
        request.models?.[0],
        request.ensemble
      );
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to make prediction');
      console.error('Prediction error:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const predictBatch = useCallback(async (texts: string[], modelName?: string) => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.predictBatch(texts, modelName);
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to make batch prediction');
      console.error('Batch prediction error:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { predict, predictBatch, loading, error };
};

// Hook for model management
export const useModelManagement = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadModel = useCallback(async (modelName: string) => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.loadModel(modelName);
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load model');
      console.error('Model loading error:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const unloadModel = useCallback(async (modelName: string) => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.unloadModel(modelName);
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to unload model');
      console.error('Model unloading error:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const reloadModel = useCallback(async (modelName: string) => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.reloadModel(modelName);
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reload model');
      console.error('Model reloading error:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const reloadAllModels = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.reloadAllModels();
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reload all models');
      console.error('Model reloading error:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    loadModel,
    unloadModel,
    reloadModel,
    reloadAllModels,
    loading,
    error
  };
};

// Hook for model information
export const useModelInfo = (modelName?: string) => {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchModelInfo = useCallback(async (name: string) => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.getModelInfo(name);
      setModelInfo(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch model info');
      console.error('Model info error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (modelName) {
      fetchModelInfo(modelName);
    }
  }, [modelName, fetchModelInfo]);

  return { modelInfo, loading, error, refetch: fetchModelInfo };
};

// Hook for cache management
export const useCacheManagement = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const clearCache = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.clearModelCache();
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to clear cache');
      console.error('Cache clearing error:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const preloadModel = useCallback(async (modelName: string) => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.preloadModel(modelName);
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to preload model');
      console.error('Model preloading error:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { clearCache, preloadModel, loading, error };
};

// Hook for model API dashboard
export const useModelAPIDashboard = () => {
  const modelAPI = useModelAPI();
  
  const isLoading = modelAPI.loading;
  const hasError = modelAPI.error;

  const refetchAll = useCallback(() => {
    modelAPI.refetch();
  }, [modelAPI.refetch]);

  return {
    ...modelAPI,
    loading: isLoading,
    error: hasError,
    refetch: refetchAll
  };
};
