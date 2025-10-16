import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/apiService';

// Types for Analytics Service
export interface RedTeamTestResult {
  test_id: string;
  model_name: string;
  model_type: string;
  model_version?: string;
  model_source?: string;
  total_attacks: number;
  vulnerabilities_found: number;
  detection_rate: number;
  test_duration_seconds?: number;
  batch_size?: number;
  attack_categories?: string[];
  attack_results?: any[];
}

export interface ModelPerformance {
  model_name: string;
  model_type: string;
  model_version?: string;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  training_duration_seconds?: number;
  dataset_size?: number;
}

export interface RedTeamSummary {
  summary: Array<{
    model_name: string;
    model_type: string;
    total_tests: number;
    avg_detection_rate: string;
    avg_attacks: string;
    avg_vulnerabilities: string;
    last_test: string;
  }>;
}

export interface ModelComparison {
  model_name: string;
  pretrained?: {
    avg_detection_rate: string | null;
    avg_attacks: string | null;
    avg_vulnerabilities: string | null;
    test_count: number;
  };
  trained?: {
    avg_detection_rate: string | null;
    avg_attacks: string | null;
    avg_vulnerabilities: string | null;
    test_count: number;
  };
  improvement?: {
    detection_rate_improvement: number;
    vulnerability_detection_improvement: number;
  };
}

export interface PerformanceTrends {
  trends: Array<{
    test_date: string;
    model_name: string;
    model_type: string;
    avg_detection_rate: string;
    test_count: number;
  }>;
}

// Custom hooks for Analytics Service
export const useRedTeamAnalytics = (days: number = 7) => {
  const [data, setData] = useState<RedTeamSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.getRedTeamSummary(days);
      setData(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch red team analytics');
      console.error('Red team analytics error:', err);
    } finally {
      setLoading(false);
    }
  }, [days]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

export const useAnalyticsTrends = (days: number = 30) => {
  const [data, setData] = useState<PerformanceTrends | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.getAnalyticsTrends(days);
      setData(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch analytics trends');
      console.error('Analytics trends error:', err);
    } finally {
      setLoading(false);
    }
  }, [days]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

export const useModelComparison = (modelName: string, days: number = 30) => {
  const [data, setData] = useState<ModelComparison | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    if (modelName === 'all') {
      setData(null);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const response = await apiService.getModelComparison(modelName, days);
      setData(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch model comparison');
      console.error('Model comparison error:', err);
    } finally {
      setLoading(false);
    }
  }, [modelName, days]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

// Hook for storing red team results
export const useStoreRedTeamResults = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const storeResults = useCallback(async (results: RedTeamTestResult) => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.storeRedTeamResults(results);
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to store red team results');
      console.error('Store red team results error:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { storeResults, loading, error };
};

// Hook for storing model performance
export const useStoreModelPerformance = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const storePerformance = useCallback(async (performance: ModelPerformance) => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.storeModelPerformance(performance);
      return response;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to store model performance');
      console.error('Store model performance error:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { storePerformance, loading, error };
};

// Hook for analytics dashboard data
export const useAnalyticsDashboard = (timeRange: number = 7) => {
  const redTeamData = useRedTeamAnalytics(timeRange);
  const trendsData = useAnalyticsTrends(timeRange);
  
  const [selectedModel, setSelectedModel] = useState<string>('all');
  const comparisonData = useModelComparison(selectedModel, timeRange);

  const isLoading = redTeamData.loading || trendsData.loading || comparisonData.loading;
  const hasError = redTeamData.error || trendsData.error || comparisonData.error;

  const refetchAll = useCallback(() => {
    redTeamData.refetch();
    trendsData.refetch();
    comparisonData.refetch();
  }, [redTeamData, trendsData, comparisonData]);

  return {
    redTeamData: redTeamData.data,
    trendsData: trendsData.data,
    comparisonData: comparisonData.data,
    selectedModel,
    setSelectedModel,
    loading: isLoading,
    error: hasError,
    refetch: refetchAll
  };
};
