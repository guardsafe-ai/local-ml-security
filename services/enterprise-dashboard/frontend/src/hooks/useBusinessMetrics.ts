import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/apiService';

// Types for Business Metrics Service
export interface AttackSuccessRate {
  total_attacks: number;
  successful_attacks: number;
  success_rate: number;
  by_category: Record<string, number>;
  by_model: Record<string, number>;
  trend_7d: number;
  trend_30d: number;
}

export interface ModelDriftMetrics {
  model_name: string;
  drift_detected: boolean;
  drift_score: number;
  confidence_interval: [number, number];
  last_drift_check: string;
  features_drifted: string[];
  severity: 'low' | 'medium' | 'high' | 'critical';
}

export interface CostMetrics {
  total_cost_usd: number;
  compute_cost: number;
  storage_cost: number;
  api_calls_cost: number;
  model_training_cost: number;
  cost_per_prediction: number;
  cost_trend_7d: number;
  cost_trend_30d: number;
}

export interface SystemEffectiveness {
  overall_effectiveness: number;
  detection_accuracy: number;
  false_positive_rate: number;
  false_negative_rate: number;
  response_time_p95: number;
  availability_percent: number;
  user_satisfaction_score: number;
}

export interface BusinessKPI {
  timestamp: string;
  attack_success_rate: AttackSuccessRate;
  model_drift: ModelDriftMetrics[];
  cost_metrics: CostMetrics;
  system_effectiveness: SystemEffectiveness;
  recommendations: string[];
}

// Custom hooks for Business Metrics Service
export const useBusinessMetrics = (timeRange: number = 7) => {
  const [kpis, setKpis] = useState<BusinessKPI | null>(null);
  const [attackSuccessRate, setAttackSuccessRate] = useState<AttackSuccessRate | null>(null);
  const [modelDrift, setModelDrift] = useState<ModelDriftMetrics[]>([]);
  const [costMetrics, setCostMetrics] = useState<CostMetrics | null>(null);
  const [systemEffectiveness, setSystemEffectiveness] = useState<SystemEffectiveness | null>(null);
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Fetch all business metrics data in parallel
      const [
        kpisData,
        attackSuccessRateData,
        modelDriftData,
        costMetricsData,
        systemEffectivenessData,
        recommendationsData
      ] = await Promise.all([
        apiService.getBusinessKPIs(),
        apiService.getAttackSuccessRate(timeRange),
        apiService.getModelDrift(),
        apiService.getCostMetrics(),
        apiService.getSystemEffectiveness(),
        apiService.getRecommendations()
      ]);

      setKpis(kpisData);
      setAttackSuccessRate(attackSuccessRateData);
      setModelDrift(modelDriftData);
      setCostMetrics(costMetricsData);
      setSystemEffectiveness(systemEffectivenessData);
      setRecommendations(recommendationsData.recommendations || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch business metrics');
      console.error('Business metrics error:', err);
    } finally {
      setLoading(false);
    }
  }, [timeRange]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return {
    kpis,
    attackSuccessRate,
    modelDrift,
    costMetrics,
    systemEffectiveness,
    recommendations,
    loading,
    error,
    refetch: fetchData
  };
};

// Hook for individual business metrics
export const useAttackSuccessRate = (days: number = 7) => {
  const [data, setData] = useState<AttackSuccessRate | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.getAttackSuccessRate(days);
      setData(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch attack success rate');
      console.error('Attack success rate error:', err);
    } finally {
      setLoading(false);
    }
  }, [days]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

export const useModelDrift = () => {
  const [data, setData] = useState<ModelDriftMetrics[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.getModelDrift();
      setData(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch model drift');
      console.error('Model drift error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

export const useCostMetrics = () => {
  const [data, setData] = useState<CostMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.getCostMetrics();
      setData(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch cost metrics');
      console.error('Cost metrics error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

export const useSystemEffectiveness = () => {
  const [data, setData] = useState<SystemEffectiveness | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.getSystemEffectiveness();
      setData(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch system effectiveness');
      console.error('System effectiveness error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

export const useRecommendations = () => {
  const [data, setData] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.getRecommendations();
      setData(response.recommendations || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch recommendations');
      console.error('Recommendations error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

// Hook for business metrics dashboard
export const useBusinessMetricsDashboard = (timeRange: number = 7) => {
  const businessMetrics = useBusinessMetrics(timeRange);
  
  const isLoading = businessMetrics.loading;
  const hasError = businessMetrics.error;

  const refetchAll = useCallback(() => {
    businessMetrics.refetch();
  }, [businessMetrics.refetch]);

  return {
    ...businessMetrics,
    loading: isLoading,
    error: hasError,
    refetch: refetchAll
  };
};
