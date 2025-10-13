import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface PerformanceMetric {
  timestamp: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  latency: number;
}

interface DriftAlert {
  id: string;
  modelName: string;
  driftScore: number;
  threshold: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  timestamp: string;
  status: 'active' | 'acknowledged' | 'resolved';
}

interface AnalyticsState {
  performanceMetrics: PerformanceMetric[];
  driftAlerts: DriftAlert[];
  modelComparison: Array<{
    modelName: string;
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    latency: number;
  }>;
  costMetrics: {
    total: number;
    training: number;
    inference: number;
    storage: number;
    monitoring: number;
  };
  loading: boolean;
  error: string | null;
}

const initialState: AnalyticsState = {
  performanceMetrics: [],
  driftAlerts: [],
  modelComparison: [],
  costMetrics: {
    total: 0,
    training: 0,
    inference: 0,
    storage: 0,
    monitoring: 0,
  },
  loading: false,
  error: null,
};

const analyticsSlice = createSlice({
  name: 'analytics',
  initialState,
  reducers: {
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    setPerformanceMetrics: (state, action: PayloadAction<PerformanceMetric[]>) => {
      state.performanceMetrics = action.payload;
    },
    addPerformanceMetric: (state, action: PayloadAction<PerformanceMetric>) => {
      state.performanceMetrics.push(action.payload);
    },
    setDriftAlerts: (state, action: PayloadAction<DriftAlert[]>) => {
      state.driftAlerts = action.payload;
    },
    addDriftAlert: (state, action: PayloadAction<DriftAlert>) => {
      state.driftAlerts.unshift(action.payload);
    },
    updateDriftAlert: (state, action: PayloadAction<{ id: string; updates: Partial<DriftAlert> }>) => {
      const { id, updates } = action.payload;
      const index = state.driftAlerts.findIndex(alert => alert.id === id);
      if (index !== -1) {
        state.driftAlerts[index] = { ...state.driftAlerts[index], ...updates };
      }
    },
    setModelComparison: (state, action: PayloadAction<AnalyticsState['modelComparison']>) => {
      state.modelComparison = action.payload;
    },
    setCostMetrics: (state, action: PayloadAction<AnalyticsState['costMetrics']>) => {
      state.costMetrics = action.payload;
    },
  },
});

export const {
  setLoading,
  setError,
  setPerformanceMetrics,
  addPerformanceMetric,
  setDriftAlerts,
  addDriftAlert,
  updateDriftAlert,
  setModelComparison,
  setCostMetrics,
} = analyticsSlice.actions;

export default analyticsSlice.reducer;
