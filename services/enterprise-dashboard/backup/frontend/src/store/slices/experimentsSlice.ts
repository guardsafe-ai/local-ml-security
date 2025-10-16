import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface Experiment {
  id: string;
  name: string;
  description: string;
  createdAt: string;
  lastModified: string;
  runCount: number;
  bestRun?: {
    id: string;
    metrics: Record<string, number>;
    parameters: Record<string, any>;
  };
}

interface ExperimentRun {
  id: string;
  experimentId: string;
  name: string;
  status: 'running' | 'finished' | 'failed' | 'killed';
  startTime: string;
  endTime?: string;
  duration?: number;
  metrics: Record<string, number>;
  parameters: Record<string, any>;
  tags: string[];
  notes?: string;
}

interface ModelVersion {
  name: string;
  version: string;
  stage: 'staging' | 'production' | 'archived';
  description?: string;
  createdAt: string;
  runId: string;
  metrics: Record<string, number>;
  tags: string[];
}

interface ExperimentsState {
  experiments: Experiment[];
  runs: ExperimentRun[];
  modelRegistry: ModelVersion[];
  selectedExperiment: string | null;
  selectedRun: string | null;
  loading: boolean;
  error: string | null;
}

const initialState: ExperimentsState = {
  experiments: [],
  runs: [],
  modelRegistry: [],
  selectedExperiment: null,
  selectedRun: null,
  loading: false,
  error: null,
};

const experimentsSlice = createSlice({
  name: 'experiments',
  initialState,
  reducers: {
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    setExperiments: (state, action: PayloadAction<Experiment[]>) => {
      state.experiments = action.payload;
    },
    addExperiment: (state, action: PayloadAction<Experiment>) => {
      state.experiments.unshift(action.payload);
    },
    updateExperiment: (state, action: PayloadAction<{ id: string; updates: Partial<Experiment> }>) => {
      const { id, updates } = action.payload;
      const index = state.experiments.findIndex(exp => exp.id === id);
      if (index !== -1) {
        state.experiments[index] = { ...state.experiments[index], ...updates };
      }
    },
    deleteExperiment: (state, action: PayloadAction<string>) => {
      state.experiments = state.experiments.filter(exp => exp.id !== action.payload);
    },
    setRuns: (state, action: PayloadAction<ExperimentRun[]>) => {
      state.runs = action.payload;
    },
    addRun: (state, action: PayloadAction<ExperimentRun>) => {
      state.runs.unshift(action.payload);
    },
    updateRun: (state, action: PayloadAction<{ id: string; updates: Partial<ExperimentRun> }>) => {
      const { id, updates } = action.payload;
      const index = state.runs.findIndex(run => run.id === id);
      if (index !== -1) {
        state.runs[index] = { ...state.runs[index], ...updates };
      }
    },
    setModelRegistry: (state, action: PayloadAction<ModelVersion[]>) => {
      state.modelRegistry = action.payload;
    },
    addModelVersion: (state, action: PayloadAction<ModelVersion>) => {
      state.modelRegistry.unshift(action.payload);
    },
    updateModelVersion: (state, action: PayloadAction<{ name: string; version: string; updates: Partial<ModelVersion> }>) => {
      const { name, version, updates } = action.payload;
      const index = state.modelRegistry.findIndex(model => model.name === name && model.version === version);
      if (index !== -1) {
        state.modelRegistry[index] = { ...state.modelRegistry[index], ...updates };
      }
    },
    setSelectedExperiment: (state, action: PayloadAction<string | null>) => {
      state.selectedExperiment = action.payload;
    },
    setSelectedRun: (state, action: PayloadAction<string | null>) => {
      state.selectedRun = action.payload;
    },
  },
});

export const {
  setLoading,
  setError,
  setExperiments,
  addExperiment,
  updateExperiment,
  deleteExperiment,
  setRuns,
  addRun,
  updateRun,
  setModelRegistry,
  addModelVersion,
  updateModelVersion,
  setSelectedExperiment,
  setSelectedRun,
} = experimentsSlice.actions;

export default experimentsSlice.reducer;
