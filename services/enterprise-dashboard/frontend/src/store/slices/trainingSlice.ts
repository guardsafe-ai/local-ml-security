import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface TrainingJob {
  id: string;
  modelName: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  startTime: string;
  endTime?: string;
  duration?: number;
  config: {
    epochs: number;
    learningRate: number;
    batchSize: number;
    maxSequenceLength: number;
    warmupSteps: number;
    weightDecay: number;
  };
  metrics: {
    loss: number;
    accuracy: number;
    f1: number;
    precision: number;
    recall: number;
  };
  logs: string[];
  error?: string;
}

interface TrainingConfig {
  id: string;
  name: string;
  modelName: string;
  config: TrainingJob['config'];
  createdAt: string;
  updatedAt: string;
}

interface TrainingState {
  jobs: TrainingJob[];
  configs: TrainingConfig[];
  queue: TrainingJob[];
  activeJobs: TrainingJob[];
  completedJobs: TrainingJob[];
  failedJobs: TrainingJob[];
  loading: boolean;
  error: string | null;
  selectedJob: string | null;
}

const initialState: TrainingState = {
  jobs: [],
  configs: [],
  queue: [],
  activeJobs: [],
  completedJobs: [],
  failedJobs: [],
  loading: false,
  error: null,
  selectedJob: null,
};

const trainingSlice = createSlice({
  name: 'training',
  initialState,
  reducers: {
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    setJobs: (state, action: PayloadAction<TrainingJob[]>) => {
      state.jobs = action.payload;
      state.activeJobs = action.payload.filter(job => job.status === 'running');
      state.completedJobs = action.payload.filter(job => job.status === 'completed');
      state.failedJobs = action.payload.filter(job => job.status === 'failed');
      state.queue = action.payload.filter(job => job.status === 'pending');
    },
    addJob: (state, action: PayloadAction<TrainingJob>) => {
      state.jobs.unshift(action.payload);
      if (action.payload.status === 'pending') {
        state.queue.unshift(action.payload);
      } else if (action.payload.status === 'running') {
        state.activeJobs.unshift(action.payload);
      }
    },
    updateJob: (state, action: PayloadAction<{ id: string; updates: Partial<TrainingJob> }>) => {
      const { id, updates } = action.payload;
      const jobIndex = state.jobs.findIndex(job => job.id === id);
      if (jobIndex !== -1) {
        state.jobs[jobIndex] = { ...state.jobs[jobIndex], ...updates };
        
        // Update specific arrays
        if (updates.status) {
          const oldStatus = state.jobs[jobIndex].status;
          const newStatus = updates.status;
          
          // Remove from old array
          if (oldStatus === 'pending') {
            state.queue = state.queue.filter(job => job.id !== id);
          } else if (oldStatus === 'running') {
            state.activeJobs = state.activeJobs.filter(job => job.id !== id);
          } else if (oldStatus === 'completed') {
            state.completedJobs = state.completedJobs.filter(job => job.id !== id);
          } else if (oldStatus === 'failed') {
            state.failedJobs = state.failedJobs.filter(job => job.id !== id);
          }
          
          // Add to new array
          if (newStatus === 'pending') {
            state.queue.push(state.jobs[jobIndex]);
          } else if (newStatus === 'running') {
            state.activeJobs.push(state.jobs[jobIndex]);
          } else if (newStatus === 'completed') {
            state.completedJobs.push(state.jobs[jobIndex]);
          } else if (newStatus === 'failed') {
            state.failedJobs.push(state.jobs[jobIndex]);
          }
        }
      }
    },
    setConfigs: (state, action: PayloadAction<TrainingConfig[]>) => {
      state.configs = action.payload;
    },
    addConfig: (state, action: PayloadAction<TrainingConfig>) => {
      state.configs.unshift(action.payload);
    },
    updateConfig: (state, action: PayloadAction<{ id: string; updates: Partial<TrainingConfig> }>) => {
      const { id, updates } = action.payload;
      const configIndex = state.configs.findIndex(config => config.id === id);
      if (configIndex !== -1) {
        state.configs[configIndex] = { ...state.configs[configIndex], ...updates };
      }
    },
    deleteConfig: (state, action: PayloadAction<string>) => {
      state.configs = state.configs.filter(config => config.id !== action.payload);
    },
    setSelectedJob: (state, action: PayloadAction<string | null>) => {
      state.selectedJob = action.payload;
    },
    addLog: (state, action: PayloadAction<{ jobId: string; log: string }>) => {
      const { jobId, log } = action.payload;
      const job = state.jobs.find(j => j.id === jobId);
      if (job) {
        job.logs.push(log);
      }
    },
    clearLogs: (state, action: PayloadAction<string>) => {
      const job = state.jobs.find(j => j.id === action.payload);
      if (job) {
        job.logs = [];
      }
    },
  },
});

export const {
  setLoading,
  setError,
  setJobs,
  addJob,
  updateJob,
  setConfigs,
  addConfig,
  updateConfig,
  deleteConfig,
  setSelectedJob,
  addLog,
  clearLogs,
} = trainingSlice.actions;

export default trainingSlice.reducer;
