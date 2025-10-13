import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface Model {
  id: string;
  name: string;
  type: string;
  status: 'loaded' | 'unloaded' | 'loading' | 'error';
  size: number;
  accuracy?: number;
  f1?: number;
  lastUsed?: string;
  source: 'huggingface' | 'mlflow' | 'local';
  version?: string;
  stage?: 'staging' | 'production' | 'archived';
}

interface ModelsState {
  available: Model[];
  loaded: Model[];
  selected: string[];
  loading: boolean;
  error: string | null;
  predictionHistory: Array<{
    id: string;
    text: string;
    model: string;
    prediction: string;
    confidence: number;
    timestamp: string;
  }>;
}

const initialState: ModelsState = {
  available: [],
  loaded: [],
  selected: [],
  loading: false,
  error: null,
  predictionHistory: [],
};

const modelsSlice = createSlice({
  name: 'models',
  initialState,
  reducers: {
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    setAvailableModels: (state, action: PayloadAction<Model[]>) => {
      state.available = action.payload;
    },
    setLoadedModels: (state, action: PayloadAction<Model[]>) => {
      state.loaded = action.payload;
    },
    updateModelStatus: (state, action: PayloadAction<{ id: string; status: Model['status'] }>) => {
      const { id, status } = action.payload;
      const model = state.available.find(m => m.id === id);
      if (model) {
        model.status = status;
      }
    },
    selectModel: (state, action: PayloadAction<string>) => {
      if (!state.selected.includes(action.payload)) {
        state.selected.push(action.payload);
      }
    },
    deselectModel: (state, action: PayloadAction<string>) => {
      state.selected = state.selected.filter(id => id !== action.payload);
    },
    selectAllModels: (state) => {
      state.selected = state.available.map(m => m.id);
    },
    deselectAllModels: (state) => {
      state.selected = [];
    },
    addPrediction: (state, action: PayloadAction<{
      text: string;
      model: string;
      prediction: string;
      confidence: number;
    }>) => {
      const prediction = {
        id: Date.now().toString(),
        ...action.payload,
        timestamp: new Date().toISOString(),
      };
      state.predictionHistory.unshift(prediction);
      // Keep only last 100 predictions
      if (state.predictionHistory.length > 100) {
        state.predictionHistory = state.predictionHistory.slice(0, 100);
      }
    },
    clearPredictionHistory: (state) => {
      state.predictionHistory = [];
    },
  },
});

export const {
  setLoading,
  setError,
  setAvailableModels,
  setLoadedModels,
  updateModelStatus,
  selectModel,
  deselectModel,
  selectAllModels,
  deselectAllModels,
  addPrediction,
  clearPredictionHistory,
} = modelsSlice.actions;

export default modelsSlice.reducer;
