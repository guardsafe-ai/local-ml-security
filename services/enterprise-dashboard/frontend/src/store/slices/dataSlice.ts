import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface Dataset {
  id: string;
  name: string;
  description: string;
  size: number;
  recordCount: number;
  createdAt: string;
  modifiedAt: string;
  qualityScore: number;
  privacyClassification: 'public' | 'internal' | 'confidential' | 'restricted';
  tags: string[];
  format: 'jsonl' | 'csv' | 'parquet';
  path: string;
}

interface DataState {
  datasets: Dataset[];
  selectedDatasets: string[];
  uploadProgress: Record<string, number>;
  loading: boolean;
  error: string | null;
}

const initialState: DataState = {
  datasets: [],
  selectedDatasets: [],
  uploadProgress: {},
  loading: false,
  error: null,
};

const dataSlice = createSlice({
  name: 'data',
  initialState,
  reducers: {
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    setDatasets: (state, action: PayloadAction<Dataset[]>) => {
      state.datasets = action.payload;
    },
    addDataset: (state, action: PayloadAction<Dataset>) => {
      state.datasets.unshift(action.payload);
    },
    updateDataset: (state, action: PayloadAction<{ id: string; updates: Partial<Dataset> }>) => {
      const { id, updates } = action.payload;
      const index = state.datasets.findIndex(dataset => dataset.id === id);
      if (index !== -1) {
        state.datasets[index] = { ...state.datasets[index], ...updates };
      }
    },
    deleteDataset: (state, action: PayloadAction<string>) => {
      state.datasets = state.datasets.filter(dataset => dataset.id !== action.payload);
    },
    selectDataset: (state, action: PayloadAction<string>) => {
      if (!state.selectedDatasets.includes(action.payload)) {
        state.selectedDatasets.push(action.payload);
      }
    },
    deselectDataset: (state, action: PayloadAction<string>) => {
      state.selectedDatasets = state.selectedDatasets.filter(id => id !== action.payload);
    },
    selectAllDatasets: (state) => {
      state.selectedDatasets = state.datasets.map(d => d.id);
    },
    deselectAllDatasets: (state) => {
      state.selectedDatasets = [];
    },
    setUploadProgress: (state, action: PayloadAction<{ fileId: string; progress: number }>) => {
      const { fileId, progress } = action.payload;
      state.uploadProgress[fileId] = progress;
    },
    clearUploadProgress: (state, action: PayloadAction<string>) => {
      delete state.uploadProgress[action.payload];
    },
  },
});

export const {
  setLoading,
  setError,
  setDatasets,
  addDataset,
  updateDataset,
  deleteDataset,
  selectDataset,
  deselectDataset,
  selectAllDatasets,
  deselectAllDatasets,
  setUploadProgress,
  clearUploadProgress,
} = dataSlice.actions;

export default dataSlice.reducer;
