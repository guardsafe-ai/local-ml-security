import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface CachedModel {
  name: string;
  size: number;
  hitCount: number;
  missCount: number;
  lastAccessed: string;
  cachedAt: string;
}

interface CacheState {
  cachedModels: CachedModel[];
  cacheSize: number;
  maxCacheSize: number;
  hitRate: number;
  missRate: number;
  evictionPolicy: 'lru' | 'lfu' | 'fifo';
  loading: boolean;
  error: string | null;
}

const initialState: CacheState = {
  cachedModels: [],
  cacheSize: 0,
  maxCacheSize: 1024 * 1024 * 1024, // 1GB
  hitRate: 0,
  missRate: 0,
  evictionPolicy: 'lru',
  loading: false,
  error: null,
};

const cacheSlice = createSlice({
  name: 'cache',
  initialState,
  reducers: {
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    setCachedModels: (state, action: PayloadAction<CachedModel[]>) => {
      state.cachedModels = action.payload;
    },
    addCachedModel: (state, action: PayloadAction<CachedModel>) => {
      state.cachedModels.push(action.payload);
    },
    removeCachedModel: (state, action: PayloadAction<string>) => {
      state.cachedModels = state.cachedModels.filter(model => model.name !== action.payload);
    },
    updateCachedModel: (state, action: PayloadAction<{ name: string; updates: Partial<CachedModel> }>) => {
      const { name, updates } = action.payload;
      const index = state.cachedModels.findIndex(model => model.name === name);
      if (index !== -1) {
        state.cachedModels[index] = { ...state.cachedModels[index], ...updates };
      }
    },
    setCacheSize: (state, action: PayloadAction<number>) => {
      state.cacheSize = action.payload;
    },
    setMaxCacheSize: (state, action: PayloadAction<number>) => {
      state.maxCacheSize = action.payload;
    },
    setHitRate: (state, action: PayloadAction<number>) => {
      state.hitRate = action.payload;
    },
    setMissRate: (state, action: PayloadAction<number>) => {
      state.missRate = action.payload;
    },
    setEvictionPolicy: (state, action: PayloadAction<CacheState['evictionPolicy']>) => {
      state.evictionPolicy = action.payload;
    },
    clearCache: (state) => {
      state.cachedModels = [];
      state.cacheSize = 0;
    },
  },
});

export const {
  setLoading,
  setError,
  setCachedModels,
  addCachedModel,
  removeCachedModel,
  updateCachedModel,
  setCacheSize,
  setMaxCacheSize,
  setHitRate,
  setMissRate,
  setEvictionPolicy,
  clearCache,
} = cacheSlice.actions;

export default cacheSlice.reducer;
