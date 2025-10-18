import { configureStore } from '@reduxjs/toolkit';
import { setupListeners } from '@reduxjs/toolkit/query';
import { api } from './api';
import modelsSlice from './slices/modelsSlice';
import trainingSlice from './slices/trainingSlice';
import dataSlice from './slices/dataSlice';
import analyticsSlice from './slices/analyticsSlice';
import privacySlice from './slices/privacySlice';
import monitoringSlice from './slices/monitoringSlice';
import cacheSlice from './slices/cacheSlice';
import experimentsSlice from './slices/experimentsSlice';
import systemSlice from './slices/systemSlice';

export const store = configureStore({
  reducer: {
    [api.reducerPath]: api.reducer,
    models: modelsSlice,
    training: trainingSlice,
    data: dataSlice,
    analytics: analyticsSlice,
    privacy: privacySlice,
    monitoring: monitoringSlice,
    cache: cacheSlice,
    experiments: experimentsSlice,
    system: systemSlice,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: [api.util.resetApiState.type],
      },
    }).concat(api.middleware),
});

setupListeners(store.dispatch);

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
