import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

// Define the base URL for the API
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8007';

export const api = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({
    baseUrl: API_BASE_URL,
    prepareHeaders: (headers) => {
      // Add any auth headers here if needed
      headers.set('Content-Type', 'application/json');
      return headers;
    },
  }),
  tagTypes: [
    'Models',
    'TrainingJobs',
    'Datasets',
    'Analytics',
    'BusinessMetrics',
    'Privacy',
    'Monitoring',
    'Cache',
    'Experiments',
    'SystemHealth',
  ],
  endpoints: (builder) => ({
    // Health endpoints
    getHealth: builder.query<any, void>({
      query: () => '/health',
      providesTags: ['SystemHealth'],
    }),
    getServicesHealth: builder.query<any, void>({
      query: () => '/services/health',
      providesTags: ['SystemHealth'],
    }),

    // Model endpoints
    getModels: builder.query<any, void>({
      query: () => '/models',
      providesTags: ['Models'],
    }),
    getModelInfo: builder.query<any, string>({
      query: (modelName) => `/models/info/${modelName}`,
      providesTags: ['Models'],
    }),
    loadModel: builder.mutation<any, { modelName: string }>({
      query: ({ modelName }) => ({
        url: `/models/load`,
        method: 'POST',
        body: { model_name: modelName },
      }),
      invalidatesTags: ['Models'],
    }),
    unloadModel: builder.mutation<any, { modelName: string }>({
      query: ({ modelName }) => ({
        url: `/models/unload`,
        method: 'POST',
        body: { model_name: modelName },
      }),
      invalidatesTags: ['Models'],
    }),
    predictModel: builder.mutation<any, { text: string; modelName?: string; ensemble?: boolean }>({
      query: (body) => ({
        url: '/models/predict',
        method: 'POST',
        body,
      }),
    }),

    // Training endpoints
    getTrainingJobs: builder.query<any, void>({
      query: () => '/training/jobs',
      providesTags: ['TrainingJobs'],
    }),
    getTrainingJob: builder.query<any, string>({
      query: (jobId) => `/training/jobs/${jobId}`,
      providesTags: ['TrainingJobs'],
    }),
    startTraining: builder.mutation<any, any>({
      query: (body) => ({
        url: '/training/start',
        method: 'POST',
        body,
      }),
      invalidatesTags: ['TrainingJobs'],
    }),
    stopTraining: builder.mutation<any, string>({
      query: (jobId) => ({
        url: `/training/stop/${jobId}`,
        method: 'POST',
      }),
      invalidatesTags: ['TrainingJobs'],
    }),

    // Data endpoints
    getDatasets: builder.query<any, void>({
      query: () => '/data/datasets',
      providesTags: ['Datasets'],
    }),
    uploadData: builder.mutation<any, FormData>({
      query: (formData) => ({
        url: '/data/upload',
        method: 'POST',
        body: formData,
      }),
      invalidatesTags: ['Datasets'],
    }),

    // Analytics endpoints
    getAnalyticsSummary: builder.query<any, void>({
      query: () => '/analytics/summary',
      providesTags: ['Analytics'],
    }),
    getPerformanceTrends: builder.query<any, { days?: number }>({
      query: ({ days = 30 } = {}) => `/analytics/trends?days=${days}`,
      providesTags: ['Analytics'],
    }),

    // Business Metrics endpoints
    getBusinessMetrics: builder.query<any, void>({
      query: () => '/business-metrics/summary',
      providesTags: ['BusinessMetrics'],
    }),

    // Privacy endpoints
    getPrivacySummary: builder.query<any, void>({
      query: () => '/data-privacy/summary',
      providesTags: ['Privacy'],
    }),
    classifyData: builder.mutation<any, { data: any; dataId: string }>({
      query: ({ data, dataId }) => ({
        url: '/data-privacy/classify',
        method: 'POST',
        body: data,
        params: { data_id: dataId },
      }),
    }),

    // Monitoring endpoints
    getMonitoringMetrics: builder.query<any, void>({
      query: () => '/monitoring/metrics',
      providesTags: ['Monitoring'],
    }),

    // Cache endpoints
    getCacheStatus: builder.query<any, void>({
      query: () => '/cache/status',
      providesTags: ['Cache'],
    }),

    // MLflow endpoints
    getExperiments: builder.query<any, void>({
      query: () => '/mlflow/experiments',
      providesTags: ['Experiments'],
    }),
    getModelRegistry: builder.query<any, void>({
      query: () => '/mlflow/registry',
      providesTags: ['Experiments'],
    }),
  }),
});

export const {
  useGetHealthQuery,
  useGetServicesHealthQuery,
  useGetModelsQuery,
  useGetModelInfoQuery,
  useLoadModelMutation,
  useUnloadModelMutation,
  usePredictModelMutation,
  useGetTrainingJobsQuery,
  useGetTrainingJobQuery,
  useStartTrainingMutation,
  useStopTrainingMutation,
  useGetDatasetsQuery,
  useUploadDataMutation,
  useGetAnalyticsSummaryQuery,
  useGetPerformanceTrendsQuery,
  useGetBusinessMetricsQuery,
  useGetPrivacySummaryQuery,
  useClassifyDataMutation,
  useGetMonitoringMetricsQuery,
  useGetCacheStatusQuery,
  useGetExperimentsQuery,
  useGetModelRegistryQuery,
} = api;
