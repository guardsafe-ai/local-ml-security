import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8007',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Create separate axios instance for model operations (model-api service)
const modelApi = axios.create({
  baseURL: process.env.REACT_APP_MODEL_API_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Create separate axios instance for training service (training service)
const trainingApi = axios.create({
  baseURL: process.env.REACT_APP_TRAINING_API_URL || 'http://localhost:8002',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('❌ API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Add same interceptors to modelApi
modelApi.interceptors.request.use(
  (config) => {
    console.log(`🚀 Model API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('❌ Model API Request Error:', error);
    return Promise.reject(error);
  }
);

modelApi.interceptors.response.use(
  (response) => {
    console.log(`✅ Model API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error(`❌ Model API Error: ${error.response?.status} ${error.config?.url}`, error.response?.data);
    return Promise.reject(error);
  }
);

// Add interceptors to trainingApi
trainingApi.interceptors.request.use(
  (config) => {
    console.log(`🚀 Training API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('❌ Training API Request Error:', error);
    return Promise.reject(error);
  }
);

trainingApi.interceptors.response.use(
  (response) => {
    console.log(`✅ Training API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error(`❌ Training API Error: ${error.response?.status} ${error.config?.url}`, error.response?.data);
    return Promise.reject(error);
  }
);

// =============================================================================
// COMPREHENSIVE API SERVICE - ALL ENDPOINTS UTILIZED
// =============================================================================

export const apiService = {
  // =============================================================================
  // DASHBOARD & SYSTEM HEALTH
  // =============================================================================
  getDashboardMetrics: () => api.get('/dashboard/metrics').then(res => res.data),
  getServicesHealth: () => api.get('/services/health').then(res => res.data),
  getSystemStatus: () => api.get('/system/status').then(res => res.data),
  getHealthCheck: () => api.get('/health').then(res => res.data),

  // =============================================================================
  // MODEL MANAGEMENT - COMPLETE MODEL LIFECYCLE
  // =============================================================================
  
  // Model Registry & Information (Model API Service)
  getModelRegistry: () => api.get('/models/registry').then(res => res.data),
  getAvailableModels: () => api.get('/models/available').then(res => res.data),
  getModelsOverview: () => api.get('/dashboard/models/overview').then(res => res.data),
  getModelInfo: (modelName: string) => api.get(`/models/${modelName}`).then(res => res.data),
  getModelVersions: (modelName: string) => api.get(`/models/${modelName}/versions`).then(res => res.data),
  getModelsInfo: () => api.get('/models/info').then(res => res.data),
  
  // Model Loading & Management (Model API Service)
  loadModel: (modelName: string, version?: string) => 
    modelApi.post('/load', { model_name: modelName, version }).then(res => res.data),
  unloadModel: (modelName: string) => 
    modelApi.post('/unload', { model_name: modelName }).then(res => res.data),
  reloadModel: (modelName: string) => 
    modelApi.post(`/models/${modelName}/reload`).then(res => res.data),
  reloadAllModels: () => modelApi.post('/models/reload').then(res => res.data),
  
  // Model Predictions & Testing (Model API Service)
  predictModel: (text: string, modelName?: string, ensemble?: boolean) => 
    modelApi.post('/predict', { text, models: modelName ? [modelName] : undefined, ensemble }).then(res => res.data),
  predictBatch: (texts: string[], modelName?: string) => 
    modelApi.post('/predict/batch', { texts, model_name: modelName }).then(res => res.data),
  predictTrained: (text: string, modelName?: string) => 
    modelApi.post('/predict/trained', { text, model_name: modelName }).then(res => res.data),

  // Model Cache Service (port 8003)
  getModelCacheLogs: (modelName?: string, limit?: number) => 
    api.get(`http://localhost:8003/logs${modelName ? `?model_name=${modelName}` : ''}${limit ? `&limit=${limit}` : ''}`).then(res => res.data),
  getModelCacheModelLogs: (modelName: string, limit?: number) => 
    api.get(`http://localhost:8003/models/${modelName}/logs${limit ? `?limit=${limit}` : ''}`).then(res => res.data),
  
  // Model Status & Health
  getModelStatus: (modelName: string) => 
    api.get(`/models/status/${modelName}`).then(res => res.data),
  getModelHealth: (modelName: string) => 
    api.get(`/models/health/${modelName}`).then(res => res.data),

  // =============================================================================
  // TRAINING MANAGEMENT - COMPLETE TRAINING LIFECYCLE
  // =============================================================================
  
  // Training Jobs
  getTrainingJobs: () => api.get('/training/jobs').then(res => res.data),
  getJobLogs: (jobId: string) => api.get(`/training/jobs/${jobId}/logs`).then(res => res.data),
  getTrainingJob: (jobId: string) => api.get(`/training/jobs/${jobId}`).then(res => res.data),
  getTrainingLogs: () => api.get('/training/logs').then(res => res.data),
  
  // Training Operations
  startTraining: (config: any) => api.post('/training/start', config).then(res => res.data),
  trainModel: (config: any) => api.post('/training/train', config).then(res => res.data),
  trainLoadedModel: (config: any) => api.post('/training/train/loaded-model', config).then(res => res.data),
  retrainModel: (config: any) => api.post('/training/retrain', config).then(res => res.data),
  cancelTrainingJob: (jobId: string) => api.delete(`/training/jobs/${jobId}`).then(res => res.data),
  
  // Training Data Management
  getTrainingData: () => api.get('/training/data/data-statistics').then(res => res.data),
  uploadTrainingData: (data: any) => api.post('/training/data/upload-data', data).then(res => res.data),
  
  // Training Models & Configuration
  getTrainingModels: () => api.get('/training/models').then(res => res.data),
  getTrainingConfig: (modelName: string) => api.get(`/training/config/${modelName}`).then(res => res.data),
  updateTrainingConfig: (modelName: string, config: any) => 
    api.put(`/training/config/${modelName}`, config).then(res => res.data),

  // =============================================================================
  // RED TEAM TESTING - COMPREHENSIVE SECURITY TESTING
  // =============================================================================
  
  // Red Team Status & Results
  getRedTeamStatus: () => api.get('/red-team/status').then(res => res.data),
  getRedTeamResults: () => api.get('/dashboard/red-team/overview').then(res => res.data),
  getRedTeamMetrics: () => api.get('/red-team/metrics').then(res => res.data),
  
  // Red Team Operations
  startRedTeamTesting: () => api.post('/red-team/start').then(res => res.data),
  stopRedTeamTesting: () => api.post('/red-team/stop').then(res => res.data),
  runRedTeamTest: (config: any) => api.post('/red-team/test', config).then(res => res.data),
  runAdvancedTest: (config: any) => api.post('/red-team/test/advanced', config).then(res => res.data),
  
  // Red Team Analysis
  getRedTeamSummary: (days?: number) => 
    api.get(`/red-team/summary${days ? `?days=${days}` : ''}`).then(res => res.data),
  getRedTeamTrends: (days?: number) => 
    api.get(`/red-team/trends${days ? `?days=${days}` : ''}`).then(res => res.data),
  getRedTeamComparison: (modelName: string) => 
    api.get(`/red-team/comparison/${modelName}`).then(res => res.data),
  
  // Red Team Learning & Retraining
  getLearningStatus: () => api.get('/red-team/learning/status').then(res => res.data),
  triggerLearning: () => api.post('/red-team/learning/trigger').then(res => res.data),
  getVulnerabilities: () => api.get('/red-team/vulnerabilities').then(res => res.data),

  // =============================================================================
  // ANALYTICS & MONITORING - COMPREHENSIVE ANALYTICS
  // =============================================================================
  
  // Analytics Summary & Trends
  getAnalyticsSummary: (days?: number) => api.get(`/analytics/summary${days ? `?days=${days}` : ''}`).then(res => res.data),
  getAnalyticsTrends: (days?: number) => 
    api.get(`/analytics/trends${days ? `?days=${days}` : ''}`).then(res => res.data),
  getAnalyticsComparison: (modelName: string) => 
    api.get(`/analytics/comparison/${modelName}`).then(res => res.data),
  
  // Red Team Analytics (Analytics Service)
  getRedTeamAnalytics: (days?: number) => 
    api.get(`/analytics/summary${days ? `?days=${days}` : ''}`).then(res => res.data),
  getRedTeamAnalyticsTrends: (days?: number) => 
    api.get(`/analytics/trends${days ? `?days=${days}` : ''}`).then(res => res.data),
  
  // Model Performance Analytics (Analytics Service)
  getModelPerformance: (modelName: string) => 
    api.get(`/analytics/model/performance/${modelName}`).then(res => res.data),
  getModelComparison: (modelName: string, days?: number) => 
    api.get(`/red-team/comparison/${modelName}${days ? `?days=${days}` : ''}`).then(res => res.data),
  
  // Store Analytics Data
  storeRedTeamResults: (results: any) => 
    api.post('/red-team/results', results).then(res => res.data),
  storeModelPerformance: (performance: any) => 
    api.post('/model/performance', performance).then(res => res.data),
  
  // System Analytics
  getSystemMetrics: () => api.get('/analytics/system/metrics').then(res => res.data),
  getResourceUsage: () => api.get('/analytics/system/resources').then(res => res.data),
  getPerformanceMetrics: () => api.get('/analytics/system/performance').then(res => res.data),

  // =============================================================================
  // MLFLOW INTEGRATION - EXPERIMENT TRACKING
  // =============================================================================
  
  // MLflow Experiments
  getMLflowExperiments: () => api.get('/mlflow/experiments').then(res => res.data),
  getMLflowExperiment: (experimentId: string) => 
    api.get(`/mlflow/experiments/${experimentId}`).then(res => res.data),
  getMLflowExperimentSummary: () => api.get('/mlflow/experiment/summary').then(res => res.data),
  
  // MLflow Models
  getMLflowModels: () => api.get('/mlflow/models').then(res => res.data),
  getMLflowModel: (modelName: string) => 
    api.get(`/mlflow/models/${modelName}`).then(res => res.data),
  getMLflowModelPerformance: (modelName: string) => 
    api.get(`/mlflow/models/performance/${modelName}`).then(res => res.data),
  getMLflowModelCompare: () => api.get('/mlflow/models/compare').then(res => res.data),
  
  // MLflow Runs
  getMLflowRuns: (experimentId?: string) => 
    api.get(`/mlflow/runs${experimentId ? `?experiment_id=${experimentId}` : ''}`).then(res => res.data),
  getMLflowRun: (runId: string) => api.get(`/mlflow/runs/${runId}`).then(res => res.data),
  
  // MLflow Artifacts & Datasets
  getMLflowArtifacts: (runId: string) => 
    api.get(`/mlflow/runs/${runId}/artifacts`).then(res => res.data),
  getMLflowDatasets: () => api.get('/mlflow/datasets').then(res => res.data),
  getMLflowCleanup: () => api.get('/mlflow/cleanup').then(res => res.data),
  
  // MLflow Red Team Integration
  getMLflowRedTeamLog: () => api.get('/mlflow/red-team/log').then(res => res.data),

  // =============================================================================
  // BUSINESS METRICS & DATA PRIVACY
  // =============================================================================
  
  // Business Metrics (Business Metrics Service)
  getBusinessKPIs: () => api.get('/kpis').then(res => res.data),
  getAttackSuccessRate: (days?: number) => 
    api.get(`/attack-success-rate${days ? `?days=${days}` : ''}`).then(res => res.data),
  getModelDrift: () => api.get('/model-drift').then(res => res.data),
  getCostMetrics: () => api.get('/cost-metrics').then(res => res.data),
  getSystemEffectiveness: () => api.get('/system-effectiveness').then(res => res.data),
  getRecommendations: () => api.get('/recommendations').then(res => res.data),
  
  // Legacy Business Metrics (if needed)
  getBusinessMetrics: () => api.get('/business/metrics').then(res => res.data),
  getCostAnalysis: () => api.get('/business/cost-analysis').then(res => res.data),
  getROIAnalysis: () => api.get('/business/roi-analysis').then(res => res.data),
  getResourceUtilization: () => api.get('/business/resource-utilization').then(res => res.data),
  
  // Data Privacy (Data Privacy Service)
  getDataPrivacyCompliance: () => api.get('/compliance').then(res => res.data),
  getDataPrivacyAudit: (filters?: any) => 
    api.get('/audit-logs', { params: filters }).then(res => res.data),
  getDataSubjects: () => api.get('/data-subjects').then(res => res.data),
  getRetentionPolicies: () => api.get('/retention-policies').then(res => res.data),
  anonymizeData: (text: string) => 
    api.post('/anonymize', null, { params: { text } }).then(res => res.data),
  registerDataSubject: (subjectData: any) => 
    api.post('/data-subjects', subjectData).then(res => res.data),
  withdrawConsent: (subjectId: string) => 
    api.post(`/data-subjects/${subjectId}/withdraw-consent`).then(res => res.data),
  deleteDataSubject: (subjectId: string) => 
    api.delete(`/data-subjects/${subjectId}`).then(res => res.data),
  cleanupExpiredData: () => api.post('/cleanup').then(res => res.data),
  
  // Legacy Data Privacy (if needed)
  getDataPrivacyStatus: () => api.get('/data-privacy/status').then(res => res.data),
  getDataPrivacyReports: () => api.get('/data-privacy/reports').then(res => res.data),

  // =============================================================================
  // MODEL CACHE & SERVING
  // =============================================================================
  
  // Model Cache (Model API Service)
  getModelCacheStatus: () => api.get('/model-cache/status').then(res => res.data),
  getModelCacheStats: () => api.get('/cache/stats').then(res => res.data),
  clearModelCache: () => api.delete('/cache/clear').then(res => res.data),
  preloadModel: (modelName: string) => 
    api.post('/model-cache/preload', { model_name: modelName }).then(res => res.data),
  

  // =============================================================================
  // MONITORING & OBSERVABILITY
  // =============================================================================
  
  // Prometheus Metrics
  getPrometheusMetrics: () => api.get('/monitoring/prometheus/metrics').then(res => res.data),
  getPrometheusTargets: () => api.get('/monitoring/prometheus/targets').then(res => res.data),
  
  // Grafana Integration
  getGrafanaDashboards: () => api.get('/monitoring/grafana/dashboards').then(res => res.data),
  getGrafanaPanels: (dashboardId: string) => 
    api.get(`/monitoring/grafana/panels/${dashboardId}`).then(res => res.data),
  
  // Jaeger Tracing
  getJaegerTraces: () => api.get('/monitoring/jaeger/traces').then(res => res.data),
  getJaegerServices: () => api.get('/monitoring/jaeger/services').then(res => res.data),
  getJaegerOperations: (service: string) => 
    api.get(`/monitoring/jaeger/operations/${service}`).then(res => res.data),

  // =============================================================================
  // SYSTEM ADMINISTRATION
  // =============================================================================
  
  // System Configuration
  getSystemConfig: () => api.get('/admin/config').then(res => res.data),
  updateSystemConfig: (config: any) => api.put('/admin/config', config).then(res => res.data),
  getSystemLogs: () => api.get('/admin/logs').then(res => res.data),
  
  // Service Management
  restartService: (serviceName: string) => 
    api.post(`/admin/services/${serviceName}/restart`).then(res => res.data),
  getServiceLogs: (serviceName: string) => 
    api.get(`/admin/services/${serviceName}/logs`).then(res => res.data),
  
  // Database Management
  getDatabaseStatus: () => api.get('/admin/database/status').then(res => res.data),
  getDatabaseStats: () => api.get('/admin/database/stats').then(res => res.data),
  runDatabaseQuery: (query: string) => 
    api.post('/admin/database/query', { query }).then(res => res.data),

  // =============================================================================
  // UTILITY FUNCTIONS
  // =============================================================================
  
  // Health Checks
  checkServiceHealth: (serviceName: string) => 
    api.get(`/health/${serviceName}`).then(res => res.data),
  
  // System Information
  getSystemInfo: () => api.get('/system/info').then(res => res.data),
  getVersionInfo: () => api.get('/system/version').then(res => res.data),
  
  // Export & Import
  exportData: (type: string) => api.get(`/export/${type}`).then(res => res.data),
  importData: (type: string, data: any) => 
    api.post(`/import/${type}`, data).then(res => res.data),

  // =============================================================================
  // PERFORMANCE CACHE
  // =============================================================================
  
  // Cache Management
  getCacheStatus: () => 
    api.get('/training/performance/cache/status').then(res => res.data),
  
  getCachedData: () => 
    api.get('/training/performance/cache/data').then(res => res.data),
  
  invalidateCache: () => 
    api.post('/training/performance/cache/invalidate').then(res => res.data),
  
  refreshCache: () => 
    api.post('/training/performance/cache/refresh').then(res => res.data),

  // =============================================================================
  // DATA MANAGEMENT
  // =============================================================================
  
  // File Upload and Management (Training Service)
  uploadLargeFile: (formData: FormData) => 
    trainingApi.post('/data/efficient/upload-large-file', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300000 // 5 minutes
    }).then(res => res.data),
  
  getUploadProgress: (fileId: string) => 
    trainingApi.get(`/data/efficient/upload-progress/${fileId}`).then(res => res.data),
  
  getStagedFiles: (status?: string) => 
    trainingApi.get('/data/efficient/staged-files', { params: { status } }).then(res => res.data),
  
  processFile: (fileId: string, validationRules?: any) =>
    trainingApi.post(`/data/efficient/process-file/${fileId}`, validationRules).then(res => res.data),
  
  downloadFile: (fileId: string) =>
    trainingApi.get(`/data/efficient/download-file/${fileId}`, { 
      responseType: 'blob',
      headers: {
        'Accept': 'application/jsonl, application/json, text/plain, */*'
      }
    }),
  
  getFileInfo: (fileId: string) =>
    trainingApi.get(`/data/efficient/file-info/${fileId}`).then(res => res.data),
  
  retryFailedFile: (fileId: string) =>
    trainingApi.post(`/data/efficient/retry-failed-file/${fileId}`).then(res => res.data),
  
  cleanupFailedUploads: (hoursOld: number = 24) =>
    trainingApi.delete('/data/efficient/cleanup-failed-uploads', { params: { hours_old: hoursOld } }).then(res => res.data),
  
  // Traditional Data Management (Training Service)
  getFreshData: () => 
    trainingApi.get('/data/fresh-data').then(res => res.data),
  
  getUsedData: () => 
    trainingApi.get('/data/used-data').then(res => res.data),
  
  getDataStatistics: () => 
    trainingApi.get('/data/data-statistics').then(res => res.data),
  
  createSampleData: () => 
    trainingApi.post('/data/create-sample-data').then(res => res.data),
  
  cleanupOldData: (daysOld: number = 30) =>
    trainingApi.delete('/data/cleanup-old-data', { params: { days_old: daysOld } }).then(res => res.data),
  
  // Efficient Data Manager Health and Metrics
  getEfficientDataHealth: () => 
    trainingApi.get('/data/efficient/health').then(res => res.data),
  
  getEfficientDataMetrics: () => 
    trainingApi.get('/data/efficient/metrics').then(res => res.data),
};

export default api;