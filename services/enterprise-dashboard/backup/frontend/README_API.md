# Enterprise Dashboard Frontend - API Documentation

## Overview

The Enterprise Dashboard Frontend provides a comprehensive React-based web interface for the ML Security platform. This document details the frontend API integration, component interfaces, hooks, services, and data flow patterns used to interact with the backend services.

## Base Configuration

### Environment Variables
```bash
# API Endpoints
REACT_APP_API_URL=http://localhost:8007
REACT_APP_MODEL_API_URL=http://localhost:8000
REACT_APP_TRAINING_API_URL=http://localhost:8002
REACT_APP_WS_URL=ws://localhost:8007/ws

# Application Configuration
REACT_APP_ENVIRONMENT=development
REACT_APP_VERSION=1.0.0
```

### API Service Configuration
```typescript
// Multiple API instances for different services
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8007',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' }
});

const modelApi = axios.create({
  baseURL: process.env.REACT_APP_MODEL_API_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' }
});

const trainingApi = axios.create({
  baseURL: process.env.REACT_APP_TRAINING_API_URL || 'http://localhost:8002',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' }
});
```

---

## API Service Layer

### 1. Dashboard & System Health

#### getDashboardMetrics()
**Description**: Get comprehensive dashboard metrics

**Returns**: `Promise<DashboardMetrics>`
```typescript
interface DashboardMetrics {
  total_models: number;
  loaded_models: number;
  active_training_jobs: number;
  completed_training_jobs: number;
  total_red_team_tests: number;
  detection_rate: number;
  system_health: 'healthy' | 'unhealthy' | 'degraded';
  last_updated: string;
}
```

**Usage**:
```typescript
const { data: metrics, isLoading } = useQuery('dashboard-metrics', apiService.getDashboardMetrics);
```

#### getServicesHealth()
**Description**: Get health status of all ML Security services

**Returns**: `Promise<ServiceHealth[]>`
```typescript
interface ServiceHealth {
  name: string;
  status: 'healthy' | 'unhealthy' | 'degraded' | 'unknown';
  response_time?: number;
  last_check: string;
  details?: {
    service: string;
    status: string;
    timestamp: string;
    uptime_seconds?: number;
    dependencies?: Record<string, boolean>;
  };
}
```

**Usage**:
```typescript
const { data: health } = useQuery('services-health', apiService.getServicesHealth);
```

#### getSystemStatus()
**Description**: Get comprehensive system status

**Returns**: `Promise<SystemStatus>`
```typescript
interface SystemStatus {
  overall_status: 'healthy' | 'unhealthy' | 'degraded';
  services: ServiceHealth[];
  last_updated: string;
  system_metrics: {
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
    network_usage: number;
  };
}
```

### 2. Model Management

#### getAvailableModels()
**Description**: Get all available models

**Returns**: `Promise<{ models: Record<string, ModelInfo>; count: number }>`
```typescript
interface ModelInfo {
  name: string;
  type: 'pretrained' | 'trained';
  version?: string;
  source: 'Hugging Face' | 'MLflow' | 'Local';
  loaded: boolean;
  size_gb?: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  last_trained?: string;
  description?: string;
  tags?: string[];
  status: 'available' | 'loading' | 'loaded' | 'error';
  loading_progress?: number;
  error_message?: string;
}
```

**Usage**:
```typescript
const { data: models } = useQuery('available-models', apiService.getAvailableModels);
```

#### loadModel(modelName: string, version?: string)
**Description**: Load a model

**Parameters**:
- `modelName` (string): Name of the model to load
- `version` (string, optional): Model version

**Returns**: `Promise<{ status: string; message: string; data: any }>`

**Usage**:
```typescript
const loadModel = useMutation(apiService.loadModel, {
  onSuccess: () => {
    queryClient.invalidateQueries('available-models');
  }
});

// Usage
loadModel.mutate({ modelName: 'security-classifier', version: '1.0.0' });
```

#### unloadModel(modelName: string)
**Description**: Unload a model

**Parameters**:
- `modelName` (string): Name of the model to unload

**Returns**: `Promise<{ status: string; message: string; data: any }>`

**Usage**:
```typescript
const unloadModel = useMutation(apiService.unloadModel, {
  onSuccess: () => {
    queryClient.invalidateQueries('available-models');
  }
});
```

#### predictModel(text: string, modelName?: string, ensemble?: boolean)
**Description**: Make a model prediction

**Parameters**:
- `text` (string): Text to analyze
- `modelName` (string, optional): Model name to use
- `ensemble` (boolean, optional): Use ensemble prediction

**Returns**: `Promise<ModelPrediction>`
```typescript
interface ModelPrediction {
  text: string;
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  model_predictions?: Record<string, any>;
  ensemble_used: boolean;
  processing_time_ms: number;
  timestamp: string;
  from_cache?: boolean;
  model_name?: string;
}
```

**Usage**:
```typescript
const predictModel = useMutation(apiService.predictModel);

// Usage
predictModel.mutate({ 
  text: 'This is a test message', 
  modelName: 'security-classifier',
  ensemble: false 
});
```

#### predictBatch(texts: string[], modelName?: string)
**Description**: Make batch predictions

**Parameters**:
- `texts` (string[]): List of texts to analyze
- `modelName` (string, optional): Model name to use

**Returns**: `Promise<{ predictions: ModelPrediction[]; total_processing_time_ms: number; timestamp: string }>`

### 3. Training Management

#### getTrainingJobs()
**Description**: Get all training jobs

**Returns**: `Promise<{ jobs: TrainingJob[]; count: number }>`
```typescript
interface TrainingJob {
  job_id: string;
  model_name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  start_time: string;
  end_time?: string;
  duration_seconds?: number;
  training_data_path?: string;
  learning_rate?: number;
  batch_size?: number;
  num_epochs?: number;
  max_length?: number;
  config?: TrainingConfig;
  metrics?: TrainingMetrics;
  logs?: string[];
  error?: string;
  error_message?: string;
}
```

**Usage**:
```typescript
const { data: jobs } = useQuery('training-jobs', apiService.getTrainingJobs);
```

#### startTraining(config: TrainingConfig)
**Description**: Start a new training job

**Parameters**:
- `config` (TrainingConfig): Training configuration

**Returns**: `Promise<{ status: string; message: string; data: any }>`
```typescript
interface TrainingConfig {
  model_name: string;
  training_data_path: string;
  hyperparameters: {
    learning_rate?: number;
    batch_size?: number;
    epochs?: number;
    optimizer?: string;
    scheduler?: string;
  };
  validation_split?: number;
  test_split?: number;
  early_stopping?: boolean;
  patience?: number;
  metric_for_best_model?: string;
}
```

**Usage**:
```typescript
const startTraining = useMutation(apiService.startTraining, {
  onSuccess: () => {
    queryClient.invalidateQueries('training-jobs');
  }
});

// Usage
startTraining.mutate({
  model_name: 'security-classifier',
  training_data_path: '/data/training.csv',
  hyperparameters: {
    learning_rate: 0.001,
    batch_size: 32,
    epochs: 10
  }
});
```

#### getJobLogs(jobId: string)
**Description**: Get logs for a specific training job

**Parameters**:
- `jobId` (string): Job identifier

**Returns**: `Promise<{ logs: string[]; count: number }>`

**Usage**:
```typescript
const { data: logs } = useQuery(['job-logs', jobId], () => apiService.getJobLogs(jobId));
```

### 4. Red Team Testing

#### getRedTeamResults()
**Description**: Get red team testing results

**Returns**: `Promise<RedTeamTestResult[]>`
```typescript
interface RedTeamTestResult {
  test_id: string;
  model_name: string;
  model_type: string;
  model_version?: string;
  total_attacks: number;
  vulnerabilities_found: number;
  detection_rate: number;
  overall_status: 'PASS' | 'FAIL';
  pass_count: number;
  fail_count: number;
  pass_rate: number;
  test_summary: {
    total_tests: number;
    passed: number;
    failed: number;
    success_rate: number;
    overall_status: string;
  };
  security_risk_distribution: {
    CRITICAL: number;
    HIGH: number;
    MEDIUM: number;
    LOW: number;
  };
  duration_ms: number;
  timestamp: string;
  results: AttackResult[];
}
```

**Usage**:
```typescript
const { data: results } = useQuery('red-team-results', apiService.getRedTeamResults);
```

#### runRedTeamTest(config: RedTeamTestConfig)
**Description**: Run red team testing

**Parameters**:
- `config` (RedTeamTestConfig): Test configuration

**Returns**: `Promise<{ status: string; message: string; data: any }>`
```typescript
interface RedTeamTestConfig {
  model_name: string;
  test_count: number;
  attack_categories?: string[];
  test_texts?: string[];
}
```

**Usage**:
```typescript
const runRedTeamTest = useMutation(apiService.runRedTeamTest, {
  onSuccess: () => {
    queryClient.invalidateQueries('red-team-results');
  }
});
```

### 5. Analytics & Monitoring

#### getAnalyticsSummary(days?: number)
**Description**: Get analytics summary

**Parameters**:
- `days` (number, optional): Number of days to analyze

**Returns**: `Promise<AnalyticsSummary>`
```typescript
interface AnalyticsSummary {
  summary: Array<{
    model_name: string;
    model_type: string;
    total_tests: number;
    avg_detection_rate: number;
    avg_attacks: number;
    avg_vulnerabilities: number;
    last_test: string;
  }>;
}
```

**Usage**:
```typescript
const { data: analytics } = useQuery(['analytics-summary', days], () => 
  apiService.getAnalyticsSummary(days)
);
```

#### getAnalyticsTrends(days?: number)
**Description**: Get analytics trends

**Parameters**:
- `days` (number, optional): Number of days to analyze

**Returns**: `Promise<AnalyticsTrends>`
```typescript
interface AnalyticsTrends {
  trends: Array<{
    test_date: string;
    model_name: string;
    model_type: string;
    avg_detection_rate: number;
    test_count: number;
  }>;
}
```

### 6. Business Metrics

#### getBusinessKPIs()
**Description**: Get business KPIs

**Returns**: `Promise<BusinessMetrics>`
```typescript
interface BusinessMetrics {
  total_cost: number;
  cost_per_prediction: number;
  cost_per_training_hour: number;
  roi_percentage: number;
  resource_utilization: {
    cpu_usage: number;
    memory_usage: number;
    storage_usage: number;
    network_usage: number;
  };
  performance_metrics: {
    average_response_time: number;
    throughput_per_second: number;
    error_rate: number;
    availability_percentage: number;
  };
  cost_breakdown: {
    compute_costs: number;
    storage_costs: number;
    network_costs: number;
    licensing_costs: number;
  };
}
```

**Usage**:
```typescript
const { data: kpis } = useQuery('business-kpis', apiService.getBusinessKPIs);
```

### 7. Data Privacy

#### getDataPrivacyCompliance()
**Description**: Get data privacy compliance status

**Returns**: `Promise<DataPrivacyStatus>`
```typescript
interface DataPrivacyStatus {
  gdpr_compliant: boolean;
  data_anonymization_enabled: boolean;
  audit_logging_enabled: boolean;
  consent_management_enabled: boolean;
  data_retention_policy: string;
  last_audit_date: string;
  compliance_score: number;
  violations: Array<{
    type: string;
    severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
    description: string;
    timestamp: string;
    resolved: boolean;
  }>;
}
```

**Usage**:
```typescript
const { data: privacy } = useQuery('data-privacy', apiService.getDataPrivacyCompliance);
```

#### anonymizeData(text: string)
**Description**: Anonymize data

**Parameters**:
- `text` (string): Text to anonymize

**Returns**: `Promise<{ anonymized_text: string; pii_detected: string[] }>`

**Usage**:
```typescript
const anonymizeData = useMutation(apiService.anonymizeData);

// Usage
anonymizeData.mutate({ text: 'John Doe, email: john@example.com' });
```

---

## Custom Hooks

### 1. useModelAPI()
**Description**: Custom hook for model operations

**Returns**: `{ loadModel, unloadModel, predictModel, predictBatch }`

**Usage**:
```typescript
const { loadModel, unloadModel, predictModel } = useModelAPI();

// Load a model
loadModel.mutate({ modelName: 'security-classifier' });

// Make a prediction
predictModel.mutate({ 
  text: 'Test message', 
  modelName: 'security-classifier' 
});
```

### 2. useAnalytics()
**Description**: Custom hook for analytics operations

**Returns**: `{ getSummary, getTrends, getComparison }`

**Usage**:
```typescript
const { getSummary, getTrends } = useAnalytics();

// Get analytics summary
const { data: summary } = getSummary({ days: 30 });

// Get trends
const { data: trends } = getTrends({ days: 7 });
```

### 3. useBusinessMetrics()
**Description**: Custom hook for business metrics

**Returns**: `{ getKPIs, getCostAnalysis, getROIAnalysis }`

**Usage**:
```typescript
const { getKPIs, getCostAnalysis } = useBusinessMetrics();

// Get KPIs
const { data: kpis } = getKPIs();

// Get cost analysis
const { data: costs } = getCostAnalysis();
```

### 4. useDataPrivacy()
**Description**: Custom hook for data privacy operations

**Returns**: `{ getCompliance, anonymizeData, getAuditLogs }`

**Usage**:
```typescript
const { getCompliance, anonymizeData } = useDataPrivacy();

// Get compliance status
const { data: compliance } = getCompliance();

// Anonymize data
const anonymize = useMutation(anonymizeData);
```

---

## WebSocket Integration

### 1. WebSocket Provider
**Description**: Context provider for WebSocket connections

**Usage**:
```typescript
// Wrap your app with WebSocketProvider
<WebSocketProvider>
  <App />
</WebSocketProvider>
```

### 2. useWebSocket()
**Description**: Hook for WebSocket operations

**Returns**: `{ isConnected, lastMessage, sendMessage, subscribe, unsubscribe }`

**Usage**:
```typescript
const { isConnected, sendMessage, subscribe } = useWebSocket();

// Subscribe to events
useEffect(() => {
  subscribe('model-updated', (data) => {
    console.log('Model updated:', data);
  });
}, [subscribe]);

// Send message
sendMessage({
  type: 'ping',
  data: {},
  timestamp: new Date().toISOString()
});
```

### 3. WebSocket Message Types
```typescript
interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

// Common message types
const MESSAGE_TYPES = {
  PING: 'ping',
  PONG: 'pong',
  MODEL_UPDATED: 'model-updated',
  TRAINING_PROGRESS: 'training-progress',
  RED_TEAM_RESULT: 'red-team-result',
  SYSTEM_ALERT: 'system-alert'
};
```

---

## Component Interfaces

### 1. Model Management Components

#### ModelCard Props
```typescript
interface ModelCardProps {
  model: ModelInfo;
  onSelect: (model: ModelInfo) => void;
  onLoad?: (modelName: string) => void;
  onUnload?: (modelName: string) => void;
  onTest?: (modelName: string) => void;
}
```

#### ModelList Props
```typescript
interface ModelListProps {
  models: ModelInfo[];
  onSelect: (model: ModelInfo) => void;
  loading?: boolean;
  error?: string;
}
```

#### ModelDetails Props
```typescript
interface ModelDetailsProps {
  model: ModelInfo;
  onClose: () => void;
  onLoad?: (modelName: string) => void;
  onUnload?: (modelName: string) => void;
}
```

### 2. Training Components

#### TrainingJobCard Props
```typescript
interface TrainingJobCardProps {
  job: TrainingJob;
  onViewLogs: (jobId: string) => void;
  onCancel?: (jobId: string) => void;
  onRetry?: (jobId: string) => void;
}
```

#### TrainingForm Props
```typescript
interface TrainingFormProps {
  onSubmit: (config: TrainingConfig) => void;
  loading?: boolean;
  error?: string;
  initialValues?: Partial<TrainingConfig>;
}
```

### 3. Analytics Components

#### ChartContainer Props
```typescript
interface ChartContainerProps {
  data: any[];
  type: 'line' | 'bar' | 'pie' | 'area';
  title: string;
  xAxisKey?: string;
  yAxisKey?: string;
  height?: number;
  width?: number;
}
```

#### MetricCard Props
```typescript
interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'stable';
  trendValue?: string;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error';
  icon?: React.ReactNode;
}
```

### 4. Data Table Components

#### DataGrid Props
```typescript
interface DataGridProps {
  columns: TableColumn[];
  data: any[];
  loading?: boolean;
  pagination?: {
    page: number;
    pageSize: number;
    total: number;
  };
  onPageChange?: (page: number) => void;
  onPageSizeChange?: (pageSize: number) => void;
  onSort?: (column: string, direction: 'asc' | 'desc') => void;
  onFilter?: (filters: Filter[]) => void;
}
```

#### TableColumn Interface
```typescript
interface TableColumn {
  key: string;
  label: string;
  sortable?: boolean;
  filterable?: boolean;
  render?: (value: any, row: any) => React.ReactNode;
  width?: number;
  align?: 'left' | 'center' | 'right';
}
```

---

## State Management

### 1. Redux Store Structure
```typescript
interface RootState {
  models: {
    models: ModelInfo[];
    selectedModel: ModelInfo | null;
    loading: boolean;
    error: string | null;
  };
  training: {
    jobs: TrainingJob[];
    selectedJob: TrainingJob | null;
    loading: boolean;
    error: string | null;
  };
  analytics: {
    summary: AnalyticsSummary | null;
    trends: AnalyticsTrends | null;
    loading: boolean;
    error: string | null;
  };
  redTeam: {
    results: RedTeamTestResult[];
    selectedResult: RedTeamTestResult | null;
    loading: boolean;
    error: string | null;
  };
  businessMetrics: {
    kpis: BusinessMetrics | null;
    loading: boolean;
    error: string | null;
  };
  dataPrivacy: {
    compliance: DataPrivacyStatus | null;
    loading: boolean;
    error: string | null;
  };
  system: {
    health: ServiceHealth[];
    status: SystemStatus | null;
    loading: boolean;
    error: string | null;
  };
}
```

### 2. Redux Actions
```typescript
// Model actions
const modelsSlice = createSlice({
  name: 'models',
  initialState,
  reducers: {
    setModels: (state, action) => {
      state.models = action.payload;
    },
    setSelectedModel: (state, action) => {
      state.selectedModel = action.payload;
    },
    setLoading: (state, action) => {
      state.loading = action.payload;
    },
    setError: (state, action) => {
      state.error = action.payload;
    }
  }
});

export const { setModels, setSelectedModel, setLoading, setError } = modelsSlice.actions;
```

### 3. React Query Integration
```typescript
// Query keys
export const QUERY_KEYS = {
  MODELS: 'models',
  TRAINING_JOBS: 'training-jobs',
  ANALYTICS_SUMMARY: 'analytics-summary',
  RED_TEAM_RESULTS: 'red-team-results',
  BUSINESS_KPIS: 'business-kpis',
  DATA_PRIVACY: 'data-privacy',
  SYSTEM_HEALTH: 'system-health'
};

// Query functions
export const useModels = () => {
  return useQuery(QUERY_KEYS.MODELS, apiService.getAvailableModels, {
    staleTime: 5 * 60 * 1000, // 5 minutes
    cacheTime: 10 * 60 * 1000, // 10 minutes
  });
};
```

---

## Error Handling

### 1. API Error Handling
```typescript
// Global error handler
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle authentication error
      redirectToLogin();
    } else if (error.response?.status >= 500) {
      // Handle server error
      showErrorNotification('Server error occurred');
    }
    return Promise.reject(error);
  }
);
```

### 2. Component Error Handling
```typescript
// Error boundary component
class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  
  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }
  
  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }
  
  render() {
    if (this.state.hasError) {
      return <ErrorFallback error={this.state.error} />;
    }
    
    return this.props.children;
  }
}
```

### 3. Query Error Handling
```typescript
// Query with error handling
const { data, isLoading, error } = useQuery('models', apiService.getAvailableModels, {
  onError: (error) => {
    console.error('Failed to fetch models:', error);
    showErrorNotification('Failed to load models');
  },
  retry: (failureCount, error) => {
    if (error.status === 404) return false;
    return failureCount < 3;
  }
});
```

---

## Performance Optimization

### 1. Code Splitting
```typescript
// Lazy loading components
const ModelRegistry = lazy(() => import('./pages/Models/ModelRegistry'));
const TrainingQueue = lazy(() => import('./pages/Training/TrainingQueue'));

// Route-based code splitting
<Route path="/models" element={
  <Suspense fallback={<LoadingSkeleton />}>
    <ModelRegistry />
  </Suspense>
} />
```

### 2. Memoization
```typescript
// Memoized components
const ModelCard = React.memo<ModelCardProps>(({ model, onSelect }) => {
  return (
    <Card onClick={() => onSelect(model)}>
      <CardContent>
        <Typography variant="h6">{model.name}</Typography>
        <StatusIndicator status={model.status} />
      </CardContent>
    </Card>
  );
});

// Memoized callbacks
const handleModelSelect = useCallback((model: ModelInfo) => {
  setSelectedModel(model);
}, []);
```

### 3. Virtual Scrolling
```typescript
// Virtual scrolling for large lists
const VirtualizedList: React.FC<VirtualizedListProps> = ({ items, renderItem }) => {
  return (
    <FixedSizeList
      height={400}
      itemCount={items.length}
      itemSize={50}
      itemData={items}
    >
      {({ index, style, data }) => (
        <div style={style}>
          {renderItem(data[index], index)}
        </div>
      )}
    </FixedSizeList>
  );
};
```

---

## Testing

### 1. Component Testing
```typescript
// Component unit tests
describe('ModelCard', () => {
  it('renders model information correctly', () => {
    const model = { name: 'test-model', status: 'loaded' };
    render(<ModelCard model={model} onSelect={jest.fn()} />);
    
    expect(screen.getByText('test-model')).toBeInTheDocument();
    expect(screen.getByText('loaded')).toBeInTheDocument();
  });
});
```

### 2. Hook Testing
```typescript
// Custom hook tests
describe('useModelAPI', () => {
  it('loads model successfully', async () => {
    const { result } = renderHook(() => useModelAPI());
    
    act(() => {
      result.current.loadModel.mutate({ modelName: 'test-model' });
    });
    
    await waitFor(() => {
      expect(result.current.loadModel.isSuccess).toBe(true);
    });
  });
});
```

### 3. Integration Testing
```typescript
// API integration tests
describe('Model API Integration', () => {
  it('loads models successfully', async () => {
    const { result } = renderHook(() => useQuery('models', apiService.getAvailableModels));
    
    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
      expect(result.current.data).toBeDefined();
    });
  });
});
```

---

## Usage Examples

### 1. Complete Model Management Flow
```typescript
const ModelManagementPage: React.FC = () => {
  const { data: models, isLoading } = useQuery('models', apiService.getAvailableModels);
  const { loadModel, unloadModel, predictModel } = useModelAPI();
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null);
  
  const handleLoadModel = (modelName: string) => {
    loadModel.mutate({ modelName }, {
      onSuccess: () => {
        queryClient.invalidateQueries('models');
        showSuccessNotification('Model loaded successfully');
      }
    });
  };
  
  const handlePredict = (text: string) => {
    predictModel.mutate({ 
      text, 
      modelName: selectedModel?.name 
    }, {
      onSuccess: (result) => {
        console.log('Prediction result:', result);
      }
    });
  };
  
  return (
    <DashboardLayout>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <ModelList 
            models={models?.models || []} 
            onSelect={setSelectedModel}
            loading={isLoading}
          />
        </Grid>
        <Grid item xs={12} md={6}>
          {selectedModel && (
            <ModelDetails 
              model={selectedModel}
              onLoad={handleLoadModel}
              onUnload={(name) => unloadModel.mutate({ modelName: name })}
              onTest={handlePredict}
            />
          )}
        </Grid>
      </Grid>
    </DashboardLayout>
  );
};
```

### 2. Real-time Updates with WebSocket
```typescript
const RealTimeDashboard: React.FC = () => {
  const { isConnected, subscribe } = useWebSocket();
  const [notifications, setNotifications] = useState<NotificationData[]>([]);
  
  useEffect(() => {
    subscribe('model-updated', (data) => {
      setNotifications(prev => [...prev, {
        id: Date.now().toString(),
        type: 'info',
        title: 'Model Updated',
        message: `Model ${data.model_name} has been updated`,
        timestamp: new Date().toISOString(),
        read: false
      }]);
    });
    
    subscribe('training-progress', (data) => {
      // Update training progress
      queryClient.setQueryData(['training-jobs', data.job_id], (old: any) => ({
        ...old,
        progress: data.progress
      }));
    });
  }, [subscribe]);
  
  return (
    <DashboardLayout>
      <StatusIndicator 
        status={isConnected ? 'connected' : 'disconnected'} 
        label="WebSocket" 
      />
      <NotificationList notifications={notifications} />
    </DashboardLayout>
  );
};
```

### 3. Data Visualization
```typescript
const AnalyticsDashboard: React.FC = () => {
  const { data: summary } = useQuery('analytics-summary', apiService.getAnalyticsSummary);
  const { data: trends } = useQuery('analytics-trends', apiService.getAnalyticsTrends);
  
  const chartData = useMemo(() => {
    if (!trends?.trends) return [];
    
    return trends.trends.map(trend => ({
      date: trend.test_date,
      detectionRate: trend.avg_detection_rate,
      testCount: trend.test_count
    }));
  }, [trends]);
  
  return (
    <DashboardLayout>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <MetricCard
            title="Total Tests"
            value={summary?.summary?.[0]?.total_tests || 0}
            color="primary"
          />
        </Grid>
        <Grid item xs={12} md={6}>
          <MetricCard
            title="Detection Rate"
            value={`${(summary?.summary?.[0]?.avg_detection_rate || 0) * 100}%`}
            color="success"
          />
        </Grid>
        <Grid item xs={12}>
          <ChartContainer
            data={chartData}
            type="line"
            title="Detection Rate Trends"
            xAxisKey="date"
            yAxisKey="detectionRate"
          />
        </Grid>
      </Grid>
    </DashboardLayout>
  );
};
```

---

**Enterprise Dashboard Frontend API** - Complete reference for frontend API integration, component interfaces, custom hooks, state management, WebSocket communication, error handling, performance optimization, and usage examples for the ML Security platform's React-based web interface.
