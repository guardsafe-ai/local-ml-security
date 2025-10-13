# ML Security Analytics Service

## Service Architecture & Purpose

### Core Purpose
The Analytics Service is the **central intelligence hub** of the ML Security platform. It provides comprehensive analytics, drift detection, model performance monitoring, and automated retraining capabilities to ensure optimal model performance and security.

### Why This Service Exists
- **Model Performance Monitoring**: Tracks model accuracy, precision, recall, and F1 scores over time
- **Drift Detection**: Identifies data drift and model drift to maintain model reliability
- **Automated Retraining**: Triggers model retraining when performance degrades
- **Red Team Integration**: Processes and analyzes red team testing results
- **Business Intelligence**: Provides insights for decision-making and model optimization

## Complete API Documentation for Frontend Development

### Base URL
```
http://analytics:8001
```

### Authentication
All endpoints require authentication headers:
```javascript
headers: {
  'Authorization': 'Bearer <token>',
  'Content-Type': 'application/json'
}
```

### Core Analytics Endpoints

#### `GET /analytics/trends`
**Purpose**: Get performance trends over time
**Query Parameters**:
- `period` (optional): Time period (7d, 30d, 90d) - default: 30d
- `model_name` (optional): Filter by specific model
- `metric` (optional): Specific metric (accuracy, f1_score, precision, recall)

**Frontend Usage**:
```javascript
// Get trends for last 30 days
const response = await fetch('/analytics/trends?period=30d');
const trends = await response.json();

// Get specific model trends
const response = await fetch('/analytics/trends?model_name=bert-base&period=7d');
const modelTrends = await response.json();
```

**Response Model**:
```typescript
interface TrendsResponse {
  trends: {
    accuracy: {
      current: number;
      previous: number;
      change: number;
      trend: 'improving' | 'declining' | 'stable';
    };
    f1_score: {
      current: number;
      previous: number;
      change: number;
      trend: 'improving' | 'declining' | 'stable';
    };
    precision: {
      current: number;
      previous: number;
      change: number;
      trend: 'improving' | 'declining' | 'stable';
    };
    recall: {
      current: number;
      previous: number;
      change: number;
      trend: 'improving' | 'declining' | 'stable';
    };
  };
  period: string;
  timestamp: string;
}
```

### Red Team Integration Endpoints

#### `POST /red-team/results`
**Purpose**: Store red team test results
**Request Body**:
```typescript
interface RedTeamTestResult {
  test_id: string;
  model_name: string;
  test_type: string;
  attack_success: boolean;
  confidence_score: number;
  attack_details: {
    method: string;
    payload: string;
    success_rate: number;
  };
  timestamp: string;
}
```

**Frontend Usage**:
```javascript
const testResult = {
  test_id: 'test_123',
  model_name: 'bert-base',
  test_type: 'prompt_injection',
  attack_success: true,
  confidence_score: 0.95,
  attack_details: {
    method: 'jailbreak',
    payload: 'Ignore previous instructions...',
    success_rate: 0.8
  },
  timestamp: new Date().toISOString()
};

const response = await fetch('/red-team/results', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(testResult)
});
```

#### `GET /red-team/summary`
**Purpose**: Get red team test summary
**Query Parameters**:
- `days` (optional): Number of days to look back - default: 7

**Frontend Usage**:
```javascript
// Get last 7 days summary
const response = await fetch('/red-team/summary?days=7');
const summary = await response.json();

// Display in dashboard
summary.models.forEach(model => {
  console.log(`${model.model_name}: ${model.success_rate}% success rate`);
});
```

**Response Model**:
```typescript
interface RedTeamSummary {
  total_tests: number;
  successful_attacks: number;
  success_rate: number;
  models: Array<{
    model_name: string;
    test_count: number;
    success_rate: number;
    vulnerabilities: string[];
  }>;
  period_days: number;
  timestamp: string;
}
```

#### `GET /red-team/comparison/{model_name}`
**Purpose**: Get comparison between pre-trained and trained versions
**Path Parameters**:
- `model_name`: Name of the model to compare

**Frontend Usage**:
```javascript
const response = await fetch('/red-team/comparison/bert-base?days=30');
const comparison = await response.json();
```

**Response Model**:
```typescript
interface ModelComparison {
  model_name: string;
  pretrained_performance: {
    accuracy: number;
    f1_score: number;
    vulnerability_rate: number;
  };
  trained_performance: {
    accuracy: number;
    f1_score: number;
    vulnerability_rate: number;
  };
  improvement: {
    accuracy_gain: number;
    f1_gain: number;
    vulnerability_reduction: number;
  };
  period_days: number;
}
```

### Model Performance Endpoints

#### `POST /model/performance`
**Purpose**: Store model performance metrics
**Request Body**:
```typescript
interface ModelPerformance {
  model_name: string;
  version: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  inference_time_ms: number;
  timestamp: string;
  test_data_size: number;
}
```

**Frontend Usage**:
```javascript
const performance = {
  model_name: 'bert-base',
  version: '1.2.0',
  accuracy: 0.95,
  precision: 0.94,
  recall: 0.93,
  f1_score: 0.935,
  inference_time_ms: 45.2,
  timestamp: new Date().toISOString(),
  test_data_size: 1000
};

const response = await fetch('/model/performance', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(performance)
});
```

### Drift Detection Endpoints

#### `POST /drift/data-drift`
**Purpose**: Detect data drift between reference and current data
**Request Body**:
```typescript
interface DriftDetectionRequest {
  current_data: Array<Record<string, any>>;
  reference_data?: Array<Record<string, any>>;
  feature_columns?: string[];
}
```

**Frontend Usage**:
```javascript
const driftRequest = {
  current_data: [
    { text: "sample text 1", length: 12 },
    { text: "sample text 2", length: 15 }
  ],
  reference_data: [
    { text: "reference text 1", length: 10 },
    { text: "reference text 2", length: 14 }
  ],
  feature_columns: ['length', 'word_count']
};

const response = await fetch('/drift/data-drift', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(driftRequest)
});
```

**Response Model**:
```typescript
interface DriftDetectionResponse {
  drift_detected: boolean;
  overall_drift_score: number;
  drifted_features: string[];
  drift_summary: {
    total_drifted_features: number;
    drift_percentage: number;
    severe_drift_features: string[];
    moderate_drift_features: string[];
    minor_drift_features: string[];
  };
  statistical_tests: Record<string, {
    feature: string;
    is_drifted: boolean;
    drift_severity: 'minor' | 'moderate' | 'severe';
    ks_statistic: number;
    ks_pvalue: number;
    psi_value: number;
    reference_mean: number;
    current_mean: number;
  }>;
  timestamp: string;
}
```

#### `POST /drift/model-drift`
**Purpose**: Detect model performance drift
**Request Body**:
```typescript
interface ModelDriftRequest {
  current_predictions: Array<Record<string, any>>;
  reference_predictions?: Array<Record<string, any>>;
  current_labels?: string[];
  reference_labels?: string[];
}
```

#### `POST /drift/set-reference`
**Purpose**: Set reference data for drift detection
**Request Body**:
```typescript
interface SetReferenceRequest {
  data: Array<Record<string, any>>;
  predictions?: Array<Record<string, any>>;
}
```

#### `GET /drift/history`
**Purpose**: Get drift detection history
**Query Parameters**:
- `days` (optional): Number of days to look back - default: 30

**Frontend Usage**:
```javascript
// Get drift history for last 30 days
const response = await fetch('/drift/history?days=30');
const history = await response.json();

// Display in timeline chart
history.history.forEach(entry => {
  console.log(`${entry.timestamp}: ${entry.drift_score} drift score`);
});
```

#### `GET /drift/summary`
**Purpose**: Get drift detection summary
**Query Parameters**:
- `days` (optional): Number of days to look back - default: 30

#### `GET /drift/config`
**Purpose**: Get current drift detection configuration

**Response Model**:
```typescript
interface DriftConfig {
  ks_threshold: number;
  chi2_threshold: number;
  psi_threshold: number;
  psi_minor_threshold: number;
  psi_moderate_threshold: number;
  psi_severe_threshold: number;
  accuracy_drop_threshold: number;
  f1_drop_threshold: number;
  reference_window_days: number;
  detection_window_days: number;
  min_samples: number;
}
```

#### `POST /drift/config`
**Purpose**: Update drift detection configuration
**Request Body**:
```typescript
interface DriftConfigRequest {
  ks_threshold?: number;
  chi2_threshold?: number;
  psi_threshold?: number;
  accuracy_drop_threshold?: number;
  f1_drop_threshold?: number;
}
```

#### `GET /drift/alerts`
**Purpose**: Get drift alerts for the last N days
**Query Parameters**:
- `days` (optional): Number of days to look back - default: 7

**Frontend Usage**:
```javascript
// Get recent drift alerts
const response = await fetch('/drift/alerts?days=7');
const alerts = await response.json();

// Display alerts in notification panel
alerts.alerts.forEach(alert => {
  console.log(`${alert.type}: ${alert.message} (${alert.severity})`);
});
```

#### `POST /drift/test-drift`
**Purpose**: Test drift detection with sample data
**Frontend Usage**:
```javascript
// Test drift detection functionality
const response = await fetch('/drift/test-drift', {
  method: 'POST'
});
const testResults = await response.json();
```

#### `POST /drift/model-performance-drift`
**Purpose**: Compare performance between old and new models
**Request Body**:
```typescript
interface ModelPerformanceDriftRequest {
  old_model_predictions: Array<Record<string, any>>;
  new_model_predictions: Array<Record<string, any>>;
  ground_truth?: string[];
}
```

#### `POST /drift/check-and-retrain`
**Purpose**: Check for drift and automatically trigger retraining
**Request Body**:
```typescript
interface CheckAndRetrainRequest {
  model_name: string;
  current_data: Array<Record<string, any>>;
  reference_data?: Array<Record<string, any>>;
  training_data_path: string;
  feature_columns?: string[];
}
```

#### `POST /drift/evaluate-model`
**Purpose**: Evaluate a model for promotion from Staging to Production
**Request Body**:
```typescript
interface ModelEvaluationRequest {
  model_name: string;
  version: string;
  force: boolean;
  test_data?: Array<Record<string, any>>;
}
```

#### `POST /drift/promote-model`
**Purpose**: Promote a model from Staging to Production
**Request Body**:
```typescript
interface ModelPromotionRequest {
  model_name: string;
  version: string;
  force: boolean;
  test_data?: Array<Record<string, any>>;
}
```

#### `GET /drift/promotion-history`
**Purpose**: Get promotion evaluation history
**Query Parameters**:
- `model_name` (optional): Filter by specific model

#### `GET /drift/promotion-criteria`
**Purpose**: Get current model promotion criteria configuration

#### `POST /drift/test-email`
**Purpose**: Test email notification service

#### `GET /drift/recent-results`
**Purpose**: Get recent drift results for auto-retrain service
**Query Parameters**:
- `model_name` (optional): Filter by specific model
- `hours` (optional): Hours to look back - default: 24

#### `GET /drift/health`
**Purpose**: Drift detection health check

#### `POST /drift/comprehensive-metrics`
**Purpose**: Calculate comprehensive performance metrics
**Request Body**:
```typescript
interface ComprehensiveMetricsRequest {
  y_true: string[];
  y_pred: string[];
  y_prob?: number[];
  model_name: string;
}
```

#### `GET /drift/data/production-inference`
**Purpose**: Get production inference data for retraining
**Query Parameters**:
- `hours` (optional): Hours of data to retrieve - default: 24
- `model_name` (optional): Specific model to filter by

#### `POST /drift/drift/production-data`
**Purpose**: Detect drift using actual production inference data
**Query Parameters**:
- `model_name`: Model name to check for drift
- `hours` (optional): Hours of production data to analyze - default: 24

### Auto-Retrain Endpoints

#### `GET /auto-retrain/config`
**Purpose**: Get current auto-retrain configuration

**Response Model**:
```typescript
interface RetrainConfig {
  data_drift_threshold: number;
  model_drift_threshold: number;
  performance_drop_threshold: number;
  min_samples_for_retrain: number;
  retrain_cooldown_hours: number;
  max_retrain_attempts: number;
  target_models: string[];
  retrain_priority: string;
}
```

#### `PUT /auto-retrain/config`
**Purpose**: Update auto-retrain configuration
**Request Body**:
```typescript
interface RetrainConfigRequest {
  data_drift_threshold?: number;
  model_drift_threshold?: number;
  performance_drop_threshold?: number;
  min_samples_for_retrain?: number;
  retrain_cooldown_hours?: number;
  max_retrain_attempts?: number;
  target_models?: string[];
  retrain_priority?: string;
}
```

#### `POST /auto-retrain/trigger/{model_name}`
**Purpose**: Manually trigger retraining for a specific model
**Path Parameters**:
- `model_name`: Name of the model to retrain

**Request Body**:
```typescript
interface ManualRetrainRequest {
  model_name: string;
  reason: string;
  priority: string;
}
```

#### `GET /auto-retrain/status`
**Purpose**: Get status of retrain events
**Query Parameters**:
- `model_name` (optional): Filter by specific model

**Response Model**:
```typescript
interface RetrainStatusResponse {
  total_events: number;
  events: Array<{
    event_id: string;
    model_name: string;
    trigger: string;
    drift_score: number;
    threshold: number;
    status: string;
    timestamp: string;
    retrain_job_id?: string;
    error_message?: string;
  }>;
  last_retrain_times: Record<string, string>;
  daily_counts: Record<string, number>;
}
```

#### `GET /auto-retrain/status/{model_name}`
**Purpose**: Get retrain status for a specific model
**Path Parameters**:
- `model_name`: Name of the model

#### `POST /auto-retrain/check-drift/{model_name}`
**Purpose**: Manually check drift and trigger retraining if needed
**Path Parameters**:
- `model_name`: Name of the model to check

#### `GET /auto-retrain/events`
**Purpose**: Get retrain events with optional filtering
**Query Parameters**:
- `limit` (optional): Maximum number of events to return - default: 50
- `status` (optional): Filter by event status

#### `DELETE /auto-retrain/events/{event_id}`
**Purpose**: Cancel a retrain event
**Path Parameters**:
- `event_id`: ID of the event to cancel

#### `POST /auto-retrain/start-monitoring`
**Purpose**: Start the auto-retrain monitoring loop

#### `GET /auto-retrain/health`
**Purpose**: Health check for auto-retrain service

### Health and Status Endpoints

#### `GET /health`
**Purpose**: Service health check

**Response Model**:
```typescript
interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  service: string;
  timestamp: string;
  uptime_seconds: number;
  dependencies: {
    database: boolean;
    redis: boolean;
    drift_detector: boolean;
    auto_retrain: boolean;
  };
}
```

#### `GET /metrics`
**Purpose**: Prometheus metrics endpoint

### Frontend Integration Examples

#### Dashboard Component
```typescript
// React component for analytics dashboard
const AnalyticsDashboard = () => {
  const [trends, setTrends] = useState(null);
  const [driftAlerts, setDriftAlerts] = useState([]);
  const [retrainStatus, setRetrainStatus] = useState(null);

  useEffect(() => {
    // Load dashboard data
    Promise.all([
      fetch('/analytics/trends').then(r => r.json()),
      fetch('/drift/alerts?days=7').then(r => r.json()),
      fetch('/auto-retrain/status').then(r => r.json())
    ]).then(([trendsData, alertsData, retrainData]) => {
      setTrends(trendsData);
      setDriftAlerts(alertsData.alerts);
      setRetrainStatus(retrainData);
    });
  }, []);

  return (
    <div className="analytics-dashboard">
      <TrendsChart data={trends} />
      <DriftAlertsPanel alerts={driftAlerts} />
      <RetrainStatus status={retrainStatus} />
    </div>
  );
};
```

#### Drift Detection Component
```typescript
// Component for drift detection
const DriftDetection = () => {
  const [config, setConfig] = useState(null);
  const [history, setHistory] = useState([]);

  const updateConfig = async (newConfig) => {
    await fetch('/drift/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newConfig)
    });
    // Reload config
    const response = await fetch('/drift/config');
    setConfig(await response.json());
  };

  const runDriftTest = async () => {
    const response = await fetch('/drift/test-drift', {
      method: 'POST'
    });
    const results = await response.json();
    // Display results
  };

  return (
    <div className="drift-detection">
      <ConfigPanel config={config} onUpdate={updateConfig} />
      <HistoryChart data={history} />
      <TestButton onClick={runDriftTest} />
    </div>
  );
};
```

#### Model Performance Component
```typescript
// Component for model performance tracking
const ModelPerformance = () => {
  const [performance, setPerformance] = useState(null);

  const submitPerformance = async (data) => {
    await fetch('/model/performance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
  };

  return (
    <div className="model-performance">
      <PerformanceForm onSubmit={submitPerformance} />
      <PerformanceChart data={performance} />
    </div>
  );
};
```

## Architecture Deep Dive

### Service Components

#### 1. Analytics Engine
**Purpose**: Core analytics processing and data aggregation
**How it works**:
- Collects metrics from all platform services
- Performs statistical analysis and trend detection
- Generates insights and recommendations
- Provides real-time and historical analytics

#### 2. Drift Detection System
**Purpose**: Monitors data and model drift
**How it works**:
- Implements statistical tests (KS, Chi-square, PSI)
- Compares current data against reference datasets
- Detects performance degradation patterns
- Triggers alerts and automated responses

#### 3. Auto-Retrain Service
**Purpose**: Automated model retraining based on drift detection
**How it works**:
- Monitors drift thresholds and performance metrics
- Triggers retraining jobs when conditions are met
- Manages retraining queue and priorities
- Integrates with training service for model updates

#### 4. Performance Monitor
**Purpose**: Tracks model performance metrics
**How it works**:
- Collects prediction accuracy and confidence scores
- Monitors response times and throughput
- Tracks error rates and failure patterns
- Provides performance dashboards and reports

### Data Flow Architecture

```
All Services → Analytics Engine → Drift Detection → Auto-Retrain → Model Updates
     ↓              ↓                ↓              ↓              ↓
  Metrics       Aggregation      Statistical    Threshold      Training
     ↓              ↓                ↓              ↓              ↓
  Collection    Processing       Analysis       Monitoring      Service
     ↓              ↓                ↓              ↓              ↓
  Storage       Insights         Alerts         Triggers        Updates
```

## Configuration & Environment

### Environment Variables
```bash
# Service Configuration
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8001
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql://mlflow:password@postgres:5432/ml_security_consolidated
REDIS_URL=redis://redis:6379/1

# Drift Detection Configuration
DRIFT_THRESHOLD=0.5
DRIFT_CHECK_INTERVAL=3600

# Auto-Retrain Configuration
AUTO_RETRAIN_ENABLED=true
PERFORMANCE_THRESHOLD=0.8
RETRAIN_COOLDOWN=86400
```

## Security & Compliance

### Data Protection
- **Encryption**: All data encrypted in transit and at rest
- **Access Control**: Role-based access to analytics data
- **Audit Logging**: Comprehensive audit trail for all operations
- **Data Anonymization**: PII detection and anonymization support

### Compliance Features
- **GDPR Compliance**: Data privacy and right to be forgotten
- **SOC 2**: Security and availability controls
- **ISO 27001**: Information security management
- **NIST Framework**: Cybersecurity framework compliance

## Performance Characteristics

### Scalability
- **Horizontal Scaling**: Stateless design supports multiple instances
- **Database Optimization**: Connection pooling and query optimization
- **Caching Strategy**: Redis caching for frequently accessed data
- **Async Processing**: Non-blocking I/O for high throughput

### Reliability
- **Fault Tolerance**: Graceful degradation on service failures
- **Data Consistency**: ACID compliance for critical operations
- **Backup & Recovery**: Automated backup and point-in-time recovery
- **Circuit Breakers**: Protection against cascading failures

## Troubleshooting Guide

### Common Issues
1. **Database Connection Failures**: Check connection string and network
2. **Redis Timeouts**: Verify Redis configuration and memory usage
3. **Drift Detection Errors**: Check data quality and feature consistency
4. **Auto-Retrain Failures**: Verify training service connectivity

### Debug Commands
```bash
# Check service health
curl http://localhost:8001/health

# Test drift detection
curl -X POST http://localhost:8001/drift/test-drift

# Get recent drift results
curl http://localhost:8001/drift/recent-results?hours=24
```

## Future Enhancements

### Planned Features
- **ML-based Drift Detection**: Advanced ML algorithms for drift detection
- **Real-time Streaming**: Kafka integration for real-time data processing
- **Advanced Analytics**: Time series forecasting and anomaly detection
- **Custom Dashboards**: User-configurable analytics dashboards
- **Automated Remediation**: Self-healing systems based on analytics insights