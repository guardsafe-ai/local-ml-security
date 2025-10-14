# Analytics Service - API Reference

## Base URL
```
http://analytics:8006
```

## Authentication
All endpoints require authentication headers:
```javascript
headers: {
  'Authorization': 'Bearer <token>',
  'Content-Type': 'application/json'
}
```

## Endpoints

### Health & Status

#### `GET /`
**Purpose**: Root endpoint with service status

**Response**:
```typescript
interface RootResponse {
  service: string;
  version: string;
  status: string;
  description: string;
  timestamp: string;
  uptime_seconds: number;
}
```

**Example Response**:
```json
{
  "service": "analytics",
  "version": "1.0.0",
  "status": "running",
  "description": "Analytics service for ML security platform",
  "timestamp": "2025-01-09T10:30:00Z",
  "uptime_seconds": 3600
}
```

#### `GET /health`
**Purpose**: Comprehensive health check with dependencies

**Response**:
```typescript
interface HealthResponse {
  status: "healthy" | "unhealthy" | "degraded";
  service: string;
  timestamp: string;
  uptime_seconds: number;
  dependencies: {
    postgres: boolean;
    redis: boolean;
    training: boolean;
    model_api: boolean;
  };
  drift_detection: {
    has_reference_data: boolean;
    recent_checks: number;
    config: {
      ks_threshold: number;
      psi_threshold: number;
    };
  };
}
```

**Frontend Integration**:
```javascript
const checkHealth = async () => {
  try {
    const response = await fetch('/health');
    const health = await response.json();
    
    if (health.status === 'healthy') {
      console.log('✅ Analytics service is healthy');
      console.log(`📊 Recent drift checks: ${health.drift_detection.recent_checks}`);
    } else {
      console.error('❌ Analytics service is unhealthy:', health);
    }
  } catch (error) {
    console.error('Failed to check health:', error);
  }
};
```

#### `GET /metrics`
**Purpose**: Prometheus metrics endpoint

**Response**: Prometheus-formatted metrics
```
# HELP analytics_drift_detections_total Total number of drift detections
# TYPE analytics_drift_detections_total counter
analytics_drift_detections_total{model_name="bert-base",drift_type="data"} 25

# HELP analytics_model_evaluations_total Total number of model evaluations
# TYPE analytics_model_evaluations_total counter
analytics_model_evaluations_total{model_name="roberta-base",status="success"} 15
```

### Drift Detection

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

**Response**:
```typescript
interface DriftDetectionResponse {
  drift_detected: boolean;
  overall_drift_score: number;
  drifted_features: string[];
  drift_summary: {
    total_features: number;
    total_drifted_features: number;
    drift_percentage: number;
    severe_drift_features: string[];
    moderate_drift_features: string[];
    minor_drift_features: string[];
  };
  statistical_tests: Record<string, {
    feature: string;
    is_drifted: boolean;
    drift_severity: "none" | "minor" | "moderate" | "severe";
    ks_statistic: number;
    ks_pvalue: number;
    psi_value: number;
    reference_mean: number;
    current_mean: number;
  }>;
  timestamp: string;
}
```

**Frontend Usage**:
```javascript
const detectDataDrift = async (currentData, referenceData, featureColumns) => {
  try {
    const response = await fetch('/drift/data-drift', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        current_data: currentData,
        reference_data: referenceData,
        feature_columns: featureColumns
      })
    });
    
    const result = await response.json();
    console.log(`Drift detected: ${result.drift_detected}`);
    console.log(`Drifted features: ${result.drifted_features.join(', ')}`);
    return result;
  } catch (error) {
    console.error('Drift detection failed:', error);
  }
};
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

**Response**:
```typescript
interface ModelDriftResponse {
  model_drift_detected: boolean;
  performance_change: number;
  accuracy_change: number;
  f1_change: number;
  prediction_agreement: number;
  confidence_change: number;
  statistical_significance: {
    p_value: number;
    is_significant: boolean;
  };
  recommendations: string[];
  timestamp: string;
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

**Response**:
```typescript
interface SetReferenceResponse {
  message: string;
  samples: number;
  features: number;
  timestamp: string;
}
```

#### `GET /drift/history`
**Purpose**: Get drift detection history

**Query Parameters**:
- `days` (optional): Number of days to look back - default: 30, range: 1-365

**Response**:
```typescript
interface DriftHistoryResponse {
  period_days: number;
  total_checks: number;
  history: Array<{
    timestamp: string;
    model_name?: string;
    drift_detected: boolean;
    drift_score: number;
    drifted_features: string[];
    data_source: string;
  }>;
}
```

#### `GET /drift/summary`
**Purpose**: Get drift detection summary

**Query Parameters**:
- `days` (optional): Number of days to summarize - default: 30, range: 1-365

**Response**:
```typescript
interface DriftSummaryResponse {
  period_days: number;
  total_checks: number;
  drift_detection_rate: number;
  average_drift_score: number;
  most_drifted_features: Array<{
    feature: string;
    drift_count: number;
    average_psi: number;
  }>;
  model_performance: {
    models_checked: number;
    models_with_drift: number;
    average_performance_drop: number;
  };
}
```

#### `GET /drift/config`
**Purpose**: Get current drift detection configuration

**Response**:
```typescript
interface DriftConfigResponse {
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

**Response**:
```typescript
interface DriftConfigUpdateResponse {
  message: string;
  config: {
    ks_threshold: number;
    chi2_threshold: number;
    psi_threshold: number;
    accuracy_drop_threshold: number;
    f1_drop_threshold: number;
  };
}
```

#### `GET /drift/alerts`
**Purpose**: Get drift alerts for the last N days

**Query Parameters**:
- `days` (optional): Number of days to look back - default: 7, range: 1-30

**Response**:
```typescript
interface DriftAlertsResponse {
  period_days: number;
  total_alerts: number;
  alerts: Array<{
    type: "data_drift" | "model_drift";
    timestamp: string;
    severity: "low" | "medium" | "high";
    message: string;
    details: Record<string, any>;
  }>;
}
```

#### `POST /drift/test-drift`
**Purpose**: Test drift detection with sample data

**Response**:
```typescript
interface TestDriftResponse {
  message: string;
  test_results: {
    drift_detected: boolean;
    overall_drift_score: number;
    drifted_features: string[];
    drift_summary: Record<string, any>;
    statistical_tests: Record<string, any>;
  };
  reference_samples: number;
  current_samples: number;
}
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

**Response**:
```typescript
interface ModelPerformanceDriftResponse {
  model_performance_drift: {
    performance_improvement: boolean;
    accuracy_change: number;
    f1_change: number;
    precision_change: number;
    recall_change: number;
    statistical_significance: {
      p_value: number;
      is_significant: boolean;
    };
    recommendations: string[];
  };
  timestamp: string;
}
```

#### `POST /drift/check-and-retrain`
**Purpose**: Check for drift and automatically trigger retraining if needed

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

**Response**:
```typescript
interface CheckAndRetrainResponse {
  drift_detection: {
    drift_detected: boolean;
    overall_drift_score: number;
    drifted_features: string[];
  };
  model_performance_drift?: {
    performance_improvement: boolean;
    accuracy_change: number;
  };
  auto_retraining: {
    retrain_triggered: boolean;
    job_id?: string;
    reason?: string;
  };
  timestamp: string;
}
```

### Model Evaluation & Promotion

#### `POST /drift/evaluate-model`
**Purpose**: Evaluate a model for promotion from Staging to Production

**Request Body**:
```typescript
interface ModelEvaluationRequest {
  model_name: string;
  version: string;
  test_data?: Array<Record<string, any>>;
}
```

**Response**:
```typescript
interface ModelEvaluationResponse {
  evaluation: {
    status: "approved" | "rejected" | "needs_review";
    score: number;
    criteria_met: boolean;
    metrics: {
      accuracy: number;
      f1_score: number;
      precision: number;
      recall: number;
      auc_roc: number;
    };
    reasons: string[];
    recommendations: string[];
    timestamp: string;
  };
  timestamp: string;
}
```

**Frontend Usage**:
```javascript
const evaluateModel = async (modelName, version, testData) => {
  try {
    const response = await fetch('/drift/evaluate-model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_name: modelName,
        version: version,
        test_data: testData
      })
    });
    
    const result = await response.json();
    console.log(`Evaluation status: ${result.evaluation.status}`);
    console.log(`Score: ${result.evaluation.score}`);
    return result;
  } catch (error) {
    console.error('Model evaluation failed:', error);
  }
};
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

**Response**:
```typescript
interface ModelPromotionResponse {
  promotion: {
    status: "success" | "failed" | "skipped";
    message: string;
    model_name: string;
    version: string;
    previous_version?: string;
    evaluation_score?: number;
    timestamp: string;
  };
  timestamp: string;
}
```

#### `GET /drift/promotion-history`
**Purpose**: Get promotion evaluation history

**Query Parameters**:
- `model_name` (optional): Filter by specific model

**Response**:
```typescript
interface PromotionHistoryResponse {
  history: Array<{
    model_name: string;
    version: string;
    evaluation_score: number;
    status: string;
    timestamp: string;
    reasons: string[];
  }>;
  total_evaluations: number;
  timestamp: string;
}
```

#### `GET /drift/promotion-criteria`
**Purpose**: Get current model promotion criteria configuration

**Response**:
```typescript
interface PromotionCriteriaResponse {
  criteria: {
    performance: {
      min_accuracy_improvement: number;
      min_f1_improvement: number;
      min_precision_improvement: number;
      min_recall_improvement: number;
    };
    statistical: {
      max_p_value: number;
      min_sample_size: number;
    };
    drift: {
      max_psi_value: number;
      min_prediction_agreement: number;
    };
    confidence: {
      max_confidence_variance: number;
    };
  };
  timestamp: string;
}
```

### Auto-Retrain Management

#### `GET /auto-retrain/config`
**Purpose**: Get current auto-retrain configuration

**Response**:
```typescript
interface RetrainConfigResponse {
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

**Response**:
```typescript
interface RetrainConfigUpdateResponse {
  message: string;
  config: Record<string, any>;
}
```

#### `POST /auto-retrain/trigger/{model_name}`
**Purpose**: Manually trigger retraining for a specific model

**Path Parameters**:
- `model_name`: Name of the model to retrain

**Request Body**:
```typescript
interface ManualRetrainRequest {
  reason: string;
  priority: string;
}
```

**Response**:
```typescript
interface ManualRetrainResponse {
  message: string;
  event_id: string;
  job_id: string;
  status: string;
}
```

#### `GET /auto-retrain/status`
**Purpose**: Get status of retrain events

**Query Parameters**:
- `model_name` (optional): Filter by specific model

**Response**:
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

#### `POST /auto-retrain/check-drift/{model_name}`
**Purpose**: Manually check drift and trigger retraining if needed

**Path Parameters**:
- `model_name`: Name of the model to check

**Response**:
```typescript
interface CheckDriftResponse {
  message: string;
  event_id?: string;
  trigger?: string;
  job_id?: string;
  status?: string;
  drift_within_limits?: boolean;
}
```

#### `GET /auto-retrain/events`
**Purpose**: Get retrain events with optional filtering

**Query Parameters**:
- `limit` (optional): Maximum number of events to return - default: 50
- `status` (optional): Filter by event status

**Response**:
```typescript
interface RetrainEventsResponse {
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
  total_count: number;
}
```

#### `DELETE /auto-retrain/events/{event_id}`
**Purpose**: Cancel a retrain event (if still pending)

**Path Parameters**:
- `event_id`: Unique identifier for the event

**Response**:
```typescript
interface CancelEventResponse {
  message: string;
}
```

#### `POST /auto-retrain/start-monitoring`
**Purpose**: Start the auto-retrain monitoring loop

**Response**:
```typescript
interface StartMonitoringResponse {
  message: string;
}
```

#### `GET /auto-retrain/health`
**Purpose**: Health check for auto-retrain service

**Response**:
```typescript
interface AutoRetrainHealthResponse {
  status: "healthy" | "unhealthy";
  service: string;
  timestamp: string;
  config: {
    target_models: string[];
    data_drift_threshold: number;
    model_drift_threshold: number;
    performance_drop_threshold: number;
  };
  stats: {
    total_events: number;
    active_events: number;
    completed_events: number;
    failed_events: number;
  };
}
```

### Analytics & Performance

#### `GET /analytics/trends`
**Purpose**: Get performance trends over time

**Query Parameters**:
- `days` (optional): Number of days to analyze - default: 30

**Response**:
```typescript
interface PerformanceTrendsResponse {
  period_days: number;
  trends: {
    accuracy_trend: Array<{
      date: string;
      value: number;
    }>;
    f1_trend: Array<{
      date: string;
      value: number;
    }>;
    drift_trend: Array<{
      date: string;
      value: number;
    }>;
  };
  summary: {
    average_accuracy: number;
    average_f1: number;
    average_drift_score: number;
    trend_direction: "improving" | "stable" | "declining";
  };
}
```

#### `POST /drift/comprehensive-metrics`
**Purpose**: Calculate comprehensive performance metrics for model evaluation

**Request Body**:
```typescript
interface ComprehensiveMetricsRequest {
  y_true: string[];
  y_pred: string[];
  y_prob?: number[];
  model_name: string;
}
```

**Response**:
```typescript
interface ComprehensiveMetricsResponse {
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    auc_roc: number;
    confusion_matrix: number[][];
    classification_report: Record<string, any>;
    statistical_tests: Record<string, any>;
  };
  timestamp: string;
}
```

### Production Data & Monitoring

#### `GET /drift/data/production-inference`
**Purpose**: Get production inference data for retraining

**Query Parameters**:
- `hours` (optional): Hours of data to retrieve - default: 24
- `model_name` (optional): Specific model to filter by

**Response**:
```typescript
interface ProductionDataResponse {
  s3_path: string;
  local_path: string;
  sample_count: number;
  timestamp: string;
  model_name?: string;
}
```

#### `POST /drift/production-data`
**Purpose**: Detect drift using actual production inference data

**Query Parameters**:
- `model_name`: Model name to check for drift
- `hours` (optional): Hours of production data to analyze - default: 24

**Response**:
```typescript
interface ProductionDriftResponse {
  drift_detected: boolean;
  overall_drift_score: number;
  drifted_features: string[];
  drift_summary: Record<string, any>;
  statistical_tests: Record<string, any>;
  data_source: string;
  timestamp: string;
}
```

### Email Notifications

#### `POST /drift/test-email`
**Purpose**: Test email notification service

**Response**:
```typescript
interface TestEmailResponse {
  status: string;
  message: string;
  drift_alert_sent: boolean;
  performance_alert_sent: boolean;
  dummy_mode: boolean;
  recipients: string[];
  timestamp: string;
}
```

### Recent Results

#### `GET /drift/recent-results`
**Purpose**: Get recent drift detection results for auto-retrain service

**Query Parameters**:
- `model_name` (optional): Model name to filter by
- `hours` (optional): Hours to look back for results - default: 24

**Response**:
```typescript
interface RecentResultsResponse {
  data_drift_score: number;
  model_drift_score: number;
  performance_drop: number;
  last_check: string | null;
  total_checks: number;
  model_name?: string;
  hours_looked_back: number;
}
```

## Error Handling

### Common Error Responses

#### 400 Bad Request
```json
{
  "error": true,
  "status_code": 400,
  "message": "Validation error",
  "details": [
    {
      "loc": ["body", "current_data"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ],
  "path": "/drift/data-drift",
  "timestamp": "2025-01-09T10:30:00Z"
}
```

#### 404 Not Found
```json
{
  "error": true,
  "status_code": 404,
  "message": "Model not found",
  "path": "/drift/evaluate-model",
  "timestamp": "2025-01-09T10:30:00Z"
}
```

#### 500 Internal Server Error
```json
{
  "error": true,
  "status_code": 500,
  "message": "Drift detection failed: Insufficient data",
  "path": "/drift/data-drift",
  "timestamp": "2025-01-09T10:30:00Z"
}
```

### Frontend Error Handling

```javascript
const handleApiError = (error, response) => {
  if (response?.status === 400) {
    console.error('Validation error:', response.data.details);
    // Show validation errors to user
  } else if (response?.status === 404) {
    console.error('Resource not found:', response.data.message);
    // Show not found message
  } else if (response?.status === 500) {
    console.error('Server error:', response.data.message);
    // Show generic error message
  } else {
    console.error('Unknown error:', error);
    // Show generic error message
  }
};
```

## Rate Limiting

The Analytics Service implements rate limiting to prevent abuse:

- **Drift Detection**: 20 requests per minute per IP
- **Model Evaluation**: 10 requests per minute per IP
- **Auto-Retrain**: 5 requests per minute per IP
- **Configuration Updates**: 10 requests per minute per IP

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 20
X-RateLimit-Remaining: 15
X-RateLimit-Reset: 1641234567
```

## WebSocket Support

For real-time updates on drift detection and retraining events, the service supports WebSocket connections:

```javascript
const ws = new WebSocket('ws://analytics:8006/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'drift_detected') {
    console.log(`Drift detected in model ${data.model_name}: ${data.drift_score}`);
  } else if (data.type === 'retrain_completed') {
    console.log(`Retraining completed for model ${data.model_name}`);
  }
};
```

## Request/Response Examples

### Data Drift Detection
```bash
curl -X POST http://analytics:8006/drift/data-drift \
  -H "Content-Type: application/json" \
  -d '{
    "current_data": [
      {"text": "Sample text 1", "length": 12, "word_count": 2},
      {"text": "Sample text 2", "length": 15, "word_count": 3}
    ],
    "reference_data": [
      {"text": "Reference text 1", "length": 10, "word_count": 2},
      {"text": "Reference text 2", "length": 14, "word_count": 3}
    ],
    "feature_columns": ["length", "word_count"]
  }'
```

### Model Evaluation
```bash
curl -X POST http://analytics:8006/drift/evaluate-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "security-classifier",
    "version": "2",
    "test_data": [
      {"text": "Test input 1", "label": "prompt_injection"},
      {"text": "Test input 2", "label": "benign"}
    ]
  }'
```

### Auto-Retrain Configuration
```bash
curl -X PUT http://analytics:8006/auto-retrain/config \
  -H "Content-Type: application/json" \
  -d '{
    "data_drift_threshold": 0.2,
    "model_drift_threshold": 0.15,
    "performance_drop_threshold": 0.05,
    "target_models": ["bert-base", "roberta-base"]
  }'
```

---

**Analytics Service API** - Complete reference for all endpoints, request/response schemas, and integration examples for the ML Security Analytics Service.
