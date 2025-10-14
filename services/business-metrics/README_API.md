# Business Metrics Service - API Reference

## Base URL
```
http://business-metrics:8004
```

## Authentication
All endpoints require authentication headers:
```javascript
headers: {
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
  "service": "business-metrics",
  "version": "1.0.0",
  "status": "running",
  "description": "Business metrics collection and analytics service",
  "timestamp": "2025-01-09T10:30:00Z",
  "uptime_seconds": 3600
}
```

#### `GET /health`
**Purpose**: Comprehensive health check with system status

**Response**:
```typescript
interface SystemHealthResponse {
  status: "healthy" | "unhealthy" | "degraded";
  timestamp: string;
  uptime_seconds: number;
  metrics_collected: number;
  database_status: "connected" | "disconnected" | "error";
  cache_status: "connected" | "disconnected" | "error";
  performance_metrics: {
    avg_response_time: number;
    requests_per_second: number;
    error_rate: number;
    memory_usage_mb: number;
    cpu_usage_percent: number;
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
      console.log('✅ Business Metrics service is healthy');
      console.log(`📊 Metrics collected: ${health.metrics_collected}`);
      console.log(`⚡ Avg response time: ${health.performance_metrics.avg_response_time}ms`);
    } else {
      console.error('❌ Business Metrics service is unhealthy:', health);
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
# HELP business_metrics_total Total number of metrics collected
# TYPE business_metrics_total counter
business_metrics_total{service="business-metrics"} 15000

# HELP business_metrics_response_time_seconds Response time in seconds
# TYPE business_metrics_response_time_seconds histogram
business_metrics_response_time_seconds_bucket{le="0.1"} 1200
business_metrics_response_time_seconds_bucket{le="0.5"} 1400
```

### Metrics Collection

#### `POST /metrics`
**Purpose**: Record business metrics (single or batch)

**Request Body** (Single Metric):
```typescript
interface MetricData {
  metric_name: string;
  value: number;
  timestamp: string;
  tags?: Record<string, string>;
  metadata?: Record<string, any>;
}
```

**Request Body** (Batch Metrics):
```typescript
interface BatchMetricsRequest {
  metrics: MetricData[];
}
```

**Response**:
```typescript
interface MetricsResponse {
  status: "success" | "error";
  message: string;
  metrics_recorded: number;
  timestamp: string;
}
```

**Frontend Usage**:
```javascript
// Record single metric
const recordMetric = async (metricName, value, tags = {}) => {
  try {
    const response = await fetch('/metrics', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        metric_name: metricName,
        value: value,
        timestamp: new Date().toISOString(),
        tags: tags
      })
    });
    
    const result = await response.json();
    console.log(`Metric recorded: ${result.status}`);
    return result;
  } catch (error) {
    console.error('Failed to record metric:', error);
  }
};

// Record batch metrics
const recordBatchMetrics = async (metrics) => {
  try {
    const response = await fetch('/metrics', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ metrics: metrics })
    });
    
    const result = await response.json();
    console.log(`Batch metrics recorded: ${result.metrics_recorded}`);
    return result;
  } catch (error) {
    console.error('Failed to record batch metrics:', error);
  }
};
```

**Example Request**:
```json
{
  "metric_name": "model_inference_latency",
  "value": 85.2,
  "timestamp": "2025-01-09T10:30:00Z",
  "tags": {
    "model_name": "bert-base",
    "environment": "production",
    "region": "us-east-1"
  },
  "metadata": {
    "request_id": "req_123",
    "user_id": "user_456",
    "session_id": "session_789"
  }
}
```

#### `GET /metrics`
**Purpose**: Retrieve business metrics with filtering

**Query Parameters**:
- `metric_name` (optional): Filter by specific metric name
- `start_time` (optional): Start time for filtering (ISO 8601)
- `end_time` (optional): End time for filtering (ISO 8601)
- `limit` (optional): Maximum number of metrics to return - default: 1000
- `tags` (optional): Filter by tags (JSON string)
- `aggregation` (optional): Aggregation method (sum, avg, min, max, count)

**Response**:
```typescript
interface BusinessMetricsResponse {
  metrics: Array<{
    id: number;
    metric_name: string;
    value: number;
    timestamp: string;
    tags: Record<string, string>;
    metadata: Record<string, any>;
    created_at: string;
  }>;
  total_count: number;
  time_range: {
    start: string;
    end: string;
  };
  aggregations: {
    sum?: number;
    average?: number;
    minimum?: number;
    maximum?: number;
    count?: number;
  };
}
```

**Frontend Usage**:
```javascript
// Get metrics with filtering
const getMetrics = async (filters = {}) => {
  try {
    const params = new URLSearchParams();
    if (filters.metricName) params.append('metric_name', filters.metricName);
    if (filters.startTime) params.append('start_time', filters.startTime);
    if (filters.endTime) params.append('end_time', filters.endTime);
    if (filters.limit) params.append('limit', filters.limit);
    if (filters.tags) params.append('tags', JSON.stringify(filters.tags));
    if (filters.aggregation) params.append('aggregation', filters.aggregation);
    
    const response = await fetch(`/metrics?${params.toString()}`);
    const data = await response.json();
    
    console.log(`Retrieved ${data.total_count} metrics`);
    return data;
  } catch (error) {
    console.error('Failed to get metrics:', error);
  }
};
```

#### `GET /metrics/summary`
**Purpose**: Get metrics summary with caching

**Query Parameters**:
- `metric_name` (optional): Filter by specific metric name
- `time_range` (optional): Time range for summary (1h, 24h, 7d, 30d) - default: 24h
- `granularity` (optional): Summary granularity (1m, 5m, 1h, 1d) - default: 1h

**Response**:
```typescript
interface MetricsSummaryResponse {
  metric_name?: string;
  time_range: string;
  granularity: string;
  summary: {
    count: number;
    sum: number;
    average: number;
    minimum: number;
    maximum: number;
    standard_deviation: number;
    median: number;
    p95: number;
    p99: number;
  };
  time_series: Array<{
    timestamp: string;
    count: number;
    sum: number;
    average: number;
    minimum: number;
    maximum: number;
  }>;
  trends: {
    direction: "increasing" | "decreasing" | "stable";
    change_percent: number;
    change_value: number;
  };
  cached: boolean;
  cache_ttl: number;
  timestamp: string;
}
```

**Frontend Usage**:
```javascript
// Get metrics summary
const getMetricsSummary = async (metricName, timeRange = '24h') => {
  try {
    const response = await fetch(
      `/metrics/summary?metric_name=${metricName}&time_range=${timeRange}`
    );
    const summary = await response.json();
    
    console.log(`Average ${metricName}: ${summary.summary.average}`);
    console.log(`Trend: ${summary.trends.direction} (${summary.trends.change_percent}%)`);
    return summary;
  } catch (error) {
    console.error('Failed to get metrics summary:', error);
  }
};
```

### Advanced Analytics

#### `GET /metrics/trends`
**Purpose**: Get trend analysis for metrics

**Query Parameters**:
- `metric_name`: Name of the metric to analyze
- `time_range` (optional): Time range for analysis - default: 24h
- `granularity` (optional): Analysis granularity - default: 1h
- `trend_type` (optional): Type of trend analysis (linear, exponential, polynomial) - default: linear

**Response**:
```typescript
interface TrendsResponse {
  metric_name: string;
  time_range: string;
  granularity: string;
  trend_analysis: {
    direction: "increasing" | "decreasing" | "stable";
    strength: number; // 0-1, correlation coefficient
    slope: number;
    r_squared: number;
    p_value: number;
    is_significant: boolean;
  };
  predictions: Array<{
    timestamp: string;
    predicted_value: number;
    confidence_interval: {
      lower: number;
      upper: number;
    };
  }>;
  anomalies: Array<{
    timestamp: string;
    value: number;
    expected_value: number;
    deviation: number;
    severity: "low" | "medium" | "high";
  }>;
  timestamp: string;
}
```

#### `GET /metrics/correlation`
**Purpose**: Get correlation analysis between metrics

**Query Parameters**:
- `metric_names`: Comma-separated list of metric names
- `time_range` (optional): Time range for analysis - default: 24h
- `correlation_type` (optional): Type of correlation (pearson, spearman, kendall) - default: pearson

**Response**:
```typescript
interface CorrelationResponse {
  metric_names: string[];
  time_range: string;
  correlation_matrix: Record<string, Record<string, number>>;
  strongest_correlations: Array<{
    metric1: string;
    metric2: string;
    correlation: number;
    significance: number;
    is_significant: boolean;
  }>;
  insights: string[];
  timestamp: string;
}
```

#### `GET /metrics/performance`
**Purpose**: Get performance metrics for the service

**Response**:
```typescript
interface PerformanceResponse {
  service_metrics: {
    uptime_seconds: number;
    requests_total: number;
    requests_per_second: number;
    avg_response_time_ms: number;
    p95_response_time_ms: number;
    p99_response_time_ms: number;
    error_rate: number;
    success_rate: number;
  };
  resource_metrics: {
    memory_usage_mb: number;
    memory_usage_percent: number;
    cpu_usage_percent: number;
    disk_usage_mb: number;
    disk_usage_percent: number;
  };
  database_metrics: {
    connection_pool_size: number;
    active_connections: number;
    query_count: number;
    avg_query_time_ms: number;
    slow_queries: number;
  };
  cache_metrics: {
    hit_rate: number;
    miss_rate: number;
    eviction_count: number;
    memory_usage_mb: number;
  };
  timestamp: string;
}
```

### KPI Management

#### `GET /kpis`
**Purpose**: Get key performance indicators

**Query Parameters**:
- `time_range` (optional): Time range for KPIs - default: 24h
- `category` (optional): Filter by KPI category

**Response**:
```typescript
interface KPIsResponse {
  time_range: string;
  kpis: Array<{
    name: string;
    value: number;
    target: number;
    unit: string;
    status: "excellent" | "good" | "warning" | "critical";
    trend: "up" | "down" | "stable";
    change_percent: number;
    description: string;
    category: string;
  }>;
  overall_score: number;
  timestamp: string;
}
```

#### `POST /kpis/calculate`
**Purpose**: Calculate KPIs from metrics

**Request Body**:
```typescript
interface KPICalculationRequest {
  kpi_name: string;
  metric_names: string[];
  calculation_method: "sum" | "average" | "count" | "max" | "min" | "custom";
  time_range: string;
  custom_formula?: string;
}
```

**Response**:
```typescript
interface KPICalculationResponse {
  kpi_name: string;
  value: number;
  calculation_details: {
    method: string;
    metric_values: Record<string, number>;
    formula_used: string;
  };
  timestamp: string;
}
```

### Data Management

#### `DELETE /metrics/cleanup`
**Purpose**: Clean up old metrics data

**Query Parameters**:
- `older_than_days` (optional): Delete metrics older than N days - default: 90
- `dry_run` (optional): Preview what would be deleted without actually deleting - default: false

**Response**:
```typescript
interface CleanupResponse {
  status: "success" | "error";
  message: string;
  metrics_deleted: number;
  space_freed_mb: number;
  dry_run: boolean;
  timestamp: string;
}
```

#### `POST /metrics/export`
**Purpose**: Export metrics data

**Request Body**:
```typescript
interface ExportRequest {
  metric_names?: string[];
  start_time?: string;
  end_time?: string;
  format: "json" | "csv" | "xlsx";
  include_metadata?: boolean;
}
```

**Response**:
```typescript
interface ExportResponse {
  status: "success" | "error";
  message: string;
  download_url: string;
  file_size_mb: number;
  records_exported: number;
  expires_at: string;
  timestamp: string;
}
```

### Real-time Monitoring

#### `GET /metrics/realtime`
**Purpose**: Get real-time metrics stream

**Query Parameters**:
- `metric_names` (optional): Comma-separated list of metric names
- `window_seconds` (optional): Time window for real-time data - default: 60

**Response**:
```typescript
interface RealtimeResponse {
  window_seconds: number;
  current_time: string;
  metrics: Array<{
    metric_name: string;
    current_value: number;
    previous_value: number;
    change_percent: number;
    trend: "up" | "down" | "stable";
    rate_per_second: number;
  }>;
  alerts: Array<{
    metric_name: string;
    alert_type: "threshold" | "anomaly" | "trend";
    severity: "low" | "medium" | "high" | "critical";
    message: string;
    timestamp: string;
  }>;
  timestamp: string;
}
```

#### `GET /metrics/alerts`
**Purpose**: Get active alerts

**Query Parameters**:
- `severity` (optional): Filter by alert severity
- `metric_name` (optional): Filter by metric name
- `active_only` (optional): Show only active alerts - default: true

**Response**:
```typescript
interface AlertsResponse {
  alerts: Array<{
    id: string;
    metric_name: string;
    alert_type: string;
    severity: "low" | "medium" | "high" | "critical";
    message: string;
    threshold?: number;
    current_value?: number;
    status: "active" | "resolved" | "acknowledged";
    created_at: string;
    updated_at: string;
  }>;
  total_count: number;
  active_count: number;
  timestamp: string;
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
      "loc": ["body", "metric_name"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ],
  "path": "/metrics",
  "timestamp": "2025-01-09T10:30:00Z"
}
```

#### 404 Not Found
```json
{
  "error": true,
  "status_code": 404,
  "message": "Metric not found",
  "path": "/metrics/summary",
  "timestamp": "2025-01-09T10:30:00Z"
}
```

#### 500 Internal Server Error
```json
{
  "error": true,
  "status_code": 500,
  "message": "Failed to process metrics",
  "path": "/metrics",
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

The Business Metrics Service implements rate limiting to prevent abuse:

- **Metrics Collection**: 1000 requests per minute per IP
- **Metrics Retrieval**: 100 requests per minute per IP
- **Analytics Queries**: 50 requests per minute per IP
- **Export Operations**: 10 requests per minute per IP

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 850
X-RateLimit-Reset: 1641234567
```

## WebSocket Support

For real-time metrics updates, the service supports WebSocket connections:

```javascript
const ws = new WebSocket('ws://business-metrics:8004/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'metric_update') {
    console.log(`Metric ${data.metric_name} updated: ${data.value}`);
  } else if (data.type === 'alert') {
    console.log(`Alert: ${data.message}`);
  }
};
```

## Request/Response Examples

### Record Single Metric
```bash
curl -X POST http://business-metrics:8004/metrics \
  -H "Content-Type: application/json" \
  -d '{
    "metric_name": "model_accuracy",
    "value": 0.95,
    "timestamp": "2025-01-09T10:30:00Z",
    "tags": {
      "model_name": "bert-base",
      "environment": "production"
    }
  }'
```

### Get Metrics Summary
```bash
curl "http://business-metrics:8004/metrics/summary?metric_name=model_accuracy&time_range=24h"
```

### Get Performance Metrics
```bash
curl http://business-metrics:8004/metrics/performance
```

### Export Metrics Data
```bash
curl -X POST http://business-metrics:8004/metrics/export \
  -H "Content-Type: application/json" \
  -d '{
    "metric_names": ["model_accuracy", "model_throughput"],
    "start_time": "2025-01-01T00:00:00Z",
    "end_time": "2025-01-09T23:59:59Z",
    "format": "csv"
  }'
```

---

**Business Metrics Service API** - Complete reference for all endpoints, request/response schemas, and integration examples for the ML Security Business Metrics Service.
