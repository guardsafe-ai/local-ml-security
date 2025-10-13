# ML Security Business Metrics Service

## Service Architecture & Purpose

### Core Purpose
The Business Metrics Service is the **enterprise-grade metrics collection and business intelligence engine** of the ML Security platform. It provides comprehensive KPI tracking, operational metrics, and business analytics for ML operations and security performance.

### Why This Service Exists
- **KPI Tracking**: Monitor key performance indicators across all platform services
- **Business Intelligence**: Provide insights for decision-making and resource allocation
- **Operational Metrics**: Track system performance, usage patterns, and efficiency
- **Cost Management**: Monitor resource usage and operational costs
- **Compliance Reporting**: Generate metrics for regulatory and compliance requirements

## Complete API Documentation for Frontend Development

### Base URL
```
http://business-metrics:8004
```

### Authentication
All endpoints require authentication headers:
```javascript
headers: {
  'Authorization': 'Bearer <token>',
  'Content-Type': 'application/json'
}
```

### Core Metrics Endpoints

#### `POST /metrics`
**Purpose**: Record business metrics and KPIs
**Request Body**:
```typescript
interface MetricData {
  metric_name: string;
  value: number;
  unit: string;
  category: 'performance' | 'security' | 'cost' | 'usage' | 'compliance';
  service: string;
  model_name?: string;
  user_id?: string;
  session_id?: string;
  metadata?: Record<string, any>;
  timestamp?: string;
}
```

**Frontend Usage**:
```javascript
const metricData = {
  metric_name: 'prediction_accuracy',
  value: 0.95,
  unit: 'percentage',
  category: 'performance',
  service: 'model-api',
  model_name: 'bert-base',
  user_id: 'user_123',
  metadata: {
    confidence_threshold: 0.8,
    batch_size: 32
  },
  timestamp: new Date().toISOString()
};

const response = await fetch('/metrics', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(metricData)
});

const result = await response.json();
console.log(`Metric recorded: ${result.message}`);
```

**Response Model**:
```typescript
interface MetricResponse {
  success: boolean;
  message: string;
  metric_id: string;
  timestamp: string;
}
```

#### `GET /metrics`
**Purpose**: Retrieve metrics with filtering and aggregation
**Query Parameters**:
- `metric_name` (optional): Filter by specific metric name
- `category` (optional): Filter by category (performance, security, cost, usage, compliance)
- `service` (optional): Filter by service name
- `model_name` (optional): Filter by model name
- `start_time` (optional): Start time for time range (ISO 8601)
- `end_time` (optional): End time for time range (ISO 8601)
- `aggregation` (optional): Aggregation method (sum, avg, min, max, count)
- `group_by` (optional): Group by field (service, model_name, category)
- `limit` (optional): Maximum number of records - default: 1000
- `offset` (optional): Pagination offset - default: 0

**Frontend Usage**:
```javascript
// Get all performance metrics for last 24 hours
const response = await fetch('/metrics?category=performance&start_time=2025-01-01T00:00:00Z&end_time=2025-01-02T00:00:00Z');
const metrics = await response.json();

// Get aggregated accuracy metrics by model
const response = await fetch('/metrics?metric_name=prediction_accuracy&aggregation=avg&group_by=model_name');
const accuracyByModel = await response.json();

// Get cost metrics for specific service
const response = await fetch('/metrics?category=cost&service=training&aggregation=sum');
const trainingCosts = await response.json();
```

**Response Model**:
```typescript
interface MetricsResponse {
  metrics: Array<{
    metric_id: string;
    metric_name: string;
    value: number;
    unit: string;
    category: string;
    service: string;
    model_name?: string;
    user_id?: string;
    session_id?: string;
    metadata?: Record<string, any>;
    timestamp: string;
  }>;
  total_count: number;
  has_more: boolean;
  aggregation?: {
    method: string;
    value: number;
    group_by?: string;
  };
  time_range?: {
    start: string;
    end: string;
  };
}
```

#### `GET /metrics/summary`
**Purpose**: Get cached metrics summary for dashboard
**Query Parameters**:
- `time_range` (optional): Time range (1h, 24h, 7d, 30d) - default: 24h
- `category` (optional): Filter by category

**Frontend Usage**:
```javascript
// Get 24-hour summary
const response = await fetch('/metrics/summary?time_range=24h');
const summary = await response.json();

// Get performance metrics summary
const response = await fetch('/metrics/summary?category=performance&time_range=7d');
const performanceSummary = await response.json();
```

**Response Model**:
```typescript
interface MetricsSummary {
  time_range: string;
  total_metrics: number;
  categories: {
    performance: {
      count: number;
      avg_value: number;
      top_metrics: Array<{
        metric_name: string;
        value: number;
        trend: 'up' | 'down' | 'stable';
      }>;
    };
    security: {
      count: number;
      avg_value: number;
      top_metrics: Array<{
        metric_name: string;
        value: number;
        trend: 'up' | 'down' | 'stable';
      }>;
    };
    cost: {
      count: number;
      total_value: number;
      top_services: Array<{
        service: string;
        cost: number;
        trend: 'up' | 'down' | 'stable';
      }>;
    };
    usage: {
      count: number;
      total_requests: number;
      top_services: Array<{
        service: string;
        requests: number;
        trend: 'up' | 'down' | 'stable';
      }>;
    };
  };
  trends: Array<{
    metric_name: string;
    current_value: number;
    previous_value: number;
    change_percent: number;
    trend: 'up' | 'down' | 'stable';
  }>;
  alerts: Array<{
    metric_name: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    message: string;
    threshold: number;
    current_value: number;
  }>;
  last_updated: string;
}
```

### Performance Metrics Endpoints

#### `GET /metrics/performance`
**Purpose**: Get performance-specific metrics
**Query Parameters**:
- `model_name` (optional): Filter by model name
- `service` (optional): Filter by service
- `time_range` (optional): Time range - default: 24h
- `aggregation` (optional): Aggregation method - default: avg

**Frontend Usage**:
```javascript
// Get performance metrics for specific model
const response = await fetch('/metrics/performance?model_name=bert-base&time_range=7d');
const modelPerformance = await response.json();

// Get overall performance metrics
const response = await fetch('/metrics/performance?aggregation=avg&time_range=24h');
const overallPerformance = await response.json();
```

**Response Model**:
```typescript
interface PerformanceMetrics {
  accuracy: {
    current: number;
    average: number;
    trend: 'up' | 'down' | 'stable';
    by_model: Record<string, number>;
  };
  latency: {
    current_ms: number;
    average_ms: number;
    p95_ms: number;
    p99_ms: number;
    trend: 'up' | 'down' | 'stable';
  };
  throughput: {
    requests_per_second: number;
    predictions_per_second: number;
    trend: 'up' | 'down' | 'stable';
  };
  error_rate: {
    current: number;
    average: number;
    trend: 'up' | 'down' | 'stable';
    by_service: Record<string, number>;
  };
  model_performance: Array<{
    model_name: string;
    accuracy: number;
    latency_ms: number;
    throughput: number;
    error_rate: number;
    last_updated: string;
  }>;
}
```

### Security Metrics Endpoints

#### `GET /metrics/security`
**Purpose**: Get security-specific metrics
**Query Parameters**:
- `time_range` (optional): Time range - default: 24h
- `threat_type` (optional): Filter by threat type
- `severity` (optional): Filter by severity level

**Frontend Usage**:
```javascript
// Get security metrics for last 7 days
const response = await fetch('/metrics/security?time_range=7d');
const securityMetrics = await response.json();

// Get high severity threats
const response = await fetch('/metrics/security?severity=high&time_range=24h');
const highSeverityThreats = await response.json();
```

**Response Model**:
```typescript
interface SecurityMetrics {
  threat_detection: {
    total_threats: number;
    threats_blocked: number;
    false_positives: number;
    false_negatives: number;
    detection_rate: number;
    accuracy: number;
  };
  threat_types: Array<{
    type: string;
    count: number;
    severity: 'low' | 'medium' | 'high' | 'critical';
    trend: 'up' | 'down' | 'stable';
  }>;
  attack_vectors: Array<{
    vector: string;
    count: number;
    success_rate: number;
    trend: 'up' | 'down' | 'stable';
  }>;
  compliance: {
    gdpr_compliance: number;
    soc2_compliance: number;
    iso27001_compliance: number;
    overall_score: number;
  };
  incidents: Array<{
    incident_id: string;
    type: string;
    severity: string;
    status: string;
    detected_at: string;
    resolved_at?: string;
  }>;
}
```

### Cost Metrics Endpoints

#### `GET /metrics/cost`
**Purpose**: Get cost and resource usage metrics
**Query Parameters**:
- `time_range` (optional): Time range - default: 30d
- `service` (optional): Filter by service
- `resource_type` (optional): Filter by resource type (compute, storage, network)

**Frontend Usage**:
```javascript
// Get cost metrics for last 30 days
const response = await fetch('/metrics/cost?time_range=30d');
const costMetrics = await response.json();

// Get compute costs by service
const response = await fetch('/metrics/cost?resource_type=compute&group_by=service');
const computeCosts = await response.json();
```

**Response Model**:
```typescript
interface CostMetrics {
  total_cost: number;
  cost_by_service: Record<string, number>;
  cost_by_resource: {
    compute: number;
    storage: number;
    network: number;
    other: number;
  };
  cost_trends: Array<{
    date: string;
    total_cost: number;
    compute_cost: number;
    storage_cost: number;
    network_cost: number;
  }>;
  resource_usage: {
    cpu_hours: number;
    memory_gb_hours: number;
    storage_gb: number;
    network_gb: number;
  };
  cost_efficiency: {
    cost_per_prediction: number;
    cost_per_accuracy_point: number;
    cost_per_threat_detected: number;
  };
  budget_status: {
    monthly_budget: number;
    current_spend: number;
    projected_spend: number;
    budget_remaining: number;
    alert_threshold: number;
  };
}
```

### Usage Metrics Endpoints

#### `GET /metrics/usage`
**Purpose**: Get usage and activity metrics
**Query Parameters**:
- `time_range` (optional): Time range - default: 24h
- `service` (optional): Filter by service
- `user_id` (optional): Filter by user

**Frontend Usage**:
```javascript
// Get usage metrics for last 24 hours
const response = await fetch('/metrics/usage?time_range=24h');
const usageMetrics = await response.json();

// Get usage by service
const response = await fetch('/metrics/usage?group_by=service&time_range=7d');
const usageByService = await response.json();
```

**Response Model**:
```typescript
interface UsageMetrics {
  total_requests: number;
  requests_by_service: Record<string, number>;
  requests_by_user: Record<string, number>;
  peak_usage: {
    time: string;
    requests: number;
  };
  average_usage: {
    requests_per_hour: number;
    requests_per_user: number;
  };
  user_activity: Array<{
    user_id: string;
    requests: number;
    last_activity: string;
    active_sessions: number;
  }>;
  service_activity: Array<{
    service: string;
    requests: number;
    avg_response_time: number;
    error_rate: number;
    uptime: number;
  }>;
  feature_usage: Array<{
    feature: string;
    usage_count: number;
    unique_users: number;
    trend: 'up' | 'down' | 'stable';
  }>;
}
```

### Compliance Metrics Endpoints

#### `GET /metrics/compliance`
**Purpose**: Get compliance and regulatory metrics
**Query Parameters**:
- `framework` (optional): Filter by compliance framework (gdpr, soc2, iso27001, nist)
- `time_range` (optional): Time range - default: 30d

**Frontend Usage**:
```javascript
// Get GDPR compliance metrics
const response = await fetch('/metrics/compliance?framework=gdpr&time_range=30d');
const gdprMetrics = await response.json();

// Get overall compliance status
const response = await fetch('/metrics/compliance?time_range=7d');
const complianceStatus = await response.json();
```

**Response Model**:
```typescript
interface ComplianceMetrics {
  overall_score: number;
  frameworks: {
    gdpr: {
      score: number;
      status: 'compliant' | 'non_compliant' | 'partial';
      issues: Array<{
        issue: string;
        severity: 'low' | 'medium' | 'high';
        status: 'open' | 'resolved';
      }>;
    };
    soc2: {
      score: number;
      status: 'compliant' | 'non_compliant' | 'partial';
      controls: Array<{
        control: string;
        status: 'implemented' | 'partial' | 'not_implemented';
        evidence: string;
      }>;
    };
    iso27001: {
      score: number;
      status: 'compliant' | 'non_compliant' | 'partial';
      controls: Array<{
        control: string;
        status: 'implemented' | 'partial' | 'not_implemented';
        evidence: string;
      }>;
    };
  };
  audit_trail: Array<{
    action: string;
    user: string;
    timestamp: string;
    details: string;
  }>;
  data_protection: {
    data_encrypted: number;
    data_anonymized: number;
    pii_detected: number;
    pii_removed: number;
  };
  access_control: {
    total_users: number;
    active_users: number;
    failed_logins: number;
    privilege_escalations: number;
  };
}
```

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
  };
  metrics: {
    total_metrics: number;
    metrics_per_second: number;
    cache_hit_rate: number;
  };
}
```

#### `GET /metrics/prometheus`
**Purpose**: Prometheus metrics endpoint

### Frontend Integration Examples

#### Business Dashboard Component
```typescript
// React component for business metrics dashboard
const BusinessDashboard = () => {
  const [summary, setSummary] = useState(null);
  const [performance, setPerformance] = useState(null);
  const [costs, setCosts] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadDashboardData = async () => {
      setLoading(true);
      try {
        const [summaryRes, performanceRes, costsRes] = await Promise.all([
          fetch('/metrics/summary?time_range=24h'),
          fetch('/metrics/performance?time_range=24h'),
          fetch('/metrics/cost?time_range=30d')
        ]);

        setSummary(await summaryRes.json());
        setPerformance(await performanceRes.json());
        setCosts(await costsRes.json());
      } catch (error) {
        console.error('Failed to load dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadDashboardData();
  }, []);

  if (loading) return <div>Loading...</div>;

  return (
    <div className="business-dashboard">
      <MetricsSummary summary={summary} />
      <PerformanceChart data={performance} />
      <CostAnalysis costs={costs} />
      <ComplianceStatus />
    </div>
  );
};
```

#### Metrics Recording Component
```typescript
// Component for recording custom metrics
const MetricsRecorder = () => {
  const [metricName, setMetricName] = useState('');
  const [value, setValue] = useState('');
  const [category, setCategory] = useState('performance');
  const [service, setService] = useState('');

  const recordMetric = async () => {
    try {
      const response = await fetch('/metrics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          metric_name: metricName,
          value: parseFloat(value),
          unit: 'count',
          category,
          service,
          timestamp: new Date().toISOString()
        })
      });

      const result = await response.json();
      if (result.success) {
        alert('Metric recorded successfully');
        setMetricName('');
        setValue('');
      } else {
        alert('Failed to record metric');
      }
    } catch (error) {
      console.error('Failed to record metric:', error);
    }
  };

  return (
    <div className="metrics-recorder">
      <h3>Record Custom Metric</h3>
      
      <div className="form-group">
        <label>Metric Name:</label>
        <input
          type="text"
          value={metricName}
          onChange={(e) => setMetricName(e.target.value)}
          placeholder="e.g., custom_kpi"
        />
      </div>

      <div className="form-group">
        <label>Value:</label>
        <input
          type="number"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="e.g., 95.5"
        />
      </div>

      <div className="form-group">
        <label>Category:</label>
        <select value={category} onChange={(e) => setCategory(e.target.value)}>
          <option value="performance">Performance</option>
          <option value="security">Security</option>
          <option value="cost">Cost</option>
          <option value="usage">Usage</option>
          <option value="compliance">Compliance</option>
        </select>
      </div>

      <div className="form-group">
        <label>Service:</label>
        <input
          type="text"
          value={service}
          onChange={(e) => setService(e.target.value)}
          placeholder="e.g., model-api"
        />
      </div>

      <button onClick={recordMetric} disabled={!metricName || !value}>
        Record Metric
      </button>
    </div>
  );
};
```

#### Cost Analysis Component
```typescript
// Component for cost analysis and budgeting
const CostAnalysis = () => {
  const [costs, setCosts] = useState(null);
  const [budget, setBudget] = useState(10000);
  const [timeRange, setTimeRange] = useState('30d');

  useEffect(() => {
    const loadCosts = async () => {
      const response = await fetch(`/metrics/cost?time_range=${timeRange}`);
      const data = await response.json();
      setCosts(data);
    };

    loadCosts();
  }, [timeRange]);

  if (!costs) return <div>Loading...</div>;

  const budgetUtilization = (costs.total_cost / budget) * 100;
  const isOverBudget = budgetUtilization > 100;

  return (
    <div className="cost-analysis">
      <h3>Cost Analysis</h3>
      
      <div className="budget-overview">
        <div className="budget-bar">
          <div 
            className={`budget-fill ${isOverBudget ? 'over-budget' : ''}`}
            style={{ width: `${Math.min(budgetUtilization, 100)}%` }}
          />
        </div>
        <div className="budget-text">
          ${costs.total_cost.toLocaleString()} / ${budget.toLocaleString()}
          ({budgetUtilization.toFixed(1)}%)
        </div>
      </div>

      <div className="cost-breakdown">
        <h4>Cost by Service</h4>
        {Object.entries(costs.cost_by_service).map(([service, cost]) => (
          <div key={service} className="service-cost">
            <span className="service-name">{service}</span>
            <span className="service-cost-value">${cost.toLocaleString()}</span>
          </div>
        ))}
      </div>

      <div className="cost-trends">
        <h4>Cost Trends</h4>
        <CostTrendChart data={costs.cost_trends} />
      </div>
    </div>
  );
};
```

## Architecture Deep Dive

### Service Components

#### 1. Metrics Collector
**Purpose**: Collects and processes metrics from all platform services
**How it works**:
- Receives metrics via REST API and message queues
- Validates and normalizes metric data
- Implements buffering and batch processing
- Provides real-time and historical data access

#### 2. Analytics Engine
**Purpose**: Processes and analyzes collected metrics
**How it works**:
- Performs statistical analysis and trend detection
- Calculates KPIs and business metrics
- Implements alerting and threshold monitoring
- Generates insights and recommendations

#### 3. Data Storage
**Purpose**: Efficient storage and retrieval of metrics data
**How it works**:
- Uses PostgreSQL for persistent storage
- Implements Redis for real-time caching
- Provides time-series data optimization
- Supports data retention and archival

#### 4. Reporting Engine
**Purpose**: Generates reports and dashboards
**How it works**:
- Creates real-time and scheduled reports
- Provides data visualization support
- Implements export and sharing capabilities
- Supports custom report generation

### Data Flow Architecture

```
All Services → Metrics Collector → Analytics Engine → Data Storage → Reporting
     ↓              ↓                ↓              ↓            ↓
  Metrics       Validation       Processing      Storage     Dashboards
     ↓              ↓                ↓              ↓            ↓
  Collection    Normalization    Aggregation    Caching     Export
     ↓              ↓                ↓              ↓            ↓
  Buffering    Batch Processing  Alerting      Retention    Sharing
```

## Configuration & Environment

### Environment Variables
```bash
# Service Configuration
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8004
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql://mlflow:password@postgres:5432/ml_security_consolidated
REDIS_URL=redis://redis:6379/1

# Metrics Configuration
METRICS_RETENTION_DAYS=365
METRICS_BATCH_SIZE=1000
METRICS_FLUSH_INTERVAL=60
CACHE_TTL_SECONDS=300

# Alerting Configuration
ALERT_THRESHOLDS_ENABLED=true
ALERT_EMAIL_ENABLED=false
ALERT_WEBHOOK_URL=
```

## Security & Compliance

### Data Protection
- **Encryption**: All metrics data encrypted in transit and at rest
- **Access Control**: Role-based access to metrics and reports
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
- **Data Processing**: Efficient batch processing and aggregation
- **Caching Strategy**: Multi-level caching for performance
- **Async Processing**: Non-blocking I/O for high throughput

### Reliability
- **Fault Tolerance**: Graceful handling of service failures
- **Data Consistency**: ACID compliance for critical operations
- **Backup & Recovery**: Automated backup and point-in-time recovery
- **Circuit Breakers**: Protection against cascading failures

## Troubleshooting Guide

### Common Issues
1. **Metrics Not Recording**: Check service connectivity and authentication
2. **High Memory Usage**: Monitor metrics volume and retention settings
3. **Slow Queries**: Check database performance and indexing
4. **Cache Issues**: Verify Redis connectivity and memory usage

### Debug Commands
```bash
# Check service health
curl http://localhost:8004/health

# Get metrics summary
curl http://localhost:8004/metrics/summary

# Record test metric
curl -X POST http://localhost:8004/metrics \
  -H "Content-Type: application/json" \
  -d '{"metric_name": "test_metric", "value": 1, "unit": "count", "category": "usage", "service": "test"}'
```

## Future Enhancements

### Planned Features
- **Real-time Streaming**: Kafka integration for real-time metrics
- **Advanced Analytics**: Machine learning-based insights
- **Custom Dashboards**: User-configurable dashboards
- **Automated Reporting**: Scheduled and triggered reports
- **Cost Optimization**: AI-driven cost optimization recommendations

### Research Areas
- **Predictive Analytics**: Forecasting and trend prediction
- **Anomaly Detection**: ML-based anomaly detection
- **Cost Optimization**: Advanced cost analysis and optimization
- **Performance Tuning**: Automated performance optimization