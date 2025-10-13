# ML Security Tracing Service

## Service Architecture & Purpose

### Core Purpose
The Tracing Service is the **distributed observability and tracing engine** of the ML Security platform. It provides comprehensive request tracing, performance monitoring, and distributed system observability to ensure optimal system performance and debugging capabilities.

### Why This Service Exists
- **Distributed Tracing**: Track requests across multiple services and components
- **Performance Monitoring**: Identify bottlenecks and performance issues
- **Debugging Support**: Provide detailed request flow and error tracking
- **System Observability**: Gain insights into system behavior and dependencies
- **Root Cause Analysis**: Quickly identify and resolve issues across the platform

## Complete API Documentation for Frontend Development

### Base URL
```
http://tracing:8007
```

### Authentication
All endpoints require authentication headers:
```javascript
headers: {
  'Authorization': 'Bearer <token>',
  'Content-Type': 'application/json'
}
```

### Core Tracing Endpoints

#### `GET /`
**Purpose**: Root endpoint with tracing service status

**Response Model**:
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

#### `GET /health`
**Purpose**: Comprehensive health check for tracing service

**Frontend Usage**:
```javascript
const response = await fetch('/health');
const health = await response.json();

if (health.status === 'healthy') {
  console.log('Tracing service is healthy');
  console.log(`Active traces: ${health.active_traces}`);
  console.log(`Traces processed: ${health.traces_processed}`);
} else {
  console.error('Tracing service is unhealthy:', health.error);
}
```

**Response Model**:
```typescript
interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  service: string;
  timestamp: string;
  uptime_seconds: number;
  active_traces: number;
  traces_processed: number;
  spans_processed: number;
  dependencies: {
    jaeger: boolean;
    zipkin: boolean;
    opentelemetry: boolean;
    database: boolean;
  };
  error?: string;
}
```

### Trace Management Endpoints

#### `GET /traces`
**Purpose**: List traces with filtering and pagination

**Query Parameters**:
- `service` (optional): Filter by service name
- `operation` (optional): Filter by operation name
- `start_time` (optional): Start time for time range (ISO 8601)
- `end_time` (optional): End time for time range (ISO 8601)
- `duration_min` (optional): Minimum duration in milliseconds
- `duration_max` (optional): Maximum duration in milliseconds
- `limit` (optional): Maximum number of traces - default: 100
- `offset` (optional): Pagination offset - default: 0

**Frontend Usage**:
```javascript
// Get traces for specific service
const response = await fetch('/traces?service=model-api&limit=50');
const traces = await response.json();

// Get traces within time range
const response = await fetch('/traces?start_time=2025-01-01T00:00:00Z&end_time=2025-01-01T23:59:59Z');
const timeRangeTraces = await response.json();

// Get slow traces
const response = await fetch('/traces?duration_min=1000&limit=20');
const slowTraces = await response.json();
```

**Response Model**:
```typescript
interface TracesResponse {
  traces: Array<{
    trace_id: string;
    service_name: string;
    operation_name: string;
    start_time: string;
    duration_ms: number;
    status: 'success' | 'error' | 'timeout';
    tags: Record<string, string>;
    spans_count: number;
    error_message?: string;
  }>;
  total_count: number;
  has_more: boolean;
  time_range: {
    start: string;
    end: string;
  };
}
```

#### `GET /traces/{trace_id}`
**Purpose**: Get detailed trace information by ID

**Path Parameters**:
- `trace_id`: Unique identifier for the trace

**Frontend Usage**:
```javascript
const response = await fetch('/traces/abc123def456');
const trace = await response.json();

console.log(`Trace: ${trace.trace_id}`);
console.log(`Duration: ${trace.duration_ms}ms`);
console.log(`Spans: ${trace.spans.length}`);
```

**Response Model**:
```typescript
interface TraceDetails {
  trace_id: string;
  service_name: string;
  operation_name: string;
  start_time: string;
  end_time: string;
  duration_ms: number;
  status: 'success' | 'error' | 'timeout';
  tags: Record<string, string>;
  spans: Array<{
    span_id: string;
    parent_span_id?: string;
    operation_name: string;
    service_name: string;
    start_time: string;
    duration_ms: number;
    status: 'success' | 'error' | 'timeout';
    tags: Record<string, string>;
    logs: Array<{
      timestamp: string;
      level: string;
      message: string;
      fields: Record<string, any>;
    }>;
  }>;
  error_spans: Array<{
    span_id: string;
    error_message: string;
    error_type: string;
    stack_trace?: string;
  }>;
  performance_metrics: {
    total_spans: number;
    error_spans: number;
    average_span_duration: number;
    slowest_span_duration: number;
    fastest_span_duration: number;
  };
}
```

#### `GET /traces/{trace_id}/spans`
**Purpose**: Get spans for a specific trace

**Path Parameters**:
- `trace_id`: Unique identifier for the trace

**Query Parameters**:
- `operation` (optional): Filter by operation name
- `status` (optional): Filter by span status

**Frontend Usage**:
```javascript
const response = await fetch('/traces/abc123def456/spans');
const spans = await response.json();

// Display spans in timeline
spans.spans.forEach(span => {
  console.log(`${span.operation_name}: ${span.duration_ms}ms`);
});
```

### Service Analysis Endpoints

#### `GET /services`
**Purpose**: List all services with tracing information

**Frontend Usage**:
```javascript
const response = await fetch('/services');
const services = await response.json();

// Display service information
services.services.forEach(service => {
  console.log(`${service.name}: ${service.traces_count} traces, ${service.avg_duration_ms}ms avg`);
});
```

**Response Model**:
```typescript
interface ServicesResponse {
  services: Array<{
    name: string;
    traces_count: number;
    spans_count: number;
    avg_duration_ms: number;
    error_rate: number;
    last_trace: string;
    operations: Array<{
      name: string;
      count: number;
      avg_duration_ms: number;
      error_rate: number;
    }>;
  }>;
  total_services: number;
  total_traces: number;
  total_spans: number;
}
```

#### `GET /services/{service_name}/operations`
**Purpose**: Get operations for a specific service

**Path Parameters**:
- `service_name`: Name of the service

**Frontend Usage**:
```javascript
const response = await fetch('/services/model-api/operations');
const operations = await response.json();

// Display operations
operations.operations.forEach(op => {
  console.log(`${op.name}: ${op.count} calls, ${op.avg_duration_ms}ms avg`);
});
```

#### `GET /services/{service_name}/performance`
**Purpose**: Get performance metrics for a specific service

**Query Parameters**:
- `time_range` (optional): Time range (1h, 24h, 7d) - default: 24h
- `operation` (optional): Filter by specific operation

**Frontend Usage**:
```javascript
// Get performance metrics for last 24 hours
const response = await fetch('/services/model-api/performance?time_range=24h');
const performance = await response.json();

// Display performance metrics
console.log(`Average Response Time: ${performance.avg_response_time_ms}ms`);
console.log(`Error Rate: ${performance.error_rate}%`);
console.log(`Throughput: ${performance.throughput_per_second} req/s`);
```

**Response Model**:
```typescript
interface ServicePerformance {
  service_name: string;
  time_range: string;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  error_rate: number;
  avg_response_time_ms: number;
  p50_response_time_ms: number;
  p95_response_time_ms: number;
  p99_response_time_ms: number;
  throughput_per_second: number;
  operations: Array<{
    name: string;
    count: number;
    avg_duration_ms: number;
    error_rate: number;
    throughput_per_second: number;
  }>;
  trends: {
    response_time_trend: 'up' | 'down' | 'stable';
    error_rate_trend: 'up' | 'down' | 'stable';
    throughput_trend: 'up' | 'down' | 'stable';
  };
}
```

### Error Analysis Endpoints

#### `GET /errors`
**Purpose**: List errors and exceptions across the platform

**Query Parameters**:
- `service` (optional): Filter by service name
- `error_type` (optional): Filter by error type
- `start_time` (optional): Start time for time range
- `end_time` (optional): End time for time range
- `severity` (optional): Filter by severity level
- `limit` (optional): Maximum number of errors - default: 100

**Frontend Usage**:
```javascript
// Get recent errors
const response = await fetch('/errors?limit=50');
const errors = await response.json();

// Get errors for specific service
const response = await fetch('/errors?service=model-api&severity=high');
const serviceErrors = await response.json();

// Display errors
errors.errors.forEach(error => {
  console.log(`${error.error_type}: ${error.message} (${error.service_name})`);
});
```

**Response Model**:
```typescript
interface ErrorsResponse {
  errors: Array<{
    error_id: string;
    trace_id: string;
    span_id: string;
    service_name: string;
    operation_name: string;
    error_type: string;
    error_message: string;
    stack_trace?: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    timestamp: string;
    tags: Record<string, string>;
  }>;
  total_count: number;
  error_types: Array<{
    type: string;
    count: number;
    severity: string;
  }>;
  services_with_errors: Array<{
    service_name: string;
    error_count: number;
    last_error: string;
  }>;
}
```

#### `GET /errors/{error_id}`
**Purpose**: Get detailed error information by ID

**Path Parameters**:
- `error_id`: Unique identifier for the error

**Frontend Usage**:
```javascript
const response = await fetch('/errors/error_123');
const error = await response.json();

console.log(`Error: ${error.error_type}`);
console.log(`Message: ${error.error_message}`);
console.log(`Service: ${error.service_name}`);
console.log(`Trace: ${error.trace_id}`);
```

### Performance Analysis Endpoints

#### `GET /performance/slowest`
**Purpose**: Get slowest traces and operations

**Query Parameters**:
- `service` (optional): Filter by service name
- `operation` (optional): Filter by operation name
- `time_range` (optional): Time range - default: 24h
- `limit` (optional): Maximum number of results - default: 20

**Frontend Usage**:
```javascript
// Get slowest traces
const response = await fetch('/performance/slowest?limit=10');
const slowest = await response.json();

// Get slowest operations for specific service
const response = await fetch('/performance/slowest?service=model-api&time_range=7d');
const slowestOperations = await response.json();
```

**Response Model**:
```typescript
interface SlowestTracesResponse {
  traces: Array<{
    trace_id: string;
    service_name: string;
    operation_name: string;
    duration_ms: number;
    start_time: string;
    status: string;
    tags: Record<string, string>;
  }>;
  operations: Array<{
    service_name: string;
    operation_name: string;
    avg_duration_ms: number;
    max_duration_ms: number;
    count: number;
    error_rate: number;
  }>;
  time_range: string;
}
```

#### `GET /performance/trends`
**Purpose**: Get performance trends over time

**Query Parameters**:
- `service` (optional): Filter by service name
- `operation` (optional): Filter by operation name
- `time_range` (optional): Time range - default: 24h
- `granularity` (optional): Data granularity (1m, 5m, 1h) - default: 5m

**Frontend Usage**:
```javascript
// Get performance trends
const response = await fetch('/performance/trends?time_range=7d&granularity=1h');
const trends = await response.json();

// Display trends data for charts
trends.data_points.forEach(point => {
  console.log(`${point.timestamp}: ${point.avg_duration_ms}ms avg`);
});
```

**Response Model**:
```typescript
interface PerformanceTrends {
  service_name?: string;
  operation_name?: string;
  time_range: string;
  granularity: string;
  data_points: Array<{
    timestamp: string;
    avg_duration_ms: number;
    p95_duration_ms: number;
    p99_duration_ms: number;
    error_rate: number;
    throughput_per_second: number;
  }>;
  trends: {
    duration_trend: 'up' | 'down' | 'stable';
    error_rate_trend: 'up' | 'down' | 'stable';
    throughput_trend: 'up' | 'down' | 'stable';
  };
}
```

### Dependency Analysis Endpoints

#### `GET /dependencies`
**Purpose**: Get service dependency graph and analysis

**Frontend Usage**:
```javascript
const response = await fetch('/dependencies');
const dependencies = await response.json();

// Display dependency graph
dependencies.services.forEach(service => {
  console.log(`${service.name} -> ${service.dependencies.join(', ')}`);
});
```

**Response Model**:
```typescript
interface DependenciesResponse {
  services: Array<{
    name: string;
    dependencies: string[];
    dependents: string[];
    call_count: number;
    avg_latency_ms: number;
    error_rate: number;
  }>;
  graph: {
    nodes: Array<{
      id: string;
      name: string;
      type: 'service' | 'external';
    }>;
    edges: Array<{
      source: string;
      target: string;
      weight: number;
      avg_latency_ms: number;
    }>;
  };
  critical_paths: Array<{
    path: string[];
    total_latency_ms: number;
    bottleneck_service: string;
  }>;
}
```

#### `GET /dependencies/{service_name}/impact`
**Purpose**: Get impact analysis for a specific service

**Path Parameters**:
- `service_name`: Name of the service

**Frontend Usage**:
```javascript
const response = await fetch('/dependencies/model-api/impact');
const impact = await response.json();

console.log(`Service: ${impact.service_name}`);
console.log(`Dependencies: ${impact.dependencies.length}`);
console.log(`Dependents: ${impact.dependents.length}`);
console.log(`Critical Paths: ${impact.critical_paths.length}`);
```

### Health and Status Endpoints

#### `GET /metrics`
**Purpose**: Prometheus metrics endpoint

**Frontend Usage**:
```javascript
// Get metrics for monitoring dashboard
const response = await fetch('/metrics');
const metrics = await response.text();

// Parse Prometheus metrics if needed
const lines = metrics.split('\n');
const traceCount = lines.find(line => 
  line.startsWith('tracing_traces_total')
)?.split(' ')[1] || '0';
```

### Frontend Integration Examples

#### Tracing Dashboard Component
```typescript
// React component for tracing dashboard
const TracingDashboard = () => {
  const [traces, setTraces] = useState([]);
  const [services, setServices] = useState([]);
  const [errors, setErrors] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadDashboardData = async () => {
      setLoading(true);
      try {
        const [tracesRes, servicesRes, errorsRes] = await Promise.all([
          fetch('/traces?limit=20'),
          fetch('/services'),
          fetch('/errors?limit=10')
        ]);

        setTraces(await tracesRes.json());
        setServices(await servicesRes.json());
        setErrors(await errorsRes.json());
      } catch (error) {
        console.error('Failed to load dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadDashboardData();
    
    // Refresh data every 30 seconds
    const interval = setInterval(loadDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) return <div>Loading...</div>;

  return (
    <div className="tracing-dashboard">
      <TracesList traces={traces.traces} />
      <ServicesOverview services={services.services} />
      <ErrorsPanel errors={errors.errors} />
    </div>
  );
};
```

#### Trace Analysis Component
```typescript
// Component for analyzing traces
const TraceAnalysis = () => {
  const [traces, setTraces] = useState([]);
  const [selectedTrace, setSelectedTrace] = useState(null);
  const [traceDetails, setTraceDetails] = useState(null);

  const searchTraces = async (filters) => {
    try {
      const queryParams = new URLSearchParams();
      if (filters.service) queryParams.append('service', filters.service);
      if (filters.operation) queryParams.append('operation', filters.operation);
      if (filters.startTime) queryParams.append('start_time', filters.startTime);
      if (filters.endTime) queryParams.append('end_time', filters.endTime);
      if (filters.durationMin) queryParams.append('duration_min', filters.durationMin);
      if (filters.limit) queryParams.append('limit', filters.limit);

      const response = await fetch(`/traces?${queryParams}`);
      const data = await response.json();
      setTraces(data.traces);
    } catch (error) {
      console.error('Failed to search traces:', error);
    }
  };

  const loadTraceDetails = async (traceId) => {
    try {
      const response = await fetch(`/traces/${traceId}`);
      const data = await response.json();
      setTraceDetails(data);
    } catch (error) {
      console.error('Failed to load trace details:', error);
    }
  };

  return (
    <div className="trace-analysis">
      <h3>Trace Analysis</h3>
      
      <TraceSearchForm onSearch={searchTraces} />
      
      <div className="traces-list">
        {traces.map(trace => (
          <TraceCard
            key={trace.trace_id}
            trace={trace}
            onSelect={() => {
              setSelectedTrace(trace);
              loadTraceDetails(trace.trace_id);
            }}
          />
        ))}
      </div>
      
      {traceDetails && (
        <TraceDetailsPanel
          trace={traceDetails}
          onClose={() => setTraceDetails(null)}
        />
      )}
    </div>
  );
};
```

#### Service Performance Component
```typescript
// Component for service performance analysis
const ServicePerformance = () => {
  const [services, setServices] = useState([]);
  const [selectedService, setSelectedService] = useState(null);
  const [performance, setPerformance] = useState(null);

  const loadServices = async () => {
    try {
      const response = await fetch('/services');
      const data = await response.json();
      setServices(data.services);
    } catch (error) {
      console.error('Failed to load services:', error);
    }
  };

  const loadServicePerformance = async (serviceName) => {
    try {
      const response = await fetch(`/services/${serviceName}/performance?time_range=24h`);
      const data = await response.json();
      setPerformance(data);
    } catch (error) {
      console.error('Failed to load performance data:', error);
    }
  };

  useEffect(() => {
    loadServices();
  }, []);

  return (
    <div className="service-performance">
      <h3>Service Performance</h3>
      
      <div className="services-list">
        {services.map(service => (
          <ServiceCard
            key={service.name}
            service={service}
            onSelect={() => {
              setSelectedService(service);
              loadServicePerformance(service.name);
            }}
          />
        ))}
      </div>
      
      {selectedService && performance && (
        <div className="performance-details">
          <h4>Performance for {selectedService.name}</h4>
          <PerformanceMetrics metrics={performance} />
          <PerformanceChart data={performance} />
        </div>
      )}
    </div>
  );
};
```

#### Error Analysis Component
```typescript
// Component for error analysis
const ErrorAnalysis = () => {
  const [errors, setErrors] = useState([]);
  const [errorTypes, setErrorTypes] = useState([]);
  const [loading, setLoading] = useState(false);

  const loadErrors = async (filters = {}) => {
    setLoading(true);
    try {
      const queryParams = new URLSearchParams();
      if (filters.service) queryParams.append('service', filters.service);
      if (filters.errorType) queryParams.append('error_type', filters.errorType);
      if (filters.severity) queryParams.append('severity', filters.severity);
      if (filters.limit) queryParams.append('limit', filters.limit);

      const response = await fetch(`/errors?${queryParams}`);
      const data = await response.json();
      setErrors(data.errors);
      setErrorTypes(data.error_types);
    } catch (error) {
      console.error('Failed to load errors:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadErrors({ limit: 50 });
  }, []);

  return (
    <div className="error-analysis">
      <h3>Error Analysis</h3>
      
      <ErrorFilters onFilter={loadErrors} />
      
      <div className="error-summary">
        <ErrorTypesChart data={errorTypes} />
        <ErrorTimeline data={errors} />
      </div>
      
      <div className="errors-list">
        {errors.map(error => (
          <ErrorCard
            key={error.error_id}
            error={error}
            onViewTrace={() => {/* Navigate to trace */}}
          />
        ))}
      </div>
    </div>
  );
};
```

## Architecture Deep Dive

### Service Components

#### 1. Trace Collector
**Purpose**: Collects and processes distributed traces
**How it works**:
- Receives trace data from OpenTelemetry, Jaeger, and Zipkin
- Processes and normalizes trace information
- Stores traces in time-series database
- Implements trace sampling and filtering

#### 2. Span Processor
**Purpose**: Processes individual spans within traces
**How it works**:
- Analyzes span relationships and dependencies
- Extracts performance metrics and error information
- Implements span aggregation and correlation
- Provides span-level debugging information

#### 3. Performance Analyzer
**Purpose**: Analyzes performance patterns and bottlenecks
**How it works**:
- Identifies slow operations and critical paths
- Calculates performance trends and statistics
- Detects performance anomalies and regressions
- Provides performance optimization recommendations

#### 4. Error Tracker
**Purpose**: Tracks and analyzes errors across the platform
**How it works**:
- Collects error information from traces and logs
- Categorizes and prioritizes errors by severity
- Provides error correlation and root cause analysis
- Implements error alerting and notification

### Data Flow Architecture

```
All Services → Trace Collection → Span Processing → Performance Analysis → Error Tracking
     ↓              ↓                ↓                ↓                    ↓
  OpenTelemetry   Normalization   Aggregation     Trend Analysis       Alerting
     ↓              ↓                ↓                ↓                    ↓
  Jaeger/Zipkin   Storage         Correlation     Bottleneck Detection  Reporting
     ↓              ↓                ↓                ↓                    ↓
  Custom Spans    Indexing        Dependency      Optimization         Dashboard
```

## Configuration & Environment

### Environment Variables
```bash
# Service Configuration
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8007
LOG_LEVEL=INFO

# Tracing Configuration
JAEGER_ENDPOINT=http://jaeger:14268/api/traces
ZIPKIN_ENDPOINT=http://zipkin:9411/api/v2/spans
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317

# Database Configuration
DATABASE_URL=postgresql://mlflow:password@postgres:5432/ml_security_consolidated
REDIS_URL=redis://redis:6379/1

# Performance Configuration
TRACE_SAMPLING_RATE=0.1
MAX_TRACES_PER_SECOND=1000
TRACE_RETENTION_DAYS=7
```

## Security & Compliance

### Data Protection
- **Encryption**: All trace data encrypted in transit and at rest
- **Access Control**: Role-based access to trace data and analysis
- **Audit Logging**: Comprehensive audit trail for all operations
- **Data Retention**: Configurable trace retention and archival policies

### Compliance Features
- **SOC 2**: Security and availability controls
- **ISO 27001**: Information security management
- **NIST Framework**: Cybersecurity framework compliance
- **GDPR Compliance**: Data privacy and right to be forgotten

## Performance Characteristics

### Scalability
- **Horizontal Scaling**: Stateless design supports multiple instances
- **Data Processing**: Efficient trace processing and aggregation
- **Caching Strategy**: Multi-level caching for performance
- **Async Processing**: Non-blocking I/O for high throughput

### Reliability
- **Fault Tolerance**: Graceful handling of trace collection failures
- **Data Consistency**: Eventual consistency for trace data
- **Backup & Recovery**: Automated backup and point-in-time recovery
- **Circuit Breakers**: Protection against cascading failures

## Troubleshooting Guide

### Common Issues
1. **Trace Collection Failures**: Check tracing agent configuration and connectivity
2. **Performance Issues**: Monitor trace processing and database performance
3. **Missing Traces**: Verify sampling configuration and service instrumentation
4. **High Memory Usage**: Check trace retention settings and cleanup processes

### Debug Commands
```bash
# Check service health
curl http://localhost:8007/health

# List recent traces
curl http://localhost:8007/traces?limit=10

# Get service performance
curl http://localhost:8007/services/model-api/performance

# View errors
curl http://localhost:8007/errors?limit=20
```

## Future Enhancements

### Planned Features
- **AI-Powered Analysis**: Machine learning-based anomaly detection and root cause analysis
- **Real-time Alerting**: Intelligent alerting based on trace patterns and performance
- **Custom Dashboards**: User-configurable tracing dashboards and visualizations
- **Distributed Profiling**: Advanced profiling and performance analysis
- **Trace Replay**: Ability to replay traces for debugging and testing

### Research Areas
- **Anomaly Detection**: Advanced ML algorithms for trace anomaly detection
- **Predictive Analysis**: Forecasting performance issues and system behavior
- **Intelligent Sampling**: AI-driven trace sampling optimization
- **Automated Debugging**: AI-assisted debugging and root cause analysis