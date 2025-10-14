# ML Security Monitoring Service

## Service Architecture & Purpose

### Core Purpose
The Monitoring Service is the **real-time observability and alerting engine** of the ML Security platform. It provides comprehensive system monitoring, performance tracking, health checks, and alerting capabilities to ensure platform reliability, performance, and security.

### Why This Service Exists
- **System Reliability**: Ensures all platform services are running optimally
- **Performance Monitoring**: Tracks system performance metrics and identifies bottlenecks
- **Proactive Alerting**: Provides early warning of issues before they impact users
- **Operational Visibility**: Gives operations teams real-time insight into system health
- **Capacity Planning**: Monitors resource usage for scaling decisions

## Complete API Documentation for Frontend Development

### Base URL
```
http://monitoring:8006
```


### Core Monitoring Endpoints

#### `GET /`
**Purpose**: Root endpoint with service status

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
**Purpose**: Comprehensive health check for the monitoring service

**Frontend Usage**:
```javascript
const response = await fetch('/health');
const health = await response.json();

if (health.status === 'healthy') {
  console.log('Monitoring service is healthy');
  console.log(`Uptime: ${health.uptime_seconds} seconds`);
} else {
  console.error('Monitoring service is unhealthy:', health.error);
}
```

**Response Model**:
```typescript
interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  service: string;
  timestamp: string;
  uptime_seconds: number;
  dependencies: {
    redis: boolean;
    postgres: boolean;
    data_collector: boolean;
  };
  metrics: {
    total_checks: number;
    successful_checks: number;
    failed_checks: number;
    average_response_time_ms: number;
  };
}
```

#### `GET /dashboard-data`
**Purpose**: Get complete dashboard data for all platform services

**Frontend Usage**:
```javascript
const response = await fetch('/dashboard-data');
const dashboardData = await response.json();

// Display service status
Object.entries(dashboardData.services).forEach(([service, data]) => {
  console.log(`${service}: ${data.status} (${data.response_time_ms}ms)`);
});

// Display system metrics
console.log(`Total CPU: ${dashboardData.system_metrics.total_cpu_usage}%`);
console.log(`Total Memory: ${dashboardData.system_metrics.total_memory_usage_mb} MB`);
```

**Response Model**:
```typescript
interface DashboardData {
  services: Record<string, {
    status: 'healthy' | 'unhealthy' | 'degraded';
    response_time_ms: number;
    cpu_usage: number;
    memory_usage_mb: number;
    last_updated: string;
    error_rate?: number;
    throughput?: number;
  }>;
  system_metrics: {
    total_cpu_usage: number;
    total_memory_usage_mb: number;
    disk_usage_percent: number;
    network_io_mbps: number;
    load_average: number[];
  };
  alerts: {
    active: number;
    critical: number;
    warning: number;
    info: number;
  };
  trends: {
    cpu_trend: 'up' | 'down' | 'stable';
    memory_trend: 'up' | 'down' | 'stable';
    response_time_trend: 'up' | 'down' | 'stable';
  };
  last_updated: string;
}
```

### Service-Specific Monitoring Endpoints

#### `GET /model-loading-status`
**Purpose**: Monitor model loading operations across the platform

**Frontend Usage**:
```javascript
const response = await fetch('/model-loading-status');
const modelStatus = await response.json();

// Display model loading status
modelStatus.models.forEach(model => {
  console.log(`${model.name}: ${model.status} (${model.progress}%)`);
});
```

**Response Model**:
```typescript
interface ModelLoadingStatus {
  models: Array<{
    name: string;
    status: 'loading' | 'loaded' | 'failed' | 'unloaded';
    progress: number;
    memory_usage_mb: number;
    loading_time_ms?: number;
    error_message?: string;
    loaded_at?: string;
  }>;
  total_models: number;
  loaded_models: number;
  loading_models: number;
  failed_models: number;
  total_memory_usage_mb: number;
}
```

#### `GET /training-status`
**Purpose**: Monitor training job status and performance

**Frontend Usage**:
```javascript
const response = await fetch('/training-status');
const trainingStatus = await response.json();

// Display training jobs
trainingStatus.jobs.forEach(job => {
  console.log(`${job.model_name}: ${job.status} (${job.progress}%)`);
});
```

**Response Model**:
```typescript
interface TrainingStatus {
  jobs: Array<{
    job_id: string;
    model_name: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    progress: number;
    started_at: string;
    estimated_completion?: string;
    resource_usage: {
      cpu_usage: number;
      memory_usage_mb: number;
      gpu_usage?: number;
    };
  }>;
  queue_status: {
    pending_jobs: number;
    running_jobs: number;
    completed_jobs: number;
    failed_jobs: number;
  };
  resource_utilization: {
    total_cpu_usage: number;
    total_memory_usage_mb: number;
    gpu_utilization?: number;
  };
}
```

#### `GET /system-metrics`
**Purpose**: Collect and return system-level metrics

**Frontend Usage**:
```javascript
const response = await fetch('/system-metrics');
const systemMetrics = await response.json();

// Display system metrics
console.log(`CPU Usage: ${systemMetrics.cpu_usage}%`);
console.log(`Memory Usage: ${systemMetrics.memory_usage_mb} MB`);
console.log(`Disk Usage: ${systemMetrics.disk_usage_percent}%`);
```

**Response Model**:
```typescript
interface SystemMetrics {
  cpu_usage: number;
  memory_usage_mb: number;
  disk_usage_percent: number;
  network_io_mbps: number;
  load_average: number[];
  uptime_seconds: number;
  processes: number;
  threads: number;
  file_descriptors: number;
  timestamp: string;
}
```

#### `GET /service-health`
**Purpose**: Check health status of all platform services

**Frontend Usage**:
```javascript
const response = await fetch('/service-health');
const serviceHealth = await response.json();

// Display service health
Object.entries(serviceHealth.services).forEach(([service, health]) => {
  console.log(`${service}: ${health.status} (${health.response_time_ms}ms)`);
});
```

**Response Model**:
```typescript
interface ServiceHealth {
  services: Record<string, {
    status: 'healthy' | 'unhealthy' | 'degraded';
    response_time_ms: number;
    last_check: string;
    error_message?: string;
    dependencies: Record<string, boolean>;
  }>;
  overall_status: 'healthy' | 'unhealthy' | 'degraded';
  healthy_services: number;
  total_services: number;
  last_updated: string;
}
```

### Alerting Endpoints

#### `GET /alerts`
**Purpose**: Get current alerts and their status

**Query Parameters**:
- `severity` (optional): Filter by severity (critical, warning, info)
- `service` (optional): Filter by service name
- `status` (optional): Filter by status (active, resolved, acknowledged)
- `limit` (optional): Maximum number of alerts - default: 50

**Frontend Usage**:
```javascript
// Get all active alerts
const response = await fetch('/alerts?status=active');
const alerts = await response.json();

// Get critical alerts only
const response = await fetch('/alerts?severity=critical&limit=10');
const criticalAlerts = await response.json();

// Display alerts
alerts.alerts.forEach(alert => {
  console.log(`${alert.severity.toUpperCase()}: ${alert.message}`);
  console.log(`Service: ${alert.service}, Time: ${alert.timestamp}`);
});
```

**Response Model**:
```typescript
interface AlertsResponse {
  alerts: Array<{
    alert_id: string;
    severity: 'critical' | 'warning' | 'info';
    status: 'active' | 'resolved' | 'acknowledged';
    service: string;
    message: string;
    description: string;
    timestamp: string;
    acknowledged_at?: string;
    acknowledged_by?: string;
    resolved_at?: string;
    metadata?: Record<string, any>;
  }>;
  total_count: number;
  active_count: number;
  critical_count: number;
  warning_count: number;
  info_count: number;
}
```

#### `POST /alerts/acknowledge`
**Purpose**: Acknowledge and manage alerts

**Request Body**:
```typescript
interface AcknowledgeAlertRequest {
  alert_id: string;
  acknowledged_by: string;
  comment?: string;
}
```

**Frontend Usage**:
```javascript
const acknowledgeRequest = {
  alert_id: 'alert_123',
  acknowledged_by: 'user_456',
  comment: 'Investigating the issue'
};

const response = await fetch('/alerts/acknowledge', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(acknowledgeRequest)
});
```

#### `POST /alerts/resolve`
**Purpose**: Mark alerts as resolved

**Request Body**:
```typescript
interface ResolveAlertRequest {
  alert_id: string;
  resolved_by: string;
  resolution_notes?: string;
}
```

### Visualization Endpoints

#### `GET /charts/model-loading`
**Purpose**: Generate model loading visualization chart

**Query Parameters**:
- `time_range` (optional): Time range (1h, 24h, 7d) - default: 24h
- `format` (optional): Chart format (json, png, svg) - default: json

**Frontend Usage**:
```javascript
// Get model loading chart data
const response = await fetch('/charts/model-loading?time_range=24h');
const chartData = await response.json();

// Use with charting library (e.g., Chart.js, D3.js)
const chart = new Chart(ctx, {
  type: 'line',
  data: chartData,
  options: { /* chart options */ }
});
```

**Response Model**:
```typescript
interface ModelLoadingChart {
  labels: string[];
  datasets: Array<{
    label: string;
    data: number[];
    borderColor: string;
    backgroundColor: string;
  }>;
  time_range: string;
  total_models: number;
  loading_trend: 'up' | 'down' | 'stable';
}
```

#### `GET /charts/training-progress`
**Purpose**: Generate training progress visualization

**Query Parameters**:
- `model_name` (optional): Filter by specific model
- `time_range` (optional): Time range - default: 24h

**Frontend Usage**:
```javascript
// Get training progress chart
const response = await fetch('/charts/training-progress?time_range=7d');
const progressChart = await response.json();
```

#### `GET /charts/system-metrics`
**Purpose**: Generate system metrics visualization

**Query Parameters**:
- `metric` (optional): Specific metric (cpu, memory, disk, network)
- `time_range` (optional): Time range - default: 24h

**Frontend Usage**:
```javascript
// Get CPU usage chart
const response = await fetch('/charts/system-metrics?metric=cpu&time_range=24h');
const cpuChart = await response.json();

// Get memory usage chart
const response = await fetch('/charts/system-metrics?metric=memory&time_range=7d');
const memoryChart = await response.json();
```

#### `GET /charts/service-health`
**Purpose**: Generate service health visualization

**Query Parameters**:
- `service` (optional): Filter by specific service
- `time_range` (optional): Time range - default: 24h

#### `GET /charts/alerts-timeline`
**Purpose**: Generate alerts timeline visualization

**Query Parameters**:
- `severity` (optional): Filter by severity
- `time_range` (optional): Time range - default: 7d

### Metrics and Statistics Endpoints

#### `GET /metrics/summary`
**Purpose**: Get comprehensive metrics summary

**Query Parameters**:
- `time_range` (optional): Time range - default: 24h
- `service` (optional): Filter by service

**Frontend Usage**:
```javascript
const response = await fetch('/metrics/summary?time_range=7d');
const metricsSummary = await response.json();

// Display key metrics
console.log(`Total Requests: ${metricsSummary.total_requests}`);
console.log(`Average Response Time: ${metricsSummary.avg_response_time_ms}ms`);
console.log(`Error Rate: ${metricsSummary.error_rate}%`);
```

**Response Model**:
```typescript
interface MetricsSummary {
  time_range: string;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  error_rate: number;
  avg_response_time_ms: number;
  p95_response_time_ms: number;
  p99_response_time_ms: number;
  throughput_per_second: number;
  service_metrics: Record<string, {
    requests: number;
    error_rate: number;
    avg_response_time_ms: number;
    uptime_percent: number;
  }>;
  trends: {
    requests_trend: 'up' | 'down' | 'stable';
    response_time_trend: 'up' | 'down' | 'stable';
    error_rate_trend: 'up' | 'down' | 'stable';
  };
}
```

#### `GET /metrics/detailed`
**Purpose**: Get detailed metrics for specific service or metric

**Query Parameters**:
- `service` (optional): Filter by service
- `metric` (optional): Specific metric name
- `time_range` (optional): Time range - default: 24h
- `granularity` (optional): Data granularity (1m, 5m, 1h) - default: 5m

### Health Check Endpoints

#### `GET /health/services`
**Purpose**: Get health status of all services

**Frontend Usage**:
```javascript
const response = await fetch('/health/services');
const servicesHealth = await response.json();

// Display service health status
servicesHealth.services.forEach(service => {
  const statusColor = service.status === 'healthy' ? 'green' : 'red';
  console.log(`${service.name}: ${statusColor}`);
});
```

#### `GET /health/dependencies`
**Purpose**: Check health of external dependencies

**Response Model**:
```typescript
interface DependenciesHealth {
  dependencies: Record<string, {
    status: 'healthy' | 'unhealthy';
    response_time_ms: number;
    last_check: string;
    error_message?: string;
  }>;
  overall_status: 'healthy' | 'unhealthy';
  healthy_dependencies: number;
  total_dependencies: number;
}
```

### Frontend Integration Examples

#### Monitoring Dashboard Component
```typescript
// React component for monitoring dashboard
const MonitoringDashboard = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadDashboardData = async () => {
      setLoading(true);
      try {
        const [dashboardRes, alertsRes] = await Promise.all([
          fetch('/dashboard-data'),
          fetch('/alerts?status=active&limit=10')
        ]);

        setDashboardData(await dashboardRes.json());
        setAlerts(await alertsRes.json());
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
    <div className="monitoring-dashboard">
      <ServiceStatusGrid services={dashboardData?.services} />
      <SystemMetricsPanel metrics={dashboardData?.system_metrics} />
      <AlertsPanel alerts={alerts.alerts} />
      <ChartsSection />
    </div>
  );
};
```

#### Service Status Component
```typescript
// Component for displaying service status
const ServiceStatusGrid = ({ services }) => {
  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy': return 'green';
      case 'degraded': return 'yellow';
      case 'unhealthy': return 'red';
      default: return 'gray';
    }
  };

  return (
    <div className="service-status-grid">
      <h3>Service Status</h3>
      <div className="services-grid">
        {Object.entries(services || {}).map(([serviceName, serviceData]) => (
          <div key={serviceName} className="service-card">
            <div className="service-header">
              <h4>{serviceName}</h4>
              <div 
                className={`status-indicator ${getStatusColor(serviceData.status)}`}
              />
            </div>
            <div className="service-metrics">
              <div>Response Time: {serviceData.response_time_ms}ms</div>
              <div>CPU: {serviceData.cpu_usage}%</div>
              <div>Memory: {serviceData.memory_usage_mb} MB</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
```

#### Alerts Management Component
```typescript
// Component for managing alerts
const AlertsPanel = ({ alerts }) => {
  const acknowledgeAlert = async (alertId) => {
    try {
      await fetch('/alerts/acknowledge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          alert_id: alertId,
          acknowledged_by: 'current_user',
          comment: 'Acknowledged via dashboard'
        })
      });
      
      // Refresh alerts
      window.location.reload();
    } catch (error) {
      console.error('Failed to acknowledge alert:', error);
    }
  };

  const resolveAlert = async (alertId) => {
    try {
      await fetch('/alerts/resolve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          alert_id: alertId,
          resolved_by: 'current_user',
          resolution_notes: 'Resolved via dashboard'
        })
      });
      
      // Refresh alerts
      window.location.reload();
    } catch (error) {
      console.error('Failed to resolve alert:', error);
    }
  };

  return (
    <div className="alerts-panel">
      <h3>Active Alerts ({alerts.length})</h3>
      <div className="alerts-list">
        {alerts.map(alert => (
          <div key={alert.alert_id} className={`alert-item ${alert.severity}`}>
            <div className="alert-header">
              <span className="severity">{alert.severity.toUpperCase()}</span>
              <span className="service">{alert.service}</span>
              <span className="timestamp">{alert.timestamp}</span>
            </div>
            <div className="alert-message">{alert.message}</div>
            <div className="alert-actions">
              <button onClick={() => acknowledgeAlert(alert.alert_id)}>
                Acknowledge
              </button>
              <button onClick={() => resolveAlert(alert.alert_id)}>
                Resolve
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
```

#### Real-time Monitoring Component
```typescript
// Component for real-time monitoring with WebSocket
const RealTimeMonitoring = () => {
  const [metrics, setMetrics] = useState(null);
  const [ws, setWs] = useState(null);

  useEffect(() => {
    // Connect to WebSocket for real-time updates
    const websocket = new WebSocket('ws://monitoring:8006/ws/metrics');
    
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setMetrics(data);
    };

    websocket.onopen = () => {
      console.log('Connected to monitoring WebSocket');
    };

    websocket.onclose = () => {
      console.log('Disconnected from monitoring WebSocket');
    };

    setWs(websocket);

    return () => {
      websocket.close();
    };
  }, []);

  return (
    <div className="real-time-monitoring">
      <h3>Real-time Metrics</h3>
      {metrics && (
        <div className="metrics-display">
          <div className="metric">
            <label>CPU Usage:</label>
            <span>{metrics.cpu_usage}%</span>
          </div>
          <div className="metric">
            <label>Memory Usage:</label>
            <span>{metrics.memory_usage_mb} MB</span>
          </div>
          <div className="metric">
            <label>Active Alerts:</label>
            <span>{metrics.active_alerts}</span>
          </div>
        </div>
      )}
    </div>
  );
};
```

## Architecture Deep Dive

### Service Components

#### 1. Data Collector
**Purpose**: Collects metrics and health data from all platform services
**How it works**:
- Implements health check probes for all services
- Collects system metrics (CPU, memory, disk, network)
- Gathers application metrics (response times, error rates, throughput)
- Provides real-time data aggregation and processing

#### 2. Visualization Engine
**Purpose**: Creates charts, graphs, and dashboards for monitoring data
**How it works**:
- Generates real-time charts and visualizations
- Creates interactive dashboards for different user roles
- Provides historical trend analysis and forecasting
- Supports custom visualization configurations

#### 3. Alert Manager
**Purpose**: Manages alerting rules and notification delivery
**How it works**:
- Defines alerting rules and thresholds
- Evaluates metrics against alert conditions
- Manages alert state and escalation policies
- Delivers notifications via multiple channels

#### 4. Health Check Engine
**Purpose**: Performs comprehensive health checks across the platform
**How it works**:
- Implements service-specific health check logic
- Monitors dependencies and external services
- Tracks service availability and performance
- Provides health status aggregation and reporting

### Data Flow Architecture

```
All Services → Data Collector → Metrics Processing → Visualization → Dashboards
     ↓              ↓                ↓                ↓            ↓
  Health Checks   Aggregation    Trend Analysis   Real-time     Alerts
     ↓              ↓                ↓                ↓            ↓
  System Metrics  Data Storage   Forecasting     Historical    Notifications
     ↓              ↓                ↓                ↓            ↓
  App Metrics     Caching        Anomaly Detection  Reports     Escalation
```

## Configuration & Environment

### Environment Variables
```bash
# Service Configuration
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8006
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql://mlflow:password@postgres:5432/ml_security_consolidated
REDIS_URL=redis://redis:6379/1

# Monitoring Configuration
HEALTH_CHECK_INTERVAL=30  # seconds
METRICS_RETENTION_DAYS=30
ALERT_RETENTION_DAYS=90
DASHBOARD_REFRESH_INTERVAL=5  # seconds
```

## Security & Compliance

### Data Protection
- **Encryption**: All monitoring data encrypted in transit and at rest
- **Access Control**: Role-based access to monitoring data and alerts
- **Audit Logging**: Comprehensive audit trail for all operations
- **Data Retention**: Configurable data retention and archival policies

### Compliance Features
- **SOC 2**: Monitoring and logging controls
- **ISO 27001**: Information security monitoring
- **NIST Framework**: Cybersecurity monitoring and response
- **GDPR Compliance**: Privacy-preserving monitoring and data handling

## Performance Characteristics

### Scalability
- **Horizontal Scaling**: Stateless design supports multiple instances
- **Data Processing**: Efficient metrics processing and aggregation
- **Caching Strategy**: Multi-level caching for performance
- **Async Processing**: Non-blocking I/O for high throughput

### Reliability
- **Fault Tolerance**: Graceful handling of service failures
- **Data Consistency**: Eventual consistency for monitoring data
- **Backup & Recovery**: Automated backup and point-in-time recovery
- **Circuit Breakers**: Protection against cascading failures

## Troubleshooting Guide

### Common Issues
1. **Service Health Check Failures**: Check service connectivity and configuration
2. **Metrics Collection Issues**: Verify data collection configuration and permissions
3. **Alert Delivery Problems**: Check notification channels and configuration
4. **Dashboard Performance**: Monitor dashboard load times and optimize queries

### Debug Commands
```bash
# Check service health
curl http://localhost:8006/health

# Get dashboard data
curl http://localhost:8006/dashboard-data

# Check specific service health
curl http://localhost:8006/service-health

# View current alerts
curl http://localhost:8006/alerts
```

## Future Enhancements

### Planned Features
- **Machine Learning Monitoring**: ML-based anomaly detection and forecasting
- **Custom Dashboards**: User-configurable monitoring dashboards
- **Advanced Alerting**: Intelligent alerting with ML-based threshold adjustment
- **Distributed Tracing**: End-to-end request tracing and correlation
- **Capacity Planning**: Predictive scaling recommendations

### Research Areas
- **Anomaly Detection**: Advanced ML algorithms for anomaly detection
- **Predictive Monitoring**: Forecasting system behavior and failures
- **Intelligent Alerting**: AI-driven alert prioritization and noise reduction
- **Automated Remediation**: Self-healing systems based on monitoring insights