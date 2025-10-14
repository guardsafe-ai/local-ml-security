# ML Security Platform - Infrastructure Services Documentation

## Overview

The ML Security platform relies on a comprehensive infrastructure stack to support machine learning operations, data management, monitoring, and observability. This document provides detailed information about all infrastructure services, their configurations, and integration patterns.

## Infrastructure Services Overview

### Core Infrastructure Services

| Service | Port | Purpose | Technology |
|---------|------|---------|------------|
| PostgreSQL | 5432 | Primary Database | PostgreSQL 15+ |
| Redis | 6379 | Caching & Session Storage | Redis 7+ |
| MinIO | 9000/9001 | Object Storage (S3-compatible) | MinIO |
| MLflow | 5000 | ML Experiment Tracking | MLflow 2.0+ |
| Jaeger | 16686/14268 | Distributed Tracing | Jaeger |
| Prometheus | 9090 | Metrics Collection | Prometheus |
| Grafana | 3001 | Metrics Visualization | Grafana |

---

## PostgreSQL Database

### Overview
PostgreSQL serves as the primary database for the ML Security platform, storing metadata, configurations, experiment data, and business metrics.

### Configuration
```yaml
postgres:
  image: postgres:15-alpine
  container_name: local-ml-security-postgres-1
  environment:
    POSTGRES_DB: ml_security_consolidated
    POSTGRES_USER: mlflow
    POSTGRES_PASSWORD: password
    POSTGRES_INITDB_ARGS: "--encoding=UTF-8 --lc-collate=C --lc-ctype=C"
  ports:
    - "5432:5432"
  volumes:
    - postgres_data:/var/lib/postgresql/data
    - ./data/postgres/init:/docker-entrypoint-initdb.d
  networks:
    - ml-network
  restart: unless-stopped
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U mlflow -d ml_security_consolidated"]
    interval: 30s
    timeout: 10s
    retries: 3
```

### Database Schema
The database contains multiple schemas for different services:

#### MLflow Schema
- **experiments**: MLflow experiment metadata
- **runs**: MLflow run data and metrics
- **metrics**: Metric values and timestamps
- **params**: Run parameters
- **tags**: Run and experiment tags
- **model_versions**: Model version information
- **registered_models**: Registered model metadata

#### Analytics Schema
- **model_performance**: Model performance metrics
- **drift_detection**: Data and model drift results
- **red_team_results**: Red team testing results
- **attack_patterns**: Attack pattern definitions
- **vulnerabilities**: Security vulnerability data

#### Business Metrics Schema
- **kpis**: Key performance indicators
- **cost_metrics**: Cost tracking data
- **resource_utilization**: Resource usage metrics
- **sla_metrics**: Service level agreement metrics

#### Data Privacy Schema
- **data_classifications**: Data classification results
- **pii_detections**: PII detection records
- **anonymization_logs**: Data anonymization logs
- **compliance_reports**: Privacy compliance reports
- **data_subjects**: Data subject information
- **consent_records**: Consent management data

### Connection Configuration
```python
# Database connection string
DATABASE_URL = "postgresql://mlflow:password@postgres:5432/ml_security_consolidated"

# Connection pool settings
DB_POOL_SIZE = 20
DB_MAX_OVERFLOW = 30
DB_POOL_TIMEOUT = 30
DB_POOL_RECYCLE = 3600
```

### Backup & Recovery
```bash
# Create backup
docker exec local-ml-security-postgres-1 pg_dump -U mlflow ml_security_consolidated > backup.sql

# Restore backup
docker exec -i local-ml-security-postgres-1 psql -U mlflow ml_security_consolidated < backup.sql
```

---

## Redis Cache

### Overview
Redis provides high-performance caching, session storage, and pub/sub messaging for the ML Security platform.

### Configuration
```yaml
redis:
  image: redis:7-alpine
  container_name: local-ml-security-redis-1
  command: redis-server --appendonly yes --requirepass redis_password
  ports:
    - "6379:6379"
  volumes:
    - redis_data:/data
  networks:
    - ml-network
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "redis-cli", "ping"]
    interval: 30s
    timeout: 10s
    retries: 3
```

### Usage Patterns

#### Caching
```python
# Model prediction caching
import redis
import json

redis_client = redis.Redis(host='redis', port=6379, password='redis_password')

def cache_prediction(text, model_name, result):
    key = f"prediction:{hash(text)}:{model_name}"
    redis_client.setex(key, 3600, json.dumps(result))  # 1 hour TTL

def get_cached_prediction(text, model_name):
    key = f"prediction:{hash(text)}:{model_name}"
    cached = redis_client.get(key)
    return json.loads(cached) if cached else None
```

#### Session Storage
```python
# User session management
def store_session(session_id, user_data):
    key = f"session:{session_id}"
    redis_client.setex(key, 86400, json.dumps(user_data))  # 24 hours TTL

def get_session(session_id):
    key = f"session:{session_id}"
    session_data = redis_client.get(key)
    return json.loads(session_data) if session_data else None
```

#### Pub/Sub Messaging
```python
# Real-time notifications
def publish_notification(channel, message):
    redis_client.publish(channel, json.dumps(message))

def subscribe_to_notifications(channel, callback):
    pubsub = redis_client.pubsub()
    pubsub.subscribe(channel)
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            callback(json.loads(message['data']))
```

### Memory Management
```bash
# Redis memory usage
docker exec local-ml-security-redis-1 redis-cli info memory

# Clear cache
docker exec local-ml-security-redis-1 redis-cli flushall

# Monitor Redis
docker exec local-ml-security-redis-1 redis-cli monitor
```

---

## MinIO Object Storage

### Overview
MinIO provides S3-compatible object storage for ML artifacts, datasets, and model files.

### Configuration
```yaml
minio:
  image: minio/minio:latest
  container_name: local-ml-security-minio-1
  command: server /data --console-address ":9001"
  environment:
    MINIO_ROOT_USER: minioadmin
    MINIO_ROOT_PASSWORD: minioadmin
    MINIO_DEFAULT_BUCKETS: mlflow-artifacts,training-data,model-cache,datasets
  ports:
    - "9000:9000"  # API
    - "9001:9001"  # Console
  volumes:
    - minio_data:/data
  networks:
    - ml-network
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
    interval: 30s
    timeout: 10s
    retries: 3
```

### Bucket Structure
```
mlflow-artifacts/
├── experiments/
│   ├── {experiment_id}/
│   │   └── {run_id}/
│   │       ├── artifacts/
│   │       ├── models/
│   │       └── metrics/
├── models/
│   ├── {model_name}/
│   │   └── {version}/
│   │       ├── model.pkl
│   │       ├── metadata.json
│   │       └── requirements.txt
└── datasets/
    ├── training/
    ├── validation/
    └── test/

training-data/
├── raw/
├── processed/
└── augmented/

model-cache/
├── {model_name}/
│   ├── weights/
│   ├── config/
│   └── tokenizer/
```

### S3 Client Configuration
```python
import boto3
from botocore.client import Config

# MinIO S3 client
s3_client = boto3.client(
    's3',
    endpoint_url='http://minio:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin',
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

# Upload file
def upload_file(bucket, key, file_path):
    s3_client.upload_file(file_path, bucket, key)

# Download file
def download_file(bucket, key, file_path):
    s3_client.download_file(bucket, key, file_path)

# List objects
def list_objects(bucket, prefix=''):
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return response.get('Contents', [])
```

### Access Control
```python
# Bucket policies
bucket_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": "*",
            "Action": ["s3:GetObject"],
            "Resource": ["arn:aws:s3:::mlflow-artifacts/*"]
        }
    ]
}

# Apply bucket policy
s3_client.put_bucket_policy(
    Bucket='mlflow-artifacts',
    Policy=json.dumps(bucket_policy)
)
```

---

## MLflow Experiment Tracking

### Overview
MLflow provides comprehensive experiment tracking, model registry, and artifact management for the ML Security platform.

### Configuration
```yaml
mlflow:
  build:
    context: ./services/mlflow
    dockerfile: Dockerfile
  container_name: local-ml-security-mlflow-1
  ports:
    - "5000:5000"
  environment:
    MLFLOW_BACKEND_STORE_URI: postgresql://mlflow:password@postgres:5432/ml_security_consolidated
    MLFLOW_DEFAULT_ARTIFACT_ROOT: s3://mlflow-artifacts
    AWS_ACCESS_KEY_ID: minioadmin
    AWS_SECRET_ACCESS_KEY: minioadmin
    MLFLOW_S3_ENDPOINT_URL: http://minio:9000
  depends_on:
    - postgres
    - minio
  volumes:
    - ./data/mlflow:/mlflow
  networks:
    - ml-network
  restart: unless-stopped
```

### MLflow Integration
```python
import mlflow
import mlflow.sklearn
import mlflow.pytorch

# Configure MLflow
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("ml-security-experiments")

# Log experiment
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 10)
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("f1_score", 0.92)
    mlflow.log_metric("precision", 0.94)
    mlflow.log_metric("recall", 0.90)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("training_data.csv")
    mlflow.log_artifact("model_config.json")
```

### Model Registry
```python
# Register model
model_name = "security-classifier"
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name=model_name
)

# Transition model stage
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production"
)

# Load model for inference
model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
```

### Experiment Management
```python
# List experiments
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"Experiment: {exp.name} (ID: {exp.experiment_id})")

# Search runs
runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"]
)

# Compare runs
mlflow.compare_runs(run_ids)
```

---

## Jaeger Distributed Tracing

### Overview
Jaeger provides distributed tracing capabilities for monitoring request flows across the ML Security platform.

### Configuration
```yaml
jaeger:
  image: jaegertracing/all-in-one:latest
  container_name: local-ml-security-jaeger-1
  ports:
    - "16686:16686"  # Web UI
    - "14268:14268"  # HTTP collector
    - "14250:14250"  # gRPC collector
  environment:
    COLLECTOR_OTLP_ENABLED: true
  networks:
    - ml-network
  restart: unless-stopped
```

### Tracing Integration
```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=14268,
)

# Configure tracer provider
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Add span processor
span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Create spans
def predict_with_tracing(text, model_name):
    with tracer.start_as_current_span("model_prediction") as span:
        span.set_attribute("model_name", model_name)
        span.set_attribute("input_length", len(text))
        
        # Model prediction logic
        result = model.predict(text)
        
        span.set_attribute("prediction", result.prediction)
        span.set_attribute("confidence", result.confidence)
        
        return result
```

### Trace Analysis
```python
# Query traces
from jaeger_client import Config

config = Config(
    config={
        'sampler': {'type': 'const', 'param': 1},
        'logging': True,
    },
    service_name='ml-security-service'
)

tracer = config.initialize_tracer()

# Search traces
def search_traces(service_name, operation_name, start_time, end_time):
    # Implementation for trace search
    pass
```

---

## Prometheus Metrics Collection

### Overview
Prometheus collects and stores metrics from all ML Security services for monitoring and alerting.

### Configuration
```yaml
prometheus:
  image: prom/prometheus:latest
  container_name: local-ml-security-prometheus-1
  ports:
    - "9090:9090"
  volumes:
    - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
    - prometheus_data:/prometheus
  command:
    - '--config.file=/etc/prometheus/prometheus.yml'
    - '--storage.tsdb.path=/prometheus'
    - '--web.console.libraries=/etc/prometheus/console_libraries'
    - '--web.console.templates=/etc/prometheus/consoles'
    - '--storage.tsdb.retention.time=200h'
    - '--web.enable-lifecycle'
  networks:
    - ml-network
  restart: unless-stopped
```

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'ml-security-services'
    static_configs:
      - targets: 
        - 'model-api:8000'
        - 'training:8002'
        - 'analytics:8006'
        - 'business-metrics:8004'
        - 'data-privacy:8008'
        - 'model-cache:8003'
        - 'tracing:8009'
        - 'enterprise-dashboard-backend:8007'
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'infrastructure'
    static_configs:
      - targets:
        - 'postgres:5432'
        - 'redis:6379'
        - 'minio:9000'
        - 'mlflow:5000'
        - 'jaeger:16686'
```

### Custom Metrics
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
MODEL_LOAD_TIME = Histogram('model_load_duration_seconds', 'Model loading time')

# Record metrics
def record_request(method, endpoint, duration):
    REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
    REQUEST_DURATION.observe(duration)

def record_model_load(duration):
    MODEL_LOAD_TIME.observe(duration)

# Start metrics server
start_http_server(8000)
```

### Alert Rules
```yaml
# alert_rules.yml
groups:
  - name: ml-security-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          
      - alert: ModelLoadFailure
        expr: increase(model_load_failures_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Model loading failures detected"
          
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
```

---

## Grafana Visualization

### Overview
Grafana provides comprehensive dashboards for visualizing metrics, logs, and traces from the ML Security platform.

### Configuration
```yaml
grafana:
  image: grafana/grafana:latest
  container_name: local-ml-security-grafana-1
  ports:
    - "3001:3000"
  environment:
    GF_SECURITY_ADMIN_PASSWORD: admin
    GF_INSTALL_PLUGINS: grafana-piechart-panel,grafana-worldmap-panel
  volumes:
    - grafana_data:/var/lib/grafana
    - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
  networks:
    - ml-network
  restart: unless-stopped
```

### Dashboard Configuration
```json
{
  "dashboard": {
    "title": "ML Security Platform Overview",
    "panels": [
      {
        "title": "System Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"ml-security-services\"}",
            "legendFormat": "{{instance}}"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{instance}}"
          }
        ]
      },
      {
        "title": "Model Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "model_accuracy",
            "legendFormat": "{{model_name}}"
          }
        ]
      }
    ]
  }
}
```

### Data Sources
```yaml
# datasources.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    
  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
```

---

## Network Configuration

### Docker Network
```yaml
networks:
  ml-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Service Communication
```python
# Service discovery
SERVICE_ENDPOINTS = {
    'model-api': 'http://model-api:8000',
    'training': 'http://training:8002',
    'analytics': 'http://analytics:8006',
    'business-metrics': 'http://business-metrics:8004',
    'data-privacy': 'http://data-privacy:8008',
    'model-cache': 'http://model-cache:8003',
    'tracing': 'http://tracing:8009',
    'enterprise-dashboard-backend': 'http://enterprise-dashboard-backend:8007'
}
```

---

## Monitoring & Observability

### Health Checks
```yaml
# Service health check example
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### Logging Configuration
```python
import logging
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
```

### Metrics Collection
```python
# Custom metrics for ML operations
from prometheus_client import Counter, Histogram, Gauge

# Model metrics
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total model predictions', ['model_name', 'status'])
MODEL_LOAD_TIME = Histogram('model_load_duration_seconds', 'Model loading time', ['model_name'])
ACTIVE_MODELS = Gauge('active_models', 'Number of active models')

# Training metrics
TRAINING_JOBS = Counter('training_jobs_total', 'Total training jobs', ['status'])
TRAINING_DURATION = Histogram('training_duration_seconds', 'Training duration', ['model_name'])

# Security metrics
ATTACKS_DETECTED = Counter('attacks_detected_total', 'Total attacks detected', ['attack_type', 'severity'])
DETECTION_RATE = Gauge('detection_rate', 'Current detection rate')
```

---

## Backup & Recovery

### Database Backup
```bash
#!/bin/bash
# Database backup script

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
docker exec local-ml-security-postgres-1 pg_dump -U mlflow ml_security_consolidated > "${BACKUP_DIR}/postgres_${DATE}.sql"

# Compress backup
gzip "${BACKUP_DIR}/postgres_${DATE}.sql"

# Cleanup old backups (keep 30 days)
find "${BACKUP_DIR}" -name "postgres_*.sql.gz" -mtime +30 -delete
```

### MinIO Backup
```bash
#!/bin/bash
# MinIO backup script

BACKUP_DIR="/backups/minio"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "${BACKUP_DIR}/${DATE}"

# Sync MinIO data
docker exec local-ml-security-minio-1 mc mirror /data "${BACKUP_DIR}/${DATE}/"

# Cleanup old backups
find "${BACKUP_DIR}" -type d -mtime +30 -exec rm -rf {} \;
```

### Configuration Backup
```bash
#!/bin/bash
# Configuration backup script

CONFIG_BACKUP_DIR="/backups/config"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup Docker Compose files
tar -czf "${CONFIG_BACKUP_DIR}/docker-compose_${DATE}.tar.gz" docker-compose.yml

# Backup monitoring configurations
tar -czf "${CONFIG_BACKUP_DIR}/monitoring_${DATE}.tar.gz" monitoring/

# Backup service configurations
tar -czf "${CONFIG_BACKUP_DIR}/services_${DATE}.tar.gz" services/
```

---

## Security Considerations

### Network Security
- **Internal Network**: All services communicate through internal Docker network
- **Port Exposure**: Only necessary ports are exposed to host
- **Firewall Rules**: Implement firewall rules for external access
- **TLS/SSL**: Use HTTPS for external communications

### Data Security
- **Encryption at Rest**: Enable encryption for sensitive data
- **Encryption in Transit**: Use TLS for all communications
- **Access Control**: Implement proper authentication and authorization
- **Audit Logging**: Log all access and modifications

### Container Security
- **Image Scanning**: Scan container images for vulnerabilities
- **Non-root Users**: Run containers as non-root users
- **Resource Limits**: Set appropriate resource limits
- **Security Updates**: Regularly update base images

---

## Performance Optimization

### Database Optimization
```sql
-- Create indexes for better performance
CREATE INDEX idx_model_performance_model_name ON model_performance(model_name);
CREATE INDEX idx_red_team_results_timestamp ON red_team_results(timestamp);
CREATE INDEX idx_training_jobs_status ON training_jobs(status);

-- Analyze tables for query optimization
ANALYZE model_performance;
ANALYZE red_team_results;
ANALYZE training_jobs;
```

### Redis Optimization
```bash
# Redis configuration for performance
maxmemory 2gb
maxmemory-policy allkeys-lru
tcp-keepalive 60
timeout 300
```

### MinIO Optimization
```bash
# MinIO configuration for performance
MINIO_CACHE_DRIVES="/tmp/cache1,/tmp/cache2"
MINIO_CACHE_EXCLUDE="*.pdf,*.mp4"
MINIO_CACHE_QUOTA=80
MINIO_CACHE_AFTER=3
MINIO_CACHE_WATERMARK_LOW=70
MINIO_CACHE_WATERMARK_HIGH=90
```

---

## Troubleshooting

### Common Issues

#### Database Connection Issues
```bash
# Check database connectivity
docker exec local-ml-security-postgres-1 pg_isready -U mlflow -d ml_security_consolidated

# Check database logs
docker logs local-ml-security-postgres-1

# Check database size
docker exec local-ml-security-postgres-1 psql -U mlflow -d ml_security_consolidated -c "SELECT pg_size_pretty(pg_database_size('ml_security_consolidated'));"
```

#### Redis Issues
```bash
# Check Redis connectivity
docker exec local-ml-security-redis-1 redis-cli ping

# Check Redis memory usage
docker exec local-ml-security-redis-1 redis-cli info memory

# Check Redis logs
docker logs local-ml-security-redis-1
```

#### MinIO Issues
```bash
# Check MinIO connectivity
curl -f http://localhost:9000/minio/health/live

# Check MinIO logs
docker logs local-ml-security-minio-1

# Check MinIO status
docker exec local-ml-security-minio-1 mc admin info local
```

#### MLflow Issues
```bash
# Check MLflow connectivity
curl -f http://localhost:5000/health

# Check MLflow logs
docker logs local-ml-security-mlflow-1

# Check MLflow database connection
docker exec local-ml-security-mlflow-1 mlflow db upgrade
```

### Performance Monitoring
```bash
# Monitor resource usage
docker stats

# Monitor network traffic
docker exec local-ml-security-postgres-1 netstat -i

# Monitor disk usage
docker exec local-ml-security-postgres-1 df -h
```

---

**ML Security Platform Infrastructure** - Comprehensive documentation covering all infrastructure services, their configurations, integration patterns, monitoring, security, and troubleshooting for the ML Security platform's supporting infrastructure.
