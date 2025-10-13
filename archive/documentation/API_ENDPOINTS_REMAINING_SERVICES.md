# Remaining Services - API Endpoints

This document covers the API endpoints for the remaining services in the ML Security Service ecosystem.

---

## Infrastructure Services

### MLflow Service (Port 5000)

**Base URL**: `http://localhost:5000`  
**Purpose**: ML experiment tracking and model registry

#### Key Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | MLflow UI |
| GET | `/health` | Health check |
| GET | `/api/2.0/mlflow/experiments` | List experiments |
| GET | `/api/2.0/mlflow/runs` | List runs |
| GET | `/api/2.0/mlflow/models` | List models |

#### Example Usage

```bash
# Access MLflow UI
open http://localhost:5000

# Health check
curl http://localhost:5000/health

# List experiments
curl http://localhost:5000/api/2.0/mlflow/experiments

# List runs
curl http://localhost:5000/api/2.0/mlflow/runs

# List models
curl http://localhost:5000/api/2.0/mlflow/models
```

---

### PostgreSQL Service (Port 5433)

**Base URL**: `http://localhost:5433`  
**Purpose**: Database for MLflow and analytics

#### Connection Details

```bash
# Connection string
postgresql://mlflow:password@localhost:5432/mlflow

# Direct connection
psql -h localhost -p 5433 -U mlflow -d mlflow
```

#### Key Tables

- `experiments` - MLflow experiments
- `runs` - MLflow runs
- `model_versions` - Model versions
- `red_team_results` - Red team test results
- `model_performance` - Model performance metrics
- `business_metrics` - Business KPIs
- `data_subjects` - GDPR data subjects

---

### MinIO Service (Port 9000)

**Base URL**: `http://localhost:9000`  
**Purpose**: S3-compatible object storage for MLflow artifacts

#### Access Details

- **Username**: `minioadmin`
- **Password**: `minioadmin`
- **Access Key**: `minioadmin`
- **Secret Key**: `minioadmin`

#### Key Buckets

- `mlflow` - MLflow artifacts
- `models` - Model files
- `training-data` - Training datasets
- `logs` - Service logs

#### Example Usage

```bash
# Access MinIO UI
open http://localhost:9000

# List buckets
mc ls local/

# List MLflow artifacts
mc ls local/mlflow/

# Download model
mc cp local/mlflow/0/abc123/models/deberta-v3-base ./model/
```

---

### Redis Service (Port 6380)

**Base URL**: `redis://localhost:6380`  
**Purpose**: Caching and session storage

#### Key Usage

- Model registry cache
- Red team results cache
- Session storage
- Rate limiting

#### Example Usage

```bash
# Connect to Redis
redis-cli -h localhost -p 6380

# List keys
redis-cli -h localhost -p 6380 keys "*"

# Get model registry
redis-cli -h localhost -p 6380 get "model_registry"

# Get red team results
redis-cli -h localhost -p 6380 get "red_team_results"
```

---

## Monitoring Services

### Prometheus Service (Port 9090)

**Base URL**: `http://localhost:9090`  
**Purpose**: Metrics collection and monitoring

#### Key Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Prometheus UI |
| GET | `/api/v1/targets` | List targets |
| GET | `/api/v1/query` | Query metrics |
| GET | `/api/v1/query_range` | Query time series |

#### Example Usage

```bash
# Access Prometheus UI
open http://localhost:9090

# Query metrics
curl "http://localhost:9090/api/v1/query?query=up"

# Query time series
curl "http://localhost:9090/api/v1/query_range?query=up&start=2024-01-01T00:00:00Z&end=2024-01-01T23:59:59Z&step=1m"

# List targets
curl http://localhost:9090/api/v1/targets
```

---

### Grafana Service (Port 3000)

**Base URL**: `http://localhost:3000`  
**Purpose**: Metrics visualization and dashboards

#### Access Details

- **Username**: `admin`
- **Password**: `admin`

#### Key Dashboards

- **ML Security Overview** - System overview
- **Model Performance** - Model metrics
- **Red Team Testing** - Security testing metrics
- **Business KPIs** - Business metrics
- **System Health** - Infrastructure metrics

#### Example Usage

```bash
# Access Grafana UI
open http://localhost:3000

# Import dashboard
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @dashboard.json
```

---

### Jaeger Service (Port 16686)

**Base URL**: `http://localhost:16686`  
**Purpose**: Distributed tracing

#### Key Features

- Request tracing
- Performance analysis
- Error tracking
- Service dependency mapping

#### Example Usage

```bash
# Access Jaeger UI
open http://localhost:16686

# Search traces
curl "http://localhost:16686/api/traces?service=model-api&limit=10"

# Get trace details
curl "http://localhost:16686/api/traces/abc123"
```

---

## Service Health Checks

### Comprehensive Health Check

```bash
#!/bin/bash

# Check all services
services=(
  "model-api:8000"
  "training:8002"
  "red-team:8001"
  "model-cache:8003"
  "model-serving:8080"
  "analytics:8006"
  "business-metrics:8004"
  "data-privacy:8005"
  "monitoring:8501"
  "mlflow:5000"
  "postgres:5433"
  "minio:9000"
  "redis:6380"
  "prometheus:9090"
  "grafana:3000"
  "jaeger:16686"
)

echo "Checking ML Security Service Health..."
echo "======================================"

for service in "${services[@]}"; do
  name=$(echo $service | cut -d: -f1)
  port=$(echo $service | cut -d: -f2)
  
  if curl -s "http://localhost:$port/" > /dev/null 2>&1; then
    echo "✅ $name ($port) - Healthy"
  else
    echo "❌ $name ($port) - Unhealthy"
  fi
done

echo "======================================"
echo "Health check completed"
```

### Individual Service Checks

```bash
# Model API
curl http://localhost:8000/health

# Training Service
curl http://localhost:8002/

# Red Team Service
curl http://localhost:8001/

# Model Cache
curl http://localhost:8003/health

# Model Serving
curl http://localhost:8080/health

# Analytics
curl http://localhost:8006/

# Business Metrics
curl http://localhost:8004/

# Data Privacy
curl http://localhost:8005/

# Monitoring Dashboard
curl http://localhost:8501/

# MLflow
curl http://localhost:5000/health

# PostgreSQL
pg_isready -h localhost -p 5433

# MinIO
curl http://localhost:9000/minio/health/live

# Redis
redis-cli -h localhost -p 6380 ping

# Prometheus
curl http://localhost:9090/-/healthy

# Grafana
curl http://localhost:3000/api/health

# Jaeger
curl http://localhost:16686/api/services
```

---

## Service Dependencies

### Dependency Graph

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MONITORING    │    │   RED TEAM      │    │   TRAINING      │
│   DASHBOARD     │◄──►│   SERVICE       │◄──►│   SERVICE       │
│   (Streamlit)   │    │   (FastAPI)     │    │   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MODEL API     │    │   MODEL CACHE   │    │   MODEL SERVING │
│   (FastAPI)     │◄──►│   (FastAPI)     │◄──►│   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ANALYTICS     │    │   BUSINESS      │    │   DATA PRIVACY  │
│   (FastAPI)     │    │   METRICS       │    │   (FastAPI)     │
│                 │    │   (FastAPI)     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MLFLOW        │    │   POSTGRESQL    │    │   MINIO         │
│   (Tracking)    │    │   (Database)    │    │   (Storage)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   REDIS         │    │   PROMETHEUS    │    │   GRAFANA       │
│   (Cache)       │    │   (Metrics)     │    │   (Visualization)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐
│   JAEGER        │    │   MONITORING    │
│   (Tracing)     │    │   DASHBOARD     │
└─────────────────┘    └─────────────────┘
```

### Service Startup Order

1. **Infrastructure Services**
   - PostgreSQL
   - Redis
   - MinIO
   - MLflow

2. **Core ML Services**
   - Model API
   - Training Service
   - Model Cache
   - Model Serving

3. **Security Services**
   - Red Team Service
   - Data Privacy Service

4. **Analytics Services**
   - Analytics Service
   - Business Metrics Service

5. **Monitoring Services**
   - Prometheus
   - Grafana
   - Jaeger
   - Monitoring Dashboard

---

## Configuration Management

### Environment Variables

#### Common Variables

```bash
# Database
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=mlflow
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=password

# Redis
REDIS_URL=redis://redis:6379

# MinIO
MINIO_URL=http://minio:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
MLFLOW_S3_ENDPOINT_URL=http://minio:9000

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# Logging
LOG_LEVEL=INFO
```

#### Service-Specific Variables

```bash
# Model API
MODEL_SERVING_URL=http://model-serving:8080

# Training Service
TRAINING_DATA_PATH=/app/training_data

# Red Team Service
RED_TEAM_BATCH_SIZE=10

# Analytics Service
ANALYTICS_RETENTION_DAYS=90

# Business Metrics
BUSINESS_METRICS_UPDATE_INTERVAL=300

# Data Privacy
DATA_PRIVACY_RETENTION_DAYS=365
```

---

## Troubleshooting Guide

### Common Issues

#### 1. Service Not Starting

```bash
# Check Docker status
docker-compose ps

# Check service logs
docker-compose logs service-name

# Restart service
docker-compose restart service-name

# Rebuild service
docker-compose up --build service-name
```

#### 2. Database Connection Issues

```bash
# Check PostgreSQL status
docker-compose logs postgres

# Check database connectivity
docker-compose exec postgres psql -U mlflow -d mlflow -c "SELECT 1;"

# Restart database
docker-compose restart postgres
```

#### 3. Memory Issues

```bash
# Check memory usage
docker stats

# Check service memory limits
docker-compose config

# Increase memory limits in docker-compose.yml
```

#### 4. Network Issues

```bash
# Check network connectivity
docker-compose exec service-name ping other-service

# Check DNS resolution
docker-compose exec service-name nslookup other-service

# Restart network
docker-compose down && docker-compose up -d
```

#### 5. Storage Issues

```bash
# Check disk space
df -h

# Check MinIO status
curl http://localhost:9000/minio/health/live

# Check MinIO logs
docker-compose logs minio
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
docker-compose up -d

# Check debug logs
docker-compose logs -f service-name
```

### Performance Monitoring

```bash
# Monitor resource usage
docker stats

# Check service health
curl http://localhost:8000/health

# Check metrics
curl http://localhost:9090/api/v1/query?query=up
```

---

## Maintenance Tasks

### Regular Maintenance

#### Daily Tasks

```bash
# Check service health
./health_check.sh

# Check disk space
df -h

# Check logs for errors
docker-compose logs --since=24h | grep ERROR
```

#### Weekly Tasks

```bash
# Clean up old logs
docker system prune -f

# Check database size
docker-compose exec postgres psql -U mlflow -d mlflow -c "SELECT pg_size_pretty(pg_database_size('mlflow'));"

# Update service images
docker-compose pull
```

#### Monthly Tasks

```bash
# Full system backup
docker-compose exec postgres pg_dump -U mlflow mlflow > backup.sql

# Clean up old data
docker-compose exec analytics python cleanup.py

# Review performance metrics
curl http://localhost:9090/api/v1/query?query=up
```

### Backup and Recovery

#### Backup

```bash
# Database backup
docker-compose exec postgres pg_dump -U mlflow mlflow > backup_$(date +%Y%m%d).sql

# MinIO backup
mc mirror local/mlflow ./backup/mlflow/

# Configuration backup
cp docker-compose.yml backup/
cp .env backup/
```

#### Recovery

```bash
# Database recovery
docker-compose exec postgres psql -U mlflow -d mlflow < backup.sql

# MinIO recovery
mc mirror ./backup/mlflow/ local/mlflow

# Service recovery
docker-compose up -d
```

---

## Security Considerations

### Network Security

- All services run on internal Docker network
- External access only through specific ports
- No direct database access from outside

### Data Security

- All data encrypted in transit
- Sensitive data encrypted at rest
- Regular security updates

### Access Control

- Service-to-service authentication
- API key management
- Role-based access control

### Compliance

- GDPR compliance through Data Privacy Service
- Audit logging for all operations
- Data retention policies

---

## Performance Optimization

### Resource Allocation

```yaml
# docker-compose.yml
services:
  model-api:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### Caching Strategy

- Redis for session and model registry cache
- Model Cache Service for model preloading
- MinIO for model artifact storage

### Monitoring

- Prometheus for metrics collection
- Grafana for visualization
- Jaeger for distributed tracing

---

## Conclusion

The ML Security Service provides a comprehensive, enterprise-grade platform for machine learning security with 12 microservices, 81 API endpoints, and advanced capabilities including continuous learning, real-time monitoring, business metrics, and GDPR compliance.

All services are designed to work together seamlessly, providing a complete solution for ML security that can be deployed locally without any cloud dependencies.
