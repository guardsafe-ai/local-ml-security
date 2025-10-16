# Enterprise Dashboard Backend Services

## Modular API Client Architecture

This directory contains a comprehensive, modular API client architecture that provides **100% API coverage** for all ML Security services. The architecture is designed for maintainability, scalability, and ease of use.

## Architecture Overview

```
MainAPIClient (Orchestrator)
├── ModelAPIClient (port 8000)      - Inference & model management
├── TrainingClient (port 8002)      - ML model training & MLflow integration  
├── ModelCacheClient (port 8003)    - Intelligent model preloading
├── BusinessMetricsClient (port 8004) - KPI tracking & business intelligence
├── AnalyticsClient (port 8006)     - Performance analytics & drift detection
├── DataPrivacyClient (port 8008)   - GDPR compliance & data protection
├── TracingClient (port 8009)       - Distributed tracing & monitoring
└── MLflowClient (port 5000)        - Experiment tracking & model registry
```

## Key Features

### ✅ 100% API Coverage
- Every endpoint from every service is fully integrated
- No API functionality is lost or missing
- Complete feature parity with individual services

### ✅ Modular Design
- Each service has its own dedicated client
- Easy to maintain and extend individual services
- Clear separation of concerns

### ✅ Unified Interface
- Single entry point through `MainAPIClient`
- Unified prediction endpoints that try multiple services
- Comprehensive dashboard aggregation

### ✅ Error Handling & Resilience
- Graceful fallbacks between services
- Comprehensive error handling
- Circuit breaker patterns

### ✅ Caching & Performance
- Redis-based caching for frequently accessed data
- Configurable cache TTL per endpoint
- Optimized for high-performance scenarios

### ✅ Backward Compatibility
- `APIClient` alias for existing code
- No breaking changes to existing integrations

## Usage Examples

### Basic Usage

```python
from services import MainAPIClient

# Initialize the main client
client = MainAPIClient()

# Get health of all services
health = await client.get_all_services_health()

# Unified prediction (tries cache first, falls back to model API)
prediction = await client.predict_unified(
    text="This is a test prompt",
    model_name="security-classifier",
    use_cache=True
)

# Get comprehensive dashboard data
dashboard = await client.get_dashboard_overview()
```

### Individual Service Access

```python
# Access specific service clients
model_api = client.model_api
training = client.training
analytics = client.analytics

# Use service-specific methods
models = await model_api.get_models()
jobs = await training.get_training_jobs()
drift = await analytics.get_drift_analysis()
```

### Advanced Usage

```python
# Batch operations
predictions = await client.predict_batch_unified(
    texts=["text1", "text2", "text3"],
    ensemble=True
)

# Get system-wide metrics
metrics = await client.get_all_metrics()

# Model management overview
model_overview = await client.get_model_management_overview()

# Security and compliance overview
security_overview = await client.get_security_overview()
```

## Service Clients

### ModelAPIClient (port 8000)
- **Health**: `/health`, `/health/deep`, `/health/ready`, `/health/live`
- **Predictions**: `/predict`, `/predict/batch`
- **Model Management**: `/models`, `/models/{name}`, `/models/load`, `/models/unload`
- **Batch Operations**: `/models/batch-load`, `/models/warm-cache/{name}`
- **Metrics**: `/metrics`

### TrainingClient (port 8002)
- **Health**: `/health`, `/health/deep`
- **Model Management**: `/models/models`, `/models/model-registry`, `/models/latest-models`
- **Training**: `/training/jobs`, `/training/train`, `/training/retrain`
- **Data Management**: `/data/upload-data`, `/data/efficient/upload-large-file`
- **Configuration**: `/training/config/{model_name}`

### ModelCacheClient (port 8003)
- **Health**: `/health`, `/`
- **Inference**: `/predict`, `/predict/batch`
- **Model Management**: `/models`, `/models/{name}/load`, `/models/{name}/unload`
- **Cache Management**: `/stats`, `/clear-cache`, `/config`
- **Logging**: `/logs`, `/models/{name}/logs`

### BusinessMetricsClient (port 8004)
- **Health**: `/health`, `/`
- **Metrics**: `/metrics`, `/metrics/summary`, `/metrics/kpis`
- **Performance**: `/performance`, `/throughput`, `/latency`
- **Security**: `/security`, `/threats`, `/compliance`
- **Analytics**: `/analytics/trends`, `/analytics/correlation`
- **Reporting**: `/reports/generate`, `/metrics/export`

### AnalyticsClient (port 8006)
- **Health**: `/health`, `/`
- **Analytics**: `/analytics/trends`, `/analytics/summary`, `/analytics/drift`
- **Red Team**: `/red-team/summary`, `/red-team/trends`, `/red-team/patterns`
- **Performance**: `/model-performance/{name}`, `/performance/metrics`
- **Drift Detection**: `/drift/status`, `/drift/history`, `/drift/alerts`
- **Auto-Retrain**: `/auto-retrain/config`, `/auto-retrain/trigger/{name}`

### DataPrivacyClient (port 8008)
- **Health**: `/health`, `/health/health`, `/`
- **Classification**: `/classify`, `/classify/history`
- **Anonymization**: `/anonymize`, `/anonymize/batch`
- **Data Subjects**: `/data-subjects`, `/data-subjects/{id}`
- **Consent**: `/consent/record`, `/consent/status/{user_id}`, `/consent/withdraw`
- **Retention**: `/retention/policies`, `/retention/status`, `/retention/cleanup`
- **Audit**: `/audit/logs`, `/compliance/status`, `/compliance/violations`

### TracingClient (port 8009)
- **Health**: `/health`, `/`
- **Traces**: `/traces`, `/traces/{id}`, `/traces/{id}/spans`
- **Spans**: `/traces/{id}/spans/{span_id}`
- **Metrics**: `/metrics/traces`, `/metrics/performance`, `/metrics/errors`
- **Services**: `/services`, `/services/{name}/operations`
- **Dashboard**: `/dashboard`, `/service-map`, `/search/traces`

### MLflowClient (port 5000)
- **Health**: `/health`, `/`
- **Experiments**: `/api/2.0/mlflow/experiments/list`, `/api/2.0/mlflow/experiments/create`
- **Runs**: `/api/2.0/mlflow/runs/search`, `/api/2.0/mlflow/runs/get`
- **Models**: `/api/2.0/mlflow/registered-models/list`, `/api/2.0/mlflow/registered-models/create`
- **Versions**: `/api/2.0/mlflow/model-versions/create`, `/api/2.0/mlflow/model-versions/list`
- **Artifacts**: `/api/2.0/mlflow/artifacts/list`, `/api/2.0/mlflow/artifacts/get-uri`

## Configuration

### Environment Variables
```bash
# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6380
REDIS_DB=0

# Service URLs (automatically configured)
MODEL_API_URL=http://model-api:8000
TRAINING_URL=http://training:8002
MODEL_CACHE_URL=http://model-cache:8003
BUSINESS_METRICS_URL=http://business-metrics:8004
ANALYTICS_URL=http://analytics:8006
DATA_PRIVACY_URL=http://data-privacy:8008
TRACING_URL=http://tracing:8009
MLFLOW_URL=http://mlflow:5000
```

### Caching Configuration
- **Default TTL**: 300 seconds (5 minutes)
- **Health Checks**: 60 seconds
- **Metrics**: 60 seconds
- **Model Info**: 300 seconds
- **Experiments**: 300 seconds

## Error Handling

The architecture includes comprehensive error handling:

1. **Service Unavailable**: Graceful fallbacks and error reporting
2. **Timeout Handling**: Configurable timeouts for different operations
3. **Circuit Breakers**: Protection against cascading failures
4. **Retry Logic**: Automatic retries with exponential backoff
5. **Caching Fallbacks**: Cached data when services are unavailable

## Performance Optimizations

1. **Parallel Requests**: Multiple service calls executed concurrently
2. **Intelligent Caching**: Redis-based caching with appropriate TTLs
3. **Connection Pooling**: Reused HTTP connections
4. **Batch Operations**: Efficient batch processing where possible
5. **Lazy Loading**: Services loaded only when needed

## Monitoring & Observability

- **Health Checks**: Comprehensive health monitoring for all services
- **Metrics Collection**: Prometheus metrics from all services
- **Distributed Tracing**: Full trace correlation across services
- **Error Tracking**: Detailed error logging and reporting
- **Performance Monitoring**: Latency and throughput metrics

## Migration Guide

### From Old API Client
```python
# Old way
from services.api_client import APIClient
client = APIClient()

# New way (backward compatible)
from services import MainAPIClient
client = MainAPIClient()

# Or use the alias for existing code
from services import APIClient
client = APIClient()  # Same as MainAPIClient
```

### New Features Available
```python
# Unified predictions with fallbacks
prediction = await client.predict_unified(text, use_cache=True)

# Comprehensive dashboard data
dashboard = await client.get_dashboard_overview()

# Individual service access
model_api = client.model_api
analytics = client.analytics

# System-wide health and metrics
health = await client.get_all_services_health()
metrics = await client.get_all_metrics()
```

## Testing

The architecture includes comprehensive testing capabilities:

```python
# Test individual services
health = await client.get_service_health("model_api")

# Test all services
all_health = await client.get_all_services_health()

# Test unified predictions
prediction = await client.predict_unified("test text")

# Test dashboard aggregation
dashboard = await client.get_dashboard_overview()
```

## Future Enhancements

1. **Service Discovery**: Automatic service discovery and registration
2. **Load Balancing**: Intelligent load balancing across service instances
3. **Rate Limiting**: Built-in rate limiting and throttling
4. **Authentication**: Service-to-service authentication
5. **Metrics Dashboard**: Real-time metrics visualization
6. **Auto-scaling**: Automatic scaling based on load

## Support

For issues or questions about the API client architecture:

1. Check the individual service documentation
2. Review the error logs for specific service failures
3. Use the health check endpoints to diagnose issues
4. Monitor the system metrics for performance insights

This modular architecture provides a robust, scalable, and maintainable foundation for the Enterprise Dashboard Backend, ensuring 100% API coverage while maintaining excellent performance and reliability.
