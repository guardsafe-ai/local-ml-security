# Tracing Service - API Documentation

## Overview

The Tracing Service provides a comprehensive REST API for distributed tracing, trace collection, and performance monitoring. This document details all available endpoints, their inputs, outputs, and usage examples.

## Base URL
```
http://localhost:8009
```

## Authentication
Currently, the service does not require authentication, but this may be added in future versions.

## Content Types
- **Request**: `application/json` for JSON payloads
- **Response**: `application/json`

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request
```json
{
  "detail": "Error message describing the issue"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error message"
}
```

---

## Health Endpoints

### GET /
**Description**: Root endpoint with service information

**Response**:
```json
{
  "service": "tracing",
  "version": "1.0.0",
  "status": "running",
  "description": "Distributed tracing service for ML Security platform"
}
```

### GET /health
**Description**: Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "service": "tracing",
  "message": "Tracing service is running"
}
```

---

## Usage Examples

### 1. Basic Health Check

```bash
# Check service health
curl http://localhost:8009/health

# Get service information
curl http://localhost:8009/
```

### 2. Service Integration

```javascript
// Check if tracing service is available
const checkTracingHealth = async () => {
  try {
    const response = await fetch('http://localhost:8009/health');
    const health = await response.json();
    
    if (health.status === 'healthy') {
      console.log('✅ Tracing service is healthy');
      return true;
    } else {
      console.error('❌ Tracing service is unhealthy:', health);
      return false;
    }
  } catch (error) {
    console.error('Failed to check tracing health:', error);
    return false;
  }
};

// Get service information
const getTracingInfo = async () => {
  try {
    const response = await fetch('http://localhost:8009/');
    const info = await response.json();
    console.log('Tracing Service Info:', info);
    return info;
  } catch (error) {
    console.error('Failed to get tracing info:', error);
  }
};
```

### 3. Python Integration

```python
import requests

def check_tracing_health():
    """Check if tracing service is healthy"""
    try:
        response = requests.get('http://localhost:8009/health')
        response.raise_for_status()
        health = response.json()
        
        if health['status'] == 'healthy':
            print('✅ Tracing service is healthy')
            return True
        else:
            print(f'❌ Tracing service is unhealthy: {health}')
            return False
    except requests.RequestException as e:
        print(f'Failed to check tracing health: {e}')
        return False

def get_tracing_info():
    """Get tracing service information"""
    try:
        response = requests.get('http://localhost:8009/')
        response.raise_for_status()
        info = response.json()
        print(f'Tracing Service Info: {info}')
        return info
    except requests.RequestException as e:
        print(f'Failed to get tracing info: {e}')
        return None
```

---

## Request/Response Models

### HealthResponse
```json
{
  "status": "healthy",
  "service": "tracing",
  "message": "Tracing service is running"
}
```

### ServiceInfoResponse
```json
{
  "service": "tracing",
  "version": "1.0.0",
  "status": "running",
  "description": "Distributed tracing service for ML Security platform"
}
```

---

## Error Handling

### Common Error Scenarios

1. **Service Unavailable (503)**
   ```json
   {
     "detail": "Tracing service temporarily unavailable"
   }
   ```

2. **Internal Server Error (500)**
   ```json
   {
     "detail": "Internal server error"
   }
   ```

### Error Response Format
All error responses follow this format:
```json
{
  "detail": "Error message describing the issue",
  "error_code": "ERROR_CODE",
  "timestamp": "2025-01-09T10:30:00.000000"
}
```

---

## Rate Limiting

Currently, the service does not implement rate limiting, but this may be added in future versions.

## CORS

The service supports CORS for cross-origin requests. All origins are currently allowed (`*`), but this should be restricted in production environments.

## Monitoring

The service provides:
- Health check endpoints for monitoring
- Service status information
- Basic error handling
- Logging for debugging

## Performance Considerations

- **Lightweight Design**: Minimal resource usage
- **Fast Response Times**: Sub-millisecond response times
- **Low Memory Footprint**: Efficient memory usage
- **Scalable Architecture**: Designed for horizontal scaling

## Integration with Other Services

The Tracing Service integrates with other services in the ML Security platform:

### 1. Jaeger Integration
- **Purpose**: Distributed tracing backend
- **Configuration**: `JAEGER_AGENT_HOST` and `JAEGER_AGENT_PORT` environment variables
- **Features**: Trace collection, storage, and visualization

### 2. OpenTelemetry Integration
- **Purpose**: Standardized tracing instrumentation
- **Configuration**: `OTEL_EXPORTER_JAEGER_ENDPOINT` environment variable
- **Features**: Automatic instrumentation and context propagation

### 3. Service Discovery
- **Purpose**: Integration with other platform services
- **Configuration**: Service discovery through Docker Compose
- **Features**: Automatic service registration and health checking

## Configuration

### Environment Variables
```bash
# Core Configuration
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8009
SERVICE_VERSION=1.0.0

# Jaeger Configuration
JAEGER_AGENT_HOST=jaeger
JAEGER_AGENT_PORT=14268
JAEGER_ENDPOINT=http://jaeger:14268/api/traces

# OpenTelemetry Configuration
OTEL_EXPORTER_JAEGER_ENDPOINT=http://jaeger:14268/api/traces
OTEL_SERVICE_NAME=tracing-service

# Performance Configuration
TRACE_SAMPLING_RATE=0.1
MAX_TRACES_PER_SECOND=1000
TRACE_RETENTION_HOURS=24
```

### Docker Configuration
```yaml
tracing:
  build:
    context: ./services/tracing
    dockerfile: Dockerfile
  container_name: local-ml-security-tracing-1
  ports:
    - "8009:8009"
  environment:
    - JAEGER_AGENT_HOST=jaeger
    - JAEGER_AGENT_PORT=14268
    - REDIS_URL=redis://redis:6379
    - POSTGRES_URL=postgresql://mlflow:password@postgres:5432/ml_security_consolidated
  depends_on:
    - jaeger
    - redis
    - postgres
  networks:
    - ml-network
  restart: unless-stopped
```

## Future Enhancements

### Planned Features
- **Trace Collection API**: Endpoints for trace collection and management
- **Trace Analysis API**: Endpoints for trace analysis and metrics
- **Performance Monitoring**: Advanced performance monitoring capabilities
- **Custom Dashboards**: User-defined trace dashboards

### Performance Improvements
- **Caching**: Intelligent caching of trace data
- **Compression**: Trace data compression
- **Batch Processing**: Batch processing for trace collection
- **Streaming**: Real-time trace streaming

## Troubleshooting

### Common Issues

1. **Service Not Responding**
   - Check if the service is running: `docker ps | grep tracing`
   - Check service logs: `docker logs local-ml-security-tracing-1`
   - Verify port configuration: `netstat -tlnp | grep 8009`

2. **Health Check Failing**
   - Check service status: `curl http://localhost:8009/health`
   - Verify service configuration
   - Check for resource constraints

3. **Integration Issues**
   - Verify Jaeger is running: `curl http://localhost:16686`
   - Check environment variables
   - Verify network connectivity

### Debug Commands

```bash
# Check service status
curl http://localhost:8009/health

# Check service information
curl http://localhost:8009/

# Check Docker container status
docker ps | grep tracing

# Check container logs
docker logs local-ml-security-tracing-1

# Check port binding
netstat -tlnp | grep 8009
```

---

**Tracing Service API** - Complete reference for all endpoints, request/response schemas, and integration examples for the ML Security Tracing Service.
