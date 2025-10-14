# Enterprise Dashboard Backend - API Documentation

## Overview

The Enterprise Dashboard Backend provides a comprehensive REST API and WebSocket interface for the ML Security platform. It serves as an API Gateway, aggregating data from all ML Security services and providing a unified interface for dashboard management, model operations, training, analytics, and real-time monitoring.

## Base URL
```
http://localhost:8007
```

## WebSocket URL
```
ws://localhost:8007/ws
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
  "error": true,
  "status_code": 400,
  "message": "Bad request",
  "path": "/api/endpoint",
  "timestamp": "2025-01-09T10:30:00.000000"
}
```

### 404 Not Found
```json
{
  "error": true,
  "status_code": 404,
  "message": "Endpoint not found: /api/endpoint",
  "available_endpoints": ["/health", "/docs", "/dashboard/metrics"],
  "path": "/api/endpoint",
  "timestamp": "2025-01-09T10:30:00.000000"
}
```

### 422 Validation Error
```json
{
  "error": true,
  "status_code": 422,
  "message": "Validation error",
  "details": [
    {
      "loc": ["body", "field"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ],
  "path": "/api/endpoint",
  "timestamp": "2025-01-09T10:30:00.000000"
}
```

### 500 Internal Server Error
```json
{
  "error": true,
  "status_code": 500,
  "message": "Internal server error",
  "path": "/api/endpoint",
  "timestamp": "2025-01-09T10:30:00.000000"
}
```

---

## Health Endpoints

### GET /
**Description**: Root endpoint with service information

**Response**:
```json
{
  "status": "healthy",
  "service": "enterprise-dashboard-backend",
  "timestamp": "2025-01-09T10:30:00.000000",
  "version": "1.0.0"
}
```

### GET /health
**Description**: Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "service": "enterprise-dashboard-backend",
  "timestamp": "2025-01-09T10:30:00.000000",
  "version": "1.0.0",
  "uptime_seconds": 3600.0
}
```

### GET /services/health
**Description**: Get health status of all ML Security services

**Response**:
```json
[
  {
    "name": "model-api",
    "status": "healthy",
    "response_time": 15.5,
    "last_check": "2025-01-09T10:30:00.000000",
    "details": {
      "version": "1.0.0",
      "uptime": 3600.0
    }
  }
]
```

### GET /status
**Description**: Get comprehensive system status

**Response**:
```json
{
  "status": "healthy",
  "overall_health_percentage": 95.5,
  "healthy_services": 8,
  "total_services": 9,
  "services": [
    {
      "name": "model-api",
      "status": "healthy",
      "response_time": 15.5
    }
  ],
  "timestamp": "2025-01-09T10:30:00.000000",
  "version": "1.0.0",
  "uptime_seconds": 3600.0
}
```

---

## Dashboard Endpoints

### GET /dashboard/metrics
**Description**: Get comprehensive dashboard metrics

**Response**:
```json
{
  "total_models": 15,
  "active_jobs": 3,
  "total_attacks": 1250,
  "detection_rate": 94.5,
  "system_health": 95.5,
  "last_updated": "2025-01-09T10:30:00.000000"
}
```

### GET /dashboard/models/overview
**Description**: Get comprehensive models overview

**Response**:
```json
{
  "total_models": 15,
  "loaded_models": 8,
  "available_models": 7,
  "model_types": {
    "pytorch": 10,
    "tensorflow": 3,
    "huggingface": 2
  },
  "performance_summary": {
    "average_accuracy": 94.5,
    "average_confidence": 0.89
  }
}
```

### GET /dashboard/training/overview
**Description**: Get training jobs overview

**Response**:
```json
{
  "active_jobs": 3,
  "completed_jobs": 45,
  "failed_jobs": 2,
  "total_jobs": 50,
  "average_duration": 1800.5,
  "success_rate": 95.7
}
```

### GET /dashboard/red-team/overview
**Description**: Get red team testing overview

**Response**:
```json
{
  "total_attacks": 1250,
  "successful_attacks": 68,
  "detection_rate": 94.6,
  "attack_categories": {
    "prompt_injection": 450,
    "jailbreak": 300,
    "system_extraction": 250,
    "code_injection": 250
  }
}
```

### GET /dashboard/system/health
**Description**: Get system health overview

**Response**:
```json
{
  "overall_health": 95.5,
  "service_health": {
    "model-api": "healthy",
    "training": "healthy",
    "analytics": "degraded"
  },
  "resource_usage": {
    "cpu": 45.2,
    "memory": 67.8,
    "disk": 23.1
  }
}
```

### GET /dashboard/activity/recent
**Description**: Get recent system activity

**Parameters**:
- `limit` (int, optional): Number of activities to return (default: 10)

**Response**:
```json
{
  "activities": [
    {
      "id": "act_001",
      "type": "model_loaded",
      "description": "Model 'security-classifier' loaded successfully",
      "timestamp": "2025-01-09T10:25:00.000000",
      "user": "system"
    }
  ],
  "count": 1
}
```

### GET /dashboard/performance/trends
**Description**: Get performance trends over time

**Parameters**:
- `hours` (int, optional): Time range in hours (default: 24)

**Response**:
```json
{
  "trends": [
    {
      "timestamp": "2025-01-09T10:00:00.000000",
      "accuracy": 94.5,
      "throughput": 150.2,
      "latency": 25.3
    }
  ],
  "time_range": "24h"
}
```

### GET /dashboard/analytics/overview
**Description**: Get analytics overview

**Response**:
```json
{
  "total_analyses": 1250,
  "successful_analyses": 1183,
  "failed_analyses": 67,
  "average_processing_time": 45.2,
  "detection_rate": 94.6
}
```

### GET /dashboard/business-metrics/overview
**Description**: Get business metrics overview

**Response**:
```json
{
  "total_predictions": 50000,
  "cost_per_prediction": 0.002,
  "total_cost": 100.0,
  "sla_compliance": 99.2,
  "user_satisfaction": 4.7
}
```

### GET /dashboard/data-privacy/overview
**Description**: Get data privacy overview

**Response**:
```json
{
  "total_classifications": 2500,
  "pii_detections": 150,
  "anonymizations": 75,
  "compliance_score": 98.5,
  "privacy_violations": 2
}
```

---

## Model Management Endpoints

### GET /models
**Description**: Get all available models (alias for /available)

**Response**:
```json
{
  "models": {
    "security-classifier": {
      "name": "security-classifier",
      "type": "pytorch",
      "status": "loaded",
      "loaded": true,
      "path": "/models/security-classifier",
      "labels": ["safe", "unsafe"],
      "performance": {
        "accuracy": 94.5,
        "f1_score": 0.92
      },
      "model_source": "huggingface",
      "model_version": "1.0.0",
      "description": "Security content classifier"
    }
  },
  "count": 1
}
```

### GET /models/available
**Description**: Get all available models

**Response**: Same as `/models`

### GET /models/registry
**Description**: Get model registry information

**Response**:
```json
{
  "experiments": [
    {
      "experiment_id": "exp_001",
      "name": "Security Classification",
      "lifecycle_stage": "active",
      "creation_time": "2025-01-09T09:00:00.000000",
      "last_update_time": "2025-01-09T10:30:00.000000",
      "run_count": 15
    }
  ],
  "models": [
    {
      "name": "security-classifier",
      "version": "1.0.0",
      "stage": "Production",
      "last_updated": "2025-01-09T10:30:00.000000"
    }
  ]
}
```

### GET /models/latest
**Description**: Get latest versions of all models

**Response**:
```json
{
  "models": [
    {
      "name": "security-classifier",
      "version": "1.0.0",
      "stage": "Production",
      "last_updated": "2025-01-09T10:30:00.000000"
    }
  ]
}
```

### GET /models/best
**Description**: Get best performing models

**Response**:
```json
{
  "models": [
    {
      "name": "security-classifier",
      "version": "1.0.0",
      "accuracy": 94.5,
      "f1_score": 0.92,
      "stage": "Production"
    }
  ]
}
```

### POST /models/load
**Description**: Load a model

**Request Body**:
```json
{
  "model_name": "security-classifier",
  "model_type": "pytorch"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Model security-classifier loaded successfully",
  "timestamp": "2025-01-09T10:30:00.000000",
  "data": {
    "model_name": "security-classifier",
    "status": "loaded"
  }
}
```

### POST /models/unload
**Description**: Unload a model

**Request Body**:
```json
{
  "model_name": "security-classifier"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Model security-classifier unloaded successfully",
  "timestamp": "2025-01-09T10:30:00.000000",
  "data": {
    "model_name": "security-classifier",
    "status": "unloaded"
  }
}
```

### POST /models/reload
**Description**: Reload a model

**Request Body**:
```json
{
  "model_name": "security-classifier"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Model security-classifier reloaded successfully",
  "timestamp": "2025-01-09T10:30:00.000000",
  "data": {
    "unload": {
      "model_name": "security-classifier",
      "status": "unloaded"
    },
    "load": {
      "model_name": "security-classifier",
      "status": "loaded"
    }
  }
}
```

### POST /models/predict
**Description**: Make a model prediction

**Request Body**:
```json
{
  "text": "This is a test message",
  "model_name": "security-classifier",
  "ensemble": false
}
```

**Response**:
```json
{
  "text": "This is a test message",
  "prediction": "safe",
  "confidence": 0.95,
  "probabilities": {
    "safe": 0.95,
    "unsafe": 0.05
  },
  "model_name": "security-classifier",
  "processing_time_ms": 25.3,
  "timestamp": "2025-01-09T10:30:00.000000"
}
```

### POST /models/predict/batch
**Description**: Make batch predictions

**Request Body**:
```json
{
  "texts": [
    "This is a test message",
    "Another test message"
  ],
  "model_name": "security-classifier",
  "ensemble": false
}
```

**Response**:
```json
{
  "predictions": [
    {
      "text": "This is a test message",
      "prediction": "safe",
      "confidence": 0.95,
      "probabilities": {
        "safe": 0.95,
        "unsafe": 0.05
      },
      "model_name": "security-classifier",
      "processing_time_ms": 25.3,
      "timestamp": "2025-01-09T10:30:00.000000"
    }
  ],
  "total_processing_time_ms": 50.6,
  "timestamp": "2025-01-09T10:30:00.000000"
}
```

### GET /models/info/{model_name}
**Description**: Get detailed information about a specific model

**Parameters**:
- `model_name` (string): Name of the model

**Response**:
```json
{
  "name": "security-classifier",
  "type": "pytorch",
  "status": "loaded",
  "loaded": true,
  "path": "/models/security-classifier",
  "labels": ["safe", "unsafe"],
  "performance": {
    "accuracy": 94.5,
    "f1_score": 0.92
  },
  "model_source": "huggingface",
  "model_version": "1.0.0",
  "description": "Security content classifier"
}
```

---

## Training Endpoints

### GET /training/jobs
**Description**: Get all training jobs with Redis caching

**Response**:
```json
{
  "jobs": [
    {
      "job_id": "job_001",
      "model_name": "security-classifier",
      "status": "running",
      "progress": 65.5,
      "start_time": "2025-01-09T09:00:00.000000",
      "end_time": null,
      "error_message": null
    }
  ],
  "count": 1
}
```

### GET /training/jobs/{job_id}
**Description**: Get specific training job

**Parameters**:
- `job_id` (string): Job identifier

**Response**:
```json
{
  "job_id": "job_001",
  "model_name": "security-classifier",
  "status": "running",
  "progress": 65.5,
  "start_time": "2025-01-09T09:00:00.000000",
  "end_time": null,
  "error_message": null
}
```

### GET /training/jobs/{job_id}/logs
**Description**: Get logs for a specific training job

**Parameters**:
- `job_id` (string): Job identifier

**Response**:
```json
{
  "logs": [
    {
      "timestamp": "2025-01-09T09:00:00.000000",
      "level": "INFO",
      "message": "Training started"
    }
  ],
  "count": 1
}
```

### POST /training/start
**Description**: Start a new training job

**Request Body**:
```json
{
  "model_name": "security-classifier",
  "training_data_path": "/data/training.csv",
  "config": {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Training job started for security-classifier",
  "timestamp": "2025-01-09T10:30:00.000000",
  "data": {
    "job_id": "job_001",
    "status": "started"
  }
}
```

### POST /training/train
**Description**: Train a model

**Request Body**:
```json
{
  "model_name": "security-classifier",
  "training_data_path": "/data/training.csv",
  "config": {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Training started for security-classifier",
  "timestamp": "2025-01-09T10:30:00.000000",
  "data": {
    "job_id": "job_001",
    "status": "started"
  }
}
```

### POST /training/train/loaded-model
**Description**: Train a loaded model

**Request Body**:
```json
{
  "model_name": "security-classifier",
  "training_data_path": "/data/training.csv",
  "config": {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Training started for loaded model security-classifier",
  "timestamp": "2025-01-09T10:30:00.000000",
  "data": {
    "job_id": "job_001",
    "status": "started"
  }
}
```

### POST /training/retrain
**Description**: Retrain an existing model

**Request Body**:
```json
{
  "model_name": "security-classifier",
  "training_data_path": "/data/training.csv",
  "config": {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Retraining job started for security-classifier",
  "timestamp": "2025-01-09T10:30:00.000000",
  "data": {
    "job_id": "job_001",
    "status": "started"
  }
}
```

### POST /training/stop/{job_id}
**Description**: Stop a training job

**Parameters**:
- `job_id` (string): Job identifier

**Response**:
```json
{
  "status": "success",
  "message": "Training job job_001 stop requested",
  "timestamp": "2025-01-09T10:30:00.000000"
}
```

### GET /training/models
**Description**: Get models available for training

**Response**:
```json
{
  "models": [],
  "count": 0
}
```

### GET /training/models/{model_name}
**Description**: Get specific training model information

**Parameters**:
- `model_name` (string): Model name

**Response**:
```json
{
  "model_name": "security-classifier",
  "status": "unknown"
}
```

### GET /training/logs
**Description**: Get training logs

**Response**:
```json
{
  "logs": [],
  "count": 0
}
```

### GET /training/config/{model_name}
**Description**: Get training configuration for a specific model

**Parameters**:
- `model_name` (string): Model name

**Response**:
```json
{
  "model_name": "security-classifier",
  "config": {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

### PUT /training/config/{model_name}
**Description**: Save training configuration for a specific model

**Parameters**:
- `model_name` (string): Model name

**Request Body**:
```json
{
  "epochs": 10,
  "batch_size": 32,
  "learning_rate": 0.001
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Configuration saved for security-classifier",
  "timestamp": "2025-01-09T10:30:00.000000"
}
```

### GET /training/configs
**Description**: List all saved training configurations

**Response**:
```json
{
  "configs": [
    {
      "model_name": "security-classifier",
      "config": {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001
      }
    }
  ],
  "count": 1
}
```

### DELETE /training/config/{model_name}
**Description**: Delete training configuration for a specific model

**Parameters**:
- `model_name` (string): Model name

**Response**:
```json
{
  "status": "success",
  "message": "Configuration deleted for security-classifier",
  "timestamp": "2025-01-09T10:30:00.000000"
}
```

---

## Analytics Endpoints

### GET /analytics/summary
**Description**: Get analytics summary

**Response**:
```json
{
  "total_tests": 1250,
  "detection_rate": 94.6,
  "vulnerability_rate": 5.4,
  "model_performance": {
    "security-classifier": {
      "accuracy": 94.5,
      "f1_score": 0.92
    }
  },
  "trend_data": [
    {
      "timestamp": "2025-01-09T10:00:00.000000",
      "detection_rate": 94.5
    }
  ],
  "last_updated": "2025-01-09T10:30:00.000000"
}
```

### GET /analytics/overview
**Description**: Get analytics overview

**Response**:
```json
{
  "total_analyses": 0,
  "successful_analyses": 0,
  "failed_analyses": 0,
  "average_processing_time": 0.0,
  "timestamp": "2025-01-09T10:30:00.000000"
}
```

### GET /analytics/trends
**Description**: Get analytics trends

**Response**:
```json
{
  "trends": [],
  "time_range": "24h",
  "timestamp": "2025-01-09T10:30:00.000000"
}
```

### GET /analytics/comparison/{model_name}
**Description**: Get model comparison data

**Parameters**:
- `model_name` (string): Model name

**Response**:
```json
{
  "model_name": "security-classifier",
  "comparison_data": {},
  "timestamp": "2025-01-09T10:30:00.000000"
}
```

### POST /analytics/performance/store
**Description**: Store model performance data

**Request Body**:
```json
{
  "model_name": "security-classifier",
  "version": "1.0.0",
  "metrics": {
    "accuracy": 94.5,
    "f1_score": 0.92
  },
  "test_data_path": "/data/test.csv",
  "timestamp": "2025-01-09T10:30:00.000000"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Model performance stored for security-classifier",
  "timestamp": "2025-01-09T10:30:00.000000"
}
```

---

## WebSocket Endpoints

### WebSocket /ws
**Description**: Main WebSocket endpoint for real-time communication

**Connection**: `ws://localhost:8007/ws`

**Message Types**:

#### Ping Message
**Send**:
```json
{
  "type": "ping",
  "data": {},
  "timestamp": "2025-01-09T10:30:00.000000"
}
```

**Receive**:
```json
{
  "type": "pong",
  "data": {
    "timestamp": "2025-01-09T10:30:00.000000"
  },
  "timestamp": "2025-01-09T10:30:00.000000",
  "success": true
}
```

#### Subscription Message
**Send**:
```json
{
  "type": "subscribe",
  "data": {
    "type": "all"
  },
  "timestamp": "2025-01-09T10:30:00.000000"
}
```

**Receive**:
```json
{
  "type": "subscribed",
  "data": {
    "subscription": "all"
  },
  "timestamp": "2025-01-09T10:30:00.000000",
  "success": true
}
```

#### Status Request
**Send**:
```json
{
  "type": "get_status",
  "data": {},
  "timestamp": "2025-01-09T10:30:00.000000"
}
```

**Receive**:
```json
{
  "type": "status",
  "data": {
    "active_connections": 5,
    "timestamp": "2025-01-09T10:30:00.000000"
  },
  "timestamp": "2025-01-09T10:30:00.000000",
  "success": true
}
```

### GET /ws/status
**Description**: Get WebSocket connection status

**Response**:
```json
{
  "active_connections": 5,
  "connections": [
    {
      "id": "conn_001",
      "connected_at": "2025-01-09T10:00:00.000000",
      "last_activity": "2025-01-09T10:30:00.000000"
    }
  ],
  "timestamp": "2025-01-09T10:30:00.000000"
}
```

---

## Request/Response Models

### PredictionRequest
```json
{
  "text": "string",
  "model_name": "string (optional)",
  "ensemble": "boolean (optional, default: false)"
}
```

### BatchPredictionRequest
```json
{
  "texts": ["string"],
  "model_name": "string (optional)",
  "ensemble": "boolean (optional, default: false)"
}
```

### TrainingRequest
```json
{
  "model_name": "string",
  "training_data_path": "string",
  "config": {
    "epochs": "integer (optional)",
    "batch_size": "integer (optional)",
    "learning_rate": "number (optional)"
  }
}
```

### RetrainingRequest
```json
{
  "model_name": "string",
  "training_data_path": "string",
  "config": {
    "epochs": "integer (optional)",
    "batch_size": "integer (optional)",
    "learning_rate": "number (optional)"
  }
}
```

### ModelLoadRequest
```json
{
  "model_name": "string",
  "model_type": "string (optional, default: pytorch)"
}
```

### ModelUnloadRequest
```json
{
  "model_name": "string"
}
```

### ModelReloadRequest
```json
{
  "model_name": "string"
}
```

### WebSocketMessage
```json
{
  "type": "string",
  "data": {},
  "timestamp": "2025-01-09T10:30:00.000000"
}
```

---

## Usage Examples

### 1. Basic Health Check

```bash
# Check service health
curl http://localhost:8007/health

# Get all services health
curl http://localhost:8007/services/health

# Get system status
curl http://localhost:8007/status
```

### 2. Model Management

```bash
# Get all models
curl http://localhost:8007/models

# Load a model
curl -X POST http://localhost:8007/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_name": "security-classifier", "model_type": "pytorch"}'

# Make a prediction
curl -X POST http://localhost:8007/models/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test message", "model_name": "security-classifier"}'
```

### 3. Training Operations

```bash
# Get training jobs
curl http://localhost:8007/training/jobs

# Start training
curl -X POST http://localhost:8007/training/start \
  -H "Content-Type: application/json" \
  -d '{"model_name": "security-classifier", "training_data_path": "/data/training.csv"}'

# Get job logs
curl http://localhost:8007/training/jobs/job_001/logs
```

### 4. Dashboard Metrics

```bash
# Get dashboard metrics
curl http://localhost:8007/dashboard/metrics

# Get models overview
curl http://localhost:8007/dashboard/models/overview

# Get training overview
curl http://localhost:8007/dashboard/training/overview
```

### 5. WebSocket Connection

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8007/ws');

// Send ping message
ws.send(JSON.stringify({
  type: 'ping',
  data: {},
  timestamp: new Date().toISOString()
}));

// Listen for messages
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Received:', message);
};
```

### 6. Python Integration

```python
import requests
import json

# Health check
response = requests.get('http://localhost:8007/health')
print(response.json())

# Load model
response = requests.post('http://localhost:8007/models/load', 
  json={'model_name': 'security-classifier'})
print(response.json())

# Make prediction
response = requests.post('http://localhost:8007/models/predict',
  json={'text': 'This is a test message', 'model_name': 'security-classifier'})
print(response.json())
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
- WebSocket connection monitoring
- Performance metrics
- Error tracking and logging

## Performance Considerations

- **Caching**: Redis-based caching for improved performance
- **Connection Pooling**: Efficient HTTP client connection management
- **WebSocket Management**: Optimized WebSocket connection handling
- **Error Handling**: Comprehensive error handling and recovery
- **Load Balancing**: Support for multiple service instances

## Integration with Other Services

The Enterprise Dashboard Backend integrates with all ML Security services:

### 1. Model API Service (Port 8000)
- Model management and predictions
- Model registry access
- Performance monitoring

### 2. Training Service (Port 8002)
- Training job management
- Configuration management
- Progress tracking

### 3. Analytics Service (Port 8006)
- Performance analytics
- Trend analysis
- Drift detection

### 4. Business Metrics Service (Port 8004)
- KPI tracking
- Cost monitoring
- SLA monitoring

### 5. Data Privacy Service (Port 8008)
- PII detection
- Data classification
- Compliance reporting

---

**Enterprise Dashboard Backend API** - Complete reference for all endpoints, request/response schemas, WebSocket communication, and integration examples for the ML Security platform's central API Gateway.
