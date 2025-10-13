# Training Service - Detailed Endpoints

**Base URL**: `http://localhost:8002`  
**Service**: Training Service  
**Purpose**: Model training, retraining, and lifecycle management

## Endpoints Overview

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Health check |
| GET | `/models/available` | Get available models for training |
| GET | `/models/registry` | Get model registry |
| GET | `/models/latest` | Get latest models |
| GET | `/models/best` | Get best models |
| GET | `/models/versions/{model_name}` | Get model versions |
| GET | `/models` | List all models |
| GET | `/models/{model_name}` | Get model info |
| GET | `/models/{model_name}/status` | Get training status |
| GET | `/model-loading/status` | Get model loading status |
| GET | `/jobs` | List training jobs |
| GET | `/jobs/{job_id}` | Get job status |
| POST | `/model-loading/start` | Start model loading |
| POST | `/model-loading/load-specific` | Load specific models |
| POST | `/model-loading/load-single` | Load single model |
| POST | `/train` | Train a model |
| POST | `/retrain` | Retrain a model |
| POST | `/create-sample-data` | Create sample data |
| GET | `/logs` | Get training logs |

---

## Detailed Endpoint Documentation

### 1. Health Check

#### `GET /`

**Purpose**: Basic health check endpoint

**Response**:
```json
{
  "status": "running",
  "service": "training"
}
```

**Example**:
```bash
curl http://localhost:8002/
```

---

### 2. Get Available Models

#### `GET /models/available`

**Purpose**: Get list of available models for loading and training

**Response**:
```json
{
  "available_models": [
    "deberta-v3-base",
    "roberta-base",
    "bert-base-uncased",
    "distilbert-base-uncased"
  ],
  "model_configs": {
    "deberta-v3-base": {
      "path": "microsoft/deberta-v3-base",
      "type": "pytorch",
      "labels": ["prompt_injection", "jailbreak", "system_extraction", "code_injection", "benign"]
    }
  }
}
```

**Example**:
```bash
curl http://localhost:8002/models/available
```

---

### 3. Get Model Registry

#### `GET /models/registry`

**Purpose**: Get model registry with latest and best models

**Response**:
```json
{
  "latest_models": {
    "deberta-v3-base": {
      "model_name": "deberta-v3-base",
      "version": "v1.0.1234",
      "run_id": "abc123",
      "f1_score": 0.95,
      "accuracy": 0.94,
      "timestamp": "2024-01-01T00:00:00Z",
      "mlflow_uri": "models:/security_deberta-v3-base/latest"
    }
  },
  "best_models": {
    "deberta-v3-base": {
      "model_name": "deberta-v3-base",
      "version": "v1.0.1234",
      "run_id": "abc123",
      "f1_score": 0.95,
      "accuracy": 0.94,
      "timestamp": "2024-01-01T00:00:00Z",
      "mlflow_uri": "models:/security_deberta-v3-base/latest"
    }
  },
  "versions": {
    "deberta-v3-base": [
      {
        "model_name": "deberta-v3-base",
        "version": "v1.0.1234",
        "run_id": "abc123",
        "f1_score": 0.95,
        "accuracy": 0.94,
        "timestamp": "2024-01-01T00:00:00Z",
        "mlflow_uri": "models:/security_deberta-v3-base/latest"
      }
    ]
  }
}
```

**Example**:
```bash
curl http://localhost:8002/models/registry
```

---

### 4. Get Latest Models

#### `GET /models/latest`

**Purpose**: Get latest version of all models

**Response**:
```json
{
  "latest_models": {
    "deberta-v3-base": {
      "model_name": "deberta-v3-base",
      "version": "v1.0.1234",
      "run_id": "abc123",
      "f1_score": 0.95,
      "accuracy": 0.94,
      "timestamp": "2024-01-01T00:00:00Z",
      "mlflow_uri": "models:/security_deberta-v3-base/latest"
    }
  }
}
```

**Example**:
```bash
curl http://localhost:8002/models/latest
```

---

### 5. Get Best Models

#### `GET /models/best`

**Purpose**: Get best performing version of all models

**Response**:
```json
{
  "best_models": {
    "deberta-v3-base": {
      "model_name": "deberta-v3-base",
      "version": "v1.0.1234",
      "run_id": "abc123",
      "f1_score": 0.95,
      "accuracy": 0.94,
      "timestamp": "2024-01-01T00:00:00Z",
      "mlflow_uri": "models:/security_deberta-v3-base/latest"
    }
  }
}
```

**Example**:
```bash
curl http://localhost:8002/models/best
```

---

### 6. Get Model Versions

#### `GET /models/versions/{model_name}`

**Purpose**: Get all versions of a specific model

**Parameters**:
- `model_name` (path): Name of the model

**Response**:
```json
{
  "model_name": "deberta-v3-base",
  "versions": [
    {
      "version": "v1.0.1234",
      "run_id": "abc123",
      "f1_score": 0.95,
      "accuracy": 0.94,
      "timestamp": "2024-01-01T00:00:00Z",
      "mlflow_uri": "models:/security_deberta-v3-base/v1.0.1234"
    },
    {
      "version": "v1.0.1233",
      "run_id": "def456",
      "f1_score": 0.92,
      "accuracy": 0.91,
      "timestamp": "2024-01-01T00:00:00Z",
      "mlflow_uri": "models:/security_deberta-v3-base/v1.0.1233"
    }
  ]
}
```

**Example**:
```bash
curl http://localhost:8002/models/versions/deberta-v3-base
```

---

### 7. List All Models

#### `GET /models`

**Purpose**: List all available models with their status

**Response**:
```json
{
  "models": {
    "deberta-v3-base": {
      "name": "deberta-v3-base",
      "type": "pytorch",
      "trained": true,
      "path": "./models/deberta-v3-base_final",
      "status": "completed",
      "performance": {
        "accuracy": 0.94,
        "f1_score": 0.95
      }
    }
  },
  "available_models": ["deberta-v3-base", "roberta-base"],
  "model_info": {
    "deberta-v3-base": {
      "name": "deberta-v3-base",
      "type": "pytorch",
      "trained": true,
      "path": "./models/deberta-v3-base_final",
      "status": "completed"
    }
  }
}
```

**Example**:
```bash
curl http://localhost:8002/models
```

---

### 8. Get Model Information

#### `GET /models/{model_name}`

**Purpose**: Get detailed information about a specific model

**Parameters**:
- `model_name` (path): Name of the model

**Response**:
```json
{
  "name": "deberta-v3-base",
  "type": "pytorch",
  "trained": true,
  "path": "./models/deberta-v3-base_final",
  "status": "completed",
  "performance": {
    "accuracy": 0.94,
    "f1_score": 0.95,
    "precision": 0.93,
    "recall": 0.94
  },
  "training_info": {
    "last_trained": "2024-01-01T00:00:00Z",
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5
  }
}
```

**Example**:
```bash
curl http://localhost:8002/models/deberta-v3-base
```

---

### 9. Get Training Status

#### `GET /models/{model_name}/status`

**Purpose**: Get training status for a model

**Parameters**:
- `model_name` (path): Name of the model

**Response**:
```json
{
  "model_name": "deberta-v3-base",
  "status": "training",
  "progress": 65.5,
  "current_epoch": 2,
  "total_epochs": 3,
  "current_loss": 0.234,
  "best_metric": 0.92,
  "start_time": "2024-01-01T00:00:00Z",
  "estimated_completion": "2024-01-01T00:30:00Z"
}
```

**Example**:
```bash
curl http://localhost:8002/models/deberta-v3-base/status
```

---

### 10. Get Model Loading Status

#### `GET /model-loading/status`

**Purpose**: Get model loading status and progress

**Response**:
```json
{
  "loading_in_progress": true,
  "loaded_models": 2,
  "total_models": 4,
  "progress": 50.0,
  "models": {
    "deberta-v3-base": {
      "status": "loaded",
      "progress": 100.0,
      "source": "Hugging Face"
    },
    "roberta-base": {
      "status": "loading",
      "progress": 75.0,
      "source": "Hugging Face"
    }
  }
}
```

**Example**:
```bash
curl http://localhost:8002/model-loading/status
```

---

### 11. List Training Jobs

#### `GET /jobs`

**Purpose**: List all training jobs

**Response**:
```json
{
  "jobs": {
    "job_123": {
      "job_id": "job_123",
      "model_name": "deberta-v3-base",
      "status": "running",
      "progress": 65.5,
      "start_time": "2024-01-01T00:00:00Z",
      "estimated_completion": "2024-01-01T00:30:00Z"
    }
  },
  "total_jobs": 1,
  "running_jobs": 1,
  "completed_jobs": 0
}
```

**Example**:
```bash
curl http://localhost:8002/jobs
```

---

### 12. Get Job Status

#### `GET /jobs/{job_id}`

**Purpose**: Get job status by ID

**Parameters**:
- `job_id` (path): Job identifier

**Response**:
```json
{
  "job_id": "job_123",
  "model_name": "deberta-v3-base",
  "status": "running",
  "progress": 65.5,
  "current_epoch": 2,
  "total_epochs": 3,
  "current_loss": 0.234,
  "best_metric": 0.92,
  "start_time": "2024-01-01T00:00:00Z",
  "estimated_completion": "2024-01-01T00:30:00Z",
  "result": null
}
```

**Example**:
```bash
curl http://localhost:8002/jobs/job_123
```

---

### 13. Start Model Loading

#### `POST /model-loading/start`

**Purpose**: Start model loading process

**Response**:
```json
{
  "message": "Model loading started",
  "total_models": 4,
  "loading_models": ["deberta-v3-base", "roberta-base", "bert-base-uncased", "distilbert-base-uncased"]
}
```

**Example**:
```bash
curl -X POST http://localhost:8002/model-loading/start
```

---

### 14. Load Specific Models

#### `POST /model-loading/load-specific`

**Purpose**: Load specific models by name

**Request Body**:
```json
{
  "models": ["deberta-v3-base", "roberta-base"]
}
```

**Response**:
```json
{
  "message": "Loading specific models started",
  "models": ["deberta-v3-base", "roberta-base"],
  "status": "loading"
}
```

**Example**:
```bash
curl -X POST http://localhost:8002/model-loading/load-specific \
  -H "Content-Type: application/json" \
  -d '{"models": ["deberta-v3-base", "roberta-base"]}'
```

---

### 15. Load Single Model

#### `POST /model-loading/load-single`

**Purpose**: Load a single model by name

**Request Body**:
```json
{
  "model_name": "deberta-v3-base"
}
```

**Response**:
```json
{
  "message": "Loading model deberta-v3-base started",
  "model_name": "deberta-v3-base",
  "status": "loading"
}
```

**Example**:
```bash
curl -X POST http://localhost:8002/model-loading/load-single \
  -H "Content-Type: application/json" \
  -d '{"model_name": "deberta-v3-base"}'
```

---

### 16. Train Model

#### `POST /train`

**Purpose**: Train a model

**Request Body**:
```json
{
  "model_name": "deberta-v3-base",
  "training_data_path": "/app/training_data/sample_training_data.jsonl",
  "config": {
    "max_length": 256,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "num_epochs": 2,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "metric_for_best_model": "f1",
    "greater_is_better": true
  }
}
```

**Response**:
```json
{
  "message": "Training started",
  "model_name": "deberta-v3-base",
  "job_id": "job_123",
  "config": {
    "model_name": "deberta-v3-base",
    "max_length": 256,
    "batch_size": 8,
    "learning_rate": 0.00002,
    "num_epochs": 2,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "metric_for_best_model": "f1",
    "greater_is_better": true
  }
}
```

**Example**:
```bash
curl -X POST http://localhost:8002/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "deberta-v3-base",
    "training_data_path": "/app/training_data/sample_training_data.jsonl"
  }'
```

---

### 17. Retrain Model

#### `POST /retrain`

**Purpose**: Retrain a model with new data

**Request Body**:
```json
{
  "model_name": "deberta-v3-base",
  "training_data_path": "/app/training_data/new_data.jsonl",
  "config": {
    "num_epochs": 2,
    "learning_rate": 1e-5
  }
}
```

**Response**:
```json
{
  "message": "Retraining started",
  "model_name": "deberta-v3-base",
  "job_id": "job_124",
  "config": {
    "model_name": "deberta-v3-base",
    "max_length": 256,
    "batch_size": 8,
    "learning_rate": 0.00001,
    "num_epochs": 2,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "metric_for_best_model": "f1",
    "greater_is_better": true
  }
}
```

**Example**:
```bash
curl -X POST http://localhost:8002/retrain \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "deberta-v3-base",
    "training_data_path": "/app/training_data/new_data.jsonl"
  }'
```

---

### 18. Create Sample Data

#### `POST /create-sample-data`

**Purpose**: Create sample training data for testing

**Response**:
```json
{
  "message": "Sample data created",
  "path": "/app/training_data/sample_training_data.jsonl",
  "count": 20,
  "categories": {
    "prompt_injection": 5,
    "jailbreak": 5,
    "system_extraction": 5,
    "code_injection": 5
  }
}
```

**Example**:
```bash
curl -X POST http://localhost:8002/create-sample-data
```

---

### 19. Get Training Logs

#### `GET /logs`

**Purpose**: Get recent training logs

**Response**:
```json
{
  "logs": [
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "level": "INFO",
      "message": "Training started for deberta-v3-base",
      "job_id": "job_123"
    },
    {
      "timestamp": "2024-01-01T00:01:00Z",
      "level": "INFO",
      "message": "Epoch 1/2 completed",
      "job_id": "job_123"
    }
  ],
  "total_logs": 2
}
```

**Example**:
```bash
curl http://localhost:8002/logs
```

---

## Error Responses

### Common Error Codes

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Model not found |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

### Error Response Format

```json
{
  "status": "error",
  "error": "Model deberta-v3-base not found",
  "code": "MODEL_NOT_FOUND",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

---

## Usage Examples

### Python Client

```python
import requests
import json

# Train a model
response = requests.post(
    "http://localhost:8002/train",
    json={
        "model_name": "deberta-v3-base",
        "training_data_path": "/app/training_data/sample_training_data.jsonl"
    }
)
result = response.json()
print(f"Job ID: {result['job_id']}")

# Check job status
job_response = requests.get(f"http://localhost:8002/jobs/{result['job_id']}")
job_status = job_response.json()
print(f"Status: {job_status['status']}")
print(f"Progress: {job_status['progress']}%")
```

### JavaScript Client

```javascript
// Train a model
const response = await fetch('http://localhost:8002/train', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model_name: 'deberta-v3-base',
    training_data_path: '/app/training_data/sample_training_data.jsonl'
  })
});

const result = await response.json();
console.log(`Job ID: ${result.job_id}`);

// Check job status
const jobResponse = await fetch(`http://localhost:8002/jobs/${result.job_id}`);
const jobStatus = await jobResponse.json();
console.log(`Status: ${jobStatus.status}`);
console.log(`Progress: ${jobStatus.progress}%`);
```

### cURL Examples

```bash
# Health check
curl http://localhost:8002/

# List models
curl http://localhost:8002/models

# Train model
curl -X POST http://localhost:8002/train \
  -H "Content-Type: application/json" \
  -d '{"model_name": "deberta-v3-base"}'

# Check job status
curl http://localhost:8002/jobs/job_123

# Get training logs
curl http://localhost:8002/logs
```
