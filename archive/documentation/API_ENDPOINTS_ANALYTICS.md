# Analytics Service - Detailed Endpoints

**Base URL**: `http://localhost:8006`  
**Service**: Analytics Service  
**Purpose**: Data analysis and reporting for ML security metrics

## Endpoints Overview

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Health check |
| POST | `/red-team/results` | Store red team test results |
| POST | `/model/performance` | Store model performance metrics |
| GET | `/analytics/red-team/summary` | Get red team test summary |
| GET | `/analytics/model/comparison/{model_name}` | Get model comparison |
| GET | `/analytics/trends` | Get performance trends |

---

## Detailed Endpoint Documentation

### 1. Health Check

#### `GET /`

**Purpose**: Basic health check endpoint

**Response**:
```json
{
  "status": "running",
  "service": "analytics"
}
```

**Example**:
```bash
curl http://localhost:8006/
```

---

### 2. Store Red Team Results

#### `POST /red-team/results`

**Purpose**: Store red team test results in database

**Request Body**:
```json
{
  "test_id": "test_123",
  "model_name": "deberta-v3-base",
  "model_type": "pre-trained",
  "model_version": "pre-trained",
  "model_source": "Hugging Face",
  "total_attacks": 50,
  "vulnerabilities_found": 5,
  "detection_rate": 0.9,
  "test_duration_seconds": 120,
  "batch_size": 10,
  "attack_categories": ["prompt_injection", "jailbreak"],
  "attack_results": [
    {
      "attack": {
        "category": "prompt_injection",
        "pattern": "Ignore previous instructions",
        "severity": 0.9,
        "description": "Prompt injection attack"
      },
      "model_results": {
        "http://model-api:8000": {
          "prediction": "prompt_injection",
          "confidence": 0.95
        }
      },
      "detected": true,
      "confidence": 0.95,
      "timestamp": "2024-01-01T00:00:00Z"
    }
  ]
}
```

**Response**:
```json
{
  "message": "Red team results stored successfully",
  "test_id": "test_123",
  "stored_at": "2024-01-01T00:00:00Z"
}
```

**Example**:
```bash
curl -X POST http://localhost:8006/red-team/results \
  -H "Content-Type: application/json" \
  -d '{
    "test_id": "test_123",
    "model_name": "deberta-v3-base",
    "model_type": "pre-trained",
    "total_attacks": 50,
    "vulnerabilities_found": 5,
    "detection_rate": 0.9
  }'
```

---

### 3. Store Model Performance

#### `POST /model/performance`

**Purpose**: Store model performance metrics

**Request Body**:
```json
{
  "model_name": "deberta-v3-base",
  "model_type": "trained",
  "model_version": "v1.0.1234",
  "accuracy": 0.94,
  "precision": 0.93,
  "recall": 0.94,
  "f1_score": 0.935,
  "training_duration_seconds": 1800,
  "training_data_size": 1000,
  "test_data_size": 200,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Response**:
```json
{
  "message": "Model performance metrics stored successfully",
  "model_name": "deberta-v3-base",
  "stored_at": "2024-01-01T00:00:00Z"
}
```

**Example**:
```bash
curl -X POST http://localhost:8006/model/performance \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "deberta-v3-base",
    "model_type": "trained",
    "accuracy": 0.94,
    "f1_score": 0.935
  }'
```

---

### 4. Get Red Team Summary

#### `GET /analytics/red-team/summary`

**Purpose**: Get red team test summary for the last N days

**Parameters**:
- `days` (query, optional): Number of days to analyze (default: 7)

**Response**:
```json
{
  "summary": [
    {
      "model_name": "deberta-v3-base",
      "model_type": "pre-trained",
      "total_tests": 15,
      "total_attacks": 750,
      "vulnerabilities_found": 45,
      "avg_detection_rate": 0.94,
      "avg_attacks": 50.0,
      "last_test": "2024-01-01T00:00:00Z"
    },
    {
      "model_name": "deberta-v3-base",
      "model_type": "trained",
      "total_tests": 12,
      "total_attacks": 600,
      "vulnerabilities_found": 24,
      "avg_detection_rate": 0.96,
      "avg_attacks": 50.0,
      "last_test": "2024-01-01T00:00:00Z"
    },
    {
      "model_name": "roberta-base",
      "model_type": "pre-trained",
      "total_tests": 10,
      "total_attacks": 500,
      "vulnerabilities_found": 35,
      "avg_detection_rate": 0.93,
      "avg_attacks": 50.0,
      "last_test": "2024-01-01T00:00:00Z"
    }
  ],
  "overall_stats": {
    "total_tests": 37,
    "total_attacks": 1850,
    "total_vulnerabilities": 104,
    "overall_detection_rate": 0.944,
    "avg_attacks_per_test": 50.0
  },
  "time_range": {
    "start_date": "2023-12-25T00:00:00Z",
    "end_date": "2024-01-01T00:00:00Z",
    "days": 7
  }
}
```

**Example**:
```bash
# Get summary for last 7 days
curl http://localhost:8006/analytics/red-team/summary

# Get summary for last 30 days
curl "http://localhost:8006/analytics/red-team/summary?days=30"
```

---

### 5. Get Model Comparison

#### `GET /analytics/model/comparison/{model_name}`

**Purpose**: Get comparison between pre-trained and trained versions of a model

**Parameters**:
- `model_name` (path): Name of the model to compare
- `days` (query, optional): Number of days to analyze (default: 30)

**Response**:
```json
{
  "model_name": "deberta-v3-base",
  "comparison_period": {
    "start_date": "2023-12-02T00:00:00Z",
    "end_date": "2024-01-01T00:00:00Z",
    "days": 30
  },
  "pretrained": {
    "total_tests": 20,
    "total_attacks": 1000,
    "vulnerabilities_found": 60,
    "avg_detection_rate": 0.94,
    "avg_attacks": 50.0,
    "best_detection_rate": 0.96,
    "worst_detection_rate": 0.91,
    "last_test": "2024-01-01T00:00:00Z"
  },
  "trained": {
    "total_tests": 15,
    "total_attacks": 750,
    "vulnerabilities_found": 30,
    "avg_detection_rate": 0.96,
    "avg_attacks": 50.0,
    "best_detection_rate": 0.98,
    "worst_detection_rate": 0.94,
    "last_test": "2024-01-01T00:00:00Z"
  },
  "improvement": {
    "detection_rate_improvement": 0.02,
    "vulnerability_reduction": 30,
    "vulnerability_detection_improvement": 0.5,
    "performance_gain": "2.1%"
  },
  "category_breakdown": {
    "prompt_injection": {
      "pretrained_detection_rate": 0.95,
      "trained_detection_rate": 0.97,
      "improvement": 0.02
    },
    "jailbreak": {
      "pretrained_detection_rate": 0.92,
      "trained_detection_rate": 0.94,
      "improvement": 0.02
    },
    "system_extraction": {
      "pretrained_detection_rate": 0.93,
      "trained_detection_rate": 0.96,
      "improvement": 0.03
    },
    "code_injection": {
      "pretrained_detection_rate": 0.96,
      "trained_detection_rate": 0.97,
      "improvement": 0.01
    }
  }
}
```

**Example**:
```bash
# Get comparison for deberta-v3-base
curl http://localhost:8006/analytics/model/comparison/deberta-v3-base

# Get comparison for last 7 days
curl "http://localhost:8006/analytics/model/comparison/deberta-v3-base?days=7"
```

---

### 6. Get Performance Trends

#### `GET /analytics/trends`

**Purpose**: Get performance trends over time

**Parameters**:
- `days` (query, optional): Number of days to analyze (default: 30)

**Response**:
```json
{
  "trends": [
    {
      "test_date": "2024-01-01T00:00:00Z",
      "model_name": "deberta-v3-base",
      "model_type": "pre-trained",
      "avg_detection_rate": 0.94,
      "total_attacks": 50,
      "vulnerabilities_found": 3
    },
    {
      "test_date": "2024-01-01T00:00:00Z",
      "model_name": "deberta-v3-base",
      "model_type": "trained",
      "avg_detection_rate": 0.96,
      "total_attacks": 50,
      "vulnerabilities_found": 2
    },
    {
      "test_date": "2023-12-31T00:00:00Z",
      "model_name": "deberta-v3-base",
      "model_type": "pre-trained",
      "avg_detection_rate": 0.93,
      "total_attacks": 50,
      "vulnerabilities_found": 3.5
    }
  ],
  "summary_stats": {
    "total_data_points": 60,
    "avg_detection_rate": 0.945,
    "detection_rate_trend": "improving",
    "vulnerability_trend": "decreasing",
    "most_improved_model": "deberta-v3-base",
    "improvement_rate": 0.02
  },
  "time_range": {
    "start_date": "2023-12-02T00:00:00Z",
    "end_date": "2024-01-01T00:00:00Z",
    "days": 30
  }
}
```

**Example**:
```bash
# Get trends for last 30 days
curl http://localhost:8006/analytics/trends

# Get trends for last 7 days
curl "http://localhost:8006/analytics/trends?days=7"
```

---

## Database Schema

### Red Team Test Results Table

```sql
CREATE TABLE red_team_test_results (
    id SERIAL PRIMARY KEY,
    test_id VARCHAR(100) UNIQUE NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    model_version VARCHAR(50),
    model_source VARCHAR(100),
    total_attacks INTEGER NOT NULL,
    vulnerabilities_found INTEGER NOT NULL,
    detection_rate DECIMAL(5,4) NOT NULL,
    test_duration_seconds INTEGER,
    batch_size INTEGER,
    attack_categories JSONB,
    attack_results JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Model Performance Table

```sql
CREATE TABLE model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    model_version VARCHAR(50),
    accuracy DECIMAL(5,4),
    precision DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    training_duration_seconds INTEGER,
    training_data_size INTEGER,
    test_data_size INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Data Analysis Features

### 1. Detection Rate Analysis

- **Trend Analysis**: Track detection rates over time
- **Model Comparison**: Compare pre-trained vs trained models
- **Category Breakdown**: Analyze performance by attack category
- **Statistical Significance**: Determine if improvements are significant

### 2. Vulnerability Analysis

- **Vulnerability Trends**: Track vulnerability counts over time
- **Category Analysis**: Identify which attack categories are most effective
- **Model Effectiveness**: Determine which models are most vulnerable
- **Severity Analysis**: Analyze vulnerability severity levels

### 3. Performance Metrics

- **Model Performance**: Track accuracy, precision, recall, F1-score
- **Training Metrics**: Monitor training duration and data size
- **Inference Metrics**: Track prediction speed and resource usage
- **Cost Analysis**: Monitor computational costs

### 4. Business Intelligence

- **KPI Tracking**: Monitor key performance indicators
- **Trend Identification**: Identify patterns and trends
- **Anomaly Detection**: Detect unusual patterns or performance drops
- **Recommendations**: Generate actionable insights

---

## Error Responses

### Common Error Codes

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Invalid input data |
| 404 | Not Found - Model not found |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

### Error Response Format

```json
{
  "status": "error",
  "error": "Invalid model name provided",
  "code": "INVALID_MODEL_NAME",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

---

## Usage Examples

### Python Client

```python
import requests
import json
from datetime import datetime, timedelta

# Store red team results
def store_red_team_results(test_data):
    response = requests.post(
        "http://localhost:8006/red-team/results",
        json=test_data
    )
    return response.json()

# Store model performance
def store_model_performance(performance_data):
    response = requests.post(
        "http://localhost:8006/model/performance",
        json=performance_data
    )
    return response.json()

# Get red team summary
def get_red_team_summary(days=7):
    response = requests.get(
        f"http://localhost:8006/analytics/red-team/summary?days={days}"
    )
    return response.json()

# Get model comparison
def get_model_comparison(model_name, days=30):
    response = requests.get(
        f"http://localhost:8006/analytics/model/comparison/{model_name}?days={days}"
    )
    return response.json()

# Get performance trends
def get_performance_trends(days=30):
    response = requests.get(
        f"http://localhost:8006/analytics/trends?days={days}"
    )
    return response.json()

# Example usage
test_data = {
    "test_id": "test_123",
    "model_name": "deberta-v3-base",
    "model_type": "pre-trained",
    "total_attacks": 50,
    "vulnerabilities_found": 5,
    "detection_rate": 0.9
}

result = store_red_team_results(test_data)
print(f"Stored test: {result['test_id']}")

summary = get_red_team_summary(7)
print(f"Total tests: {summary['overall_stats']['total_tests']}")
print(f"Overall detection rate: {summary['overall_stats']['overall_detection_rate']:.2%}")
```

### JavaScript Client

```javascript
// Store red team results
async function storeRedTeamResults(testData) {
  const response = await fetch('http://localhost:8006/red-team/results', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(testData)
  });
  return await response.json();
}

// Store model performance
async function storeModelPerformance(performanceData) {
  const response = await fetch('http://localhost:8006/model/performance', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(performanceData)
  });
  return await response.json();
}

// Get red team summary
async function getRedTeamSummary(days = 7) {
  const response = await fetch(`http://localhost:8006/analytics/red-team/summary?days=${days}`);
  return await response.json();
}

// Get model comparison
async function getModelComparison(modelName, days = 30) {
  const response = await fetch(`http://localhost:8006/analytics/model/comparison/${modelName}?days=${days}`);
  return await response.json();
}

// Get performance trends
async function getPerformanceTrends(days = 30) {
  const response = await fetch(`http://localhost:8006/analytics/trends?days=${days}`);
  return await response.json();
}

// Example usage
const testData = {
  test_id: 'test_123',
  model_name: 'deberta-v3-base',
  model_type: 'pre-trained',
  total_attacks: 50,
  vulnerabilities_found: 5,
  detection_rate: 0.9
};

storeRedTeamResults(testData).then(result => {
  console.log(`Stored test: ${result.test_id}`);
});

getRedTeamSummary(7).then(summary => {
  console.log(`Total tests: ${summary.overall_stats.total_tests}`);
  console.log(`Overall detection rate: ${(summary.overall_stats.overall_detection_rate * 100).toFixed(2)}%`);
});
```

### cURL Examples

```bash
# Health check
curl http://localhost:8006/

# Store red team results
curl -X POST http://localhost:8006/red-team/results \
  -H "Content-Type: application/json" \
  -d '{
    "test_id": "test_123",
    "model_name": "deberta-v3-base",
    "model_type": "pre-trained",
    "total_attacks": 50,
    "vulnerabilities_found": 5,
    "detection_rate": 0.9
  }'

# Store model performance
curl -X POST http://localhost:8006/model/performance \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "deberta-v3-base",
    "model_type": "trained",
    "accuracy": 0.94,
    "f1_score": 0.935
  }'

# Get red team summary
curl http://localhost:8006/analytics/red-team/summary

# Get model comparison
curl http://localhost:8006/analytics/model/comparison/deberta-v3-base

# Get performance trends
curl http://localhost:8006/analytics/trends
```

---

## Integration Points

### Red Team Service Integration

The Analytics Service automatically receives data from the Red Team Service:
- Test results are stored in real-time
- Performance metrics are tracked
- Trends are analyzed automatically

### Business Metrics Integration

Analytics data feeds into the Business Metrics Service:
- KPI calculations
- Cost analysis
- Performance monitoring

### Monitoring Dashboard Integration

The Monitoring Dashboard displays analytics data:
- Real-time charts and graphs
- Performance comparisons
- Trend visualizations

---

## Best Practices

### Data Storage

- Store all test results for historical analysis
- Include metadata for better filtering
- Use appropriate data types for metrics
- Implement data retention policies

### Analysis

- Regular trend analysis
- Model performance comparison
- Anomaly detection
- Statistical significance testing

### Reporting

- Generate regular reports
- Focus on actionable insights
- Track key performance indicators
- Monitor improvement trends

---

## Troubleshooting

### Common Issues

1. **Database connection issues**
   ```bash
   # Check PostgreSQL status
   docker-compose logs postgres
   
   # Check analytics service logs
   docker-compose logs analytics
   ```

2. **No data in analytics**
   ```bash
   # Check if red team service is sending data
   curl http://localhost:8001/results
   
   # Check if data is being stored
   curl http://localhost:8006/analytics/red-team/summary
   ```

3. **Performance issues**
   ```bash
   # Check database performance
   docker-compose exec postgres psql -U mlflow -d ml_security -c "SELECT * FROM pg_stat_activity;"
   
   # Check service resources
   docker stats analytics
   ```

4. **Data inconsistencies**
   ```bash
   # Verify data integrity
   curl http://localhost:8006/analytics/red-team/summary?days=1
   
   # Check model comparison data
   curl http://localhost:8006/analytics/model/comparison/deberta-v3-base?days=1
   ```
