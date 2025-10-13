# Business Metrics Service - Detailed Endpoints

**Base URL**: `http://localhost:8004`  
**Service**: Business Metrics Service  
**Purpose**: Business KPIs and cost tracking

## Endpoints Overview

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Health check |
| GET | `/kpis` | Get all business KPIs |
| GET | `/attack-success-rate` | Get attack success rate metrics |
| GET | `/model-drift` | Get model drift detection results |
| GET | `/cost-metrics` | Get cost metrics |
| GET | `/system-effectiveness` | Get system effectiveness metrics |
| GET | `/recommendations` | Get optimization recommendations |
| GET | `/metrics` | Get Prometheus metrics |

---

## Detailed Endpoint Documentation

### 1. Health Check

#### `GET /`

**Purpose**: Basic health check endpoint

**Response**:
```json
{
  "status": "running",
  "service": "business-metrics"
}
```

**Example**:
```bash
curl http://localhost:8004/
```

---

### 2. Get All Business KPIs

#### `GET /kpis`

**Purpose**: Get all business KPIs in one response

**Response**:
```json
{
  "attack_success_rate": {
    "total_attacks": 150,
    "successful_attacks": 12,
    "success_rate": 0.08,
    "by_category": {
      "prompt_injection": 0.12,
      "jailbreak": 0.05,
      "system_extraction": 0.08,
      "code_injection": 0.10
    },
    "by_model": {
      "deberta-v3-base_pretrained": 0.10,
      "deberta-v3-base_trained": 0.05,
      "roberta-base_pretrained": 0.12,
      "roberta-base_trained": 0.07
    },
    "trend_7d": 0.02,
    "trend_30d": -0.05
  },
  "model_drift": {
    "models_checked": 4,
    "drift_detected": 1,
    "drift_models": ["deberta-v3-base_pretrained"],
    "overall_drift_score": 0.15
  },
  "cost_metrics": {
    "total_cost_usd": 45.50,
    "cost_per_prediction": 0.001,
    "api_calls_cost": 30.00,
    "model_training_cost": 15.50,
    "storage_cost": 2.00,
    "compute_cost": 28.00
  },
  "system_effectiveness": {
    "uptime_percentage": 99.5,
    "response_time_avg_ms": 45.2,
    "throughput_per_hour": 1200,
    "error_rate": 0.001,
    "user_satisfaction": 0.92
  },
  "recommendations": [
    {
      "type": "cost_optimization",
      "priority": "high",
      "description": "Consider using smaller batch sizes to reduce compute costs",
      "potential_savings": 15.00
    },
    {
      "type": "performance",
      "priority": "medium",
      "description": "Model deberta-v3-base_pretrained shows drift, consider retraining",
      "impact": "improved_accuracy"
    }
  ],
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Example**:
```bash
curl http://localhost:8004/kpis
```

---

### 3. Get Attack Success Rate

#### `GET /attack-success-rate`

**Purpose**: Get attack success rate metrics

**Parameters**:
- `days` (query, optional): Number of days to analyze (default: 7)

**Response**:
```json
{
  "total_attacks": 150,
  "successful_attacks": 12,
  "success_rate": 0.08,
  "by_category": {
    "prompt_injection": {
      "total_attacks": 50,
      "successful_attacks": 6,
      "success_rate": 0.12,
      "severity_breakdown": {
        "high": 2,
        "medium": 3,
        "low": 1
      }
    },
    "jailbreak": {
      "total_attacks": 40,
      "successful_attacks": 2,
      "success_rate": 0.05,
      "severity_breakdown": {
        "high": 1,
        "medium": 1,
        "low": 0
      }
    },
    "system_extraction": {
      "total_attacks": 35,
      "successful_attacks": 3,
      "success_rate": 0.08,
      "severity_breakdown": {
        "high": 1,
        "medium": 2,
        "low": 0
      }
    },
    "code_injection": {
      "total_attacks": 25,
      "successful_attacks": 1,
      "success_rate": 0.04,
      "severity_breakdown": {
        "high": 1,
        "medium": 0,
        "low": 0
      }
    }
  },
  "by_model": {
    "deberta-v3-base_pretrained": {
      "total_attacks": 75,
      "successful_attacks": 8,
      "success_rate": 0.10
    },
    "deberta-v3-base_trained": {
      "total_attacks": 75,
      "successful_attacks": 4,
      "success_rate": 0.05
    }
  },
  "trend_7d": 0.02,
  "trend_30d": -0.05,
  "analysis": {
    "overall_trend": "improving",
    "most_vulnerable_category": "prompt_injection",
    "most_effective_model": "deberta-v3-base_trained",
    "recommendation": "Focus on improving prompt injection detection"
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Example**:
```bash
# Get attack success rate for last 7 days
curl http://localhost:8004/attack-success-rate

# Get attack success rate for last 30 days
curl "http://localhost:8004/attack-success-rate?days=30"
```

---

### 4. Get Model Drift Detection

#### `GET /model-drift`

**Purpose**: Get model drift detection results

**Response**:
```json
{
  "models_checked": 4,
  "drift_detected": 1,
  "overall_drift_score": 0.15,
  "drift_details": [
    {
      "model_name": "deberta-v3-base_pretrained",
      "drift_detected": true,
      "drift_score": 0.15,
      "confidence_interval": [0.12, 0.18],
      "last_drift_check": "2024-01-01T00:00:00Z",
      "features_drifted": ["accuracy", "precision"],
      "severity": "medium",
      "recommendation": "Consider retraining the model"
    },
    {
      "model_name": "deberta-v3-base_trained",
      "drift_detected": false,
      "drift_score": 0.05,
      "confidence_interval": [0.03, 0.07],
      "last_drift_check": "2024-01-01T00:00:00Z",
      "features_drifted": [],
      "severity": "low",
      "recommendation": "Model is performing well"
    },
    {
      "model_name": "roberta-base_pretrained",
      "drift_detected": false,
      "drift_score": 0.08,
      "confidence_interval": [0.06, 0.10],
      "last_drift_check": "2024-01-01T00:00:00Z",
      "features_drifted": [],
      "severity": "low",
      "recommendation": "Monitor for potential drift"
    },
    {
      "model_name": "roberta-base_trained",
      "drift_detected": false,
      "drift_score": 0.03,
      "confidence_interval": [0.01, 0.05],
      "last_drift_check": "2024-01-01T00:00:00Z",
      "features_drifted": [],
      "severity": "low",
      "recommendation": "Model is stable"
    }
  ],
  "drift_summary": {
    "high_severity": 0,
    "medium_severity": 1,
    "low_severity": 3,
    "models_needing_attention": ["deberta-v3-base_pretrained"],
    "overall_health": "good"
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Example**:
```bash
curl http://localhost:8004/model-drift
```

---

### 5. Get Cost Metrics

#### `GET /cost-metrics`

**Purpose**: Get cost metrics and analysis

**Response**:
```json
{
  "total_cost_usd": 45.50,
  "cost_breakdown": {
    "api_calls_cost": 30.00,
    "model_training_cost": 15.50,
    "storage_cost": 2.00,
    "compute_cost": 28.00,
    "infrastructure_cost": 5.00
  },
  "cost_per_prediction": 0.001,
  "cost_per_training": 15.50,
  "cost_per_hour": 1.89,
  "cost_trends": {
    "daily": {
      "current": 45.50,
      "previous": 42.30,
      "change": 0.076
    },
    "weekly": {
      "current": 318.50,
      "previous": 296.10,
      "change": 0.076
    },
    "monthly": {
      "current": 1365.00,
      "previous": 1184.40,
      "change": 0.152
    }
  },
  "optimization_opportunities": [
    {
      "type": "batch_processing",
      "description": "Increase batch sizes to reduce API call costs",
      "potential_savings": 8.50,
      "implementation_effort": "low"
    },
    {
      "type": "model_optimization",
      "description": "Use smaller models for less critical predictions",
      "potential_savings": 12.00,
      "implementation_effort": "medium"
    },
    {
      "type": "caching",
      "description": "Implement more aggressive caching",
      "potential_savings": 5.00,
      "implementation_effort": "low"
    }
  ],
  "cost_by_model": {
    "deberta-v3-base_pretrained": 12.50,
    "deberta-v3-base_trained": 15.00,
    "roberta-base_pretrained": 10.00,
    "roberta-base_trained": 8.00
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Example**:
```bash
curl http://localhost:8004/cost-metrics
```

---

### 6. Get System Effectiveness

#### `GET /system-effectiveness`

**Purpose**: Get system effectiveness metrics

**Response**:
```json
{
  "uptime_percentage": 99.5,
  "response_time_metrics": {
    "avg_ms": 45.2,
    "p50_ms": 42.1,
    "p95_ms": 78.5,
    "p99_ms": 125.3,
    "max_ms": 250.0
  },
  "throughput_metrics": {
    "requests_per_hour": 1200,
    "requests_per_day": 28800,
    "peak_throughput": 1500,
    "avg_throughput": 1200
  },
  "error_metrics": {
    "error_rate": 0.001,
    "total_errors": 12,
    "error_types": {
      "timeout": 5,
      "model_not_found": 3,
      "validation_error": 4
    },
    "error_trend": "decreasing"
  },
  "user_satisfaction": {
    "overall_score": 0.92,
    "response_time_satisfaction": 0.89,
    "accuracy_satisfaction": 0.95,
    "reliability_satisfaction": 0.92
  },
  "resource_utilization": {
    "cpu_usage": 0.65,
    "memory_usage": 0.78,
    "disk_usage": 0.45,
    "network_usage": 0.32
  },
  "performance_indicators": {
    "availability": "excellent",
    "performance": "good",
    "reliability": "excellent",
    "scalability": "good"
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Example**:
```bash
curl http://localhost:8004/system-effectiveness
```

---

### 7. Get Optimization Recommendations

#### `GET /recommendations`

**Purpose**: Get optimization recommendations based on current metrics

**Response**:
```json
{
  "recommendations": [
    {
      "id": "rec_001",
      "type": "cost_optimization",
      "priority": "high",
      "title": "Optimize Batch Processing",
      "description": "Increase batch sizes to reduce API call costs by 15%",
      "potential_savings": 8.50,
      "implementation_effort": "low",
      "estimated_impact": "high",
      "affected_services": ["model-api", "training"],
      "implementation_steps": [
        "Increase default batch size from 10 to 25",
        "Implement dynamic batch sizing based on load",
        "Add batch size monitoring"
      ],
      "success_metrics": ["cost_reduction", "throughput_increase"]
    },
    {
      "id": "rec_002",
      "type": "performance",
      "priority": "medium",
      "title": "Address Model Drift",
      "description": "Model deberta-v3-base_pretrained shows drift, consider retraining",
      "potential_improvement": "accuracy_increase",
      "implementation_effort": "medium",
      "estimated_impact": "medium",
      "affected_services": ["training", "model-api"],
      "implementation_steps": [
        "Retrain deberta-v3-base_pretrained with recent data",
        "Implement continuous drift monitoring",
        "Set up automated retraining triggers"
      ],
      "success_metrics": ["detection_rate_improvement", "drift_score_reduction"]
    },
    {
      "id": "rec_003",
      "type": "reliability",
      "priority": "low",
      "title": "Improve Caching Strategy",
      "description": "Implement more aggressive caching to reduce response times",
      "potential_improvement": "response_time_reduction",
      "implementation_effort": "low",
      "estimated_impact": "medium",
      "affected_services": ["model-cache", "model-api"],
      "implementation_steps": [
        "Increase cache TTL for frequently accessed models",
        "Implement predictive caching",
        "Add cache hit rate monitoring"
      ],
      "success_metrics": ["response_time_improvement", "cache_hit_rate_increase"]
    }
  ],
  "summary": {
    "total_recommendations": 3,
    "high_priority": 1,
    "medium_priority": 1,
    "low_priority": 1,
    "total_potential_savings": 8.50,
    "implementation_effort": "low_to_medium"
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Example**:
```bash
curl http://localhost:8004/recommendations
```

---

### 8. Get Prometheus Metrics

#### `GET /metrics`

**Purpose**: Get Prometheus-formatted metrics

**Response**:
```
# HELP business_metrics_attack_success_rate Attack success rate
# TYPE business_metrics_attack_success_rate gauge
business_metrics_attack_success_rate 0.08

# HELP business_metrics_total_attacks Total number of attacks
# TYPE business_metrics_total_attacks counter
business_metrics_total_attacks 150

# HELP business_metrics_successful_attacks Total number of successful attacks
# TYPE business_metrics_successful_attacks counter
business_metrics_successful_attacks 12

# HELP business_metrics_model_drift_score Model drift score
# TYPE business_metrics_model_drift_score gauge
business_metrics_model_drift_score{model="deberta-v3-base_pretrained"} 0.15
business_metrics_model_drift_score{model="deberta-v3-base_trained"} 0.05

# HELP business_metrics_total_cost_usd Total cost in USD
# TYPE business_metrics_total_cost_usd gauge
business_metrics_total_cost_usd 45.50

# HELP business_metrics_cost_per_prediction Cost per prediction in USD
# TYPE business_metrics_cost_per_prediction gauge
business_metrics_cost_per_prediction 0.001

# HELP business_metrics_uptime_percentage System uptime percentage
# TYPE business_metrics_uptime_percentage gauge
business_metrics_uptime_percentage 99.5

# HELP business_metrics_response_time_avg_ms Average response time in milliseconds
# TYPE business_metrics_response_time_avg_ms gauge
business_metrics_response_time_avg_ms 45.2

# HELP business_metrics_error_rate Error rate
# TYPE business_metrics_error_rate gauge
business_metrics_error_rate 0.001

# HELP business_metrics_user_satisfaction User satisfaction score
# TYPE business_metrics_user_satisfaction gauge
business_metrics_user_satisfaction 0.92
```

**Example**:
```bash
curl http://localhost:8004/metrics
```

---

## KPI Definitions

### Attack Success Rate

**Definition**: Percentage of attacks that successfully bypassed security measures

**Calculation**: (Successful Attacks / Total Attacks) × 100

**Target**: < 10% (lower is better)

**Categories**:
- Prompt Injection
- Jailbreak
- System Extraction
- Code Injection

### Model Drift

**Definition**: Measure of how much a model's performance has degraded over time

**Calculation**: Statistical distance between current and baseline performance

**Target**: < 0.1 (lower is better)

**Severity Levels**:
- High: > 0.2
- Medium: 0.1 - 0.2
- Low: < 0.1

### Cost Metrics

**Definition**: Financial metrics for system operation

**Components**:
- API Calls Cost
- Model Training Cost
- Storage Cost
- Compute Cost
- Infrastructure Cost

**Target**: Optimize for cost-effectiveness

### System Effectiveness

**Definition**: Overall system performance and reliability metrics

**Components**:
- Uptime Percentage
- Response Time
- Throughput
- Error Rate
- User Satisfaction

**Target**: > 99% uptime, < 100ms response time

---

## Error Responses

### Common Error Codes

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Resource not found |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

### Error Response Format

```json
{
  "status": "error",
  "error": "Invalid days parameter",
  "code": "INVALID_DAYS",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

---

## Usage Examples

### Python Client

```python
import requests
import json

# Get all KPIs
def get_all_kpis():
    response = requests.get("http://localhost:8004/kpis")
    return response.json()

# Get attack success rate
def get_attack_success_rate(days=7):
    response = requests.get(f"http://localhost:8004/attack-success-rate?days={days}")
    return response.json()

# Get model drift
def get_model_drift():
    response = requests.get("http://localhost:8004/model-drift")
    return response.json()

# Get cost metrics
def get_cost_metrics():
    response = requests.get("http://localhost:8004/cost-metrics")
    return response.json()

# Get system effectiveness
def get_system_effectiveness():
    response = requests.get("http://localhost:8004/system-effectiveness")
    return response.json()

# Get recommendations
def get_recommendations():
    response = requests.get("http://localhost:8004/recommendations")
    return response.json()

# Example usage
kpis = get_all_kpis()
print(f"Attack success rate: {kpis['attack_success_rate']['success_rate']:.2%}")
print(f"Total cost: ${kpis['cost_metrics']['total_cost_usd']:.2f}")
print(f"Uptime: {kpis['system_effectiveness']['uptime_percentage']:.1f}%")

# Get specific metrics
attack_rate = get_attack_success_rate(30)
print(f"30-day attack success rate: {attack_rate['success_rate']:.2%}")

drift = get_model_drift()
for model in drift['drift_details']:
    if model['drift_detected']:
        print(f"Drift detected in {model['model_name']}: {model['drift_score']:.3f}")

costs = get_cost_metrics()
print(f"Cost per prediction: ${costs['cost_per_prediction']:.4f}")

recommendations = get_recommendations()
for rec in recommendations['recommendations']:
    if rec['priority'] == 'high':
        print(f"High priority: {rec['title']} - {rec['description']}")
```

### JavaScript Client

```javascript
// Get all KPIs
async function getAllKpis() {
  const response = await fetch('http://localhost:8004/kpis');
  return await response.json();
}

// Get attack success rate
async function getAttackSuccessRate(days = 7) {
  const response = await fetch(`http://localhost:8004/attack-success-rate?days=${days}`);
  return await response.json();
}

// Get model drift
async function getModelDrift() {
  const response = await fetch('http://localhost:8004/model-drift');
  return await response.json();
}

// Get cost metrics
async function getCostMetrics() {
  const response = await fetch('http://localhost:8004/cost-metrics');
  return await response.json();
}

// Get system effectiveness
async function getSystemEffectiveness() {
  const response = await fetch('http://localhost:8004/system-effectiveness');
  return await response.json();
}

// Get recommendations
async function getRecommendations() {
  const response = await fetch('http://localhost:8004/recommendations');
  return await response.json();
}

// Example usage
getAllKpis().then(kpis => {
  console.log(`Attack success rate: ${(kpis.attack_success_rate.success_rate * 100).toFixed(2)}%`);
  console.log(`Total cost: $${kpis.cost_metrics.total_cost_usd.toFixed(2)}`);
  console.log(`Uptime: ${kpis.system_effectiveness.uptime_percentage.toFixed(1)}%`);
});

getAttackSuccessRate(30).then(attackRate => {
  console.log(`30-day attack success rate: ${(attackRate.success_rate * 100).toFixed(2)}%`);
});

getModelDrift().then(drift => {
  drift.drift_details.forEach(model => {
    if (model.drift_detected) {
      console.log(`Drift detected in ${model.model_name}: ${model.drift_score.toFixed(3)}`);
    }
  });
});

getCostMetrics().then(costs => {
  console.log(`Cost per prediction: $${costs.cost_per_prediction.toFixed(4)}`);
});

getRecommendations().then(recommendations => {
  recommendations.recommendations.forEach(rec => {
    if (rec.priority === 'high') {
      console.log(`High priority: ${rec.title} - ${rec.description}`);
    }
  });
});
```

### cURL Examples

```bash
# Health check
curl http://localhost:8004/

# Get all KPIs
curl http://localhost:8004/kpis

# Get attack success rate
curl http://localhost:8004/attack-success-rate

# Get attack success rate for 30 days
curl "http://localhost:8004/attack-success-rate?days=30"

# Get model drift
curl http://localhost:8004/model-drift

# Get cost metrics
curl http://localhost:8004/cost-metrics

# Get system effectiveness
curl http://localhost:8004/system-effectiveness

# Get recommendations
curl http://localhost:8004/recommendations

# Get Prometheus metrics
curl http://localhost:8004/metrics
```

---

## Integration Points

### Analytics Service Integration

The Business Metrics Service receives data from the Analytics Service:
- Red team test results
- Model performance metrics
- Trend analysis data

### Monitoring Dashboard Integration

The Monitoring Dashboard displays business metrics:
- Real-time KPI dashboards
- Cost analysis charts
- Performance trend graphs
- Recommendation displays

### Alerting Integration

Business metrics can trigger alerts:
- High attack success rates
- Model drift detection
- Cost threshold breaches
- Performance degradation

---

## Best Practices

### KPI Monitoring

- Set up regular monitoring of key metrics
- Establish alert thresholds
- Track trends over time
- Compare against targets

### Cost Optimization

- Regular cost analysis
- Identify optimization opportunities
- Implement cost-saving measures
- Monitor cost trends

### Performance Management

- Track system effectiveness metrics
- Monitor response times
- Ensure high uptime
- Maintain user satisfaction

### Recommendation Implementation

- Prioritize high-impact recommendations
- Track implementation progress
- Measure success metrics
- Regular review and updates

---

## Troubleshooting

### Common Issues

1. **No KPI data**
   ```bash
   # Check if analytics service is running
   curl http://localhost:8006/
   
   # Check if data is being received
   curl http://localhost:8004/kpis
   ```

2. **Incorrect cost calculations**
   ```bash
   # Check cost metrics
   curl http://localhost:8004/cost-metrics
   
   # Verify cost data sources
   curl http://localhost:8004/system-effectiveness
   ```

3. **Model drift detection issues**
   ```bash
   # Check model drift
   curl http://localhost:8004/model-drift
   
   # Verify model performance data
   curl http://localhost:8006/analytics/model/comparison/deberta-v3-base
   ```

4. **Performance issues**
   ```bash
   # Check system effectiveness
   curl http://localhost:8004/system-effectiveness
   
   # Check service logs
   docker-compose logs business-metrics
   ```
