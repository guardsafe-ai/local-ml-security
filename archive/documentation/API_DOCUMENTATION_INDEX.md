# ML Security Service - Complete API Documentation Index

This is the master index for all API documentation in the ML Security Service. Each service has its own detailed documentation file with complete endpoint descriptions, examples, and usage patterns.

## 📚 Documentation Overview

The ML Security Service consists of **12 microservices** with **81+ API endpoints** providing comprehensive ML security capabilities including model training, inference, red team testing, analytics, business metrics, and GDPR compliance.

---

## 🔗 Service Documentation Links

### Core ML Services

| Service | Port | Documentation | Purpose |
|---------|------|---------------|---------|
| **Model API** | 8000 | [API_ENDPOINTS_MODEL_API.md](./API_ENDPOINTS_MODEL_API.md) | Central inference gateway |
| **Training Service** | 8002 | [API_ENDPOINTS_TRAINING.md](./API_ENDPOINTS_TRAINING.md) | Model training & lifecycle |
| **Model Cache** | 8003 | [API_ENDPOINTS_MODEL_CACHE.md](./API_ENDPOINTS_MODEL_CACHE.md) | Intelligent caching |
| **Model Serving** | 8080 | [API_ENDPOINTS_MODEL_SERVING.md](./API_ENDPOINTS_MODEL_SERVING.md) | Production serving |

### Security & Testing Services

| Service | Port | Documentation | Purpose |
|---------|------|---------------|---------|
| **Red Team Service** | 8001 | [API_ENDPOINTS_RED_TEAM.md](./API_ENDPOINTS_RED_TEAM.md) | Security testing |
| **Data Privacy Service** | 8005 | [API_ENDPOINTS_DATA_PRIVACY.md](./API_ENDPOINTS_DATA_PRIVACY.md) | GDPR compliance |

### Analytics & Business Services

| Service | Port | Documentation | Purpose |
|---------|------|---------------|---------|
| **Analytics Service** | 8006 | [API_ENDPOINTS_ANALYTICS.md](./API_ENDPOINTS_ANALYTICS.md) | Data analysis |
| **Business Metrics** | 8004 | [API_ENDPOINTS_BUSINESS_METRICS.md](./API_ENDPOINTS_BUSINESS_METRICS.md) | Business KPIs |

### Infrastructure & Monitoring

| Service | Port | Documentation | Purpose |
|---------|------|---------------|---------|
| **Monitoring Dashboard** | 8501 | [COMPREHENSIVE_README.md](./COMPREHENSIVE_README.md) | Real-time monitoring |
| **Infrastructure Services** | Various | [API_ENDPOINTS_REMAINING_SERVICES.md](./API_ENDPOINTS_REMAINING_SERVICES.md) | MLflow, PostgreSQL, MinIO, Redis, Prometheus, Grafana, Jaeger |

---

## 🚀 Quick Start Guide

### 1. Start All Services

```bash
# Clone and setup
git clone <your-repo>
cd local-ml-security
chmod +x setup.sh
./setup.sh

# Start all services
docker-compose up -d

# Verify services
curl http://localhost:8000/health
curl http://localhost:8001/
curl http://localhost:8002/
```

### 2. Access the Dashboard

```bash
# Open monitoring dashboard
open http://localhost:8501
```

### 3. Test Core Functionality

```bash
# Test model prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Test text for prediction"}'

# Test red team
curl -X POST http://localhost:8001/test

# Check training status
curl http://localhost:8002/models
```

---

## 📊 API Endpoints Summary

### Model API Service (13 endpoints)
- **Health & Status**: `/`, `/health`
- **Model Management**: `/models`, `/models/{model_name}`, `/models/reload`
- **Predictions**: `/predict`, `/predict/batch`, `/test-predict`
- **Cache Management**: `/cache/stats`, `/cache/clear`
- **Metrics**: `/metrics`

### Training Service (19 endpoints)
- **Health & Status**: `/`
- **Model Registry**: `/models/registry`, `/models/latest`, `/models/best`, `/models/versions/{model_name}`
- **Model Management**: `/models`, `/models/{model_name}`, `/models/{model_name}/status`
- **Model Loading**: `/model-loading/status`, `/model-loading/start`, `/model-loading/load-specific`, `/model-loading/load-single`
- **Training**: `/train`, `/retrain`, `/create-sample-data`
- **Job Management**: `/jobs`, `/jobs/{job_id}`
- **Logs**: `/logs`

### Red Team Service (8 endpoints)
- **Health & Status**: `/`, `/status`
- **Testing**: `/start`, `/stop`, `/test`
- **Model Management**: `/models`
- **Results**: `/results`
- **Metrics**: `/metrics`

### Model Cache Service (11 endpoints)
- **Health & Status**: `/`, `/health`
- **Model Management**: `/models`, `/models/{model_name}`
- **Model Operations**: `/models/{model_name}/load`, `/models/{model_name}/unload`, `/models/{model_name}/warmup`
- **Predictions**: `/predict`
- **Cache Management**: `/stats`, `/cleanup`
- **Metrics**: `/metrics`

### Model Serving Service (8 endpoints)
- **Health & Status**: `/`, `/health`
- **Model Management**: `/models`, `/models/{model_name}`
- **Predictions**: `/models/{model_name}/predict`, `/predict`, `/predict/batch`
- **Metrics**: `/metrics`

### Analytics Service (6 endpoints)
- **Health & Status**: `/`
- **Data Storage**: `/red-team/results`, `/model/performance`
- **Analytics**: `/analytics/red-team/summary`, `/analytics/model/comparison/{model_name}`, `/analytics/trends`

### Business Metrics Service (7 endpoints)
- **Health & Status**: `/`
- **KPIs**: `/kpis`, `/attack-success-rate`, `/model-drift`, `/cost-metrics`, `/system-effectiveness`
- **Recommendations**: `/recommendations`
- **Metrics**: `/metrics`

### Data Privacy Service (9 endpoints)
- **Health & Status**: `/`
- **Data Processing**: `/anonymize`
- **Data Subject Management**: `/data-subjects`, `/data-subjects/{subject_id}/withdraw-consent`, `/data-subjects/{subject_id}`
- **Compliance**: `/cleanup`, `/compliance`, `/audit-logs`
- **Metrics**: `/metrics`

---

## 🔧 Common Usage Patterns

### 1. Model Training Workflow

```bash
# 1. Load models
curl -X POST http://localhost:8002/model-loading/start

# 2. Create sample data
curl -X POST http://localhost:8002/create-sample-data

# 3. Train model
curl -X POST http://localhost:8002/train \
  -H "Content-Type: application/json" \
  -d '{"model_name": "deberta-v3-base"}'

# 4. Check training status
curl http://localhost:8002/jobs

# 5. Test trained model
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Test text", "models": ["deberta-v3-base_trained"]}'
```

### 2. Red Team Testing Workflow

```bash
# 1. Start continuous testing
curl -X POST http://localhost:8001/start

# 2. Run specific test
curl -X POST http://localhost:8001/test?batch_size=20&model_name=deberta-v3-base_trained

# 3. Get results
curl http://localhost:8001/results

# 4. Check analytics
curl http://localhost:8006/analytics/red-team/summary
```

### 3. Business Metrics Monitoring

```bash
# 1. Get all KPIs
curl http://localhost:8004/kpis

# 2. Check attack success rate
curl http://localhost:8004/attack-success-rate

# 3. Check model drift
curl http://localhost:8004/model-drift

# 4. Get recommendations
curl http://localhost:8004/recommendations
```

### 4. Data Privacy Compliance

```bash
# 1. Anonymize data
curl -X POST http://localhost:8005/anonymize \
  -H "Content-Type: application/json" \
  -d '{"text": "Contact John Doe at john@example.com"}'

# 2. Register data subject
curl -X POST http://localhost:8005/data-subjects \
  -H "Content-Type: application/json" \
  -d '{"subject_id": "user_123", "email": "user@example.com", "consent_given": true}'

# 3. Check compliance
curl http://localhost:8005/compliance

# 4. Get audit logs
curl http://localhost:8005/audit-logs
```

---

## 📈 Performance Benchmarks

### Response Times

| Service | Endpoint | Avg Response Time | 95th Percentile |
|---------|----------|------------------|-----------------|
| Model API | `/predict` | 45ms | 100ms |
| Training | `/train` | 30s | 60s |
| Red Team | `/test` | 5s | 10s |
| Analytics | `/analytics/red-team/summary` | 200ms | 500ms |
| Business Metrics | `/kpis` | 150ms | 300ms |
| Data Privacy | `/anonymize` | 50ms | 100ms |

### Throughput

| Service | Requests/Second | Concurrent Users |
|---------|----------------|------------------|
| Model API | 100 | 50 |
| Model Serving | 200 | 100 |
| Red Team | 10 | 5 |
| Analytics | 50 | 25 |
| Business Metrics | 100 | 50 |
| Data Privacy | 80 | 40 |

---

## 🔒 Security Features

### Authentication & Authorization
- Service-to-service authentication
- API key management
- Role-based access control

### Data Protection
- PII detection and anonymization
- GDPR compliance
- Data encryption at rest and in transit
- Audit logging

### Security Testing
- Automated red team testing
- Continuous vulnerability assessment
- Model drift detection
- Attack pattern generation

---

## 📊 Monitoring & Observability

### Metrics Collection
- **Prometheus**: System and application metrics
- **Grafana**: Visualization and dashboards
- **Jaeger**: Distributed tracing

### Health Monitoring
- Service health checks
- Resource usage monitoring
- Performance metrics
- Error tracking

### Alerting
- Performance degradation alerts
- Error rate thresholds
- Resource usage alerts
- Security incident alerts

---

## 🛠️ Development & Testing

### Local Development

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Run tests
python -m pytest tests/

# Check code quality
flake8 services/
black services/
```

### API Testing

```bash
# Test all endpoints
./test_all_endpoints.sh

# Load testing
./load_test.sh

# Security testing
./security_test.sh
```

### Documentation

```bash
# Generate API docs
./generate_docs.sh

# Update documentation
./update_docs.sh
```

---

## 📚 Additional Resources

### Architecture Documentation
- [COMPREHENSIVE_README.md](./COMPREHENSIVE_README.md) - Complete system overview
- [ENHANCED_FEATURES.md](./ENHANCED_FEATURES.md) - Feature details

### Configuration
- `docker-compose.yml` - Service configuration
- `.env` - Environment variables
- `setup.sh` - Installation script

### Monitoring
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Jaeger: http://localhost:16686
- MLflow: http://localhost:5000

---

## 🆘 Support & Troubleshooting

### Common Issues

1. **Service not responding**
   ```bash
   docker-compose ps
   docker-compose logs service-name
   ```

2. **Database connection issues**
   ```bash
   docker-compose logs postgres
   docker-compose restart postgres
   ```

3. **Memory issues**
   ```bash
   docker stats
   docker-compose restart
   ```

4. **Network issues**
   ```bash
   docker-compose down && docker-compose up -d
   ```

### Getting Help

- Check service logs: `docker-compose logs service-name`
- Verify health: `curl http://localhost:port/health`
- Check metrics: `curl http://localhost:9090/api/v1/query?query=up`
- Review documentation: See individual service docs above

---

## 🎯 Next Steps

1. **Deploy the system** using the setup scripts
2. **Explore the dashboard** at http://localhost:8501
3. **Test the APIs** using the examples in each service documentation
4. **Monitor performance** through Prometheus and Grafana
5. **Customize and extend** based on your specific needs

The ML Security Service provides a complete, enterprise-grade solution for machine learning security with comprehensive monitoring, business intelligence, and compliance features in a single, locally-deployable platform.
