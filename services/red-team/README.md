# Red Team AI Security Service

A comprehensive AI red teaming service that provides advanced adversarial testing, compliance auditing, and security assessment capabilities for AI systems.

## 🚀 Features

### Core Capabilities
- **Advanced Adversarial Attacks**: Gradient-based, word-level, multi-turn, and agent-specific attacks
- **Compliance Frameworks**: OWASP LLM Top 10, NIST AI RMF, EU AI Act, HIPAA, PCI-DSS, ISO 42001
- **Benchmarking Integration**: HarmBench, StrongREJECT, SafetyBench
- **Behavior Analysis**: Activation, attribution, causal analysis, and anomaly detection
- **Privacy Testing**: Membership inference, model inversion, data extraction
- **Fairness Testing**: Demographic parity, counterfactual fairness, bias detection
- **Robustness Certification**: Randomized smoothing, interval bound propagation
- **Threat Intelligence**: MITRE ATLAS, CVE database, jailbreak databases
- **Incident Learning**: Pattern extraction, feedback loops, continuous improvement
- **Distributed Testing**: Celery + Redis task queue, horizontal scaling
- **Real-time Monitoring**: Prometheus metrics, WebSocket streaming, alerting
- **Comprehensive Reporting**: Executive, technical, and compliance reports

### Architecture
- **Microservices**: Modular design with separate services for different capabilities
- **Distributed Processing**: Scalable task queue with Celery workers
- **Containerized**: Docker Compose setup with all dependencies
- **Monitoring**: Prometheus, Grafana, Jaeger for observability
- **Storage**: PostgreSQL, Redis, MinIO for data persistence
- **API**: FastAPI with comprehensive documentation

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.9+
- 8GB+ RAM recommended
- GPU support (optional, for ML workloads)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd local-ml-security/services/red-team
   ```

2. **Build and start services**
   ```bash
   docker-compose up -d
   ```

3. **Verify installation**
   ```bash
   # Check service health
   curl http://localhost:8000/health
   
   # Check attack data generator
   curl http://localhost:8001/health
   
   # Check monitoring
   curl http://localhost:9090  # Prometheus
   curl http://localhost:3000  # Grafana
   ```

## 🚀 Quick Start

### 1. Generate Training Datasets
```bash
# Generate all datasets
python generate_datasets.py --dataset-type all --samples-per-category 100

# Generate specific dataset
python generate_datasets.py --dataset-type owasp --samples-per-category 50
```

### 2. Run Performance Tests
```bash
# Run all performance tests
python run_performance_tests.py --benchmark-type all

# Run specific benchmark
python run_performance_tests.py --benchmark-type latency --verbose
```

### 3. Execute Red Team Operations
```bash
# Run comprehensive benchmark
python benchmark_runner.py --benchmark-type all

# Run specific attack tests
python benchmark_runner.py --benchmark-type adversarial
```

## 📊 API Endpoints

### Main Service (Port 8000)
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `POST /attacks/execute` - Execute attack
- `GET /attacks/sessions` - List attack sessions
- `POST /compliance/check` - Run compliance check
- `GET /reports/generate` - Generate reports

### Attack Data Generator (Port 8001)
- `GET /health` - Health check
- `POST /patterns/generate` - Generate attack patterns
- `GET /patterns/list` - List available patterns
- `POST /patterns/evolve` - Evolve patterns using GA

### Monitoring
- `http://localhost:9090` - Prometheus metrics
- `http://localhost:3000` - Grafana dashboards
- `http://localhost:16686` - Jaeger tracing
- `http://localhost:5555` - Celery Flower

## 🔧 Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://red_team_user:red_team_password@postgres:5432/red_team

# Redis
REDIS_URL=redis://redis:6379

# MinIO
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# Celery
CELERY_BROKER_URL=redis://redis:6379
CELERY_RESULT_BACKEND=redis://redis:6379
```

### Performance Configuration
Edit `performance_config.yaml` to customize:
- Benchmark parameters
- Attack configurations
- Performance thresholds
- Alert settings

## 📈 Monitoring and Alerting

### Metrics
- **Performance**: Latency, throughput, memory usage, CPU usage
- **Attacks**: Success rate, failure rate, execution time
- **Compliance**: Violation rate, score trends
- **System**: Database connections, queue length, worker status

### Alerts
- High error rate (>10%)
- High latency (>1s 95th percentile)
- High memory usage (>1GB)
- Database connection failures
- Queue length exceeded
- Compliance violations
- Privacy breaches

### Dashboards
- **Security Overview**: Attack success rates, compliance scores
- **Performance**: System metrics, response times
- **Operations**: Queue status, worker health
- **Compliance**: Framework coverage, violation trends

## 🧪 Testing

### Unit Tests
```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific test category
pytest tests/unit/test_attacks.py -v
```

### Integration Tests
```bash
# Run integration tests
pytest tests/integration/ -v
```

### Performance Tests
```bash
# Run performance tests
pytest tests/performance/ -v
```

## 📚 Documentation

- **API Reference**: `docs/api/`
- **Attack Catalog**: `docs/attack_catalog/`
- **Runbooks**: `docs/runbooks/`
- **Architecture**: `docs/architecture/`

## 🔒 Security Considerations

### Data Protection
- All data encrypted at rest and in transit
- Access controls and authentication
- Audit logging for all operations
- Privacy-preserving techniques

### Compliance
- OWASP LLM Top 10 compliance
- NIST AI RMF framework
- EU AI Act compliance
- HIPAA, PCI-DSS, ISO 42001 support

### Best Practices
- Regular security updates
- Vulnerability scanning
- Penetration testing
- Security training

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check the docs/ directory
- **Issues**: Create a GitHub issue
- **Discussions**: Use GitHub Discussions
- **Security**: Report security issues privately

## 🗺️ Roadmap

### Phase 1: Core Attacks ✅
- [x] Gradient-based attacks
- [x] Word-level attacks
- [x] Multi-turn attacks
- [x] Agent-specific attacks

### Phase 2: Compliance & Benchmarking ✅
- [x] OWASP LLM Top 10
- [x] NIST AI RMF
- [x] EU AI Act
- [x] Benchmarking integration

### Phase 3: Advanced Features ✅
- [x] Pattern evolution
- [x] Threat intelligence
- [x] Incident learning
- [x] Data generation

### Phase 4: Analysis & Certification ✅
- [x] Behavior analysis
- [x] Robustness certification
- [x] Privacy attacks
- [x] Fairness testing

### Phase 5: Infrastructure ✅
- [x] Distributed processing
- [x] Caching and optimization
- [x] Monitoring and alerting

### Phase 6: Reporting & Visualization ✅
- [x] Executive reports
- [x] Technical reports
- [x] Interactive visualizations
- [x] Security dashboards

### Phase 7: Integration ✅
- [x] MLflow integration
- [x] SIEM integration
- [x] CI/CD gates

### Phase 8: Documentation & Datasets ✅
- [x] Comprehensive documentation
- [x] Training datasets
- [x] API reference

### Phase 9: Testing & Performance ✅
- [x] Test suite
- [x] Performance benchmarking
- [x] Load testing

## 📊 Performance Benchmarks

### Latency Targets
- **API Response**: <100ms (95th percentile)
- **Attack Execution**: <5s (average)
- **Report Generation**: <30s (average)

### Throughput Targets
- **API Requests**: >1000 req/s
- **Attack Processing**: >100 attacks/s
- **Report Generation**: >10 reports/s

### Resource Usage
- **Memory**: <4GB per service
- **CPU**: <80% utilization
- **Storage**: <100GB for datasets

## 🔍 Troubleshooting

### Common Issues
1. **Service won't start**: Check Docker logs
2. **Database connection failed**: Verify PostgreSQL is running
3. **High memory usage**: Check for memory leaks
4. **Slow performance**: Review resource limits

### Debug Commands
```bash
# Check service logs
docker-compose logs red-team-service

# Check database status
docker-compose exec postgres psql -U red_team_user -d red_team -c "SELECT 1;"

# Check Redis status
docker-compose exec redis redis-cli ping

# Check worker status
docker-compose exec celery-worker celery -A services.red_team.main inspect active
```

## 📞 Contact

- **Maintainer**: AI Security Team
- **Email**: security@example.com
- **Slack**: #ai-security
- **Office Hours**: Monday-Friday, 9 AM - 5 PM EST