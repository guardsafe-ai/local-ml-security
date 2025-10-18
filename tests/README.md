# ML Security Platform - Comprehensive Testing Suite

This directory contains a comprehensive testing suite for the ML Security Platform, organized by service and test type.

## 🏗️ Test Structure

```
tests/
├── conftest.py                 # Pytest configuration and fixtures
├── pytest.ini                 # Pytest settings
├── requirements.txt            # Test dependencies
├── run_tests.py               # Test runner script
├── README.md                  # This file
├── platform/                  # Platform-wide tests
│   ├── unit/                  # Unit tests for platform components
│   ├── integration/           # Integration tests between services
│   ├── e2e/                   # End-to-end workflow tests
│   └── performance/           # Performance and load tests
└── services/                  # Service-specific tests
    ├── enterprise-dashboard-backend/  # Backend gateway tests
    │   ├── test_backend_routes.py     # Route and endpoint tests
    │   └── test_backend_clients.py    # Service client tests
    ├── model-api/             # Model API service tests
    ├── training/              # Training service tests
    ├── analytics/             # Analytics service tests
    ├── business-metrics/      # Business metrics service tests
    ├── data-privacy/          # Data privacy service tests
    ├── tracing/               # Tracing service tests
    ├── mlflow/                # MLflow service tests
    └── model-cache/           # Model cache service tests
```

## 🎯 Test Categories

### Platform-Wide Tests (`tests/platform/`)
- **Unit Tests**: Test individual platform components in isolation
- **Integration Tests**: Test service interactions and data flow
- **End-to-End Tests**: Test complete user workflows across all services
- **Performance Tests**: Test system performance, load handling, and scalability

### Service-Specific Tests (`tests/services/`)
- **Enterprise Dashboard Backend**: Tests for the backend gateway and its service clients
- **Individual Services**: Tests for each of the 8 core services
- **Service Integration**: Tests for how services work together

## 🚀 Quick Start

### 1. Install Test Dependencies

```bash
cd /Users/arpitsrivastava/Documents/guardsafe-ai/local-ml-security
pip install -r tests/requirements.txt
```

### 2. Run All Tests

```bash
python tests/run_tests.py --type all
```

### 3. Run Platform-Wide Tests

```bash
# All platform tests
python tests/run_tests.py --type platform

# Specific platform test types
python tests/run_tests.py --type unit
python tests/run_tests.py --type integration
python tests/run_tests.py --type e2e
python tests/run_tests.py --type performance
```

### 4. Run Service-Specific Tests

```bash
# Backend gateway tests
python tests/run_tests.py --type backend

# Individual service tests
python tests/run_tests.py --type model-api
python tests/run_tests.py --type training
python tests/run_tests.py --type analytics
# ... etc
```

## 🧪 Test Types Explained

### Platform Tests
- **Purpose**: Test the entire ML Security Platform as a unified system
- **Scope**: All 8 services + backend gateway + integrations
- **Dependencies**: Requires all services running
- **Use Case**: System-wide validation, performance testing, end-to-end workflows

### Service Tests
- **Purpose**: Test individual services in isolation
- **Scope**: Single service functionality, API endpoints, business logic
- **Dependencies**: May require service dependencies (Redis, PostgreSQL, etc.)
- **Use Case**: Service development, debugging, unit testing

## 📊 Test Features

### Comprehensive Coverage
- **8 Core Services**: All services tested individually and together
- **Backend Gateway**: Complete API coverage testing
- **Service Clients**: All service client functionality tested
- **Error Handling**: Robust error scenario testing
- **Performance**: Load testing and performance validation
- **Data Flow**: End-to-end data consistency testing

### Advanced Testing Capabilities
- **Async Testing**: Full async/await support
- **Concurrent Testing**: Multi-threaded and multi-process testing
- **Mock Testing**: Comprehensive mocking for isolated testing
- **Performance Benchmarking**: Detailed performance metrics
- **Coverage Reporting**: Code coverage analysis
- **HTML Reports**: Detailed test reports with screenshots

## 🛠️ Advanced Usage

### Run with Coverage

```bash
python tests/run_tests.py --type all --coverage
```

### Run with HTML Report

```bash
python tests/run_tests.py --type all --html
```

### Run Performance Benchmarks

```bash
python tests/run_tests.py --type performance --benchmark
```

### Run in Parallel

```bash
python tests/run_tests.py --type all --parallel 4
```

### Run Specific Test Files

```bash
# Run backend gateway tests
pytest tests/services/enterprise-dashboard-backend/ -v

# Run platform integration tests
pytest tests/platform/integration/ -v

# Run specific test class
pytest tests/services/enterprise-dashboard-backend/test_backend_routes.py::TestBackendRoutes -v
```

## 📈 Test Metrics

### Coverage Targets
- **Platform Tests**: 90%+ code coverage across all services
- **Service Tests**: 95%+ code coverage for individual services
- **Integration Tests**: 80%+ API endpoint coverage
- **E2E Tests**: 100% critical workflow coverage

### Performance Targets
- **Response Time**: < 2 seconds average, < 5 seconds maximum
- **Concurrent Load**: 50+ concurrent requests
- **Memory Usage**: < 100MB increase under load
- **Error Rate**: < 5% under normal load

## 🔧 Configuration

### Environment Variables
```bash
# Test configuration
export TEST_BASE_URL=http://localhost:8007
export TEST_TIMEOUT=30
export TEST_RETRY_ATTEMPTS=3

# Service URLs (if different from defaults)
export MODEL_API_URL=http://localhost:8000
export TRAINING_URL=http://localhost:8002
# ... etc
```

## 🐛 Troubleshooting

### Common Issues

1. **Services Not Running**
   ```bash
   # Start all services
   docker-compose up -d
   
   # Check service status
   docker ps
   ```

2. **Port Conflicts**
   ```bash
   # Check port usage
   lsof -i :8007
   
   # Kill conflicting processes
   sudo kill -9 <PID>
   ```

3. **Test Timeouts**
   ```bash
   # Increase timeout
   export TEST_TIMEOUT=60
   python tests/run_tests.py --type integration
   ```

## 📝 Adding New Tests

### Platform Tests
1. Create test file in appropriate `tests/platform/` subdirectory
2. Follow naming convention: `test_*.py`
3. Use mocks for external dependencies
4. Keep tests fast and isolated

### Service Tests
1. Create test file in `tests/services/<service-name>/`
2. Test service-specific functionality
3. Use realistic test data
4. Test both success and error scenarios

## 🎯 Best Practices

1. **Test Isolation**: Each test should be independent
2. **Clear Naming**: Use descriptive test names
3. **Single Responsibility**: One assertion per test
4. **Fast Feedback**: Keep unit tests fast
5. **Realistic Data**: Use realistic test data
6. **Error Testing**: Test both success and failure cases
7. **Documentation**: Document complex test scenarios
8. **Maintenance**: Keep tests up to date with code changes

## 🏆 Success Criteria

A successful test run should show:
- ✅ All tests passing
- ✅ High code coverage (>90%)
- ✅ Performance targets met
- ✅ No flaky tests
- ✅ Clean HTML reports
- ✅ Fast execution times

---

**Happy Testing! 🧪✨**