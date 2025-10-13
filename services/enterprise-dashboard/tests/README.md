# 🧪 Enterprise Dashboard Tests

This directory contains comprehensive test suites for the Enterprise Dashboard service.

## 📁 **Test Organization**

### **Frontend Tests**
- `test-*.js` - Frontend functionality tests
- `debug-*.js` - Debug and troubleshooting scripts

### **Backend Tests**
- `test-*-comprehensive.js` - End-to-end backend tests
- `test-*-verification.js` - Feature verification tests

### **Reports**
- `*-REPORT.md` - Test execution reports and analysis
- `*-ANALYSIS.md` - Feature analysis and documentation

## 🚀 **Running Tests**

### **Frontend Tests**
```bash
# Test frontend accessibility
node test-frontend-accessibility.js

# Test specific features
node test-model-registry-comprehensive.js
node test-job-logs-comprehensive.js
```

### **Backend Tests**
```bash
# Test backend APIs
node test-backend-apis.js

# Test end-to-end functionality
node test-complete-rebuild.js
```

## 📊 **Test Categories**

### **1. Model Registry Tests**
- Model registration and retrieval
- Model versioning and staging
- Model performance metrics
- Model deployment status

### **2. Training Tests**
- Training job management
- Job logs and monitoring
- Training configuration
- Job status tracking

### **3. Analytics Tests**
- Performance analytics
- Data visualization
- Chart rendering
- Real-time updates

### **4. Integration Tests**
- Frontend-backend integration
- Service communication
- Data flow validation
- End-to-end workflows

## 🔧 **Test Configuration**

All tests are configured to run against:
- **Frontend**: `http://localhost:3000`
- **Backend**: `http://localhost:8007`
- **MLflow**: `http://localhost:5000`

## 📋 **Test Reports**

- `MODEL_REGISTRY_COMPREHENSIVE_REPORT.md` - Model registry testing results
- `JOB_LOGS_FIX_REPORT.md` - Training logs functionality fixes
- `ENHANCED_MODEL_REGISTRY_REPORT.md` - Enhanced model registry features

## 🎯 **Best Practices**

1. **Run tests before deployment** to ensure functionality
2. **Check test reports** for detailed analysis
3. **Use debug scripts** for troubleshooting
4. **Update tests** when adding new features
5. **Document test results** in report files
