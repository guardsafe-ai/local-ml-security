# ML Security Dashboard - Testing Guide

This guide covers comprehensive testing strategies for the ML Security Dashboard, including frontend-backend integration testing.

## 🧪 **Testing Tools Setup**

### Playwright (E2E Testing)
- **Installed**: `@playwright/test`
- **Configuration**: `playwright.config.ts`
- **Test Directory**: `./tests/`

### Available Test Commands
```bash
# Run all E2E tests
npm run test:e2e

# Run tests with UI (interactive)
npm run test:e2e:ui

# Run tests in headed mode (see browser)
npm run test:e2e:headed

# Debug tests step by step
npm run test:e2e:debug

# Run specific test suites
npm run test:api              # Backend API validation
npm run test:integration      # Frontend-backend integration
npm run test:workflows        # User workflow testing
```

## 📋 **Test Categories**

### 1. **Backend API Validation** (`backend-api-validation.spec.ts`)
Tests the backend API directly to ensure:
- ✅ All endpoints return correct status codes
- ✅ Response data structure is valid
- ✅ Error handling works properly
- ✅ CORS headers are set correctly
- ✅ Response times are acceptable
- ✅ Concurrent requests are handled

**Key Tests:**
- Health check endpoint
- Dashboard metrics validation
- Models data structure
- Red team results validation
- Analytics data integrity
- Model prediction functionality
- Error handling for invalid requests

### 2. **Frontend-Backend Integration** (`api-backend-integration.spec.ts`)
Tests the complete data flow from frontend to backend:
- ✅ Frontend makes correct API calls
- ✅ Backend responds with valid data
- ✅ Frontend displays data correctly
- ✅ Real-time updates work
- ✅ Error handling in UI
- ✅ Loading states work properly

**Key Tests:**
- Dashboard loads with real backend data
- Models page displays API data
- Red team page shows test results
- Analytics page loads summary data
- Training page displays job information
- MLflow page shows experiment data
- Error handling for failed API calls
- Real-time data refresh functionality

### 3. **User Workflow Testing** (`user-workflows.spec.ts`)
Tests complete user journeys:
- ✅ Model management workflow
- ✅ Red team testing workflow
- ✅ Training workflow
- ✅ Analytics workflow
- ✅ Navigation between pages
- ✅ Error recovery

**Key Workflows:**
- Load model → Test prediction → View results
- Configure red team test → Run test → Analyze results
- Start training job → Monitor progress → View completion
- Filter analytics → Compare models → Export data
- Navigate between all pages
- Handle errors and retry functionality

## 🚀 **Running Tests**

### Prerequisites
1. **Backend Services Running**:
   ```bash
   cd /Users/arpitsrivastava/Desktop/ITRIcometax/local-ml-security
   docker-compose up -d
   ```

2. **Frontend Running**:
   ```bash
   cd services/enterprise-dashboard/frontend
   npm start
   ```

### Quick Test Run
```bash
# Test backend APIs only (fastest)
npm run test:api

# Test frontend-backend integration
npm run test:integration

# Test complete user workflows
npm run test:workflows

# Run all tests
npm run test:e2e
```

### Debugging Tests
```bash
# Run with UI for interactive debugging
npm run test:e2e:ui

# Run in headed mode to see browser
npm run test:e2e:headed

# Debug specific test
npm run test:e2e:debug -- --grep "Dashboard loads"
```

## 📊 **Test Data Requirements**

### Backend Data
- At least one model loaded (`distilbert_trained`)
- Red team test results available
- Training jobs in various states
- MLflow experiments and runs
- Analytics data (summary and trends)

### Test Scenarios
1. **Happy Path**: All services working, data available
2. **Error Scenarios**: Network failures, invalid data
3. **Edge Cases**: Empty data, slow responses
4. **Load Testing**: Multiple concurrent requests

## 🔍 **Test Verification Points**

### API Response Validation
- Status codes (200, 404, 422, 500)
- Response structure matches expected schema
- Data types are correct
- Required fields are present
- Error messages are meaningful

### Frontend Display Validation
- Data loads without errors
- Loading states show/hide correctly
- Error messages display properly
- Real-time updates work
- Navigation functions correctly
- Forms submit successfully

### User Experience Validation
- Page loads within acceptable time
- Interactive elements respond
- Data refreshes automatically
- Error recovery works
- Mobile responsiveness

## 🐛 **Common Issues & Solutions**

### Test Failures
1. **API Timeout**: Increase timeout in config
2. **Element Not Found**: Add proper test IDs to components
3. **Network Errors**: Ensure backend services are running
4. **Data Mismatch**: Verify test data setup

### Debugging Tips
1. Use `--headed` mode to see what's happening
2. Add `await page.pause()` in tests to inspect
3. Check browser console for errors
4. Verify API responses manually with curl
5. Use `--debug` mode for step-by-step execution

## 📈 **Continuous Integration**

### GitHub Actions Example
```yaml
name: E2E Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm install
      - run: docker-compose up -d
      - run: npm run test:e2e
```

### Test Reports
- HTML reports generated in `playwright-report/`
- Screenshots on failure in `test-results/`
- Videos of failed tests
- Trace files for debugging

## 🎯 **Best Practices**

1. **Test Data**: Use consistent test data
2. **Isolation**: Each test should be independent
3. **Cleanup**: Clean up after tests
4. **Reliability**: Use proper waits, not fixed timeouts
5. **Maintainability**: Keep tests simple and focused
6. **Documentation**: Document complex test scenarios

## 📝 **Adding New Tests**

1. **Identify Test Category**: API, Integration, or Workflow
2. **Create Test File**: Follow naming convention
3. **Add Test IDs**: Update components with `data-testid`
4. **Write Test**: Use Playwright best practices
5. **Run Tests**: Verify they pass
6. **Update Documentation**: Add to this guide

---

**Happy Testing! 🚀**
