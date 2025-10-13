#!/usr/bin/env node

/**
 * Comprehensive Job Logs Test
 * Tests the complete job logs functionality end-to-end
 */

const axios = require('axios');

async function testJobLogsComprehensive() {
  console.log('🧪 Comprehensive Job Logs Test Starting...\n');
  
  const results = {
    passed: 0,
    failed: 0,
    tests: []
  };

  const test = (name, fn) => {
    results.tests.push({ name, fn });
  };

  const runTests = async () => {
    for (const testCase of results.tests) {
      try {
        console.log(`\n🔍 Testing: ${testCase.name}`);
        await testCase.fn();
        console.log(`✅ PASSED: ${testCase.name}`);
        results.passed++;
      } catch (error) {
        console.log(`❌ FAILED: ${testCase.name}`);
        console.log(`   Error: ${error.message}`);
        results.failed++;
      }
    }
  };

  // Test 1: Backend API Health
  test('Backend API Health Check', async () => {
    const response = await axios.get('http://localhost:8007/health');
    if (response.status !== 200) {
      throw new Error(`Backend health check failed: ${response.status}`);
    }
  });

  // Test 2: Training Jobs Endpoint
  test('Training Jobs Endpoint', async () => {
    const response = await axios.get('http://localhost:8007/training/jobs');
    if (response.status !== 200) {
      throw new Error(`Training jobs endpoint failed: ${response.status}`);
    }
    const jobs = response.data.jobs || response.data;
    if (!Array.isArray(jobs) || jobs.length === 0) {
      throw new Error('No training jobs found');
    }
    console.log(`   Found ${jobs.length} training jobs`);
  });

  // Test 3: Job Details Endpoint
  test('Job Details Endpoint', async () => {
    const jobsResponse = await axios.get('http://localhost:8007/training/jobs');
    const jobs = jobsResponse.data.jobs || jobsResponse.data;
    const job = jobs[0];
    
    const response = await axios.get(`http://localhost:8007/training/jobs/${job.job_id}`);
    if (response.status !== 200) {
      throw new Error(`Job details endpoint failed: ${response.status}`);
    }
    console.log(`   Job details for ${job.job_id}: ${response.data.status}`);
  });

  // Test 4: Job Logs Endpoint
  test('Job Logs Endpoint', async () => {
    const jobsResponse = await axios.get('http://localhost:8007/training/jobs');
    const jobs = jobsResponse.data.jobs || jobsResponse.data;
    const job = jobs[0];
    
    const response = await axios.get(`http://localhost:8007/training/jobs/${job.job_id}/logs`);
    if (response.status !== 200) {
      throw new Error(`Job logs endpoint failed: ${response.status}`);
    }
    
    const logs = response.data.logs || [];
    if (!Array.isArray(logs)) {
      throw new Error('Logs response is not an array');
    }
    console.log(`   Found ${logs.length} log entries for ${job.job_id}`);
    
    if (logs.length > 0) {
      const firstLog = logs[0];
      console.log(`   Sample log: ${firstLog.timestamp} - ${firstLog.level} - ${firstLog.message}`);
    }
  });

  // Test 5: Frontend Accessibility
  test('Frontend Accessibility', async () => {
    const response = await axios.get('http://localhost:3000');
    if (response.status !== 200) {
      throw new Error(`Frontend not accessible: ${response.status}`);
    }
    console.log('   Frontend is accessible');
  });

  // Test 6: Training Service Direct Access
  test('Training Service Direct Access', async () => {
    const response = await axios.get('http://localhost:8002/training/jobs');
    if (response.status !== 200) {
      throw new Error(`Training service not accessible: ${response.status}`);
    }
    console.log('   Training service is accessible');
  });

  // Test 7: Model Registry Integration
  test('Model Registry Integration', async () => {
    const response = await axios.get('http://localhost:8007/models/registry');
    if (response.status !== 200) {
      throw new Error(`Model registry not accessible: ${response.status}`);
    }
    console.log('   Model registry is accessible');
  });

  // Run all tests
  await runTests();

  // Summary
  console.log('\n📊 Test Results Summary:');
  console.log(`✅ Passed: ${results.passed}`);
  console.log(`❌ Failed: ${results.failed}`);
  console.log(`📈 Success Rate: ${((results.passed / (results.passed + results.failed)) * 100).toFixed(1)}%`);

  if (results.failed === 0) {
    console.log('\n🎉 All tests passed! Job logs functionality is working correctly.');
    console.log('\n📋 Next Steps:');
    console.log('1. Open http://localhost:3000 in your browser');
    console.log('2. Navigate to the Training page');
    console.log('3. Click on any job to view details');
    console.log('4. Check the "Training Logs" section in the job details dialog');
    console.log('5. Look for the debug info box showing loading state and log count');
    console.log('6. Check browser console for detailed debug messages');
  } else {
    console.log('\n💔 Some tests failed. Please check the errors above.');
  }
}

testJobLogsComprehensive().catch(console.error);
