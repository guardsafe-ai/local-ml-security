#!/usr/bin/env node

/**
 * Test Job Logs Functionality
 * Tests if job details and logs are working properly
 */

const axios = require('axios');

async function testJobLogs() {
  console.log('🧪 Testing Job Logs Functionality...\n');
  
  try {
    // Test 1: Get training jobs
    console.log('1. Getting training jobs...');
    const jobsResponse = await axios.get('http://localhost:8007/training/jobs');
    const jobs = jobsResponse.data.jobs || jobsResponse.data;
    
    if (jobs.length === 0) {
      console.log('❌ No training jobs found');
      return;
    }
    
    const job = jobs[0];
    console.log(`✅ Found job: ${job.job_id} (${job.status})`);
    
    // Test 2: Get job details
    console.log('\n2. Getting job details...');
    const jobDetailsResponse = await axios.get(`http://localhost:8007/training/jobs/${job.job_id}`);
    const jobDetails = jobDetailsResponse.data;
    console.log(`✅ Job details: ${jobDetails.status} - ${jobDetails.progress}%`);
    
    // Test 3: Get job logs
    console.log('\n3. Getting job logs...');
    const logsResponse = await axios.get(`http://localhost:8007/training/jobs/${job.job_id}/logs`);
    const logs = logsResponse.data;
    
    if (logs.logs && logs.logs.length > 0) {
      console.log(`✅ Found ${logs.logs.length} log entries`);
      console.log('Sample log entry:');
      console.log(`  - ${logs.logs[0].timestamp}: ${logs.logs[0].level} - ${logs.logs[0].message}`);
    } else {
      console.log('❌ No logs found for this job');
    }
    
    // Test 4: Check frontend accessibility
    console.log('\n4. Checking frontend accessibility...');
    const frontendResponse = await axios.get('http://localhost:3000');
    if (frontendResponse.status === 200) {
      console.log('✅ Frontend is accessible');
    } else {
      console.log('❌ Frontend is not accessible');
    }
    
    console.log('\n🎉 Job logs functionality test completed!');
    console.log('\nTo test the UI:');
    console.log('1. Open http://localhost:3000');
    console.log('2. Go to Training page');
    console.log('3. Click on a job to view details');
    console.log('4. Check if logs are displayed in the job details dialog');
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    if (error.response) {
      console.error('Response status:', error.response.status);
      console.error('Response data:', error.response.data);
    }
  }
}

testJobLogs();
