#!/usr/bin/env node

/**
 * Debug Job Logs UI
 * Creates a simple test to check if the job logs are working
 */

const axios = require('axios');

async function debugJobLogsUI() {
  console.log('🔍 Debugging Job Logs UI...\n');
  
  try {
    // Test 1: Check if frontend is accessible
    console.log('1. Checking frontend accessibility...');
    const frontendResponse = await axios.get('http://localhost:3000');
    console.log(`   ✅ Frontend status: ${frontendResponse.status}`);
    
    // Test 2: Check if we can get training jobs
    console.log('\n2. Checking training jobs...');
    const jobsResponse = await axios.get('http://localhost:8007/training/jobs');
    const jobs = jobsResponse.data.jobs || jobsResponse.data;
    console.log(`   ✅ Found ${jobs.length} training jobs`);
    
    if (jobs.length > 0) {
      const job = jobs[0];
      console.log(`   📋 Sample job: ${job.job_id} (${job.status})`);
      
      // Test 3: Check job details
      console.log('\n3. Checking job details...');
      const jobDetailsResponse = await axios.get(`http://localhost:8007/training/jobs/${job.job_id}`);
      console.log(`   ✅ Job details: ${jobDetailsResponse.data.status} - ${jobDetailsResponse.data.progress}%`);
      
      // Test 4: Check job logs
      console.log('\n4. Checking job logs...');
      const logsResponse = await axios.get(`http://localhost:8007/training/jobs/${job.job_id}/logs`);
      const logs = logsResponse.data.logs || [];
      console.log(`   ✅ Found ${logs.length} log entries`);
      
      if (logs.length > 0) {
        console.log('   📝 Sample logs:');
        logs.slice(0, 3).forEach((log, index) => {
          console.log(`      ${index + 1}. [${log.timestamp}] ${log.level}: ${log.message}`);
        });
      }
    }
    
    console.log('\n🎯 UI Debug Instructions:');
    console.log('1. Open http://localhost:3000 in your browser');
    console.log('2. Open Developer Tools (F12)');
    console.log('3. Go to Console tab');
    console.log('4. Navigate to Training page');
    console.log('5. Click on any job');
    console.log('6. Look for these console messages:');
    console.log('   - "Opening job details for: [job_id]"');
    console.log('   - "Job details fetched: [details]"');
    console.log('   - "🔄 Loading logs for job: [job_id]"');
    console.log('   - "📋 Received logs response: [response]"');
    console.log('   - "📝 Setting logs array: [count] entries"');
    console.log('   - "🔍 Job details logs state changed: [state]"');
    console.log('7. In the job details dialog, look for:');
    console.log('   - "Training Logs" section');
    console.log('   - Debug info box showing loading state and count');
    console.log('   - Either logs display or "No logs available" message');
    
  } catch (error) {
    console.error('❌ Debug failed:', error.message);
    if (error.response) {
      console.error('   Response status:', error.response.status);
      console.error('   Response data:', error.response.data);
    }
  }
}

debugJobLogsUI();
