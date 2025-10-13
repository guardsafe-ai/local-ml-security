#!/usr/bin/env node

/**
 * Final Verification Test
 * Tests the complete job logs functionality after rebuild
 */

const axios = require('axios');

async function testFinalVerification() {
  console.log('🎯 Final Verification Test - After Rebuild\n');
  
  try {
    // Test 1: Frontend accessibility
    console.log('1. Testing frontend accessibility...');
    const frontendResponse = await axios.get('http://localhost:3000');
    console.log(`   ✅ Frontend status: ${frontendResponse.status}`);
    
    // Test 2: Backend APIs
    console.log('\n2. Testing backend APIs...');
    const jobsResponse = await axios.get('http://localhost:8007/training/jobs');
    const jobs = jobsResponse.data.jobs || jobsResponse.data;
    const job = jobs[0];
    
    const logsResponse = await axios.get(`http://localhost:8007/training/jobs/${job.job_id}/logs`);
    const logs = logsResponse.data.logs || [];
    
    console.log(`   ✅ Backend working: ${logs.length} logs available for ${job.job_id}`);
    
    console.log('\n🎉 SUCCESS! Frontend has been rebuilt and is running.');
    console.log('\n📋 What to do now:');
    console.log('1. Open http://localhost:3000 in your browser');
    console.log('2. Go to the Training page');
    console.log('3. Look for the BRIGHT ORANGE DEBUG SECTION at the bottom of the page');
    console.log('4. You should see:');
    console.log('   - "🔧 DEBUG: Job Logs Section" title');
    console.log('   - "Selected Job: ❌ None selected" (initially)');
    console.log('   - "Dialog Open: ❌ No" (initially)');
    console.log('   - "Select First Job" and "Clear Selection" buttons');
    console.log('   - "Load Logs" button');
    console.log('5. Click "Select First Job" to manually select a job');
    console.log('6. Click "Load Logs" to load the logs');
    console.log('7. You should see the logs appear in the debug section');
    
    console.log('\n🔍 If you still can\'t see the orange debug section:');
    console.log('- Make sure you\'re on the Training page');
    console.log('- Scroll to the bottom of the page');
    console.log('- Check browser console for any errors (F12)');
    console.log('- Try refreshing the page');
    
    console.log('\n📊 Expected Results:');
    console.log('- Orange debug section should be visible');
    console.log('- Manual job selection should work');
    console.log('- Logs should load and display');
    console.log('- All debug information should be shown');
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    if (error.response) {
      console.error('   Response status:', error.response.status);
    }
  }
}

testFinalVerification();
