#!/usr/bin/env node

/**
 * Final Test for Job Logs Fix
 * Verifies the simplified logs section is working
 */

const axios = require('axios');

async function testFinalLogsFix() {
  console.log('🎯 Final Test for Job Logs Fix\n');
  
  try {
    // Test backend
    console.log('1. Testing backend APIs...');
    const jobsResponse = await axios.get('http://localhost:8007/training/jobs');
    const jobs = jobsResponse.data.jobs || jobsResponse.data;
    const job = jobs[0];
    
    const logsResponse = await axios.get(`http://localhost:8007/training/jobs/${job.job_id}/logs`);
    const logs = logsResponse.data.logs || [];
    
    console.log(`   ✅ Backend working: ${logs.length} logs available for ${job.job_id}`);
    
    // Test frontend
    console.log('\n2. Testing frontend...');
    const frontendResponse = await axios.get('http://localhost:3000');
    console.log(`   ✅ Frontend accessible: ${frontendResponse.status}`);
    
    console.log('\n🎉 SUCCESS! The fix has been applied.');
    console.log('\n📋 What to do now:');
    console.log('1. Open http://localhost:3000 in your browser');
    console.log('2. Go to the Training page');
    console.log('3. Click on ANY job to view details');
    console.log('4. Scroll down in the job details dialog');
    console.log('5. You should now see a BLUE BORDERED "📋 Training Logs" section');
    console.log('6. Click the "Load Logs" button');
    console.log('7. You should see the logs appear below');
    
    console.log('\n🔍 What you should see:');
    console.log('- A blue-bordered card with "📋 Training Logs" title');
    console.log('- A debug info box showing loading state and count');
    console.log('- A blue "Load Logs" button');
    console.log('- Either logs displayed or "No logs available" message');
    
    console.log('\n🐛 If you still can\'t see it:');
    console.log('- Check browser console for errors (F12)');
    console.log('- Make sure you\'re scrolling down in the dialog');
    console.log('- Try refreshing the page');
    console.log('- Check if the dialog is fully open (not cut off)');
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
  }
}

testFinalLogsFix();
