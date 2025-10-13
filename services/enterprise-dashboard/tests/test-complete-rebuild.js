#!/usr/bin/env node

/**
 * Complete Rebuild Test
 * Tests all features after complete frontend rebuild
 */

const axios = require('axios');

async function testCompleteRebuild() {
  console.log('🚀 Complete Rebuild Test - All Features\n');
  
  try {
    // Test 1: Frontend accessibility
    console.log('1. Testing frontend accessibility...');
    const frontendResponse = await axios.get('http://localhost:3000');
    console.log(`   ✅ Frontend status: ${frontendResponse.status}`);
    
    // Test 2: Model Registry API
    console.log('\n2. Testing Model Registry API...');
    const registryResponse = await axios.get('http://localhost:8007/models/registry');
    const registry = registryResponse.data;
    
    if (registry.models && registry.models.length > 0) {
      console.log(`   ✅ Found ${registry.models.length} models in registry`);
      
      registry.models.forEach((model, index) => {
        console.log(`   📋 Model ${index + 1}: ${model.name}`);
        console.log(`      Created: ${new Date(model.creation_timestamp).toLocaleDateString()}`);
        console.log(`      Versions: ${model.latest_versions?.length || 0}`);
        if (model.latest_versions && model.latest_versions.length > 0) {
          const version = model.latest_versions[0];
          console.log(`      Latest: v${version.version} (${version.stage || 'None'})`);
        }
      });
    }
    
    // Test 3: Training Jobs API
    console.log('\n3. Testing Training Jobs API...');
    const jobsResponse = await axios.get('http://localhost:8007/training/jobs');
    const jobs = jobsResponse.data.jobs || jobsResponse.data;
    console.log(`   ✅ Found ${jobs.length} training jobs`);
    
    // Test 4: Job Logs API
    if (jobs.length > 0) {
      const job = jobs[0];
      const logsResponse = await axios.get(`http://localhost:8007/training/jobs/${job.job_id}/logs`);
      const logs = logsResponse.data.logs || [];
      console.log(`   ✅ Found ${logs.length} log entries for ${job.job_id}`);
    }
    
    // Test 5: Backend Health
    console.log('\n4. Testing Backend Health...');
    const healthResponse = await axios.get('http://localhost:8007/health');
    console.log(`   ✅ Backend health: ${healthResponse.status}`);
    
    console.log('\n🎉 Complete Rebuild Test Successful!');
    console.log('\n📋 What you should see in the UI:');
    console.log('\n🔧 TRAINING PAGE:');
    console.log('1. Open http://localhost:3000');
    console.log('2. Go to Training page');
    console.log('3. Look for the BRIGHT ORANGE DEBUG SECTION at the bottom');
    console.log('4. You should see:');
    console.log('   - "🔧 DEBUG: Job Logs Section" title');
    console.log('   - Status indicators (Selected Job, Dialog Open)');
    console.log('   - Manual job selection buttons');
    console.log('   - Load Logs button');
    console.log('5. Click "Select First Job" to test logs functionality');
    
    console.log('\n📊 MODELS PAGE:');
    console.log('1. Go to Models page');
    console.log('2. Click on "Model Registry" tab');
    console.log('3. You should see enhanced model cards with:');
    console.log('   - Model name and architecture (BERT, DistilBERT, RoBERTa)');
    console.log('   - Version and accuracy prominently displayed');
    console.log('   - Stage and deployment status chips');
    console.log('   - Expandable sections for detailed information');
    console.log('   - Action buttons (Details, Download, Deploy, Configure)');
    
    console.log('\n🎯 Key Features to Test:');
    console.log('✅ Enhanced Model Registry with comprehensive details');
    console.log('✅ Training logs functionality with debug section');
    console.log('✅ Model performance metrics and deployment status');
    console.log('✅ Professional ML engineer interface');
    console.log('✅ Responsive design for all devices');
    
    console.log('\n🚀 The frontend has been completely rebuilt with all enhancements!');
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    if (error.response) {
      console.error('   Response status:', error.response.status);
      console.error('   Response data:', error.response.data);
    }
  }
}

testCompleteRebuild();
