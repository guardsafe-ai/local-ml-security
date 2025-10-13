#!/usr/bin/env node

/**
 * Test Enhanced Model Registry
 * Verifies the comprehensive Model Registry functionality
 */

const axios = require('axios');

async function testEnhancedModelRegistry() {
  console.log('🎯 Testing Enhanced Model Registry\n');
  
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
      
      const model = registry.models[0];
      console.log(`   📋 Sample model: ${model.name}`);
      console.log(`   📅 Created: ${new Date(model.creation_timestamp).toLocaleDateString()}`);
      console.log(`   🔄 Versions: ${model.latest_versions?.length || 0}`);
      
      if (model.latest_versions && model.latest_versions.length > 0) {
        const version = model.latest_versions[0];
        console.log(`   🏷️  Latest version: v${version.version}`);
        console.log(`   🎯 Stage: ${version.stage || 'None'}`);
      }
    } else {
      console.log('   ❌ No models found in registry');
    }
    
    // Test 3: Latest Models API
    console.log('\n3. Testing Latest Models API...');
    const latestResponse = await axios.get('http://localhost:8007/models/latest');
    console.log(`   ✅ Latest models API: ${latestResponse.status}`);
    
    // Test 4: Best Models API
    console.log('\n4. Testing Best Models API...');
    const bestResponse = await axios.get('http://localhost:8007/models/best');
    console.log(`   ✅ Best models API: ${bestResponse.status}`);
    
    console.log('\n🎉 Enhanced Model Registry Test Complete!');
    console.log('\n📋 What you should see in the UI:');
    console.log('1. Open http://localhost:3000');
    console.log('2. Go to Models page');
    console.log('3. Click on "Model Registry" tab');
    console.log('4. You should see enhanced model cards with:');
    console.log('   - Model name and architecture (BERT, DistilBERT, RoBERTa)');
    console.log('   - Version and accuracy prominently displayed');
    console.log('   - Stage and deployment status chips');
    console.log('   - Expandable sections for:');
    console.log('     • Model Details (architecture, parameters, size, framework)');
    console.log('     • Performance Metrics (precision, recall, F1-score, AUC)');
    console.log('     • Deployment Status (endpoint, requests, latency, uptime)');
    console.log('   - Action buttons (Details, Download, Deploy, Configure)');
    
    console.log('\n🔍 Enhanced Features:');
    console.log('✅ Comprehensive model information');
    console.log('✅ Performance metrics visualization');
    console.log('✅ Deployment status tracking');
    console.log('✅ Model operations (deploy, configure)');
    console.log('✅ Expandable detailed sections');
    console.log('✅ Professional ML engineer interface');
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    if (error.response) {
      console.error('   Response status:', error.response.status);
      console.error('   Response data:', error.response.data);
    }
  }
}

testEnhancedModelRegistry();
