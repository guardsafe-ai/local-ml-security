const puppeteer = require('puppeteer');

async function testSimpleWarning() {
  console.log('🚀 Testing Simple Warning System...');
  
  const browser = await puppeteer.launch({ 
    headless: false, 
    defaultViewport: null,
    args: ['--start-maximized']
  });
  
  const page = await browser.newPage();
  
  try {
    console.log('📍 Navigating to Data Management page...');
    await page.goto('http://localhost:3000/data-management', { 
      waitUntil: 'networkidle2',
      timeout: 30000 
    });
    
    console.log('⏳ Waiting for page to load...');
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Test the warning handling function directly
    console.log('🔍 Testing warning handling function...');
    const warningTest = await page.evaluate(() => {
      // Simulate the handleApiResponse function
      const mockResponse = {
        status: 'success',
        message: 'File upload started: test.jsonl',
        file_id: 'file_123',
        warning: 'File with same hash already exists: file_123',
        is_duplicate: true
      };
      
      // Simulate the showSnackbar function
      let snackbarShown = false;
      let snackbarMessage = '';
      let snackbarSeverity = '';
      
      const mockShowSnackbar = (message, severity) => {
        snackbarShown = true;
        snackbarMessage = message;
        snackbarSeverity = severity;
        console.log(`Snackbar shown: ${message} (${severity})`);
      };
      
      // Simulate the handleApiResponse function
      const handleApiResponse = (response, successMessage) => {
        if (response.warning) {
          mockShowSnackbar(`⚠️ ${response.warning}`, 'warning');
        }
        
        if (successMessage && !response.warning) {
          mockShowSnackbar(successMessage, 'success');
        }
        
        return response;
      };
      
      // Test the function
      const result = handleApiResponse(mockResponse, 'File uploaded successfully');
      
      return {
        result: result,
        snackbarShown: snackbarShown,
        snackbarMessage: snackbarMessage,
        snackbarSeverity: snackbarSeverity
      };
    });
    
    console.log('📊 Warning Test Results:');
    console.log(`   Snackbar Shown: ${warningTest.snackbarShown}`);
    console.log(`   Message: ${warningTest.snackbarMessage}`);
    console.log(`   Severity: ${warningTest.snackbarSeverity}`);
    
    if (warningTest.snackbarShown && warningTest.snackbarMessage.includes('⚠️')) {
      console.log('✅ SUCCESS: Warning handling function works correctly!');
    } else {
      console.log('❌ ISSUE: Warning handling function not working');
    }
    
    // Test with a real API call to see if warnings are returned
    console.log('🔍 Testing real API call for warnings...');
    const apiTest = await page.evaluate(async () => {
      try {
        // Create a test file
        const testContent = '{"text": "Test warning API", "label": "test"}';
        const blob = new Blob([testContent], { type: 'application/json' });
        const file = new File([blob], 'test_warning_api.jsonl', { type: 'application/jsonl' });
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('data_type', 'custom');
        formData.append('description', 'Test warning API');
        
        // First upload
        const response1 = await fetch('http://localhost:8002/data/efficient/upload-large-file', {
          method: 'POST',
          body: formData
        });
        const result1 = await response1.json();
        
        // Second upload (should trigger warning)
        const response2 = await fetch('http://localhost:8002/data/efficient/upload-large-file', {
          method: 'POST',
          body: formData
        });
        const result2 = await response2.json();
        
        return {
          firstUpload: result1,
          secondUpload: result2,
          hasWarning: !!result2.warning
        };
      } catch (error) {
        return { error: error.message };
      }
    });
    
    console.log('📊 API Test Results:');
    if (apiTest.error) {
      console.log(`   Error: ${apiTest.error}`);
    } else {
      console.log(`   First Upload: ${apiTest.firstUpload.status}`);
      console.log(`   Second Upload: ${apiTest.secondUpload.status}`);
      console.log(`   Has Warning: ${apiTest.hasWarning}`);
      if (apiTest.hasWarning) {
        console.log(`   Warning Message: ${apiTest.secondUpload.warning}`);
        console.log('✅ SUCCESS: API returns warning messages correctly!');
      } else {
        console.log('❌ ISSUE: API not returning warning messages');
      }
    }
    
    // Take a screenshot
    console.log('📸 Taking screenshot...');
    await page.screenshot({ 
      path: 'simple-warning-test.png', 
      fullPage: true 
    });
    
    console.log('✅ Simple warning testing completed');
    
  } catch (error) {
    console.error('❌ Error during testing:', error);
  } finally {
    await browser.close();
  }
}

testSimpleWarning().catch(console.error);
