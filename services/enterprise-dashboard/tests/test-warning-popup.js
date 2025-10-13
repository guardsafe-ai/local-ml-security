const puppeteer = require('puppeteer');

async function testWarningPopup() {
  console.log('🚀 Testing Warning Popup System...');
  
  const browser = await puppeteer.launch({ 
    headless: false, 
    defaultViewport: null,
    args: ['--start-maximized']
  });
  
  const page = await browser.newPage();
  
  // Track console messages and network requests
  page.on('console', msg => {
    if (msg.text().includes('warning') || msg.text().includes('Warning') || msg.text().includes('duplicate')) {
      console.log(`📱 Browser Console [${msg.type()}]:`, msg.text());
    }
  });
  
  // Track network requests
  page.on('response', response => {
    if (response.url().includes('upload-large-file')) {
      console.log(`🌐 Upload API Response: ${response.status()} ${response.url()}`);
    }
  });
  
  try {
    console.log('📍 Navigating to Data Management page...');
    await page.goto('http://localhost:3000/data-management', { 
      waitUntil: 'networkidle2',
      timeout: 30000 
    });
    
    console.log('⏳ Waiting for page to load...');
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Create a test file
    console.log('📝 Creating test file...');
    await page.evaluate(() => {
      const testContent = '{"text": "Test warning popup system", "label": "test"}';
      const blob = new Blob([testContent], { type: 'application/json' });
      const file = new File([blob], 'test_warning_popup.jsonl', { type: 'application/jsonl' });
      
      // Store file globally for later use
      window.testFile = file;
    });
    
    // Test 1: Upload file for the first time
    console.log('🔍 Test 1: Uploading file for the first time...');
    await page.evaluate(async () => {
      const file = window.testFile;
      const formData = new FormData();
      formData.append('file', file);
      formData.append('data_type', 'custom');
      formData.append('description', 'Test warning popup system');
      
      try {
        const response = await fetch('http://localhost:8002/data/efficient/upload-large-file', {
          method: 'POST',
          body: formData
        });
        const result = await response.json();
        console.log('First upload result:', result);
        window.firstUploadResult = result;
      } catch (error) {
        console.error('First upload error:', error);
      }
    });
    
    // Wait a moment
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Test 2: Upload the same file again (should trigger warning)
    console.log('🔍 Test 2: Uploading the same file again (should trigger warning)...');
    await page.evaluate(async () => {
      const file = window.testFile;
      const formData = new FormData();
      formData.append('file', file);
      formData.append('data_type', 'custom');
      formData.append('description', 'Test warning popup system - duplicate');
      
      try {
        const response = await fetch('http://localhost:8002/data/efficient/upload-large-file', {
          method: 'POST',
          body: formData
        });
        const result = await response.json();
        console.log('Second upload result:', result);
        window.secondUploadResult = result;
      } catch (error) {
        console.error('Second upload error:', error);
      }
    });
    
    // Wait for any popups to appear
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Check for warning popups in the UI
    console.log('🔍 Checking for warning popups in UI...');
    const warningPopups = await page.evaluate(() => {
      // Look for Material-UI Snackbar components with warning messages
      const snackbars = document.querySelectorAll('[role="alert"]');
      const warnings = [];
      
      snackbars.forEach(snackbar => {
        const text = snackbar.textContent || '';
        if (text.includes('warning') || text.includes('Warning') || text.includes('duplicate') || text.includes('same hash')) {
          warnings.push({
            text: text,
            visible: snackbar.offsetParent !== null
          });
        }
      });
      
      return warnings;
    });
    
    console.log('📊 Warning Popups Found:', warningPopups);
    
    // Check the upload results
    const uploadResults = await page.evaluate(() => {
      return {
        firstUpload: window.firstUploadResult,
        secondUpload: window.secondUploadResult
      };
    });
    
    console.log('📊 Upload Results:');
    console.log('   First Upload:', uploadResults.firstUpload);
    console.log('   Second Upload:', uploadResults.secondUpload);
    
    // Verify warning was included in second upload
    if (uploadResults.secondUpload && uploadResults.secondUpload.warning) {
      console.log('✅ SUCCESS: Warning message included in API response');
      console.log(`   Warning: ${uploadResults.secondUpload.warning}`);
    } else {
      console.log('❌ ISSUE: No warning message in second upload response');
    }
    
    // Check if warning popup appeared
    if (warningPopups.length > 0) {
      console.log('✅ SUCCESS: Warning popup detected in UI');
      warningPopups.forEach((popup, index) => {
        console.log(`   Popup ${index + 1}: ${popup.text} (visible: ${popup.visible})`);
      });
    } else {
      console.log('❌ ISSUE: No warning popup detected in UI');
    }
    
    // Take a screenshot
    console.log('📸 Taking screenshot...');
    await page.screenshot({ 
      path: 'warning-popup-test.png', 
      fullPage: true 
    });
    
    console.log('✅ Warning popup testing completed');
    
  } catch (error) {
    console.error('❌ Error during testing:', error);
  } finally {
    await browser.close();
  }
}

testWarningPopup().catch(console.error);
