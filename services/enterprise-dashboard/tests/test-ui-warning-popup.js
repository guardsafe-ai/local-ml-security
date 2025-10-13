const puppeteer = require('puppeteer');
const fs = require('fs');

async function testUIWarningPopup() {
  console.log('🚀 Testing UI Warning Popup System...');
  
  const browser = await puppeteer.launch({ 
    headless: false, 
    defaultViewport: null,
    args: ['--start-maximized']
  });
  
  const page = await browser.newPage();
  
  // Track console messages
  page.on('console', msg => {
    if (msg.text().includes('warning') || msg.text().includes('Warning') || msg.text().includes('duplicate') || msg.text().includes('same hash')) {
      console.log(`📱 Browser Console [${msg.type()}]:`, msg.text());
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
    
    // Create a test file on disk
    console.log('📝 Creating test file on disk...');
    const testContent = '{"text": "Test UI warning popup system", "label": "test"}';
    fs.writeFileSync('test_ui_warning.jsonl', testContent);
    
    // Test 1: Upload file through UI for the first time
    console.log('🔍 Test 1: Uploading file through UI for the first time...');
    
    // Click upload button
    const uploadButtons = await page.$$('button');
    let uploadButton = null;
    for (let button of uploadButtons) {
      const text = await button.evaluate(el => el.textContent?.toLowerCase() || '');
      if (text.includes('upload')) {
        uploadButton = button;
        break;
      }
    }
    
    if (uploadButton) {
      await uploadButton.click();
      console.log('✅ Clicked upload button');
    } else {
      console.log('❌ Upload button not found');
    }
    
    // Wait for upload dialog
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Set file input
    const fileInput = await page.$('input[type="file"]');
    if (fileInput) {
      await fileInput.uploadFile('./test_ui_warning.jsonl');
      console.log('✅ File selected');
    } else {
      console.log('❌ File input not found');
    }
    
    // Fill in form fields
    await page.select('select[name="data_type"]', 'custom');
    await page.type('input[name="description"]', 'Test UI warning popup system');
    
    // Submit upload
    const submitButtons = await page.$$('button');
    let submitButton = null;
    for (let button of submitButtons) {
      const text = await button.evaluate(el => el.textContent?.toLowerCase() || '');
      if (text.includes('submit') || text.includes('upload') || text.includes('start')) {
        submitButton = button;
        break;
      }
    }
    
    if (submitButton) {
      await submitButton.click();
      console.log('✅ Upload submitted');
    } else {
      console.log('❌ Submit button not found');
    }
    
    // Wait for first upload to complete
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Test 2: Upload the same file again through UI
    console.log('🔍 Test 2: Uploading the same file again through UI (should trigger warning)...');
    
    // Click upload button again
    const uploadButtons2 = await page.$$('button');
    let uploadButton2 = null;
    for (let button of uploadButtons2) {
      const text = await button.evaluate(el => el.textContent?.toLowerCase() || '');
      if (text.includes('upload')) {
        uploadButton2 = button;
        break;
      }
    }
    
    if (uploadButton2) {
      await uploadButton2.click();
      console.log('✅ Clicked upload button for second upload');
    }
    
    // Wait for upload dialog
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Set file input again
    const fileInput2 = await page.$('input[type="file"]');
    if (fileInput2) {
      await fileInput2.uploadFile('./test_ui_warning.jsonl');
      console.log('✅ Same file selected again');
    }
    
    // Fill in form fields
    await page.select('select[name="data_type"]', 'custom');
    await page.type('input[name="description"]', 'Test UI warning popup system - duplicate');
    
    // Submit upload
    const submitButtons2 = await page.$$('button');
    let submitButton2 = null;
    for (let button of submitButtons2) {
      const text = await button.evaluate(el => el.textContent?.toLowerCase() || '');
      if (text.includes('submit') || text.includes('upload') || text.includes('start')) {
        submitButton2 = button;
        break;
      }
    }
    
    if (submitButton2) {
      await submitButton2.click();
      console.log('✅ Second upload submitted');
    }
    
    // Wait for warning popup to appear
    console.log('⏳ Waiting for warning popup to appear...');
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Check for warning popups
    console.log('🔍 Checking for warning popups...');
    const warningPopups = await page.evaluate(() => {
      // Look for Material-UI Snackbar components
      const snackbars = document.querySelectorAll('[role="alert"], .MuiSnackbar-root, .MuiAlert-root');
      const warnings = [];
      
      snackbars.forEach(snackbar => {
        const text = snackbar.textContent || '';
        if (text.includes('warning') || text.includes('Warning') || text.includes('duplicate') || text.includes('same hash') || text.includes('⚠️')) {
          warnings.push({
            text: text.trim(),
            visible: snackbar.offsetParent !== null,
            className: snackbar.className
          });
        }
      });
      
      // Also check for any elements with warning text
      const allElements = document.querySelectorAll('*');
      allElements.forEach(el => {
        const text = el.textContent || '';
        if (text.includes('File with same hash already exists') && text.length < 200) {
          warnings.push({
            text: text.trim(),
            visible: el.offsetParent !== null,
            className: el.className,
            tagName: el.tagName
          });
        }
      });
      
      return warnings;
    });
    
    console.log('📊 Warning Popups Found:', warningPopups);
    
    if (warningPopups.length > 0) {
      console.log('✅ SUCCESS: Warning popup detected!');
      warningPopups.forEach((popup, index) => {
        console.log(`   Popup ${index + 1}: "${popup.text}"`);
        console.log(`   Visible: ${popup.visible}, Class: ${popup.className}`);
      });
    } else {
      console.log('❌ ISSUE: No warning popup detected');
      
      // Debug: Check what snackbars are present
      const allSnackbars = await page.evaluate(() => {
        const snackbars = document.querySelectorAll('[role="alert"], .MuiSnackbar-root, .MuiAlert-root, .MuiSnackbarContent-root');
        return Array.from(snackbars).map(sb => ({
          text: sb.textContent?.trim(),
          visible: sb.offsetParent !== null,
          className: sb.className
        }));
      });
      
      console.log('📊 All Snackbars Found:', allSnackbars);
    }
    
    // Take a screenshot
    console.log('📸 Taking screenshot...');
    await page.screenshot({ 
      path: 'ui-warning-popup-test.png', 
      fullPage: true 
    });
    
    console.log('✅ UI warning popup testing completed');
    
  } catch (error) {
    console.error('❌ Error during testing:', error);
  } finally {
    // Clean up test file
    try {
      fs.unlinkSync('test_ui_warning.jsonl');
    } catch (e) {
      // File might not exist
    }
    await browser.close();
  }
}

testUIWarningPopup().catch(console.error);
