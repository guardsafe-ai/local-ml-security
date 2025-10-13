const puppeteer = require('puppeteer');

async function testDataManagementUI() {
  console.log('🚀 Starting Data Management UI Testing...');
  
  const browser = await puppeteer.launch({ 
    headless: false, 
    defaultViewport: null,
    args: ['--start-maximized']
  });
  
  const page = await browser.newPage();
  
  // Enable console logging
  page.on('console', msg => {
    console.log(`📱 Browser Console [${msg.type()}]:`, msg.text());
  });
  
  // Enable network request logging
  page.on('request', request => {
    console.log(`🌐 Request: ${request.method()} ${request.url()}`);
  });
  
  // Enable response logging
  page.on('response', response => {
    if (!response.ok()) {
      console.log(`❌ Failed Response: ${response.status()} ${response.url()}`);
    }
  });
  
  try {
    console.log('📍 Navigating to Data Management page...');
    await page.goto('http://localhost:3000/data-management', { 
      waitUntil: 'networkidle2',
      timeout: 30000 
    });
    
    console.log('⏳ Waiting for page to load...');
    await page.waitForTimeout(5000);
    
    // Check for error messages
    console.log('🔍 Checking for error messages...');
    const errorElements = await page.$$('text=/Failed to load|Error|Failed/');
    if (errorElements.length > 0) {
      console.log(`❌ Found ${errorElements.length} error elements:`);
      for (let i = 0; i < errorElements.length; i++) {
        const text = await errorElements[i].evaluate(el => el.textContent);
        console.log(`   - ${text}`);
      }
    }
    
    // Check for loading indicators
    console.log('🔍 Checking for loading indicators...');
    const loadingElements = await page.$$('text=/Loading|loading/');
    if (loadingElements.length > 0) {
      console.log(`⏳ Found ${loadingElements.length} loading indicators`);
    }
    
    // Check for data management specific elements
    console.log('🔍 Checking for Data Management elements...');
    const dataElements = await page.$$('text=/Data Management|Fresh Data|Used Data|Upload|Statistics/');
    console.log(`📊 Found ${dataElements.length} data management elements`);
    
    // Check for file lists
    console.log('🔍 Checking for file lists...');
    const fileElements = await page.$$('[class*="file"], [class*="list"], [class*="item"]');
    console.log(`📁 Found ${fileElements.length} potential file elements`);
    
    // Check for buttons
    console.log('🔍 Checking for interactive elements...');
    const buttons = await page.$$('button, [role="button"], input[type="button"]');
    console.log(`🔘 Found ${buttons.length} buttons`);
    
    // Take a screenshot
    console.log('📸 Taking screenshot...');
    await page.screenshot({ 
      path: 'data-management-ui-test.png', 
      fullPage: true 
    });
    
    // Check page title
    const title = await page.title();
    console.log(`📄 Page title: ${title}`);
    
    // Check for any JavaScript errors
    console.log('🔍 Checking for JavaScript errors...');
    const jsErrors = await page.evaluate(() => {
      return window.console.errors || [];
    });
    
    if (jsErrors.length > 0) {
      console.log(`❌ Found ${jsErrors.length} JavaScript errors`);
    }
    
    console.log('✅ Data Management UI testing completed');
    
  } catch (error) {
    console.error('❌ Error during testing:', error);
  } finally {
    await browser.close();
  }
}

testDataManagementUI().catch(console.error);
