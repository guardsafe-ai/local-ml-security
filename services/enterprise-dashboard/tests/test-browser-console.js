const puppeteer = require('puppeteer');

async function testBrowserConsole() {
  console.log('🔍 Testing Browser Console...');
  
  const browser = await puppeteer.launch({ 
    headless: false, 
    defaultViewport: null,
    args: ['--start-maximized']
  });
  
  try {
    const page = await browser.newPage();
    
    // Enable all console logging
    page.on('console', msg => {
      const text = msg.text();
      console.log(`📊 [${msg.type().toUpperCase()}] ${text}`);
    });
    
    // Enable error logging
    page.on('pageerror', error => {
      console.log(`❌ [PAGE ERROR] ${error.message}`);
    });
    
    console.log('🌐 Navigating to training page...');
    await page.goto('http://localhost:3000/training', { 
      waitUntil: 'networkidle0',
      timeout: 30000 
    });
    
    // Wait for data to load
    console.log('⏳ Waiting for data to load...');
    await new Promise(resolve => setTimeout(resolve, 20000));
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Check for any error messages in the console
    const consoleErrors = await page.evaluate(() => {
      return {
        hasErrors: window.console.error.toString().includes('error'),
        hasWarnings: window.console.warn.toString().includes('warn')
      };
    });
    
    console.log('📊 Console Errors:', JSON.stringify(consoleErrors, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'test-browser-console.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check test-browser-console.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testBrowserConsole().catch(console.error);
