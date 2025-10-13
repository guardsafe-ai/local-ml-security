const puppeteer = require('puppeteer');

async function testDataDebug() {
  console.log('🔍 Testing Data Debug...');
  
  const browser = await puppeteer.launch({ 
    headless: false, 
    defaultViewport: null,
    args: ['--start-maximized']
  });
  
  try {
    const page = await browser.newPage();
    
    // Enable console logging
    page.on('console', msg => {
      console.log(`📊 [${msg.type().toUpperCase()}] ${msg.text()}`);
    });
    
    console.log('🌐 Navigating to training page...');
    await page.goto('http://localhost:3000/training', { 
      waitUntil: 'networkidle0',
      timeout: 30000 
    });
    
    // Wait for data to load
    console.log('⏳ Waiting for data to load...');
    await new Promise(resolve => setTimeout(resolve, 8000));
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Wait a bit more for the debug output
    console.log('⏳ Waiting for debug output...');
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'test-data-debug.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check test-data-debug.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testDataDebug().catch(console.error);
