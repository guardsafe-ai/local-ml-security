const puppeteer = require('puppeteer');

async function testDebugInfo() {
  console.log('🔍 Testing Debug Info...');
  
  const browser = await puppeteer.launch({ 
    headless: false, 
    defaultViewport: null,
    args: ['--start-maximized']
  });
  
  try {
    const page = await browser.newPage();
    
    // Enable console logging
    page.on('console', msg => {
      const text = msg.text();
      if (text.includes('🔍') || text.includes('Performance') || text.includes('Chart') || text.includes('Render') || text.includes('Data') || text.includes('Jobs')) {
        console.log(`📊 [${msg.type().toUpperCase()}] ${text}`);
      }
    });
    
    console.log('🌐 Navigating to training page...');
    await page.goto('http://localhost:3000/training', { 
      waitUntil: 'networkidle0',
      timeout: 30000 
    });
    
    // Wait for data to load
    console.log('⏳ Waiting for data to load...');
    await new Promise(resolve => setTimeout(resolve, 10000));
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Check the debug info
    const debugInfo = await page.evaluate(() => {
      const visiblePanel = document.querySelector('[role="tabpanel"]:not([hidden])');
      const debugBox = visiblePanel?.querySelector('[class*="warning"]');
      const debugText = debugBox?.textContent || 'No debug info found';
      
      return {
        debugText,
        hasDebugBox: !!debugBox,
        panelText: visiblePanel?.textContent?.substring(0, 500) || 'No panel found'
      };
    });
    
    console.log('📊 Debug Info:', JSON.stringify(debugInfo, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'test-debug-info.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check test-debug-info.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testDebugInfo().catch(console.error);
