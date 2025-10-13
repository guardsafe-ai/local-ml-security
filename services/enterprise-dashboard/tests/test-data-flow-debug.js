const puppeteer = require('puppeteer');

async function testDataFlowDebug() {
  console.log('🔍 Testing Complete Data Flow...');
  
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
      if (text.includes('🔍') || text.includes('🎯') || text.includes('📊') || 
          text.includes('Training jobs data changed') || text.includes('Chart data created') ||
          text.includes('PerformanceAnalyticsCharts rendered') || text.includes('useEffect triggered')) {
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
    await new Promise(resolve => setTimeout(resolve, 20000));
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Check the actual data structure
    const dataStructure = await page.evaluate(() => {
      // Try to access the React component state
      const reactRoot = document.querySelector('#root');
      if (reactRoot && reactRoot._reactInternalFiber) {
        return {
          hasReactRoot: true,
          reactRootType: typeof reactRoot._reactInternalFiber
        };
      }
      
      return {
        hasReactRoot: !!reactRoot,
        bodyText: document.body.textContent.substring(0, 1000)
      };
    });
    
    console.log('📊 Data Structure:', JSON.stringify(dataStructure, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'test-data-flow-debug.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check test-data-flow-debug.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testDataFlowDebug().catch(console.error);
