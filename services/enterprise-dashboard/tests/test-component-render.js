const puppeteer = require('puppeteer');

async function testComponentRender() {
  console.log('🔍 Testing Component Render...');
  
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
      if (text.includes('🔍') || text.includes('PerformanceAnalyticsCharts') || 
          text.includes('Chart Data Memoized') || text.includes('Force refresh') || 
          text.includes('Performance') || text.includes('Chart') || text.includes('Render') || 
          text.includes('Data') || text.includes('Jobs') || text.includes('useEffect') || 
          text.includes('Training') || text.includes('🔄')) {
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
    
    // Check for component render
    const componentInfo = await page.evaluate(() => {
      const visiblePanel = document.querySelector('[role="tabpanel"]:not([hidden])');
      const debugBoxes = document.querySelectorAll('[class*="warning"], [class*="success"]');
      const charts = document.querySelectorAll('.recharts-wrapper');
      
      return {
        panelText: visiblePanel?.textContent?.substring(0, 500) || 'No panel found',
        hasPerformanceAnalytics: visiblePanel?.textContent?.includes('Performance Analytics') || false,
        debugBoxCount: debugBoxes.length,
        debugText: Array.from(debugBoxes).map(box => box.textContent),
        chartCount: charts.length,
        hasCharts: charts.length > 0,
        hasSuccessBox: Array.from(debugBoxes).some(box => box.textContent.includes('✅ Data Loaded')),
        hasWarningBox: Array.from(debugBoxes).some(box => box.textContent.includes('Debug: Jobs='))
      };
    });
    
    console.log('📊 Component Info:', JSON.stringify(componentInfo, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'test-component-render.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check test-component-render.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testComponentRender().catch(console.error);
