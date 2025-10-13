const puppeteer = require('puppeteer');

async function debugDataFlow() {
  console.log('🔍 Debugging Data Flow...');
  
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
      if (text.includes('🔍') || text.includes('Chart Data Memoized') || 
          text.includes('Performance Analytics Render') || text.includes('Training jobs data changed') ||
          text.includes('useEffect') || text.includes('executeJobs') || text.includes('Found jobs')) {
        console.log(`📊 [${msg.type().toUpperCase()}] ${text}`);
      }
    });
    
    console.log('🌐 Navigating to training page...');
    await page.goto('http://localhost:3000/training', { 
      waitUntil: 'networkidle0',
      timeout: 30000 
    });
    
    // Wait for initial data load
    console.log('⏳ Waiting for initial data load...');
    await new Promise(resolve => setTimeout(resolve, 8000));
    
    // Check data state before clicking tab
    console.log('📊 Checking data state before tab click...');
    const beforeTabClick = await page.evaluate(() => {
      // Check if we can access the React component state
      const reactRoot = document.querySelector('#root');
      return {
        hasReactRoot: !!reactRoot,
        pageTitle: document.title,
        currentUrl: window.location.href
      };
    });
    
    console.log('📊 Before Tab Click:', JSON.stringify(beforeTabClick, null, 2));
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Check data state after clicking tab
    console.log('📊 Checking data state after tab click...');
    const afterTabClick = await page.evaluate(() => {
      const visiblePanel = document.querySelector('[role="tabpanel"]:not([hidden])');
      const charts = document.querySelectorAll('.recharts-wrapper');
      const debugBoxes = document.querySelectorAll('[class*="warning"], [class*="success"]');
      
      return {
        panelText: visiblePanel?.textContent?.substring(0, 300) || 'No panel found',
        chartCount: charts.length,
        debugBoxCount: debugBoxes.length,
        debugText: Array.from(debugBoxes).map(box => box.textContent),
        hasCharts: charts.length > 0,
        chartDataPoints: Array.from(charts).map((chart, index) => {
          const svg = chart.querySelector('svg');
          const dataPoints = svg ? svg.querySelectorAll('circle, rect[class*="bar"], path[class*="line"]') : [];
          return {
            index,
            dataPointCount: dataPoints.length,
            hasData: dataPoints.length > 0
          };
        })
      };
    });
    
    console.log('📊 After Tab Click:', JSON.stringify(afterTabClick, null, 2));
    
    // Wait for potential re-render
    console.log('⏳ Waiting for potential re-render...');
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Check final state
    const finalState = await page.evaluate(() => {
      const visiblePanel = document.querySelector('[role="tabpanel"]:not([hidden])');
      const debugBoxes = document.querySelectorAll('[class*="warning"], [class*="success"]');
      
      return {
        debugBoxCount: debugBoxes.length,
        debugText: Array.from(debugBoxes).map(box => box.textContent),
        panelText: visiblePanel?.textContent?.substring(0, 300) || 'No panel found'
      };
    });
    
    console.log('📊 Final State:', JSON.stringify(finalState, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'debug-data-flow.png', 
      fullPage: true 
    });
    
    console.log('✅ Debug completed! Check debug-data-flow.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

debugDataFlow().catch(console.error);