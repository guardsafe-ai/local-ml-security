const puppeteer = require('puppeteer');

async function testSmartLoading() {
  console.log('🔍 Testing Smart Loading Solution...');
  
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
          text.includes('useEffect') || text.includes('executeJobs') || text.includes('Found jobs') ||
          text.includes('🎯') || text.includes('✅') || text.includes('🔄') || text.includes('Pre-loading')) {
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
    await new Promise(resolve => setTimeout(resolve, 10000));
    
    // Test hover pre-loading
    console.log('🖱️ Testing hover pre-loading...');
    const performanceTab = await page.$('button[role="tab"]:nth-child(3)');
    if (performanceTab) {
      await performanceTab.hover();
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Check for data ready indicator
    const dataReadyInfo = await page.evaluate(() => {
      const dataReadyChip = document.querySelector('[class*="MuiChip-root"]');
      const debugBoxes = document.querySelectorAll('[class*="warning"], [class*="success"]');
      const charts = document.querySelectorAll('.recharts-wrapper');
      
      return {
        hasDataReadyChip: !!dataReadyChip,
        dataReadyText: dataReadyChip?.textContent || 'No chip found',
        debugBoxCount: debugBoxes.length,
        debugText: Array.from(debugBoxes).map(box => box.textContent),
        chartCount: charts.length,
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
    
    console.log('📊 Data Ready Info:', JSON.stringify(dataReadyInfo, null, 2));
    
    // Test refresh button
    console.log('🔄 Testing refresh button...');
    const refreshButton = await page.$('button[class*="MuiButton"]');
    if (refreshButton) {
      const buttonText = await refreshButton.evaluate(el => el.textContent);
      console.log('🔄 Found button with text:', buttonText);
      if (buttonText.includes('🔄 Refresh Data')) {
        await refreshButton.click();
        await new Promise(resolve => setTimeout(resolve, 3000));
      }
    }
    
    // Check final state
    const finalState = await page.evaluate(() => {
      const debugBoxes = document.querySelectorAll('[class*="warning"], [class*="success"]');
      const dataReadyChip = document.querySelector('[class*="MuiChip-root"]');
      
      return {
        debugBoxCount: debugBoxes.length,
        debugText: Array.from(debugBoxes).map(box => box.textContent),
        hasDataReadyChip: !!dataReadyChip,
        dataReadyText: dataReadyChip?.textContent || 'No chip found'
      };
    });
    
    console.log('📊 Final State:', JSON.stringify(finalState, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'test-smart-loading.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check test-smart-loading.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testSmartLoading().catch(console.error);
