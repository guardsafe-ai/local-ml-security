const puppeteer = require('puppeteer');

async function testChartsWorking() {
  console.log('🔍 Testing Charts Working Solution...');
  
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
      if (text.includes('🔍') || text.includes('Direct Chart Data') || 
          text.includes('Performance Analytics Render') || text.includes('Training jobs data changed') ||
          text.includes('useEffect') || text.includes('executeJobs') || text.includes('Found jobs') ||
          text.includes('🎯') || text.includes('✅') || text.includes('🔄') || text.includes('Pre-loading') ||
          text.includes('API Response') || text.includes('Raw jobs') || text.includes('Filtering jobs') ||
          text.includes('Jobs data structure') || text.includes('Performance data ready')) {
        console.log(`📊 [${msg.type().toUpperCase()}] ${text}`);
      }
    });
    
    // Enable network monitoring
    page.on('response', response => {
      if (response.url().includes('/training/jobs')) {
        console.log(`🌐 API Response: ${response.status()} ${response.url()}`);
      }
    });
    
    console.log('🌐 Navigating to training page...');
    await page.goto('http://localhost:3000/training', { 
      waitUntil: 'networkidle0',
      timeout: 30000 
    });
    
    // Wait for initial data load
    console.log('⏳ Waiting for initial data load...');
    await new Promise(resolve => setTimeout(resolve, 15000));
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Check for debug boxes and data
    const debugInfo = await page.evaluate(() => {
      const debugBoxes = document.querySelectorAll('[class*="warning"], [class*="success"]');
      const dataReadyChip = document.querySelector('[class*="MuiChip-root"]');
      const charts = document.querySelectorAll('.recharts-wrapper');
      
      return {
        debugBoxCount: debugBoxes.length,
        debugText: Array.from(debugBoxes).map(box => box.textContent),
        hasDataReadyChip: !!dataReadyChip,
        dataReadyText: dataReadyChip?.textContent || 'No chip found',
        chartCount: charts.length,
        chartDataPoints: Array.from(charts).map((chart, index) => {
          const svg = chart.querySelector('svg');
          const dataPoints = svg ? svg.querySelectorAll('circle, rect[class*="bar"], path[class*="line"]') : [];
          const texts = svg ? svg.querySelectorAll('text') : [];
          return {
            index,
            dataPointCount: dataPoints.length,
            textCount: texts.length,
            hasData: dataPoints.length > 0,
            hasText: texts.length > 0
          };
        })
      };
    });
    
    console.log('📊 Debug Info:', JSON.stringify(debugInfo, null, 2));
    
    // Check if we have success box with data
    if (debugInfo.debugBoxCount > 0) {
      console.log('✅ Found debug boxes!');
      debugInfo.debugText.forEach((text, index) => {
        console.log(`📊 Debug Box ${index + 1}: ${text}`);
      });
    }
    
    // Check chart data points
    if (debugInfo.chartDataPoints.length > 0) {
      console.log('📊 Chart Analysis:');
      debugInfo.chartDataPoints.forEach((chart, index) => {
        console.log(`  Chart ${index + 1}: ${chart.dataPointCount} data points, ${chart.textCount} text elements, hasData: ${chart.hasData}`);
      });
    }
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'test-charts-working.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check test-charts-working.png');
    
    // If charts are still empty, try refreshing
    if (debugInfo.chartDataPoints.every(chart => !chart.hasData)) {
      console.log('🔄 Charts still empty, trying refresh...');
      
      // Look for refresh button
      const refreshButton = await page.$('button[class*="MuiButton"]');
      if (refreshButton) {
        const buttonText = await refreshButton.evaluate(el => el.textContent);
        console.log('🔄 Found button with text:', buttonText);
        if (buttonText.includes('🔄 Refresh Data')) {
          console.log('🔄 Clicking refresh button...');
          await refreshButton.click();
          await new Promise(resolve => setTimeout(resolve, 5000));
          
          // Check again after refresh
          const refreshDebugInfo = await page.evaluate(() => {
            const debugBoxes = document.querySelectorAll('[class*="warning"], [class*="success"]');
            const charts = document.querySelectorAll('.recharts-wrapper');
            
            return {
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
          
          console.log('📊 After Refresh:', JSON.stringify(refreshDebugInfo, null, 2));
          
          // Take another screenshot
          await page.screenshot({ 
            path: 'test-charts-after-refresh.png', 
            fullPage: true 
          });
        }
      }
    }
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testChartsWorking().catch(console.error);
