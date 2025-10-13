const puppeteer = require('puppeteer');

async function testFinalCharts() {
  console.log('🔍 Testing Final Charts...');
  
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
    
    // Check for the success message and charts
    const chartStatus = await page.evaluate(() => {
      const successBox = document.querySelector('[class*="success"]');
      const charts = document.querySelectorAll('.recharts-wrapper');
      const chartData = [];
      
      charts.forEach((chart, index) => {
        const svg = chart.querySelector('svg');
        if (svg) {
          const dataPoints = svg.querySelectorAll('circle, rect[class*="bar"], path[class*="line"]');
          const texts = svg.querySelectorAll('text');
          const hasData = dataPoints.length > 0;
          
          chartData.push({
            index,
            dataPointCount: dataPoints.length,
            textCount: texts.length,
            hasData,
            svgContent: svg.innerHTML.substring(0, 200)
          });
        }
      });
      
      return {
        hasSuccessBox: !!successBox,
        successBoxText: successBox ? successBox.textContent : 'No success box',
        chartCount: charts.length,
        chartData: chartData
      };
    });
    
    console.log('📊 Chart Status:', JSON.stringify(chartStatus, null, 2));
    
    // If charts are still empty, try to force a refresh
    if (chartStatus.chartData.every(chart => !chart.hasData)) {
      console.log('🔄 Charts still empty, trying to force refresh...');
      
      // Look for refresh button and click it
      const refreshButton = await page.$('button[class*="MuiButton"]');
      if (refreshButton) {
        const buttonText = await refreshButton.evaluate(el => el.textContent);
        console.log('🔄 Found button:', buttonText);
        
        if (buttonText.includes('🔄 Refresh Data')) {
          console.log('🔄 Clicking refresh button...');
          await refreshButton.click();
          await new Promise(resolve => setTimeout(resolve, 5000));
          
          // Check again after refresh
          const refreshStatus = await page.evaluate(() => {
            const charts = document.querySelectorAll('.recharts-wrapper');
            const chartData = [];
            
            charts.forEach((chart, index) => {
              const svg = chart.querySelector('svg');
              if (svg) {
                const dataPoints = svg.querySelectorAll('circle, rect[class*="bar"], path[class*="line"]');
                chartData.push({
                  index,
                  dataPointCount: dataPoints.length,
                  hasData: dataPoints.length > 0
                });
              }
            });
            
            return {
              chartCount: charts.length,
              chartData: chartData
            };
          });
          
          console.log('📊 After Refresh:', JSON.stringify(refreshStatus, null, 2));
        }
      }
    }
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'test-final-charts.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check test-final-charts.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testFinalCharts().catch(console.error);
