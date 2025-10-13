const puppeteer = require('puppeteer');

async function testChartsFinal() {
  console.log('🎯 Testing Charts Final Solution...');
  
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
      if (text.includes('🎯') || text.includes('PerformanceAnalyticsCharts') || 
          text.includes('Chart data created') || text.includes('actualChartData') ||
          text.includes('actualBarChartData') || text.includes('Data Loaded')) {
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
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'test-charts-final.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check test-charts-final.png');
    
    // If charts are still empty, show the user what we found
    if (chartStatus.chartData.every(chart => !chart.hasData)) {
      console.log('❌ Charts are still empty. The issue is that the component is not being rendered correctly.');
      console.log('🔍 This suggests there is a compilation error preventing the component from being included in the bundle.');
    } else {
      console.log('✅ Charts are working! Data points found in charts.');
    }
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testChartsFinal().catch(console.error);
