const puppeteer = require('puppeteer');

async function testAnalyticsFixed() {
  console.log('🔍 Testing Fixed Performance Analytics...');
  
  const browser = await puppeteer.launch({ 
    headless: false, 
    defaultViewport: null,
    args: ['--start-maximized']
  });
  
  try {
    const page = await browser.newPage();
    
    // Enable console logging
    page.on('console', msg => {
      if (msg.text().includes('chart') || msg.text().includes('data') || msg.text().includes('analytics')) {
        console.log(`📊 [${msg.type().toUpperCase()}] ${msg.text()}`);
      }
    });
    
    console.log('🌐 Navigating to training page...');
    await page.goto('http://localhost:3000/training', { 
      waitUntil: 'networkidle0',
      timeout: 30000 
    });
    
    // Wait for page to load
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    console.log('📈 Checking Performance Analytics charts...');
    
    // Check the chart data
    const chartData = await page.evaluate(() => {
      const charts = document.querySelectorAll('.recharts-wrapper');
      const results = [];
      
      charts.forEach((chart, index) => {
        const svg = chart.querySelector('svg');
        if (svg) {
          // Check for data elements
          const paths = svg.querySelectorAll('path');
          const lines = svg.querySelectorAll('line');
          const rects = svg.querySelectorAll('rect');
          const circles = svg.querySelectorAll('circle');
          const texts = svg.querySelectorAll('text');
          
          // Check for specific chart elements
          const dataElements = svg.querySelectorAll('[data-key], [class*="recharts"]');
          const hasData = paths.length > 0 || lines.length > 0 || rects.length > 0 || circles.length > 0;
          
          results.push({
            index,
            hasSvg: true,
            pathCount: paths.length,
            lineCount: lines.length,
            rectCount: rects.length,
            circleCount: circles.length,
            textCount: texts.length,
            dataElementCount: dataElements.length,
            hasData,
            isEmpty: !hasData
          });
        }
      });
      
      return results;
    });
    
    console.log('📊 Chart Data:', JSON.stringify(chartData, null, 2));
    
    // Check if charts have actual data points
    const dataPoints = await page.evaluate(() => {
      const charts = document.querySelectorAll('.recharts-wrapper');
      const results = [];
      
      charts.forEach((chart, index) => {
        const svg = chart.querySelector('svg');
        if (svg) {
          // Look for actual data visualization elements
          const dataPoints = svg.querySelectorAll('circle[class*="dot"], rect[class*="bar"], path[class*="line"]');
          const axisLabels = svg.querySelectorAll('text[class*="tick"]');
          const legendItems = svg.querySelectorAll('text[class*="legend"]');
          
          results.push({
            index,
            dataPointCount: dataPoints.length,
            axisLabelCount: axisLabels.length,
            legendItemCount: legendItems.length,
            hasData: dataPoints.length > 0,
            hasAxis: axisLabels.length > 0,
            hasLegend: legendItems.length > 0
          });
        }
      });
      
      return results;
    });
    
    console.log('📊 Data Points:', JSON.stringify(dataPoints, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'analytics-fixed-test.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check analytics-fixed-test.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testAnalyticsFixed().catch(console.error);
