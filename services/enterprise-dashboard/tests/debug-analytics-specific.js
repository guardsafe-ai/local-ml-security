const puppeteer = require('puppeteer');

async function debugAnalyticsSpecific() {
  console.log('🔍 Debugging Performance Analytics Specifically...');
  
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
    
    // Wait for page to load
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    console.log('📈 Checking Performance Analytics charts...');
    
    // Check the chart data specifically
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
          
          results.push({
            index,
            hasSvg: true,
            pathCount: paths.length,
            lineCount: lines.length,
            rectCount: rects.length,
            circleCount: circles.length,
            textCount: texts.length,
            dataElementCount: dataElements.length,
            hasData: paths.length > 0 || lines.length > 0 || rects.length > 0,
            svgContent: svg.innerHTML.substring(0, 500)
          });
        }
      });
      
      return results;
    });
    
    console.log('📊 Chart Data:', JSON.stringify(chartData, null, 2));
    
    // Check if the data is being passed to the charts
    const dataCheck = await page.evaluate(() => {
      // Try to find the React component and check its props
      const chartContainers = document.querySelectorAll('.recharts-wrapper');
      const dataInfo = [];
      
      chartContainers.forEach((container, index) => {
        const svg = container.querySelector('svg');
        if (svg) {
          // Look for data-related attributes or classes
          const dataElements = svg.querySelectorAll('[data-key], [class*="recharts"]');
          const hasAxis = svg.querySelectorAll('g[class*="xAxis"], g[class*="yAxis"]').length > 0;
          const hasGrid = svg.querySelectorAll('g[class*="grid"]').length > 0;
          const hasLegend = svg.querySelectorAll('g[class*="legend"]').length > 0;
          
          dataInfo.push({
            index,
            dataElementCount: dataElements.length,
            hasAxis,
            hasGrid,
            hasLegend,
            isEmpty: dataElements.length === 0 && !hasAxis
          });
        }
      });
      
      return dataInfo;
    });
    
    console.log('📊 Data Check:', JSON.stringify(dataCheck, null, 2));
    
    // Check the actual chart data being used
    const actualData = await page.evaluate(() => {
      // Try to access the chart data from the DOM
      const chartContainers = document.querySelectorAll('.recharts-wrapper');
      const chartData = [];
      
      chartContainers.forEach((container, index) => {
        const svg = container.querySelector('svg');
        if (svg) {
          // Look for data points in the SVG
          const dataPoints = svg.querySelectorAll('circle, rect[data-key]');
          const lines = svg.querySelectorAll('path[class*="recharts-line"]');
          const bars = svg.querySelectorAll('rect[class*="recharts-bar"]');
          
          chartData.push({
            index,
            dataPointCount: dataPoints.length,
            lineCount: lines.length,
            barCount: bars.length,
            hasData: dataPoints.length > 0 || lines.length > 0 || bars.length > 0
          });
        }
      });
      
      return chartData;
    });
    
    console.log('📊 Actual Data:', JSON.stringify(actualData, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'analytics-specific-debug.png', 
      fullPage: true 
    });
    
    console.log('✅ Debug completed! Check analytics-specific-debug.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

debugAnalyticsSpecific().catch(console.error);
