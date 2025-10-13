const puppeteer = require('puppeteer');

async function testChartDataFix() {
  console.log('🔍 Testing Chart Data Fix...');
  
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
    
    // Wait for data to load
    console.log('⏳ Waiting for data to load...');
    await new Promise(resolve => setTimeout(resolve, 8000));
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Check for the debug output
    console.log('🔍 Looking for debug output...');
    const debugFound = await page.evaluate(() => {
      const performanceSection = document.querySelector('[role="tabpanel"]:not([hidden])');
      return {
        hasPerformanceAnalytics: performanceSection?.textContent?.includes('Performance Analytics') || false,
        hasDebugLog: performanceSection?.textContent?.includes('🔍 Chart Data Debug') || false,
        sectionText: performanceSection?.textContent?.substring(0, 500) || 'No section found'
      };
    });
    
    console.log('🔍 Debug Found:', JSON.stringify(debugFound, null, 2));
    
    // Check the chart data
    console.log('📈 Checking chart data...');
    const chartData = await page.evaluate(() => {
      const charts = document.querySelectorAll('.recharts-wrapper');
      const results = [];
      
      charts.forEach((chart, index) => {
        const svg = chart.querySelector('svg');
        if (svg) {
          // Check for data points
          const dataPoints = svg.querySelectorAll('circle, rect[class*="bar"], path[class*="line"]');
          const axisLabels = svg.querySelectorAll('text[class*="tick"]');
          const legendItems = svg.querySelectorAll('[class*="legend-item"]');
          
          results.push({
            chartIndex: index,
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
    
    console.log('📊 Chart Data:', JSON.stringify(chartData, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'test-chart-data-fix.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check test-chart-data-fix.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testChartDataFix().catch(console.error);
