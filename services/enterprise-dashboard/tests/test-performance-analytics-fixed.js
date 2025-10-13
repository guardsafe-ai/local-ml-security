const puppeteer = require('puppeteer');

async function testPerformanceAnalyticsFixed() {
  console.log('🔍 Testing Performance Analytics - Fixed Version...');
  
  const browser = await puppeteer.launch({ 
    headless: false, 
    defaultViewport: null,
    args: ['--start-maximized']
  });
  
  try {
    const page = await browser.newPage();
    
    // Enable console logging
    page.on('console', msg => {
      const text = msg.text();
      if (text.includes('🔍') || text.includes('Performance Analytics') || text.includes('Chart Data') || text.includes('Render')) {
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
    
    // Check the chart data
    const chartAnalysis = await page.evaluate(() => {
      const charts = document.querySelectorAll('.recharts-wrapper');
      const results = [];
      
      charts.forEach((chart, index) => {
        const svg = chart.querySelector('svg');
        if (svg) {
          // Check for data points
          const dataPoints = svg.querySelectorAll('circle, rect[class*="bar"], path[class*="line"]');
          const axisLabels = svg.querySelectorAll('text[class*="tick"]');
          const legendItems = svg.querySelectorAll('[class*="legend-item"]');
          const lines = svg.querySelectorAll('line');
          const paths = svg.querySelectorAll('path');
          
          results.push({
            chartIndex: index,
            dataPointCount: dataPoints.length,
            axisLabelCount: axisLabels.length,
            legendItemCount: legendItems.length,
            lineCount: lines.length,
            pathCount: paths.length,
            hasData: dataPoints.length > 0,
            hasAxis: axisLabels.length > 0,
            hasLegend: legendItems.length > 0
          });
        }
      });
      
      return {
        totalCharts: charts.length,
        chartDetails: results,
        hasPerformanceAnalytics: document.querySelector('[role="tabpanel"]:not([hidden])')?.textContent?.includes('Performance Analytics') || false
      };
    });
    
    console.log('📊 Chart Analysis:', JSON.stringify(chartAnalysis, null, 2));
    
    // Check for any loading states or error messages
    const uiState = await page.evaluate(() => {
      const visiblePanel = document.querySelector('[role="tabpanel"]:not([hidden])');
      const loadingElements = visiblePanel?.querySelectorAll('.MuiCircularProgress-root');
      const errorMessages = visiblePanel?.querySelectorAll('[class*="error"], [class*="Error"]');
      const noDataMessages = visiblePanel?.querySelectorAll('h6, h4');
      
      return {
        hasLoading: loadingElements?.length > 0,
        hasErrors: errorMessages?.length > 0,
        noDataMessages: Array.from(noDataMessages || []).map(el => el.textContent),
        panelText: visiblePanel?.textContent?.substring(0, 500) || 'No panel found'
      };
    });
    
    console.log('📊 UI State:', JSON.stringify(uiState, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'test-performance-analytics-fixed.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check test-performance-analytics-fixed.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testPerformanceAnalyticsFixed().catch(console.error);