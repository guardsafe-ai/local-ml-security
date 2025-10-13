const puppeteer = require('puppeteer');

async function finalPerformanceTest() {
  console.log('🔍 Final Performance Analytics Test...');
  
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
      if (text.includes('🔍') || text.includes('Performance') || text.includes('Chart') || text.includes('Render') || text.includes('Data') || text.includes('Jobs')) {
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
    
    // Check the final state
    const finalState = await page.evaluate(() => {
      const visiblePanel = document.querySelector('[role="tabpanel"]:not([hidden])');
      const charts = document.querySelectorAll('.recharts-wrapper');
      
      // Check chart data
      const chartData = [];
      charts.forEach((chart, index) => {
        const svg = chart.querySelector('svg');
        if (svg) {
          const dataPoints = svg.querySelectorAll('circle, rect[class*="bar"], path[class*="line"]');
          const axisLabels = svg.querySelectorAll('text[class*="tick"]');
          const legendItems = svg.querySelectorAll('[class*="legend-item"]');
          const lines = svg.querySelectorAll('line');
          const paths = svg.querySelectorAll('path');
          
          chartData.push({
            index,
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
        hasPerformanceAnalytics: visiblePanel?.textContent?.includes('Performance Analytics') || false,
        chartCount: charts.length,
        chartData,
        panelText: visiblePanel?.textContent?.substring(0, 300) || 'No panel found',
        hasLoading: visiblePanel?.querySelectorAll('.MuiCircularProgress-root').length > 0,
        hasNoDataMessage: visiblePanel?.textContent?.includes('No Training Data') || visiblePanel?.textContent?.includes('No Completed Training Jobs')
      };
    });
    
    console.log('📊 Final State:', JSON.stringify(finalState, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'final-performance-test.png', 
      fullPage: true 
    });
    
    console.log('✅ Final test completed! Check final-performance-test.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

finalPerformanceTest().catch(console.error);