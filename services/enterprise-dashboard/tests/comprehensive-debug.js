const puppeteer = require('puppeteer');

async function comprehensiveDebug() {
  console.log('🔍 Comprehensive Performance Analytics Debug...');
  
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
      if (text.includes('🔍') || text.includes('Performance') || text.includes('Chart') || 
          text.includes('Render') || text.includes('Data') || text.includes('Jobs') ||
          text.includes('useEffect') || text.includes('Training')) {
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
    await new Promise(resolve => setTimeout(resolve, 8000));
    
    // Check initial state
    console.log('📊 Checking initial state...');
    const initialState = await page.evaluate(() => {
      const visiblePanel = document.querySelector('[role="tabpanel"]:not([hidden])');
      return {
        panelText: visiblePanel?.textContent?.substring(0, 200) || 'No panel found',
        hasPerformanceAnalytics: visiblePanel?.textContent?.includes('Performance Analytics') || false,
        chartCount: document.querySelectorAll('.recharts-wrapper').length
      };
    });
    
    console.log('📊 Initial State:', JSON.stringify(initialState, null, 2));
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Check state after tab click
    console.log('📊 Checking state after tab click...');
    const afterTabClick = await page.evaluate(() => {
      const visiblePanel = document.querySelector('[role="tabpanel"]:not([hidden])');
      const charts = document.querySelectorAll('.recharts-wrapper');
      const debugBoxes = document.querySelectorAll('[class*="warning"], [class*="success"]');
      
      return {
        panelText: visiblePanel?.textContent?.substring(0, 300) || 'No panel found',
        hasPerformanceAnalytics: visiblePanel?.textContent?.includes('Performance Analytics') || false,
        chartCount: charts.length,
        debugBoxCount: debugBoxes.length,
        debugText: Array.from(debugBoxes).map(box => box.textContent),
        hasCharts: charts.length > 0
      };
    });
    
    console.log('📊 After Tab Click:', JSON.stringify(afterTabClick, null, 2));
    
    // Wait for potential data loading
    console.log('⏳ Waiting for potential data loading...');
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Check final state
    console.log('📊 Checking final state...');
    const finalState = await page.evaluate(() => {
      const visiblePanel = document.querySelector('[role="tabpanel"]:not([hidden])');
      const charts = document.querySelectorAll('.recharts-wrapper');
      const debugBoxes = document.querySelectorAll('[class*="warning"], [class*="success"]');
      
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
        panelText: visiblePanel?.textContent?.substring(0, 300) || 'No panel found',
        hasPerformanceAnalytics: visiblePanel?.textContent?.includes('Performance Analytics') || false,
        chartCount: charts.length,
        chartData,
        debugBoxCount: debugBoxes.length,
        debugText: Array.from(debugBoxes).map(box => box.textContent),
        hasLoading: visiblePanel?.querySelectorAll('.MuiCircularProgress-root').length > 0,
        hasNoDataMessage: visiblePanel?.textContent?.includes('No Training Data') || 
                         visiblePanel?.textContent?.includes('No Completed Training Jobs')
      };
    });
    
    console.log('📊 Final State:', JSON.stringify(finalState, null, 2));
    
    // Try to force a re-render by clicking the tab again
    console.log('🔄 Trying to force re-render...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Check state after forced re-render
    const afterRerender = await page.evaluate(() => {
      const visiblePanel = document.querySelector('[role="tabpanel"]:not([hidden])');
      const debugBoxes = document.querySelectorAll('[class*="warning"], [class*="success"]');
      return {
        debugBoxCount: debugBoxes.length,
        debugText: Array.from(debugBoxes).map(box => box.textContent),
        panelText: visiblePanel?.textContent?.substring(0, 300) || 'No panel found'
      };
    });
    
    console.log('📊 After Re-render:', JSON.stringify(afterRerender, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'comprehensive-debug.png', 
      fullPage: true 
    });
    
    console.log('✅ Comprehensive debug completed! Check comprehensive-debug.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

comprehensiveDebug().catch(console.error);
