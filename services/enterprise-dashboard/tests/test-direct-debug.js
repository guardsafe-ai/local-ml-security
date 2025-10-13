const puppeteer = require('puppeteer');

async function testDirectDebug() {
  console.log('🔍 Testing Direct Debug...');
  
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
      if (text.includes('🔍') || text.includes('Performance Analytics') || text.includes('Data Debug')) {
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
    
    // Check if the debug output appeared
    const debugOutput = await page.evaluate(() => {
      // Check if there are any console logs in the page
      const performanceSection = document.querySelector('[role="tabpanel"]:not([hidden])');
      const hasPerformanceAnalytics = performanceSection?.textContent?.includes('Performance Analytics') || false;
      
      // Check the actual data being passed to charts
      const charts = document.querySelectorAll('.recharts-wrapper');
      const chartData = [];
      
      charts.forEach((chart, index) => {
        const svg = chart.querySelector('svg');
        if (svg) {
          const dataElements = svg.querySelectorAll('[data-key], circle, rect, path');
          const textElements = svg.querySelectorAll('text');
          const lineElements = svg.querySelectorAll('line');
          
          chartData.push({
            index,
            dataElements: dataElements.length,
            textElements: textElements.length,
            lineElements: lineElements.length,
            hasData: dataElements.length > 0
          });
        }
      });
      
      return {
        hasPerformanceAnalytics,
        chartCount: charts.length,
        chartData,
        sectionText: performanceSection?.textContent?.substring(0, 300) || 'No section found'
      };
    });
    
    console.log('📊 Debug Output:', JSON.stringify(debugOutput, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'test-direct-debug.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check test-direct-debug.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testDirectDebug().catch(console.error);
