const puppeteer = require('puppeteer');

async function debugDataStructure() {
  console.log('🔍 Debugging Data Structure...');
  
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
    
    // Inject debugging code to check the actual data
    console.log('🔍 Injecting debugging code...');
    await page.evaluate(() => {
      // Override console.log to capture our debug output
      const originalLog = console.log;
      console.log = function(...args) {
        if (args[0] && args[0].includes && args[0].includes('DATA_DEBUG')) {
          originalLog(...args);
        }
        originalLog(...args);
      };
      
      // Try to access the React component state
      const reactRoot = document.querySelector('#root');
      if (reactRoot && reactRoot._reactInternalFiber) {
        console.log('DATA_DEBUG: Found React root');
      }
    });
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Check the actual data being passed to the charts
    const dataStructureDebug = await page.evaluate(() => {
      // Try to find the React component and access its state
      const reactRoot = document.querySelector('#root');
      let componentState = null;
      
      if (reactRoot && reactRoot._reactInternalFiber) {
        // Try to traverse the React fiber tree to find the Training component
        let current = reactRoot._reactInternalFiber;
        while (current) {
          if (current.type && current.type.name === 'Training') {
            componentState = current.memoizedProps;
            break;
          }
          current = current.child || current.sibling;
        }
      }
      
      // Check if we can access any global state
      const globalKeys = Object.keys(window).filter(key => 
        key.includes('training') || key.includes('job') || key.includes('Training')
      );
      
      return {
        hasReactRoot: !!reactRoot,
        componentState: componentState,
        globalKeys: globalKeys,
        windowKeys: Object.keys(window).slice(0, 20) // First 20 keys
      };
    });
    
    console.log('📊 Data Structure Debug:', JSON.stringify(dataStructureDebug, null, 2));
    
    // Check the actual chart data by looking at the DOM
    const chartDataDebug = await page.evaluate(() => {
      const charts = document.querySelectorAll('.recharts-wrapper');
      const results = [];
      
      charts.forEach((chart, index) => {
        const svg = chart.querySelector('svg');
        if (svg) {
          // Check for data elements
          const dataElements = svg.querySelectorAll('[data-key], circle, rect, path');
          const textElements = svg.querySelectorAll('text');
          const lineElements = svg.querySelectorAll('line');
          const pathElements = svg.querySelectorAll('path');
          
          // Check for specific Recharts elements
          const rechartsElements = svg.querySelectorAll('[class*="recharts"]');
          const axisElements = svg.querySelectorAll('[class*="axis"]');
          const legendElements = svg.querySelectorAll('[class*="legend"]');
          
          results.push({
            chartIndex: index,
            dataElements: dataElements.length,
            textElements: textElements.length,
            lineElements: lineElements.length,
            pathElements: pathElements.length,
            rechartsElements: rechartsElements.length,
            axisElements: axisElements.length,
            legendElements: legendElements.length,
            isEmpty: dataElements.length === 0 && textElements.length === 0
          });
        }
      });
      
      return results;
    });
    
    console.log('📊 Chart Data Debug:', JSON.stringify(chartDataDebug, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'debug-data-structure.png', 
      fullPage: true 
    });
    
    console.log('✅ Debug completed! Check debug-data-structure.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

debugDataStructure().catch(console.error);
