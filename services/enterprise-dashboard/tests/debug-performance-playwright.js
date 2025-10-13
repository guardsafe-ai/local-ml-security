const puppeteer = require('puppeteer');

async function debugPerformancePlaywright() {
  console.log('🔍 Debugging Performance Analytics with Playwright...');
  
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
    
    // Check the raw training jobs data
    console.log('📊 Checking raw training jobs data...');
    const rawData = await page.evaluate(() => {
      // Access the React component state if possible
      const reactRoot = document.querySelector('#root');
      if (reactRoot && reactRoot._reactInternalFiber) {
        console.log('Found React root');
      }
      
      // Try to find any global state or data
      const scripts = Array.from(document.querySelectorAll('script'));
      const dataScripts = scripts.filter(script => 
        script.textContent && script.textContent.includes('trainingJobs')
      );
      
      return {
        hasReactRoot: !!reactRoot,
        dataScripts: dataScripts.length,
        windowKeys: Object.keys(window).filter(key => key.includes('training') || key.includes('job'))
      };
    });
    
    console.log('📊 Raw Data Check:', JSON.stringify(rawData, null, 2));
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Check the actual chart data being passed to Recharts
    console.log('📈 Analyzing chart data...');
    const chartAnalysis = await page.evaluate(() => {
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
          
          // Check for specific Recharts elements
          const rechartsElements = svg.querySelectorAll('[class*="recharts"]');
          const dataElements = svg.querySelectorAll('[data-key]');
          const axisElements = svg.querySelectorAll('[class*="axis"]');
          const legendElements = svg.querySelectorAll('[class*="legend"]');
          
          // Check for data points specifically
          const dataPoints = svg.querySelectorAll('circle[class*="dot"], rect[class*="bar"], path[class*="line"]');
          
          results.push({
            chartIndex: index,
            hasSvg: true,
            pathCount: paths.length,
            lineCount: lines.length,
            rectCount: rects.length,
            circleCount: circles.length,
            textCount: texts.length,
            rechartsElementCount: rechartsElements.length,
            dataElementCount: dataElements.length,
            axisElementCount: axisElements.length,
            legendElementCount: legendElements.length,
            dataPointCount: dataPoints.length,
            svgContent: svg.outerHTML.substring(0, 500) + '...'
          });
        }
      });
      
      return results;
    });
    
    console.log('📊 Chart Analysis:', JSON.stringify(chartAnalysis, null, 2));
    
    // Check the actual data being passed to the charts
    console.log('🔍 Checking chart data props...');
    const chartDataProps = await page.evaluate(() => {
      // Try to find the chart components and their data props
      const chartContainers = document.querySelectorAll('[class*="recharts"]');
      const results = [];
      
      chartContainers.forEach((container, index) => {
        // Look for data attributes or props
        const dataAttr = container.getAttribute('data-chart-data');
        const children = Array.from(container.children);
        
        results.push({
          index,
          hasDataAttr: !!dataAttr,
          dataAttrValue: dataAttr,
          childCount: children.length,
          childTypes: children.map(child => child.tagName),
          containerClass: container.className
        });
      });
      
      return results;
    });
    
    console.log('📊 Chart Data Props:', JSON.stringify(chartDataProps, null, 2));
    
    // Check if there are any JavaScript errors
    console.log('🚨 Checking for JavaScript errors...');
    const errors = await page.evaluate(() => {
      return window.console.errors || [];
    });
    
    console.log('📊 JavaScript Errors:', errors);
    
    // Take a detailed screenshot
    await page.screenshot({ 
      path: 'debug-performance-playwright.png', 
      fullPage: true 
    });
    
    console.log('✅ Debug completed! Check debug-performance-playwright.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

debugPerformancePlaywright().catch(console.error);
