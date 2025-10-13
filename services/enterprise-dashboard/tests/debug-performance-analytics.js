const puppeteer = require('puppeteer');

async function debugPerformanceAnalytics() {
  console.log('🔍 Starting Performance Analytics Debug Test...');
  
  const browser = await puppeteer.launch({ 
    headless: false, 
    defaultViewport: null,
    args: ['--start-maximized']
  });
  
  try {
    const page = await browser.newPage();
    
    // Enable console logging
    page.on('console', msg => {
      if (msg.type() === 'error') {
        console.log('❌ Console Error:', msg.text());
      } else if (msg.text().includes('training') || msg.text().includes('chart') || msg.text().includes('data')) {
        console.log('📊 Console:', msg.text());
      }
    });
    
    console.log('🌐 Navigating to training page...');
    await page.goto('http://localhost:3000/training', { 
      waitUntil: 'networkidle0',
      timeout: 30000 
    });
    
    // Wait for page to load
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    console.log('📋 Checking if training jobs are loaded...');
    
    // Check if training jobs are loaded
    const jobsLoaded = await page.evaluate(() => {
      // Check if the training jobs data is available
      const jobsElement = document.querySelector('[data-testid="training-jobs-list"]') || 
                         document.querySelector('.MuiTableBody-root');
      return jobsElement ? jobsElement.children.length : 0;
    });
    
    console.log(`📊 Found ${jobsLoaded} training jobs in the list`);
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)'); // Third tab (Performance Analytics)
    
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    console.log('📈 Checking Performance Analytics content...');
    
    // Check what's in the Performance Analytics tab
    const analyticsContent = await page.evaluate(() => {
      const tabPanel = document.querySelector('[role="tabpanel"]:not([hidden])');
      if (!tabPanel) return 'No tab panel found';
      
      const charts = tabPanel.querySelectorAll('.recharts-wrapper');
      const cards = tabPanel.querySelectorAll('.MuiCard-root');
      
      return {
        tabPanelText: tabPanel.textContent,
        chartCount: charts.length,
        cardCount: cards.length,
        hasData: tabPanel.textContent.includes('No data') || tabPanel.textContent.includes('empty'),
        chartElements: Array.from(charts).map(chart => ({
          width: chart.style.width,
          height: chart.style.height,
          visible: chart.offsetWidth > 0 && chart.offsetHeight > 0
        }))
      };
    });
    
    console.log('📊 Analytics Content:', JSON.stringify(analyticsContent, null, 2));
    
    // Check if charts are rendering
    console.log('🔍 Checking chart rendering...');
    const chartInfo = await page.evaluate(() => {
      const charts = document.querySelectorAll('.recharts-wrapper');
      const results = [];
      
      charts.forEach((chart, index) => {
        const rect = chart.getBoundingClientRect();
        const svg = chart.querySelector('svg');
        const paths = chart.querySelectorAll('path, line, rect');
        
        results.push({
          index,
          visible: rect.width > 0 && rect.height > 0,
          dimensions: { width: rect.width, height: rect.height },
          hasSvg: !!svg,
          pathCount: paths.length,
          innerHTML: chart.innerHTML.substring(0, 200) + '...'
        });
      });
      
      return results;
    });
    
    console.log('📈 Chart Details:', JSON.stringify(chartInfo, null, 2));
    
    // Check the actual data being passed to charts
    console.log('🔍 Checking chart data...');
    const chartData = await page.evaluate(() => {
      // Try to find the chart data in the React component
      const reactFiber = document.querySelector('.recharts-wrapper')?._reactInternalFiber;
      
      // Check if we can access the data from the DOM
      const chartContainers = document.querySelectorAll('.recharts-wrapper');
      const dataInfo = [];
      
      chartContainers.forEach((container, index) => {
        const svg = container.querySelector('svg');
        if (svg) {
          const textElements = svg.querySelectorAll('text');
          const pathElements = svg.querySelectorAll('path');
          const lineElements = svg.querySelectorAll('line');
          
          dataInfo.push({
            index,
            textCount: textElements.length,
            pathCount: pathElements.length,
            lineCount: lineElements.length,
            hasAxis: textElements.length > 0,
            hasData: pathElements.length > 0 || lineElements.length > 0
          });
        }
      });
      
      return dataInfo;
    });
    
    console.log('📊 Chart Data Info:', JSON.stringify(chartData, null, 2));
    
    // Take a screenshot for visual debugging
    console.log('📸 Taking screenshot...');
    await page.screenshot({ 
      path: 'performance-analytics-debug.png', 
      fullPage: true 
    });
    
    console.log('✅ Debug test completed! Check performance-analytics-debug.png for visual reference.');
    
  } catch (error) {
    console.error('❌ Error during debug test:', error);
  } finally {
    await browser.close();
  }
}

debugPerformanceAnalytics().catch(console.error);
