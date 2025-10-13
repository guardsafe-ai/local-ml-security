const puppeteer = require('puppeteer');

async function debugTrainingData() {
  console.log('🔍 Debugging Training Data Issue...');
  
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
    
    console.log('🔍 Checking training jobs data...');
    
    // Check the actual training jobs data in the component
    const trainingData = await page.evaluate(() => {
      // Try to access the React component state
      const reactRoot = document.querySelector('#root')._reactInternalFiber;
      
      // Check if we can find the training jobs data
      const jobsList = document.querySelector('.MuiTableBody-root');
      const jobRows = jobsList ? jobsList.children : [];
      
      // Check for any data attributes or text content
      const tableContent = Array.from(jobRows).map(row => ({
        text: row.textContent,
        hasData: row.textContent.includes('train_') || row.textContent.includes('bert') || row.textContent.includes('distilbert')
      }));
      
      return {
        jobRowsCount: jobRows.length,
        tableContent,
        hasJobsList: !!jobsList,
        jobsListHTML: jobsList ? jobsList.innerHTML.substring(0, 500) : 'No jobs list found'
      };
    });
    
    console.log('📊 Training Data Debug:', JSON.stringify(trainingData, null, 2));
    
    // Check the API response directly
    console.log('🌐 Checking API response...');
    const apiResponse = await page.evaluate(async () => {
      try {
        const response = await fetch('http://localhost:8007/training/jobs');
        const data = await response.json();
        return {
          status: response.status,
          dataLength: data.jobs ? data.jobs.length : 0,
          firstJob: data.jobs ? data.jobs[0] : null,
          hasResults: data.jobs ? data.jobs.filter(job => job.result).length : 0
        };
      } catch (error) {
        return { error: error.message };
      }
    });
    
    console.log('🌐 API Response:', JSON.stringify(apiResponse, null, 2));
    
    // Check the Performance Analytics tab specifically
    console.log('🎯 Checking Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const analyticsData = await page.evaluate(() => {
      const tabPanel = document.querySelector('[role="tabpanel"]:not([hidden])');
      const charts = tabPanel ? tabPanel.querySelectorAll('.recharts-wrapper') : [];
      
      // Check if charts have data
      const chartData = Array.from(charts).map((chart, index) => {
        const svg = chart.querySelector('svg');
        const paths = svg ? svg.querySelectorAll('path, line, rect') : [];
        const text = svg ? svg.querySelectorAll('text') : [];
        
        return {
          index,
          hasSvg: !!svg,
          pathCount: paths.length,
          textCount: text.length,
          isEmpty: paths.length === 0 && text.length === 0
        };
      });
      
      return {
        tabPanelExists: !!tabPanel,
        chartCount: charts.length,
        chartData,
        tabPanelText: tabPanel ? tabPanel.textContent.substring(0, 200) : 'No tab panel'
      };
    });
    
    console.log('📈 Analytics Data:', JSON.stringify(analyticsData, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'training-data-debug.png', 
      fullPage: true 
    });
    
    console.log('✅ Debug completed! Check training-data-debug.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

debugTrainingData().catch(console.error);
