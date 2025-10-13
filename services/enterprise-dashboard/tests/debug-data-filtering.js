const puppeteer = require('puppeteer');

async function debugDataFiltering() {
  console.log('🔍 Debugging Data Filtering...');
  
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
    
    // Inject debugging code to check the actual data structure
    console.log('🔍 Injecting data debugging code...');
    await page.evaluate(() => {
      // Override console.log to capture our debug output
      const originalLog = console.log;
      console.log = function(...args) {
        if (args[0] && args[0].includes && args[0].includes('DATA_FILTER_DEBUG')) {
          originalLog(...args);
        }
        originalLog(...args);
      };
    });
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Check the actual data structure and filtering
    const dataFilteringDebug = await page.evaluate(() => {
      // Try to access the React component state
      const reactRoot = document.querySelector('#root');
      let componentState = null;
      
      // Try to find the Training component in the React fiber tree
      if (reactRoot && reactRoot._reactInternalFiber) {
        let current = reactRoot._reactInternalFiber;
        while (current) {
          if (current.type && current.type.name === 'Training') {
            componentState = current.memoizedProps;
            break;
          }
          current = current.child || current.sibling;
        }
      }
      
      // Check if we can find any training data in the DOM
      const performanceSection = document.querySelector('[role="tabpanel"]:not([hidden])');
      const hasPerformanceSection = !!performanceSection;
      
      // Try to find any data attributes or props
      const chartContainers = document.querySelectorAll('.recharts-wrapper');
      const chartData = [];
      
      chartContainers.forEach((chart, index) => {
        const svg = chart.querySelector('svg');
        if (svg) {
          // Check for data elements
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
        hasReactRoot: !!reactRoot,
        componentState: componentState,
        hasPerformanceSection,
        performanceSectionText: performanceSection?.textContent?.substring(0, 200) || 'No section found',
        chartData,
        totalCharts: chartContainers.length
      };
    });
    
    console.log('📊 Data Filtering Debug:', JSON.stringify(dataFilteringDebug, null, 2));
    
    // Now let's try to access the actual data by simulating the filtering logic
    const actualDataCheck = await page.evaluate(() => {
      // Simulate the data structure we expect
      const mockTrainingJobs = {
        jobs: [
          {
            job_id: "train_bert-base_1759595412",
            model_name: "bert-base",
            status: "completed",
            start_time: "2025-01-03T10:30:12.000Z",
            result: {
              status: "completed",
              metrics: {
                epoch: 2.0,
                eval_loss: 1.9591144323349,
                eval_runtime: 2.3079,
                eval_steps_per_second: 0.433,
                eval_samples_per_second: 0.867
              }
            }
          }
        ]
      };
      
      // Test the filtering logic
      const filteredJobs = mockTrainingJobs.jobs.filter(job => job.result?.metrics);
      const chartData = filteredJobs.map(job => ({
        epoch: job.result.metrics.epoch,
        loss: job.result.metrics.eval_loss,
        val_loss: job.result.metrics.eval_loss,
        model: job.model_name,
        date: new Date(job.start_time).toLocaleDateString()
      }));
      
      return {
        mockData: mockTrainingJobs,
        filteredJobsCount: filteredJobs.length,
        chartDataLength: chartData.length,
        chartData: chartData,
        filteringWorks: filteredJobs.length > 0
      };
    });
    
    console.log('📊 Actual Data Check:', JSON.stringify(actualDataCheck, null, 2));
    
    // Check if there are any JavaScript errors that might be preventing data rendering
    const errorCheck = await page.evaluate(() => {
      const errors = [];
      
      // Check for console errors
      if (window.console && window.console.errors) {
        errors.push(...window.console.errors);
      }
      
      // Check for any uncaught exceptions
      if (window.onerror) {
        errors.push('onerror handler exists');
      }
      
      return {
        errors,
        hasErrors: errors.length > 0
      };
    });
    
    console.log('📊 Error Check:', JSON.stringify(errorCheck, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'debug-data-filtering.png', 
      fullPage: true 
    });
    
    console.log('✅ Debug completed! Check debug-data-filtering.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

debugDataFiltering().catch(console.error);
