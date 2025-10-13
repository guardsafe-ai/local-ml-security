const puppeteer = require('puppeteer');

async function debugChartData() {
  console.log('🔍 Debugging Chart Data...');
  
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
    
    // Inject debugging code to check the actual data being passed to charts
    console.log('🔍 Injecting debugging code...');
    await page.evaluate(() => {
      // Override console.log to capture our debug output
      const originalLog = console.log;
      console.log = function(...args) {
        if (args[0] && args[0].includes && args[0].includes('CHART_DEBUG')) {
          originalLog(...args);
        }
      };
      
      // Try to find the React component and access its state
      const reactRoot = document.querySelector('#root');
      if (reactRoot && reactRoot._reactInternalFiber) {
        console.log('CHART_DEBUG: Found React root');
      }
    });
    
    // Check the actual data being passed to the charts
    const chartDataDebug = await page.evaluate(() => {
      // Get the raw training jobs data
      const trainingJobsData = window.trainingJobs || null;
      
      // Simulate the filtering logic from the component
      const filteredJobs = trainingJobsData?.jobs?.filter(job => job.result?.metrics) || [];
      
      // Simulate the data mapping for LineChart
      const lineChartData = filteredJobs.map(job => ({
        epoch: job.result.metrics.epoch,
        loss: job.result.metrics.eval_loss,
        val_loss: job.result.metrics.eval_loss,
        model: job.model_name,
        date: new Date(job.start_time).toLocaleDateString()
      }));
      
      // Simulate the data mapping for BarChart
      const barChartData = filteredJobs.map(job => ({
        model: job.model_name,
        samples_per_sec: job.result.metrics.eval_samples_per_second || 0,
        steps_per_sec: job.result.metrics.eval_steps_per_second || 0,
        loss: job.result.metrics.eval_loss || 0,
        runtime: job.result.metrics.eval_runtime || 0
      }));
      
      return {
        hasTrainingJobsData: !!trainingJobsData,
        trainingJobsKeys: trainingJobsData ? Object.keys(trainingJobsData) : [],
        jobsCount: trainingJobsData?.jobs?.length || 0,
        filteredJobsCount: filteredJobs.length,
        lineChartDataLength: lineChartData.length,
        barChartDataLength: barChartData.length,
        lineChartData: lineChartData,
        barChartData: barChartData,
        sampleJob: filteredJobs[0] || null
      };
    });
    
    console.log('📊 Chart Data Debug:', JSON.stringify(chartDataDebug, null, 2));
    
    // Check if the data is actually being passed to the Recharts components
    const rechartsDataCheck = await page.evaluate(() => {
      const charts = document.querySelectorAll('.recharts-wrapper');
      const results = [];
      
      charts.forEach((chart, index) => {
        // Look for data attributes or any way to access the data
        const svg = chart.querySelector('svg');
        if (svg) {
          // Check for data points in the SVG
          const dataElements = svg.querySelectorAll('[data-key], circle, rect, path[class*="line"], path[class*="bar"]');
          const hasDataElements = dataElements.length > 0;
          
          // Check for axis labels
          const axisLabels = svg.querySelectorAll('text[class*="tick"]');
          const hasAxisLabels = axisLabels.length > 0;
          
          // Check for legend items
          const legendItems = svg.querySelectorAll('[class*="legend-item"]');
          const hasLegendItems = legendItems.length > 0;
          
          results.push({
            chartIndex: index,
            hasDataElements,
            dataElementCount: dataElements.length,
            hasAxisLabels,
            axisLabelCount: axisLabels.length,
            hasLegendItems,
            legendItemCount: legendItems.length,
            isEmpty: !hasDataElements && !hasAxisLabels
          });
        }
      });
      
      return results;
    });
    
    console.log('📊 Recharts Data Check:', JSON.stringify(rechartsDataCheck, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'debug-chart-data.png', 
      fullPage: true 
    });
    
    console.log('✅ Debug completed! Check debug-chart-data.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

debugChartData().catch(console.error);
