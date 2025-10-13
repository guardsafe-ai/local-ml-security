const puppeteer = require('puppeteer');

async function testDataState() {
  console.log('🔍 Testing Data State...');
  
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
      console.log(`📊 [${msg.type().toUpperCase()}] ${text}`);
    });
    
    console.log('🌐 Navigating to training page...');
    await page.goto('http://localhost:3000/training', { 
      waitUntil: 'networkidle0',
      timeout: 30000 
    });
    
    // Wait for data to load
    console.log('⏳ Waiting for data to load...');
    await new Promise(resolve => setTimeout(resolve, 20000));
    
    // Check the exact data state by injecting a script
    const dataState = await page.evaluate(() => {
      // Try to access the React component state
      const reactRoot = document.querySelector('#root');
      if (reactRoot && reactRoot._reactInternalFiber) {
        return {
          hasReactRoot: true,
          reactRootType: typeof reactRoot._reactInternalFiber
        };
      }
      
      // Check if we can find any training data in the DOM
      const trainingElements = document.querySelectorAll('[class*="training"], [class*="job"], [class*="chart"]');
      const textContent = document.body.textContent;
      
      return {
        hasReactRoot: !!reactRoot,
        trainingElementsCount: trainingElements.length,
        hasTrainingText: textContent.includes('Training'),
        hasJobText: textContent.includes('job'),
        hasChartText: textContent.includes('chart'),
        bodyTextSample: textContent.substring(0, 1000)
      };
    });
    
    console.log('📊 Data State:', JSON.stringify(dataState, null, 2));
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Check what's actually rendered in the Performance Analytics tab
    const tabContent = await page.evaluate(() => {
      const tabPanels = document.querySelectorAll('[role="tabpanel"]');
      let performancePanel = null;
      
      for (let panel of tabPanels) {
        if (panel.textContent.includes('Performance Analytics') || panel.textContent.includes('Loading') || panel.textContent.includes('No Training')) {
          performancePanel = panel;
          break;
        }
      }
      
      return {
        tabPanelCount: tabPanels.length,
        hasPerformancePanel: !!performancePanel,
        performancePanelText: performancePanel ? performancePanel.textContent : 'Not found',
        allPanelTexts: Array.from(tabPanels).map(panel => panel.textContent.substring(0, 200))
      };
    });
    
    console.log('📊 Tab Content:', JSON.stringify(tabContent, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'test-data-state.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check test-data-state.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testDataState().catch(console.error);
