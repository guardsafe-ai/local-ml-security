const puppeteer = require('puppeteer');

async function testDebugConditional() {
  console.log('🔍 Testing Debug Conditional...');
  
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
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Check what's actually rendered
    const renderedContent = await page.evaluate(() => {
      const tabPanels = document.querySelectorAll('[role="tabpanel"]');
      let performancePanel = null;
      
      for (let panel of tabPanels) {
        if (panel.textContent.includes('Performance Analytics')) {
          performancePanel = panel;
          break;
        }
      }
      
      if (performancePanel) {
        const loadingText = performancePanel.textContent.includes('Loading Training Data');
        const noDataText = performancePanel.textContent.includes('No Training Data Available');
        const noCompletedText = performancePanel.textContent.includes('No Completed Training Jobs');
        const successText = performancePanel.textContent.includes('✅ Data Loaded');
        
        return {
          hasPerformancePanel: true,
          loadingText,
          noDataText,
          noCompletedText,
          successText,
          fullText: performancePanel.textContent
        };
      }
      
      return {
        hasPerformancePanel: false,
        tabPanelCount: tabPanels.length
      };
    });
    
    console.log('📊 Rendered Content:', JSON.stringify(renderedContent, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'test-debug-conditional.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check test-debug-conditional.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testDebugConditional().catch(console.error);
