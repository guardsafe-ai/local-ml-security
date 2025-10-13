const puppeteer = require('puppeteer');

async function testDataFlow() {
  console.log('🔍 Testing Data Flow...');
  
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
    
    // Check if we can see the Performance Analytics tab content
    const tabContent = await page.evaluate(() => {
      // Find the Performance Analytics tab panel
      const tabPanels = document.querySelectorAll('[role="tabpanel"]');
      let performancePanel = null;
      
      for (let panel of tabPanels) {
        if (panel.textContent.includes('Performance Analytics')) {
          performancePanel = panel;
          break;
        }
      }
      
      return {
        tabPanelCount: tabPanels.length,
        hasPerformancePanel: !!performancePanel,
        performancePanelText: performancePanel ? performancePanel.textContent.substring(0, 500) : 'Not found',
        performancePanelVisible: performancePanel ? !performancePanel.hasAttribute('hidden') : false
      };
    });
    
    console.log('📊 Tab Content:', JSON.stringify(tabContent, null, 2));
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Check tab content after click
    const tabContentAfter = await page.evaluate(() => {
      const tabPanels = document.querySelectorAll('[role="tabpanel"]');
      let performancePanel = null;
      
      for (let panel of tabPanels) {
        if (panel.textContent.includes('Performance Analytics')) {
          performancePanel = panel;
          break;
        }
      }
      
      return {
        tabPanelCount: tabPanels.length,
        hasPerformancePanel: !!performancePanel,
        performancePanelText: performancePanel ? performancePanel.textContent.substring(0, 500) : 'Not found',
        performancePanelVisible: performancePanel ? !performancePanel.hasAttribute('hidden') : false,
        allPanelTexts: Array.from(tabPanels).map(panel => panel.textContent.substring(0, 100))
      };
    });
    
    console.log('📊 Tab Content After Click:', JSON.stringify(tabContentAfter, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'test-data-flow.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check test-data-flow.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testDataFlow().catch(console.error);
