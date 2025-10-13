const puppeteer = require('puppeteer');

async function simpleTabTest() {
  console.log('🔍 Simple Tab Test...');
  
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
    
    console.log('🔍 Checking tab structure...');
    
    // Check the tab structure
    const tabStructure = await page.evaluate(() => {
      const tabs = document.querySelectorAll('button[role="tab"]');
      const tabPanels = document.querySelectorAll('[role="tabpanel"]');
      
      return {
        tabCount: tabs.length,
        tabLabels: Array.from(tabs).map(tab => tab.textContent),
        tabPanelCount: tabPanels.length,
        tabPanelTexts: Array.from(tabPanels).map(panel => panel.textContent.substring(0, 100))
      };
    });
    
    console.log('📊 Tab Structure:', JSON.stringify(tabStructure, null, 2));
    
    // Check which tab is active
    const activeTab = await page.evaluate(() => {
      const activeTab = document.querySelector('button[role="tab"][aria-selected="true"]');
      const visibleTabPanel = document.querySelector('[role="tabpanel"]:not([hidden])');
      
      return {
        activeTabText: activeTab ? activeTab.textContent : 'No active tab',
        visibleTabPanelText: visibleTabPanel ? visibleTabPanel.textContent.substring(0, 100) : 'No visible tab panel'
      };
    });
    
    console.log('📊 Active Tab:', JSON.stringify(activeTab, null, 2));
    
    // Try clicking on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Check what's visible after clicking
    const afterClick = await page.evaluate(() => {
      const activeTab = document.querySelector('button[role="tab"][aria-selected="true"]');
      const visibleTabPanel = document.querySelector('[role="tabpanel"]:not([hidden])');
      
      return {
        activeTabText: activeTab ? activeTab.textContent : 'No active tab',
        visibleTabPanelText: visibleTabPanel ? visibleTabPanel.textContent.substring(0, 200) : 'No visible tab panel',
        hasPerformanceAnalytics: visibleTabPanel ? visibleTabPanel.textContent.includes('Performance Analytics') : false
      };
    });
    
    console.log('📊 After Click:', JSON.stringify(afterClick, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'simple-tab-test.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check simple-tab-test.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

simpleTabTest().catch(console.error);
