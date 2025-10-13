const puppeteer = require('puppeteer');

async function testTabRendering() {
  console.log('🔍 Testing Tab Rendering...');
  
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
    await new Promise(resolve => setTimeout(resolve, 15000));
    
    // Check tab structure
    const tabStructure = await page.evaluate(() => {
      const tabs = document.querySelectorAll('button[role="tab"]');
      const tabPanels = document.querySelectorAll('[role="tabpanel"]');
      
      return {
        tabCount: tabs.length,
        tabTexts: Array.from(tabs).map(tab => tab.textContent),
        tabPanelCount: tabPanels.length,
        visibleTabPanel: Array.from(tabPanels).find(panel => !panel.hasAttribute('hidden')),
        visibleTabPanelText: Array.from(tabPanels).find(panel => !panel.hasAttribute('hidden'))?.textContent?.substring(0, 200)
      };
    });
    
    console.log('📊 Tab Structure:', JSON.stringify(tabStructure, null, 2));
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Check tab structure after click
    const tabStructureAfter = await page.evaluate(() => {
      const tabs = document.querySelectorAll('button[role="tab"]');
      const tabPanels = document.querySelectorAll('[role="tabpanel"]');
      
      return {
        tabCount: tabs.length,
        tabTexts: Array.from(tabs).map(tab => tab.textContent),
        tabPanelCount: tabPanels.length,
        visibleTabPanel: Array.from(tabPanels).find(panel => !panel.hasAttribute('hidden')),
        visibleTabPanelText: Array.from(tabPanels).find(panel => !panel.hasAttribute('hidden'))?.textContent?.substring(0, 200),
        allTabPanelTexts: Array.from(tabPanels).map(panel => panel.textContent?.substring(0, 100))
      };
    });
    
    console.log('📊 Tab Structure After Click:', JSON.stringify(tabStructureAfter, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'test-tab-rendering.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check test-tab-rendering.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testTabRendering().catch(console.error);
