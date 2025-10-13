const puppeteer = require('puppeteer');

async function debugTabIndexing() {
  console.log('🔍 Debugging Tab Indexing...');
  
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
    
    // Check all tabs and their content
    console.log('📋 Checking all tabs...');
    const allTabs = await page.evaluate(() => {
      const tabButtons = document.querySelectorAll('button[role="tab"]');
      const tabPanels = document.querySelectorAll('[role="tabpanel"]');
      
      const tabs = Array.from(tabButtons).map((tab, index) => ({
        index,
        text: tab.textContent,
        ariaSelected: tab.getAttribute('aria-selected'),
        disabled: tab.disabled,
        tabId: tab.getAttribute('id'),
        ariaControls: tab.getAttribute('aria-controls')
      }));
      
      const panels = Array.from(tabPanels).map((panel, index) => ({
        index,
        id: panel.getAttribute('id'),
        ariaLabelledBy: panel.getAttribute('aria-labelledby'),
        hidden: panel.hasAttribute('hidden'),
        display: panel.style.display,
        textContent: panel.textContent.substring(0, 200) + '...'
      }));
      
      return { tabs, panels };
    });
    
    console.log('📋 All Tabs:', JSON.stringify(allTabs, null, 2));
    
    // Click on each tab and check what content is shown
    console.log('🎯 Testing each tab...');
    for (let i = 0; i < 4; i++) {
      console.log(`\n--- Testing Tab ${i} ---`);
      
      // Click on tab i
      await page.click(`button[role="tab"]:nth-child(${i + 1})`);
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Check what content is visible
      const visibleContent = await page.evaluate(() => {
        const visiblePanels = Array.from(document.querySelectorAll('[role="tabpanel"]'))
          .filter(panel => !panel.hasAttribute('hidden') && panel.style.display !== 'none');
        
        return {
          visiblePanelCount: visiblePanels.length,
          visiblePanelText: visiblePanels[0]?.textContent?.substring(0, 200) || 'No visible panel',
          hasPerformanceAnalytics: visiblePanels[0]?.textContent?.includes('Performance Analytics') || false,
          hasTrainingJobs: visiblePanels[0]?.textContent?.includes('Training Jobs') || false,
          hasModelRetraining: visiblePanels[0]?.textContent?.includes('Model Retraining') || false,
          hasCharts: visiblePanels[0]?.querySelectorAll('.recharts-wrapper').length > 0
        };
      });
      
      console.log(`Tab ${i} Content:`, JSON.stringify(visibleContent, null, 2));
    }
    
    // Now specifically click on Performance Analytics tab (index 2)
    console.log('\n🎯 Clicking on Performance Analytics tab (index 2)...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Check the final state
    const finalState = await page.evaluate(() => {
      const visiblePanels = Array.from(document.querySelectorAll('[role="tabpanel"]'))
        .filter(panel => !panel.hasAttribute('hidden') && panel.style.display !== 'none');
      
      const charts = document.querySelectorAll('.recharts-wrapper');
      
      return {
        visiblePanelCount: visiblePanels.length,
        visiblePanelText: visiblePanels[0]?.textContent?.substring(0, 300) || 'No visible panel',
        hasPerformanceAnalytics: visiblePanels[0]?.textContent?.includes('Performance Analytics') || false,
        hasCharts: charts.length,
        chartDetails: Array.from(charts).map((chart, index) => ({
          index,
          hasSvg: !!chart.querySelector('svg'),
          dataElements: chart.querySelectorAll('[data-key], circle, rect, path').length
        }))
      };
    });
    
    console.log('📊 Final State:', JSON.stringify(finalState, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'debug-tab-indexing.png', 
      fullPage: true 
    });
    
    console.log('✅ Debug completed! Check debug-tab-indexing.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

debugTabIndexing().catch(console.error);
