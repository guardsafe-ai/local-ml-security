const puppeteer = require('puppeteer');

async function debugTabRendering() {
  console.log('🔍 Debugging Tab Rendering...');
  
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
    
    // Check all tabs
    console.log('📋 Checking all tabs...');
    const tabs = await page.evaluate(() => {
      const tabButtons = document.querySelectorAll('button[role="tab"]');
      return Array.from(tabButtons).map((tab, index) => ({
        index,
        text: tab.textContent,
        ariaSelected: tab.getAttribute('aria-selected'),
        disabled: tab.disabled
      }));
    });
    
    console.log('📋 Tabs:', JSON.stringify(tabs, null, 2));
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Check which tab panel is visible
    console.log('📊 Checking visible tab panel...');
    const visiblePanel = await page.evaluate(() => {
      const tabPanels = document.querySelectorAll('[role="tabpanel"]');
      const visiblePanels = Array.from(tabPanels).filter(panel => 
        !panel.hasAttribute('hidden') && panel.style.display !== 'none'
      );
      
      return {
        totalPanels: tabPanels.length,
        visiblePanels: visiblePanels.length,
        visiblePanelText: visiblePanels[0]?.textContent?.substring(0, 200) || 'No visible panel',
        hasPerformanceAnalytics: visiblePanels[0]?.textContent?.includes('Performance Analytics') || false
      };
    });
    
    console.log('📊 Visible Panel:', JSON.stringify(visiblePanel, null, 2));
    
    // Check if the debug console.log is being executed
    console.log('🔍 Checking for debug output...');
    const debugOutput = await page.evaluate(() => {
      // Try to find any debug information
      const performanceSection = document.querySelector('[role="tabpanel"]:not([hidden])');
      if (performanceSection) {
        return {
          hasPerformanceAnalytics: performanceSection.textContent.includes('Performance Analytics'),
          hasDebugLog: performanceSection.textContent.includes('🔍 Performance Analytics Debug'),
          hasCharts: performanceSection.querySelectorAll('.recharts-wrapper').length > 0,
          chartCount: performanceSection.querySelectorAll('.recharts-wrapper').length
        };
      }
      return { error: 'No visible tab panel found' };
    });
    
    console.log('🔍 Debug Output:', JSON.stringify(debugOutput, null, 2));
    
    // Check the actual data being passed to charts
    console.log('📈 Checking chart data in detail...');
    const detailedChartData = await page.evaluate(() => {
      const charts = document.querySelectorAll('.recharts-wrapper');
      const results = [];
      
      charts.forEach((chart, index) => {
        const svg = chart.querySelector('svg');
        if (svg) {
          // Check for all possible data elements
          const allElements = svg.querySelectorAll('*');
          const dataElements = svg.querySelectorAll('[data-key], circle, rect, path');
          const textElements = svg.querySelectorAll('text');
          const lineElements = svg.querySelectorAll('line');
          const pathElements = svg.querySelectorAll('path');
          
          // Check for specific Recharts elements
          const rechartsElements = svg.querySelectorAll('[class*="recharts"]');
          const axisElements = svg.querySelectorAll('[class*="axis"]');
          const legendElements = svg.querySelectorAll('[class*="legend"]');
          const tooltipElements = svg.querySelectorAll('[class*="tooltip"]');
          
          results.push({
            chartIndex: index,
            totalElements: allElements.length,
            dataElements: dataElements.length,
            textElements: textElements.length,
            lineElements: lineElements.length,
            pathElements: pathElements.length,
            rechartsElements: rechartsElements.length,
            axisElements: axisElements.length,
            legendElements: legendElements.length,
            tooltipElements: tooltipElements.length,
            isEmpty: dataElements.length === 0 && textElements.length === 0
          });
        }
      });
      
      return results;
    });
    
    console.log('📈 Detailed Chart Data:', JSON.stringify(detailedChartData, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'debug-tab-rendering.png', 
      fullPage: true 
    });
    
    console.log('✅ Debug completed! Check debug-tab-rendering.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

debugTabRendering().catch(console.error);
