const puppeteer = require('puppeteer');

async function debugSimpleRender() {
  console.log('🔍 Debugging Simple Render...');
  
  const browser = await puppeteer.launch({ 
    headless: false, 
    defaultViewport: null,
    args: ['--start-maximized']
  });
  
  try {
    const page = await browser.newPage();
    
    // Enable console logging
    page.on('console', msg => {
      const text = msg.text();
      if (text.includes('🔍') || text.includes('Performance') || text.includes('Chart') || text.includes('Render') || text.includes('Data')) {
        console.log(`📊 [${msg.type().toUpperCase()}] ${text}`);
      }
    });
    
    console.log('🌐 Navigating to training page...');
    await page.goto('http://localhost:3000/training', { 
      waitUntil: 'networkidle0',
      timeout: 30000 
    });
    
    // Wait for data to load
    console.log('⏳ Waiting for data to load...');
    await new Promise(resolve => setTimeout(resolve, 10000));
    
    // Check the current state before clicking Performance Analytics
    console.log('📊 Checking state before clicking Performance Analytics...');
    const beforeState = await page.evaluate(() => {
      const visiblePanel = document.querySelector('[role="tabpanel"]:not([hidden])');
      return {
        panelText: visiblePanel?.textContent?.substring(0, 200) || 'No panel found',
        hasPerformanceAnalytics: visiblePanel?.textContent?.includes('Performance Analytics') || false
      };
    });
    
    console.log('📊 Before State:', JSON.stringify(beforeState, null, 2));
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Check the state after clicking
    console.log('📊 Checking state after clicking Performance Analytics...');
    const afterState = await page.evaluate(() => {
      const visiblePanel = document.querySelector('[role="tabpanel"]:not([hidden])');
      const charts = document.querySelectorAll('.recharts-wrapper');
      
      return {
        panelText: visiblePanel?.textContent?.substring(0, 200) || 'No panel found',
        hasPerformanceAnalytics: visiblePanel?.textContent?.includes('Performance Analytics') || false,
        chartCount: charts.length,
        hasCharts: charts.length > 0
      };
    });
    
    console.log('📊 After State:', JSON.stringify(afterState, null, 2));
    
    // Try to inject some debugging code directly
    console.log('🔍 Injecting debugging code...');
    await page.evaluate(() => {
      // Override console.log to capture our debug output
      const originalLog = console.log;
      console.log = function(...args) {
        if (args[0] && args[0].includes && args[0].includes('INJECTED_DEBUG')) {
          originalLog(...args);
        }
        originalLog(...args);
      };
      
      // Try to find the Performance Analytics section
      const performanceSection = document.querySelector('[role="tabpanel"]:not([hidden])');
      if (performanceSection && performanceSection.textContent.includes('Performance Analytics')) {
        console.log('INJECTED_DEBUG: Found Performance Analytics section');
        
        // Try to find any React components or data
        const reactRoot = document.querySelector('#root');
        if (reactRoot) {
          console.log('INJECTED_DEBUG: Found React root');
        }
      } else {
        console.log('INJECTED_DEBUG: Performance Analytics section not found');
      }
    });
    
    // Wait a bit more for any debug output
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'debug-simple-render.png', 
      fullPage: true 
    });
    
    console.log('✅ Debug completed! Check debug-simple-render.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

debugSimpleRender().catch(console.error);
