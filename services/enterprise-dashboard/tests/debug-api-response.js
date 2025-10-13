const puppeteer = require('puppeteer');

async function debugApiResponse() {
  console.log('🔍 Debugging API Response...');
  
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
      if (text.includes('🔍') || text.includes('Chart Data Memoized') || 
          text.includes('Performance Analytics Render') || text.includes('Training jobs data changed') ||
          text.includes('useEffect') || text.includes('executeJobs') || text.includes('Found jobs') ||
          text.includes('🎯') || text.includes('✅') || text.includes('🔄') || text.includes('Pre-loading') ||
          text.includes('API Response') || text.includes('Raw jobs') || text.includes('Filtering jobs')) {
        console.log(`📊 [${msg.type().toUpperCase()}] ${text}`);
      }
    });
    
    // Enable network monitoring
    page.on('response', response => {
      if (response.url().includes('/training/jobs')) {
        console.log(`🌐 API Response: ${response.status()} ${response.url()}`);
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
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Check what data the component actually has
    const componentData = await page.evaluate(() => {
      // Try to access the React component state
      const reactRoot = document.querySelector('#root');
      return {
        hasReactRoot: !!reactRoot,
        pageTitle: document.title,
        currentUrl: window.location.href
      };
    });
    
    console.log('📊 Component Data:', JSON.stringify(componentData, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'debug-api-response.png', 
      fullPage: true 
    });
    
    console.log('✅ Debug completed! Check debug-api-response.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

debugApiResponse().catch(console.error);
