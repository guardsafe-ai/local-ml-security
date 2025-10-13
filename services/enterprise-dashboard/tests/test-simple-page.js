const puppeteer = require('puppeteer');

async function testSimplePage() {
  console.log('🔍 Testing Simple Page...');
  
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
    
    // Simple page check
    const pageInfo = await page.evaluate(() => {
      return {
        title: document.title,
        url: window.location.href,
        hasReactRoot: !!document.querySelector('#root'),
        bodyText: document.body.textContent.substring(0, 500)
      };
    });
    
    console.log('📊 Page Info:', JSON.stringify(pageInfo, null, 2));
    
    // Check for tabs
    const tabs = await page.$$('button[role="tab"]');
    console.log('📊 Found tabs:', tabs.length);
    
    if (tabs.length > 0) {
      const tabTexts = await Promise.all(tabs.map(tab => tab.evaluate(el => el.textContent)));
      console.log('📊 Tab texts:', tabTexts);
      
      // Click on the third tab (Performance Analytics)
      if (tabs.length >= 3) {
        console.log('🎯 Clicking on Performance Analytics tab...');
        await tabs[2].click();
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        // Check for charts
        const charts = await page.$$('.recharts-wrapper');
        console.log('📊 Found charts:', charts.length);
        
        if (charts.length > 0) {
          const chartInfo = await Promise.all(charts.map((chart, chartIndex) => 
            chart.evaluate((el, index) => {
              const svg = el.querySelector('svg');
              const dataPoints = svg ? svg.querySelectorAll('circle, rect[class*="bar"], path[class*="line"]') : [];
              return {
                index: index,
                dataPointCount: dataPoints.length,
                hasData: dataPoints.length > 0
              };
            }, chartIndex)
          ));
          console.log('📊 Chart Info:', JSON.stringify(chartInfo, null, 2));
        }
      }
    }
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'test-simple-page.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check test-simple-page.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testSimplePage().catch(console.error);
