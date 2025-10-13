const puppeteer = require('puppeteer');

async function testRefreshButton() {
  console.log('🔍 Testing Refresh Button...');
  
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
      if (text.includes('🔍') || text.includes('Chart Data Memoized') || text.includes('Force refresh') || 
          text.includes('Performance') || text.includes('Chart') || text.includes('Render') || 
          text.includes('Data') || text.includes('Jobs') || text.includes('useEffect') || 
          text.includes('Training') || text.includes('🔄')) {
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
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Check for refresh button and click it
    console.log('🔄 Looking for refresh button...');
    const refreshButton = await page.$('button[class*="MuiButton"]');
    if (refreshButton) {
      const buttonText = await refreshButton.evaluate(el => el.textContent);
      console.log('🔄 Found button with text:', buttonText);
      if (buttonText.includes('🔄 Refresh Data')) {
        console.log('🔄 Clicking refresh button...');
        await refreshButton.click();
        await new Promise(resolve => setTimeout(resolve, 3000));
      } else {
        console.log('❌ Refresh button text does not match');
      }
    } else {
      console.log('❌ No button found');
    }
    
    // Check for debug boxes after refresh
    const debugInfo = await page.evaluate(() => {
      const debugBoxes = document.querySelectorAll('[class*="warning"], [class*="success"]');
      return {
        debugBoxCount: debugBoxes.length,
        debugText: Array.from(debugBoxes).map(box => box.textContent),
        hasSuccessBox: Array.from(debugBoxes).some(box => box.textContent.includes('✅ Data Loaded')),
        hasWarningBox: Array.from(debugBoxes).some(box => box.textContent.includes('Debug: Jobs='))
      };
    });
    
    console.log('📊 Debug Info After Refresh:', JSON.stringify(debugInfo, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'test-refresh-button.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check test-refresh-button.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testRefreshButton().catch(console.error);
