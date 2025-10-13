const puppeteer = require('puppeteer');

async function testTrainingConfig() {
  console.log('🔧 Testing Training Configuration Feature...');
  
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
      if (text.includes('🔧') || text.includes('💾') || text.includes('📋') || 
          text.includes('Configuration') || text.includes('Save') || text.includes('Load')) {
        console.log(`🔧 [${msg.type().toUpperCase()}] ${text}`);
      }
    });
    
    console.log('🌐 Navigating to training page...');
    await page.goto('http://localhost:3000/training', { 
      waitUntil: 'networkidle0',
      timeout: 30000 
    });
    
    // Wait for page to load
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Click on Training Configuration tab
    console.log('🎯 Clicking on Training Configuration tab...');
    await page.click('button[role="tab"]:nth-child(2)');
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Check if Save/Load buttons are present
    const buttons = await page.$$('button[class*="MuiButton"]');
    let saveButton = null;
    let loadButton = null;
    
    for (const button of buttons) {
      const text = await button.evaluate(el => el.textContent);
      if (text && text.includes('Save Config')) {
        saveButton = button;
      }
      if (text && text.includes('Load Config')) {
        loadButton = button;
      }
    }
    
    if (saveButton && loadButton) {
      console.log('✅ Save and Load buttons are present');
    } else {
      console.log('❌ Save and Load buttons are missing');
    }
    
    // Test model selection
    console.log('🎯 Testing model selection...');
    const modelSelect = await page.$('div[role="button"]');
    if (modelSelect) {
      await modelSelect.click();
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Check if models are available
      const modelOptions = await page.$$('li[role="option"]');
      if (modelOptions.length > 0) {
        console.log(`✅ Found ${modelOptions.length} model options`);
        
        // Select first model
        await modelOptions[0].click();
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Test save configuration
        console.log('💾 Testing save configuration...');
        const saveBtn = await page.$('button[class*="MuiButton"]');
        if (saveBtn) {
          const saveBtnText = await saveBtn.evaluate(el => el.textContent);
          if (saveBtnText && saveBtnText.includes('Save Config')) {
            await saveBtn.click();
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // Check for success message
            const successAlert = await page.$('div[role="alert"]');
            if (successAlert) {
              const alertText = await successAlert.evaluate(el => el.textContent);
              if (alertText && alertText.includes('Configuration saved successfully')) {
                console.log('✅ Configuration saved successfully!');
              } else {
                console.log('❌ No success message found');
              }
            }
          }
        }
        
        // Test load configuration
        console.log('📋 Testing load configuration...');
        const loadBtn = await page.$('button[class*="MuiButton"]');
        if (loadBtn) {
          const loadBtnText = await loadBtn.evaluate(el => el.textContent);
          if (loadBtnText && loadBtnText.includes('Load Config')) {
            await loadBtn.click();
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // Check for success message
            const successAlert = await page.$('div[role="alert"]');
            if (successAlert) {
              const alertText = await successAlert.evaluate(el => el.textContent);
              if (alertText && alertText.includes('Configuration saved successfully')) {
                console.log('✅ Configuration loaded successfully!');
              } else {
                console.log('❌ No success message found for load');
              }
            }
          }
        }
      } else {
        console.log('❌ No model options found');
      }
    } else {
      console.log('❌ Model select dropdown not found');
    }
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'test-training-config.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check test-training-config.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testTrainingConfig().catch(console.error);
