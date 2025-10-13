const { chromium } = require('playwright');

async function testDatabaseBackedConfig() {
    console.log('🔧 Testing Database-Backed Training Configuration...');
    
    const browser = await chromium.launch({ headless: false });
    const page = await browser.newPage();
    
    try {
        // Navigate to training page
        console.log('🌐 Navigating to training page...');
        await page.goto('http://localhost:3000/training');
        await page.waitForLoadState('networkidle');
        
        // Click on Training Configuration tab
        console.log('🎯 Clicking on Training Configuration tab...');
        await page.click('text=Training Configuration');
        await page.waitForTimeout(2000);
        
        // Check if Save and Load buttons are present
        const saveButton = await page.locator('button').filter({ hasText: 'Save Config' });
        const loadButton = await page.locator('button').filter({ hasText: 'Load Config' });
        
        if (await saveButton.count() > 0 && await loadButton.count() > 0) {
            console.log('✅ Save and Load buttons are present');
        } else {
            console.log('❌ Save and Load buttons not found');
        }
        
        // Test model selection dropdown
        console.log('🎯 Testing model selection...');
        const modelSelect = await page.locator('select[name="model_name"]');
        
        if (await modelSelect.count() > 0) {
            const options = await modelSelect.locator('option').all();
            console.log(`📋 Found ${options.length} model options`);
            
            // Select bert-base_pretrained if available
            const bertOption = await modelSelect.locator('option[value="bert-base_pretrained"]');
            if (await bertOption.count() > 0) {
                await modelSelect.selectOption('bert-base_pretrained');
                console.log('✅ Selected bert-base_pretrained model');
                
                // Wait for auto-load to complete
                await page.waitForTimeout(2000);
                
                // Check if configuration was loaded
                const dataPathInput = await page.locator('input[name="training_data_path"]');
                const dataPathValue = await dataPathInput.inputValue();
                
                if (dataPathValue.includes('/data/training/bert')) {
                    console.log('✅ Configuration auto-loaded from database');
                } else {
                    console.log('❌ Configuration not auto-loaded');
                }
                
                // Test saving a new configuration
                console.log('🎯 Testing save configuration...');
                
                // Update some values
                await dataPathInput.fill('/data/training/bert-updated');
                const learningRateInput = await page.locator('input[name="hyperparameters.learning_rate"]');
                await learningRateInput.fill('0.0002');
                
                // Click save
                await saveButton.click();
                await page.waitForTimeout(2000);
                
                // Check for success message
                const successAlert = await page.locator('.MuiAlert-message');
                if (await successAlert.count() > 0) {
                    const alertText = await successAlert.textContent();
                    if (alertText.includes('Configuration saved successfully')) {
                        console.log('✅ Configuration saved successfully');
                    } else {
                        console.log('❌ Save failed:', alertText);
                    }
                } else {
                    console.log('❌ No success message found');
                }
                
                // Test loading configuration
                console.log('🎯 Testing load configuration...');
                
                // Clear the form first
                await dataPathInput.fill('');
                await learningRateInput.fill('');
                
                // Click load
                await loadButton.click();
                await page.waitForTimeout(2000);
                
                // Check if configuration was loaded
                const reloadedDataPath = await dataPathInput.inputValue();
                const reloadedLearningRate = await learningRateInput.inputValue();
                
                if (reloadedDataPath.includes('/data/training/bert-updated') && reloadedLearningRate === '0.0002') {
                    console.log('✅ Configuration loaded successfully from database');
                } else {
                    console.log('❌ Configuration not loaded properly');
                    console.log('Data path:', reloadedDataPath);
                    console.log('Learning rate:', reloadedLearningRate);
                }
                
            } else {
                console.log('❌ bert-base_pretrained model not found in dropdown');
            }
        } else {
            console.log('❌ Model selection dropdown not found');
        }
        
        // Take screenshot
        await page.screenshot({ path: 'test-database-config.png', fullPage: true });
        console.log('✅ Test completed! Check test-database-config.png');
        
    } catch (error) {
        console.error('❌ Test failed:', error);
        await page.screenshot({ path: 'test-database-config-error.png', fullPage: true });
    } finally {
        await browser.close();
    }
}

testDatabaseBackedConfig().catch(console.error);
