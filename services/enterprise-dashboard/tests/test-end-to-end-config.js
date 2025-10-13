const { chromium } = require('playwright');

async function testEndToEndConfigFlow() {
    console.log('🧪 Testing End-to-End Configuration Flow...');
    
    const browser = await chromium.launch({ headless: false });
    const page = await browser.newPage();
    
    try {
        // Step 1: Navigate to training page
        console.log('🌐 Step 1: Navigating to training page...');
        await page.goto('http://localhost:3000/training');
        await page.waitForLoadState('networkidle');
        
        // Step 2: Go to Training Configuration tab
        console.log('🎯 Step 2: Opening Training Configuration tab...');
        await page.click('text=Training Configuration');
        await page.waitForTimeout(2000);
        
        // Step 3: Select a model
        console.log('📋 Step 3: Selecting bert-base_pretrained model...');
        const modelSelect = await page.locator('select[name="model_name"]');
        await modelSelect.selectOption('bert-base_pretrained');
        await page.waitForTimeout(2000);
        
        // Step 4: Check if configuration auto-loaded
        console.log('🔄 Step 4: Checking if configuration auto-loaded...');
        const dataPathInput = await page.locator('input[name="training_data_path"]');
        const dataPathValue = await dataPathInput.inputValue();
        
        if (dataPathValue.includes('/data/training/bert-saved-config')) {
            console.log('✅ Configuration auto-loaded successfully from database!');
        } else {
            console.log('❌ Configuration not auto-loaded, current value:', dataPathValue);
        }
        
        // Step 5: Modify configuration
        console.log('✏️ Step 5: Modifying configuration...');
        await dataPathInput.fill('/data/training/bert-modified-via-frontend');
        
        const learningRateInput = await page.locator('input[name="hyperparameters.learning_rate"]');
        await learningRateInput.fill('0.0005');
        
        const batchSizeInput = await page.locator('input[name="hyperparameters.batch_size"]');
        await batchSizeInput.fill('8');
        
        const epochsInput = await page.locator('input[name="hyperparameters.epochs"]');
        await epochsInput.fill('10');
        
        // Step 6: Save configuration
        console.log('💾 Step 6: Saving modified configuration...');
        const saveButton = await page.locator('button').filter({ hasText: 'Save Config' });
        await saveButton.click();
        await page.waitForTimeout(2000);
        
        // Check for success message
        const successAlert = await page.locator('.MuiAlert-message');
        if (await successAlert.count() > 0) {
            const alertText = await successAlert.textContent();
            if (alertText.includes('Configuration saved successfully')) {
                console.log('✅ Configuration saved successfully!');
            } else {
                console.log('❌ Save failed:', alertText);
            }
        } else {
            console.log('❌ No success message found');
        }
        
        // Step 7: Test training with saved configuration
        console.log('🚀 Step 7: Testing training with saved configuration...');
        
        // Go to Training Jobs tab
        await page.click('text=Training Jobs');
        await page.waitForTimeout(1000);
        
        // Click Start Training
        const startTrainingButton = await page.locator('button').filter({ hasText: 'Start Training' });
        if (await startTrainingButton.count() > 0) {
            await startTrainingButton.click();
            await page.waitForTimeout(2000);
            
            // Select model in training dialog
            const trainingModelSelect = await page.locator('select[name="model_name"]');
            if (await trainingModelSelect.count() > 0) {
                await trainingModelSelect.selectOption('bert-base_pretrained');
                await page.waitForTimeout(1000);
                
                // Check if training data path is pre-filled with saved config
                const trainingDataPathInput = await page.locator('input[name="training_data_path"]');
                const trainingDataPathValue = await trainingDataPathInput.inputValue();
                
                if (trainingDataPathValue.includes('/data/training/bert-modified-via-frontend')) {
                    console.log('✅ Training dialog pre-filled with saved configuration!');
                } else {
                    console.log('❌ Training dialog not pre-filled, current value:', trainingDataPathValue);
                }
                
                // Cancel training dialog
                const cancelButton = await page.locator('button').filter({ hasText: 'Cancel' });
                await cancelButton.click();
            }
        }
        
        // Step 8: Verify configuration persistence
        console.log('🔄 Step 8: Verifying configuration persistence...');
        
        // Go back to Training Configuration tab
        await page.click('text=Training Configuration');
        await page.waitForTimeout(1000);
        
        // Reload the page to test persistence
        await page.reload();
        await page.waitForLoadState('networkidle');
        
        // Go back to Training Configuration tab
        await page.click('text=Training Configuration');
        await page.waitForTimeout(2000);
        
        // Select the model again
        const modelSelectAfterReload = await page.locator('select[name="model_name"]');
        await modelSelectAfterReload.selectOption('bert-base_pretrained');
        await page.waitForTimeout(2000);
        
        // Check if configuration persisted
        const dataPathInputAfterReload = await page.locator('input[name="training_data_path"]');
        const dataPathValueAfterReload = await dataPathInputAfterReload.inputValue();
        
        if (dataPathValueAfterReload.includes('/data/training/bert-modified-via-frontend')) {
            console.log('✅ Configuration persisted across page reload!');
        } else {
            console.log('❌ Configuration not persisted, current value:', dataPathValueAfterReload);
        }
        
        // Take final screenshot
        await page.screenshot({ path: 'test-end-to-end-config.png', fullPage: true });
        console.log('✅ End-to-end test completed! Check test-end-to-end-config.png');
        
    } catch (error) {
        console.error('❌ Test failed:', error);
        await page.screenshot({ path: 'test-end-to-end-config-error.png', fullPage: true });
    } finally {
        await browser.close();
    }
}

testEndToEndConfigFlow().catch(console.error);
