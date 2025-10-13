import { test, expect } from '@playwright/test';

test.describe('Model Unload Logs - Simple Test', () => {
  test('should navigate to models page and test unload functionality', async ({ page }) => {
    // Navigate to the main page
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
    
    // Take a screenshot to see what's on the page
    await page.screenshot({ path: 'test-results/initial-page.png' });
    
    // Look for any navigation to models
    console.log('Page title:', await page.title());
    console.log('Current URL:', page.url());
    
    // Check if we're already on a models page or need to navigate
    const modelsTab = page.locator('text=Models').first();
    const modelsLink = page.locator('a[href*="models"]').first();
    
    if (await modelsTab.isVisible()) {
      console.log('Found Models tab, clicking...');
      await modelsTab.click();
      await page.waitForLoadState('networkidle');
    } else if (await modelsLink.isVisible()) {
      console.log('Found Models link, clicking...');
      await modelsLink.click();
      await page.waitForLoadState('networkidle');
    } else {
      console.log('No Models navigation found, checking current page...');
    }
    
    // Take another screenshot after navigation
    await page.screenshot({ path: 'test-results/after-navigation.png' });
    
    // Look for any Load buttons
    const loadButtons = page.locator('button:has-text("Load")');
    const loadButtonCount = await loadButtons.count();
    console.log(`Found ${loadButtonCount} Load buttons`);
    
    if (loadButtonCount > 0) {
      console.log('Found Load buttons, proceeding with test...');
      
      // Get the first load button
      const firstLoadButton = loadButtons.first();
      
      // Get the model name from the button's parent context
      const modelCard = firstLoadButton.locator('xpath=ancestor::*[contains(@class, "MuiCard") or contains(@class, "card")]');
      const modelNameElement = modelCard.locator('h6, .MuiTypography-h6, [data-testid="model-name"]').first();
      
      let modelName = 'Unknown Model';
      if (await modelNameElement.isVisible()) {
        modelName = await modelNameElement.textContent() || 'Unknown Model';
      }
      
      console.log(`Testing with model: ${modelName}`);
      
      // Click Load button
      await firstLoadButton.click();
      console.log('Clicked Load button');
      
      // Wait for loading to complete - look for Unload button or loading indicator
      try {
        await page.waitForSelector('button:has-text("Unload")', { timeout: 30000 });
        console.log('Model loaded successfully, found Unload button');
        
        // Click Unload button
        const unloadButton = page.locator('button:has-text("Unload")').first();
        await unloadButton.click();
        console.log('Clicked Unload button');
        
        // Wait for logs modal to appear
        await page.waitForSelector('[role="dialog"]', { timeout: 10000 });
        console.log('Logs modal appeared');
        
        // Verify modal title
        const modalTitle = page.locator('[role="dialog"] h6, [role="dialog"] .MuiTypography-h6');
        await expect(modalTitle).toContainText('Model Unload Logs');
        
        // Verify logs are displayed
        const logsList = page.locator('[role="dialog"] .MuiList-root');
        await expect(logsList).toBeVisible();
        
        // Check for specific log entries
        await expect(page.locator('text=Starting unload process')).toBeVisible();
        await expect(page.locator('text=Successfully unloaded model')).toBeVisible();
        
        console.log('✅ Test passed! Logs modal is working correctly.');
        
        // Close the modal
        const closeButton = page.locator('[role="dialog"] button:has-text("Close")');
        await closeButton.click();
        
      } catch (error) {
        console.log('❌ Test failed:', error);
        await page.screenshot({ path: 'test-results/error-state.png' });
        throw error;
      }
      
    } else {
      console.log('No Load buttons found. Checking what buttons are available...');
      const allButtons = page.locator('button');
      const buttonCount = await allButtons.count();
      console.log(`Found ${buttonCount} buttons total`);
      
      for (let i = 0; i < Math.min(buttonCount, 10); i++) {
        const button = allButtons.nth(i);
        const text = await button.textContent();
        console.log(`Button ${i}: "${text}"`);
      }
      
      // Take a final screenshot
      await page.screenshot({ path: 'test-results/no-load-buttons.png' });
      
      throw new Error('No Load buttons found on the page');
    }
  });
});
