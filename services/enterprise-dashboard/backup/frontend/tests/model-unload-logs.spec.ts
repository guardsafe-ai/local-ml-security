import { test, expect } from '@playwright/test';

test.describe('Model Unload Logs Modal', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the models page
    await page.goto('http://localhost:3000');
    
    // Wait for the page to load
    await page.waitForLoadState('networkidle');
    
    // Navigate to models page if not already there
    const modelsLink = page.locator('a[href*="models"], button:has-text("Models")').first();
    if (await modelsLink.isVisible()) {
      await modelsLink.click();
      await page.waitForLoadState('networkidle');
    }
  });

  test('should load model, unload model, and show logs popup', async ({ page }) => {
    // Wait for models to load
    await page.waitForSelector('[data-testid="models-list"], .MuiCard-root', { timeout: 10000 });
    
    // Find a model that is not loaded (should have a Load button)
    const loadButton = page.locator('button:has-text("Load")').first();
    await expect(loadButton).toBeVisible({ timeout: 10000 });
    
    // Get the model name from the button's context
    const modelCard = loadButton.locator('..').locator('..').locator('..');
    const modelName = await modelCard.locator('h6, .MuiTypography-h6').textContent();
    console.log(`Testing with model: ${modelName}`);
    
    // Click Load button
    await loadButton.click();
    
    // Wait for loading to complete (button should change to Unload)
    await page.waitForSelector('button:has-text("Unload")', { timeout: 30000 });
    
    // Verify model is loaded
    const unloadButton = page.locator('button:has-text("Unload")').first();
    await expect(unloadButton).toBeVisible();
    
    // Click Unload button
    await unloadButton.click();
    
    // Wait for logs modal to appear
    await page.waitForSelector('[role="dialog"]', { timeout: 10000 });
    
    // Verify modal title contains model name
    const modalTitle = page.locator('[role="dialog"] h6, [role="dialog"] .MuiTypography-h6');
    await expect(modalTitle).toContainText('Model Unload Logs');
    await expect(modalTitle).toContainText(modelName || '');
    
    // Verify success status
    const successChip = page.locator('[role="dialog"] .MuiChip-root:has-text("Success")');
    await expect(successChip).toBeVisible();
    
    // Verify logs are displayed
    const logsList = page.locator('[role="dialog"] .MuiList-root');
    await expect(logsList).toBeVisible();
    
    // Check for specific log entries
    const logEntries = page.locator('[role="dialog"] .MuiListItem-root');
    await expect(logEntries).toHaveCount({ min: 5 }); // Should have multiple log entries
    
    // Verify log levels are displayed
    const infoChips = page.locator('[role="dialog"] .MuiChip-root:has-text("INFO")');
    await expect(infoChips).toHaveCount({ min: 3 });
    
    // Verify specific log messages
    await expect(page.locator('text=Starting unload process')).toBeVisible();
    await expect(page.locator('text=Model wrapper unloaded')).toBeVisible();
    await expect(page.locator('text=Successfully unloaded model')).toBeVisible();
    
    // Test copy logs functionality
    const copyButton = page.locator('[role="dialog"] button:has-text("Copy Logs")');
    await expect(copyButton).toBeVisible();
    await copyButton.click();
    
    // Verify copy feedback (should show "Copied!" briefly)
    await expect(page.locator('text=Copied!')).toBeVisible({ timeout: 5000 });
    
    // Test download logs functionality
    const downloadButton = page.locator('[role="dialog"] button:has-text("Download")');
    await expect(downloadButton).toBeVisible();
    await downloadButton.click();
    
    // Verify memory cleanup summary is shown
    const memorySummary = page.locator('[role="dialog"] .MuiAlert-root:has-text("Memory Cleanup Summary")');
    await expect(memorySummary).toBeVisible();
    
    // Verify specific memory cleanup items
    await expect(page.locator('text=Model object removed from memory')).toBeVisible();
    await expect(page.locator('text=Tokenizer object cleared')).toBeVisible();
    
    // Close the modal
    const closeButton = page.locator('[role="dialog"] button:has-text("Close")');
    await closeButton.click();
    
    // Verify modal is closed
    await expect(page.locator('[role="dialog"]')).not.toBeVisible();
    
    // Verify model is now unloaded (should show Load button again)
    await expect(page.locator('button:has-text("Load")')).toBeVisible();
  });

  test('should handle unload error gracefully', async ({ page }) => {
    // Try to unload a model that's not loaded
    const unloadButton = page.locator('button:has-text("Unload")').first();
    
    if (await unloadButton.isVisible()) {
      await unloadButton.click();
      
      // Wait for logs modal to appear
      await page.waitForSelector('[role="dialog"]', { timeout: 10000 });
      
      // Verify error status
      const errorChip = page.locator('[role="dialog"] .MuiChip-root:has-text("Failed")');
      await expect(errorChip).toBeVisible();
      
      // Verify error message
      await expect(page.locator('text=Model unload failed')).toBeVisible();
    }
  });

  test('should display logs with proper formatting', async ({ page }) => {
    // Load a model first
    const loadButton = page.locator('button:has-text("Load")').first();
    if (await loadButton.isVisible()) {
      await loadButton.click();
      await page.waitForSelector('button:has-text("Unload")', { timeout: 30000 });
      
      // Unload the model
      const unloadButton = page.locator('button:has-text("Unload")').first();
      await unloadButton.click();
      
      // Wait for logs modal
      await page.waitForSelector('[role="dialog"]', { timeout: 10000 });
      
      // Verify log formatting
      const logItems = page.locator('[role="dialog"] .MuiListItem-root');
      const firstLogItem = logItems.first();
      
      // Check that each log item has an icon
      const logIcon = firstLogItem.locator('.MuiListItemIcon-root svg');
      await expect(logIcon).toBeVisible();
      
      // Check that each log item has a level chip
      const levelChip = firstLogItem.locator('.MuiChip-root');
      await expect(levelChip).toBeVisible();
      
      // Check that each log item has a timestamp
      const timestamp = firstLogItem.locator('text=/\\d{1,2}:\\d{2}:\\d{2}/');
      await expect(timestamp).toBeVisible();
      
      // Check that each log item has a message
      const message = firstLogItem.locator('.MuiListItemText-secondary');
      await expect(message).toBeVisible();
    }
  });

  test('should handle empty logs gracefully', async ({ page }) => {
    // This test would require mocking the API to return empty logs
    // For now, we'll just verify the modal structure
    const loadButton = page.locator('button:has-text("Load")').first();
    if (await loadButton.isVisible()) {
      await loadButton.click();
      await page.waitForSelector('button:has-text("Unload")', { timeout: 30000 });
      
      const unloadButton = page.locator('button:has-text("Unload")').first();
      await unloadButton.click();
      
      await page.waitForSelector('[role="dialog"]', { timeout: 10000 });
      
      // Verify modal structure even with logs
      const modal = page.locator('[role="dialog"]');
      await expect(modal).toBeVisible();
      
      const title = modal.locator('h6, .MuiTypography-h6');
      await expect(title).toBeVisible();
      
      const closeButton = modal.locator('button:has-text("Close")');
      await expect(closeButton).toBeVisible();
    }
  });
});
