import { test, expect } from '@playwright/test';

test.describe('Final Model Sizes Test', () => {
  test('should display actual model sizes from Hugging Face Hub API', async ({ page }) => {
    // Navigate to models page
    await page.goto('http://localhost:3000/models');
    
    // Wait for models to load
    await page.waitForSelector('[data-testid="model-card"]', { timeout: 10000 });
    
    // Get all model cards
    const modelCards = await page.locator('[data-testid="model-card"]').all();
    expect(modelCards.length).toBe(4); // Should have 4 models
    
    // Check that we have actual sizes (not estimates)
    const firstCard = modelCards[0];
    
    // Look for size information in the model card
    const sizeText = await firstCard.locator('text=/\\d+\\.\\d+ GB/').first().textContent();
    expect(sizeText).toBeTruthy();
    
    // Verify it's a reasonable size (actual from Hugging Face Hub)
    const sizeValue = parseFloat(sizeText!.replace(' GB', ''));
    expect(sizeValue).toBeGreaterThan(0);
    expect(sizeValue).toBeLessThan(10); // Reasonable upper bound
    
    console.log(`First model size: ${sizeText}`);
    
    // Check that we have different sizes for different models
    const allSizes = await page.locator('text=/\\d+\\.\\d+ GB/').allTextContents();
    const uniqueSizes = [...new Set(allSizes)];
    expect(uniqueSizes.length).toBe(4); // Should have 4 different sizes
    
    console.log('All model sizes:', allSizes);
    
    // Verify specific model sizes are displayed
    expect(allSizes).toContain('0.062385136261582375 GB'); // distilbert
    expect(allSizes).toContain('0.10254460200667381 GB');  // bert-base
    expect(allSizes).toContain('0.11613401304930449 GB');  // roberta-base
    expect(allSizes).toContain('3.44789981842041 GB');     // deberta-v3-base
  });
  
  test('should show model source and status information', async ({ page }) => {
    await page.goto('http://localhost:3000/models');
    await page.waitForSelector('[data-testid="model-card"]', { timeout: 10000 });
    
    // Check that source information is displayed
    const sourceText = await page.locator('text=Hugging Face').first().textContent();
    expect(sourceText).toBeTruthy();
    
    // Check that status information is displayed
    const statusText = await page.locator('text=available').first().textContent();
    expect(statusText).toBeTruthy();
  });
  
  test('should display model information in correct format', async ({ page }) => {
    await page.goto('http://localhost:3000/models');
    await page.waitForSelector('[data-testid="model-card"]', { timeout: 10000 });
    
    const firstCard = page.locator('[data-testid="model-card"]').first();
    
    // Check that the model information section shows:
    // - Size in GB format
    // - Version information
    // - Source information
    // - Status information
    
    await expect(firstCard.locator('text=/Size/')).toBeVisible();
    await expect(firstCard.locator('text=/\\d+\\.\\d+ GB/')).toBeVisible();
    await expect(firstCard.locator('text=/Version/')).toBeVisible();
    await expect(firstCard.locator('text=/Source/')).toBeVisible();
    await expect(firstCard.locator('text=/Status/')).toBeVisible();
  });
});
