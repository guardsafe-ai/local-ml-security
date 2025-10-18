import { test, expect, Page } from '@playwright/test';

test.describe('Simple Debug Tests', () => {
  test('should load the dashboard page', async ({ page }) => {
    // Set up console logging before navigation
    const consoleLogs: string[] = [];
    page.on('console', msg => {
      consoleLogs.push(`${msg.type()}: ${msg.text()}`);
    });
    
    await page.goto('http://localhost:3000');
    
    // Wait for the page to load
    await page.waitForLoadState('networkidle');
    
    // Take a screenshot for debugging
    await page.screenshot({ path: 'debug-screenshot.png' });
    
    // Check if the page title contains ML Security
    const title = await page.title();
    console.log('Page title:', title);
    
    // Check if there's any text containing "Load Data"
    const loadDataText = await page.locator('text=Load Data').first();
    const isVisible = await loadDataText.isVisible();
    console.log('Load Data button visible:', isVisible);
    
    // Check for any buttons
    const buttons = await page.locator('button').count();
    console.log('Number of buttons found:', buttons);
    
    // List all button texts
    for (let i = 0; i < buttons; i++) {
      const buttonText = await page.locator('button').nth(i).textContent();
      console.log(`Button ${i}:`, buttonText);
    }
    
    // Check for the test ID
    const testIdButton = await page.locator('[data-testid="load-data-button"]').count();
    console.log('Buttons with test ID:', testIdButton);
    
    // Check for service status chips
    const serviceChips = await page.locator('text=/\\d+\\/\\d+ Services/').count();
    console.log('Service status chips found:', serviceChips);
    
    // List all service status texts
    for (let i = 0; i < serviceChips; i++) {
      const serviceText = await page.locator('text=/\\d+\\/\\d+ Services/').nth(i).textContent();
      console.log(`Service Status ${i}:`, serviceText);
    }
    
    // Wait a bit to collect console logs
    await page.waitForTimeout(3000);
    
    console.log('Console logs:', consoleLogs);
  });
});
