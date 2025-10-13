import { test, expect } from '@playwright/test';

test.describe('Models Page Test', () => {
  test('should check what is displayed on the Models page', async ({ page }) => {
    // Navigate to the models page
    await page.goto('http://localhost:3000/models');
    
    // Wait for the page to load
    await page.waitForLoadState('networkidle');
    
    // Take a screenshot for debugging
    await page.screenshot({ path: 'models-page-debug.png' });
    
    // Check if the page title exists
    const pageTitle = page.locator('h1, h2, h3, h4, h5, h6').first();
    const hasTitle = await pageTitle.isVisible();
    console.log('Has page title:', hasTitle);
    if (hasTitle) {
      const titleText = await pageTitle.textContent();
      console.log('Page title:', titleText);
    }
    
    // Check for any loading indicators
    const loadingIndicators = page.locator('text=Loading, text=loading, [data-testid*="loading"], [data-testid*="skeleton"]');
    const loadingCount = await loadingIndicators.count();
    console.log('Loading indicators found:', loadingCount);
    
    // Check for any error messages
    const errorMessages = page.locator('text=Error, text=error, [role="alert"]');
    const errorCount = await errorMessages.count();
    console.log('Error messages found:', errorCount);
    
    // Check for any cards or content sections
    const cards = page.locator('[class*="Card"], [class*="card"], .MuiCard-root');
    const cardCount = await cards.count();
    console.log('Cards found:', cardCount);
    
    // Check for any lists or tables
    const lists = page.locator('ul, ol, table, [class*="List"], [class*="Table"]');
    const listCount = await lists.count();
    console.log('Lists/Tables found:', listCount);
    
    // Check for any buttons
    const buttons = page.locator('button, [role="button"]');
    const buttonCount = await buttons.count();
    console.log('Buttons found:', buttonCount);
    
    // Check for any text content
    const bodyText = await page.locator('body').textContent();
    console.log('Page body text length:', bodyText?.length || 0);
    console.log('First 500 characters of body text:', bodyText?.substring(0, 500));
    
    // Check if there are any console errors
    const consoleMessages: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleMessages.push(`ERROR: ${msg.text()}`);
      }
    });
    
    // Wait a bit for any async operations
    await page.waitForTimeout(3000);
    
    // Print console errors
    if (consoleMessages.length > 0) {
      console.log('Console errors found:');
      consoleMessages.forEach(msg => console.log(msg));
    }
    
    // Check if the page is completely empty
    const isEmpty = cardCount === 0 && listCount === 0 && buttonCount === 0;
    console.log('Page appears empty:', isEmpty);
  });
});
