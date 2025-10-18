import { test, expect } from '@playwright/test';

test.describe('Load Data Button Test', () => {
  test('should load data when Load Data button is clicked', async ({ page }) => {
    // Set up console logging
    const consoleLogs: string[] = [];
    page.on('console', msg => {
      consoleLogs.push(`${msg.type()}: ${msg.text()}`);
    });
    
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
    
    // Check initial state - should have no service status chips
    const initialServiceChips = await page.locator('text=/\\d+\\/\\d+ Services/').count();
    console.log('Initial service chips:', initialServiceChips);
    expect(initialServiceChips).toBe(0);
    
    // Click the Load Data button
    await page.locator('button:has-text("Load Data")').first().click();
    
    // Wait for data to load
    await page.waitForTimeout(3000);
    
    // Check if service status chips appear
    const serviceChips = await page.locator('text=/\\d+\\/\\d+ Services/').count();
    console.log('Service chips after loading:', serviceChips);
    
    // Should have at least one service status chip
    expect(serviceChips).toBeGreaterThan(0);
    
    // Check the content of the service status
    if (serviceChips > 0) {
      const serviceText = await page.locator('text=/\\d+\\/\\d+ Services/').first().textContent();
      console.log('Service status text:', serviceText);
      
      // Should not show 0/0
      expect(serviceText).not.toContain('0/0');
    }
    
    // Check if metrics are loaded
    const metricCards = await page.locator('[data-testid="metric-card"]').count();
    console.log('Metric cards found:', metricCards);
    expect(metricCards).toBeGreaterThan(0);
    
    // Check console logs for any errors
    const errorLogs = consoleLogs.filter(log => log.includes('error') || log.includes('Error'));
    if (errorLogs.length > 0) {
      console.log('Error logs:', errorLogs);
    }
    
    console.log('All console logs:', consoleLogs);
  });
});
