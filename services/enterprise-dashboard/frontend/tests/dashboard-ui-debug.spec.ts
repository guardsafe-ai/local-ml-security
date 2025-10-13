import { test, expect, Page } from '@playwright/test';

test.describe('Dashboard UI Debug Tests', () => {
  let page: Page;

  test.beforeEach(async ({ page: testPage }) => {
    page = testPage;
    await page.goto('/');
    
    // Wait for the page to load
    await page.waitForLoadState('networkidle');
  });

  test('should display dashboard with proper service status', async () => {
    // Check if the dashboard loads
    await expect(page.locator('h1')).toContainText('ML Security Dashboard');
    
    // Check for the Load Data button - use the first one found
    const loadDataButton = page.locator('button:has-text("Load Data")').first();
    await expect(loadDataButton).toBeVisible();
    
    // Click the Load Data button to fetch data
    await loadDataButton.click();
    
    // Wait for loading to complete
    await page.waitForTimeout(2000);
    
    // Check console logs for debugging
    const consoleLogs: string[] = [];
    page.on('console', msg => {
      consoleLogs.push(`${msg.type()}: ${msg.text()}`);
    });
    
    // Check if service status is displayed correctly
    const serviceStatusChip = page.locator('[data-testid="service-status-chip"]');
    if (await serviceStatusChip.isVisible()) {
      const serviceText = await serviceStatusChip.textContent();
      console.log('Service Status Text:', serviceText);
      
      // Should not show 0/0 Services
      expect(serviceText).not.toContain('0/0');
    }
    
    // Check for debug information in the test counter
    const testCounter = page.locator('text=Test Counter:');
    if (await testCounter.isVisible()) {
      const counterText = await testCounter.textContent();
      console.log('Test Counter Text:', counterText);
    }
    
    // Check if metrics are loaded
    const metricsCards = page.locator('[data-testid="metric-card"]');
    const metricsCount = await metricsCards.count();
    console.log('Number of metric cards:', metricsCount);
    
    // Check for any error alerts
    const errorAlerts = page.locator('.MuiAlert-root[severity="error"]');
    const errorCount = await errorAlerts.count();
    if (errorCount > 0) {
      for (let i = 0; i < errorCount; i++) {
        const errorText = await errorAlerts.nth(i).textContent();
        console.log('Error Alert:', errorText);
      }
    }
    
    // Check for info alerts
    const infoAlerts = page.locator('.MuiAlert-root[severity="info"]');
    const infoCount = await infoAlerts.count();
    if (infoCount > 0) {
      for (let i = 0; i < infoCount; i++) {
        const infoText = await infoAlerts.nth(i).textContent();
        console.log('Info Alert:', infoText);
      }
    }
    
    // Log all console messages
    console.log('Console Logs:', consoleLogs);
  });

  test('should handle API errors gracefully', async () => {
    // Mock API responses to simulate errors
    await page.route('**/services/health', route => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Service unavailable' })
      });
    });
    
    await page.route('**/dashboard/metrics', route => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Metrics unavailable' })
      });
    });
    
    // Click Load Data button
    await page.locator('button:has-text("Load Data")').first().click();
    
    // Wait for error handling
    await page.waitForTimeout(2000);
    
    // Check if error is handled gracefully
    const errorAlerts = page.locator('.MuiAlert-root[severity="error"]');
    const errorCount = await errorAlerts.count();
    
    if (errorCount > 0) {
      console.log('Error alerts found:', errorCount);
      for (let i = 0; i < errorCount; i++) {
        const errorText = await errorAlerts.nth(i).textContent();
        console.log('Error Alert:', errorText);
      }
    }
  });

  test('should display service status correctly when data is available', async () => {
    // Mock successful API responses
    await page.route('**/services/health', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          { service: 'model-api', status: 'healthy', uptime_seconds: 3600 },
          { service: 'training-service', status: 'healthy', uptime_seconds: 7200 },
          { service: 'red-team-service', status: 'degraded', uptime_seconds: 1800 }
        ])
      });
    });
    
    await page.route('**/dashboard/metrics', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          total_models: 5,
          system_health: 85,
          detection_rate: 92.5,
          active_jobs: 2,
          total_attacks: 15
        })
      });
    });
    
    // Click Load Data button
    await page.locator('button:has-text("Load Data")').first().click();
    
    // Wait for data to load
    await page.waitForTimeout(2000);
    
    // Check service status display
    const serviceStatusText = page.locator('text=/\\d+\\/\\d+ Services/');
    await expect(serviceStatusText).toBeVisible();
    
    const statusText = await serviceStatusText.textContent();
    console.log('Service Status:', statusText);
    
    // Should show 2/3 or 3/3 services (not 0/0)
    expect(statusText).toMatch(/\d+\/\d+ Services/);
    expect(statusText).not.toContain('0/0');
    
    // Check if metrics are displayed
    const modelCount = page.locator('text=/\\d+ available/');
    await expect(modelCount).toBeVisible();
  });

  test('should show loading states properly', async () => {
    // Mock slow API responses
    await page.route('**/services/health', route => {
      setTimeout(() => {
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([
            { service: 'test-service', status: 'healthy', uptime_seconds: 3600 }
          ])
        });
      }, 1000);
    });
    
    // Click Load Data button
    await page.locator('button:has-text("Load Data")').first().click();
    
    // Check if loading state is shown
    const loadingButton = page.locator('button:has-text("Loading...")');
    await expect(loadingButton).toBeVisible();
    
    // Wait for loading to complete
    await page.waitForTimeout(2000);
    
    // Check if loading state is removed
    await expect(loadingButton).not.toBeVisible();
  });

  test('should handle empty service data correctly', async () => {
    // Mock empty service response
    await page.route('**/services/health', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([])
      });
    });
    
    // Click Load Data button
    await page.locator('button:has-text("Load Data")').first().click();
    
    // Wait for data to load
    await page.waitForTimeout(2000);
    
    // Service status should not show 0/0
    const serviceStatusChip = page.locator('[data-testid="service-status-chip"]');
    const isVisible = await serviceStatusChip.isVisible();
    
    if (isVisible) {
      const statusText = await serviceStatusChip.textContent();
      console.log('Empty services status:', statusText);
      // Should either not show or show a meaningful message
      expect(statusText).not.toContain('0/0');
    }
  });
});
