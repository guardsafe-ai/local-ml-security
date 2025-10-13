import { test, expect } from '@playwright/test';

/**
 * User Workflow Tests
 * Tests complete user workflows and interactions
 */

test.describe('User Workflows', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('Complete model management workflow', async ({ page }) => {
    // Navigate to models page
    await page.click('[data-testid="nav-models"]');
    await page.waitForLoadState('networkidle');
    
    // Wait for models to load
    await expect(page.locator('[data-testid="models-list"]')).toBeVisible();
    
    // Test model loading
    const loadButton = page.locator('[data-testid="load-model-btn"]').first();
    if (await loadButton.isVisible()) {
      await loadButton.click();
      
      // Wait for API call
      await page.waitForResponse(response => 
        response.url().includes('/models/load') && response.status() === 200
      );
      
      // Verify model status updated
      await expect(page.locator('[data-testid="model-status"]').first()).toContainText('Loaded');
    }
    
    // Test model prediction
    const predictButton = page.locator('[data-testid="predict-btn"]').first();
    if (await predictButton.isVisible()) {
      await predictButton.click();
      
      // Fill prediction form
      await page.fill('[data-testid="prediction-text"]', 'Test prompt injection attack');
      await page.click('[data-testid="submit-prediction"]');
      
      // Wait for prediction API call
      await page.waitForResponse(response => 
        response.url().includes('/models/predict') && response.status() === 200
      );
      
      // Verify prediction result
      await expect(page.locator('[data-testid="prediction-result"]')).toBeVisible();
    }
  });

  test('Complete red team testing workflow', async ({ page }) => {
    // Navigate to red team page
    await page.click('[data-testid="nav-red-team"]');
    await page.waitForLoadState('networkidle');
    
    // Wait for red team data to load
    await expect(page.locator('[data-testid="red-team-results"]')).toBeVisible();
    
    // Test running a new red team test
    const runTestButton = page.locator('[data-testid="run-test-btn"]');
    if (await runTestButton.isVisible()) {
      await runTestButton.click();
      
      // Fill test configuration
      await page.fill('[data-testid="test-count"]', '5');
      await page.check('[data-testid="attack-prompt-injection"]');
      await page.check('[data-testid="attack-jailbreak"]');
      
      // Submit test
      await page.click('[data-testid="submit-test"]');
      
      // Wait for test to start
      await page.waitForResponse(response => 
        response.url().includes('/red-team/test') && response.status() === 200
      );
      
      // Verify test is running
      await expect(page.locator('[data-testid="test-status"]')).toContainText('Running');
      
      // Wait for test to complete (with timeout)
      await page.waitForResponse(response => 
        response.url().includes('/red-team/results') && response.status() === 200,
        { timeout: 30000 }
      );
      
      // Verify test completed
      await expect(page.locator('[data-testid="test-status"]')).toContainText('Completed');
    }
  });

  test('Complete training workflow', async ({ page }) => {
    // Navigate to training page
    await page.click('[data-testid="nav-training"]');
    await page.waitForLoadState('networkidle');
    
    // Wait for training data to load
    await expect(page.locator('[data-testid="training-jobs"]')).toBeVisible();
    
    // Test starting a new training job
    const startTrainingButton = page.locator('[data-testid="start-training-btn"]');
    if (await startTrainingButton.isVisible()) {
      await startTrainingButton.click();
      
      // Fill training configuration
      await page.selectOption('[data-testid="model-select"]', 'distilbert-base-uncased');
      await page.fill('[data-testid="epochs"]', '3');
      await page.fill('[data-testid="batch-size"]', '16');
      await page.fill('[data-testid="learning-rate"]', '0.001');
      
      // Submit training
      await page.click('[data-testid="submit-training"]');
      
      // Wait for training to start
      await page.waitForResponse(response => 
        response.url().includes('/training/start') && response.status() === 200
      );
      
      // Verify training started
      await expect(page.locator('[data-testid="training-status"]')).toContainText('Running');
      
      // Check training progress
      await expect(page.locator('[data-testid="training-progress"]')).toBeVisible();
    }
  });

  test('Complete analytics workflow', async ({ page }) => {
    // Navigate to analytics page
    await page.click('[data-testid="nav-analytics"]');
    await page.waitForLoadState('networkidle');
    
    // Wait for analytics data to load
    await expect(page.locator('[data-testid="analytics-summary"]')).toBeVisible();
    
    // Test filtering by date range
    const dateFilter = page.locator('[data-testid="date-filter"]');
    if (await dateFilter.isVisible()) {
      await dateFilter.click();
      await page.selectOption('[data-testid="date-range"]', '30');
      
      // Wait for filtered data
      await page.waitForResponse(response => 
        response.url().includes('/analytics/summary') && response.status() === 200
      );
      
      // Verify filtered data is displayed
      await expect(page.locator('[data-testid="analytics-summary"]')).toBeVisible();
    }
    
    // Test model comparison
    const compareButton = page.locator('[data-testid="compare-models-btn"]');
    if (await compareButton.isVisible()) {
      await compareButton.click();
      
      // Select models to compare
      await page.check('[data-testid="model-distilbert"]');
      await page.check('[data-testid="model-bert"]');
      
      // Submit comparison
      await page.click('[data-testid="submit-comparison"]');
      
      // Wait for comparison API call
      await page.waitForResponse(response => 
        response.url().includes('/analytics/model/comparison') && response.status() === 200
      );
      
      // Verify comparison results
      await expect(page.locator('[data-testid="comparison-results"]')).toBeVisible();
    }
  });

  test('Navigation and page transitions work correctly', async ({ page }) => {
    const pages = [
      { nav: '[data-testid="nav-dashboard"]', content: '[data-testid="dashboard-metrics"]' },
      { nav: '[data-testid="nav-models"]', content: '[data-testid="models-list"]' },
      { nav: '[data-testid="nav-training"]', content: '[data-testid="training-jobs"]' },
      { nav: '[data-testid="nav-red-team"]', content: '[data-testid="red-team-results"]' },
      { nav: '[data-testid="nav-analytics"]', content: '[data-testid="analytics-summary"]' },
      { nav: '[data-testid="nav-mlflow"]', content: '[data-testid="mlflow-experiments"]' },
      { nav: '[data-testid="nav-monitoring"]', content: '[data-testid="monitoring-alerts"]' },
      { nav: '[data-testid="nav-privacy"]', content: '[data-testid="privacy-status"]' },
      { nav: '[data-testid="nav-business"]', content: '[data-testid="business-metrics"]' },
      { nav: '[data-testid="nav-settings"]', content: '[data-testid="settings-form"]' }
    ];

    for (const pageTest of pages) {
      // Click navigation
      await page.click(pageTest.nav);
      await page.waitForLoadState('networkidle');
      
      // Verify content loads
      await expect(page.locator(pageTest.content)).toBeVisible();
      
      // Verify URL changed
      expect(page.url()).toContain(pageTest.nav.replace('[data-testid="nav-', '/').replace('"]', ''));
    }
  });

  test('Error handling and recovery', async ({ page }) => {
    // Test network error handling
    await page.route('**/dashboard/metrics', route => {
      route.abort('failed');
    });

    await page.reload();
    await page.waitForLoadState('networkidle');
    
    // Verify error message is shown
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    
    // Test retry functionality
    const retryButton = page.locator('[data-testid="retry-btn"]');
    if (await retryButton.isVisible()) {
      // Restore network
      await page.unroute('**/dashboard/metrics');
      
      await retryButton.click();
      await page.waitForLoadState('networkidle');
      
      // Verify data loads after retry
      await expect(page.locator('[data-testid="dashboard-metrics"]')).toBeVisible();
    }
  });
});
