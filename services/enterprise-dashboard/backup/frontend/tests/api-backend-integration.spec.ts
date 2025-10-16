import { test, expect } from '@playwright/test';

/**
 * API Backend Integration Tests
 * Tests the frontend-backend API integration and verifies data flow
 */

test.describe('API Backend Integration', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the dashboard before each test
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('Dashboard loads with real data from backend', async ({ page }) => {
    // Check if dashboard metrics are loaded
    await expect(page.locator('[data-testid="dashboard-metrics"]')).toBeVisible();
    
    // Verify API calls are made successfully
    const response = await page.waitForResponse(response => 
      response.url().includes('/dashboard/metrics') && response.status() === 200
    );
    
    const data = await response.json();
    expect(data).toHaveProperty('total_models');
    expect(data).toHaveProperty('active_jobs');
    expect(data).toHaveProperty('total_attacks');
    expect(data).toHaveProperty('system_health');
  });

  test('Models page loads with real model data', async ({ page }) => {
    // Navigate to models page
    await page.click('[data-testid="nav-models"]');
    await page.waitForLoadState('networkidle');
    
    // Wait for models API call
    const response = await page.waitForResponse(response => 
      response.url().includes('/models/available') && response.status() === 200
    );
    
    const data = await response.json();
    expect(data).toHaveProperty('models');
    expect(data).toHaveProperty('available_models');
    
    // Verify models are displayed
    await expect(page.locator('[data-testid="models-list"]')).toBeVisible();
  });

  test('Red Team page loads with test results', async ({ page }) => {
    // Navigate to red team page
    await page.click('[data-testid="nav-red-team"]');
    await page.waitForLoadState('networkidle');
    
    // Wait for red team results API call
    const response = await page.waitForResponse(response => 
      response.url().includes('/red-team/results') && response.status() === 200
    );
    
    const data = await response.json();
    expect(data).toHaveProperty('test_id');
    expect(data).toHaveProperty('total_attacks');
    expect(data).toHaveProperty('results');
    
    // Verify red team data is displayed
    await expect(page.locator('[data-testid="red-team-results"]')).toBeVisible();
  });

  test('Analytics page loads with summary data', async ({ page }) => {
    // Navigate to analytics page
    await page.click('[data-testid="nav-analytics"]');
    await page.waitForLoadState('networkidle');
    
    // Wait for analytics API call
    const response = await page.waitForResponse(response => 
      response.url().includes('/analytics/summary') && response.status() === 200
    );
    
    const data = await response.json();
    expect(data).toHaveProperty('summary');
    
    // Verify analytics data is displayed
    await expect(page.locator('[data-testid="analytics-summary"]')).toBeVisible();
  });

  test('Training page loads with job data', async ({ page }) => {
    // Navigate to training page
    await page.click('[data-testid="nav-training"]');
    await page.waitForLoadState('networkidle');
    
    // Wait for training jobs API call
    const response = await page.waitForResponse(response => 
      response.url().includes('/training/jobs') && response.status() === 200
    );
    
    const data = await response.json();
    expect(Array.isArray(data)).toBe(true);
    
    // Verify training data is displayed
    await expect(page.locator('[data-testid="training-jobs"]')).toBeVisible();
  });

  test('MLflow page loads with experiment data', async ({ page }) => {
    // Navigate to MLflow page
    await page.click('[data-testid="nav-mlflow"]');
    await page.waitForLoadState('networkidle');
    
    // Wait for MLflow API call
    const response = await page.waitForResponse(response => 
      response.url().includes('/mlflow/experiments') && response.status() === 200
    );
    
    const data = await response.json();
    expect(Array.isArray(data)).toBe(true);
    
    // Verify MLflow data is displayed
    await expect(page.locator('[data-testid="mlflow-experiments"]')).toBeVisible();
  });

  test('Error handling for failed API calls', async ({ page }) => {
    // Mock a failed API response
    await page.route('**/dashboard/metrics', route => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal Server Error' })
      });
    });

    // Reload page to trigger the failed API call
    await page.reload();
    await page.waitForLoadState('networkidle');
    
    // Verify error handling
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
  });

  test('Real-time data updates work correctly', async ({ page }) => {
    // Get initial data
    const initialResponse = await page.waitForResponse(response => 
      response.url().includes('/dashboard/metrics') && response.status() === 200
    );
    const initialData = await initialResponse.json();
    
    // Wait for refresh interval (10 seconds for dashboard metrics)
    await page.waitForTimeout(11000);
    
    // Check if data was refreshed
    const refreshResponse = await page.waitForResponse(response => 
      response.url().includes('/dashboard/metrics') && response.status() === 200
    );
    const refreshData = await refreshResponse.json();
    
    // Data should be refreshed (timestamp should be different)
    expect(refreshData.last_updated).not.toBe(initialData.last_updated);
  });
});
