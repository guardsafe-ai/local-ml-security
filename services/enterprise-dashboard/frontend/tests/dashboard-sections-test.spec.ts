import { test, expect } from '@playwright/test';

test.describe('Dashboard Sections Test', () => {
  test('should display Model Performance Comparison and Service Status sections', async ({ page }) => {
    // Navigate to the dashboard
    await page.goto('http://localhost:3000');
    
    // Wait for the page to load
    await page.waitForLoadState('networkidle');
    
    // Click Load Data button
    await page.click('[data-testid="load-data-button"]');
    
    // Wait for data to load
    await page.waitForTimeout(3000);
    
    // Check Model Performance Comparison section
    const modelPerformanceSection = page.locator('text=Model Performance Comparison').first();
    await expect(modelPerformanceSection).toBeVisible();
    
    // Check if the chart has data (bars should be visible)
    const chartBars = page.locator('.recharts-bar-rectangle');
    await expect(chartBars).toHaveCount(16); // 4 models × 4 metrics = 16 bars
    
    // Check if model names are displayed
    await expect(page.locator('text=distilbert')).toBeVisible();
    await expect(page.locator('text=bert-base')).toBeVisible();
    await expect(page.locator('text=roberta-base')).toBeVisible();
    await expect(page.locator('text=deberta-v3-base')).toBeVisible();
    
    // Check Service Status section
    const serviceStatusSection = page.locator('text=Service Status').first();
    await expect(serviceStatusSection).toBeVisible();
    
    // Check if services are listed (should have 10 services)
    const serviceItems = page.locator('[data-testid="service-status-card"]');
    await expect(serviceItems).toHaveCount(10);
    
    // Check for specific service names in the Service Status section
    const serviceStatusCard = page.locator('text=Service Status').first().locator('..').locator('..');
    await expect(serviceStatusCard.locator('text=training')).toBeVisible();
    await expect(serviceStatusCard.locator('text=model_api')).toBeVisible();
    await expect(serviceStatusCard.locator('text=red_team')).toBeVisible();
    await expect(serviceStatusCard.locator('text=analytics')).toBeVisible();
    await expect(serviceStatusCard.locator('text=business_metrics')).toBeVisible();
    await expect(serviceStatusCard.locator('text=data_privacy')).toBeVisible();
    await expect(serviceStatusCard.locator('text=mlflow')).toBeVisible();
    await expect(serviceStatusCard.locator('text=minio')).toBeVisible();
    await expect(serviceStatusCard.locator('text=prometheus')).toBeVisible();
    await expect(serviceStatusCard.locator('text=grafana')).toBeVisible();
    
    console.log('✅ Model Performance Comparison section is populated');
    console.log('✅ Service Status section is populated with all 10 services');
  });
});
