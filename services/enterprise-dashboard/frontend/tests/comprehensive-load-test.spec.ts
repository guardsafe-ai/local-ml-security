import { test, expect } from '@playwright/test';

test.describe('Comprehensive Load Data Test', () => {
  test('should load all data successfully when Load Data button is clicked', async ({ page }) => {
    // Set up console logging to monitor API calls
    const consoleLogs: string[] = [];
    page.on('console', msg => {
      consoleLogs.push(`${msg.type()}: ${msg.text()}`);
    });
    
    // Set up network monitoring
    const apiCalls: string[] = [];
    page.on('request', request => {
      if (request.url().includes('localhost:8007')) {
        apiCalls.push(`${request.method()} ${request.url()}`);
      }
    });
    
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
    
    console.log('=== INITIAL STATE ===');
    
    // Check initial state - should have no service status chips
    const initialServiceChips = await page.locator('text=/\\d+\\/\\d+ Services/').count();
    console.log('Initial service chips:', initialServiceChips);
    expect(initialServiceChips).toBe(0);
    
    // Check initial metrics - should all be 0
    const loadedModels = await page.locator('text=/\\d+ available/').textContent();
    const activeJobs = await page.locator('text=/\\d+ running/').textContent();
    const redTeamTests = await page.locator('text=/\\d+\\.\\d+% detection rate/').textContent();
    
    console.log('Initial metrics:');
    console.log('- Loaded Models:', loadedModels);
    console.log('- Active Jobs:', activeJobs);
    console.log('- Red Team Tests:', redTeamTests);
    
    // Click the Load Data button
    console.log('=== CLICKING LOAD DATA BUTTON ===');
    await page.locator('button:has-text("Load Data")').first().click();
    
    // Wait for loading to complete
    console.log('=== WAITING FOR DATA TO LOAD ===');
    await page.waitForTimeout(5000);
    
    console.log('=== AFTER LOADING ===');
    
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
      
      // Should show actual service counts
      expect(serviceText).toMatch(/\d+\/\d+ Services/);
    }
    
    // Check if metrics are updated
    const updatedLoadedModels = await page.locator('text=/\\d+ available/').textContent();
    const updatedActiveJobs = await page.locator('text=/\\d+ running/').textContent();
    const updatedRedTeamTests = await page.locator('text=/\\d+\\.\\d+% detection rate/').textContent();
    
    console.log('Updated metrics:');
    console.log('- Loaded Models:', updatedLoadedModels);
    console.log('- Active Jobs:', updatedActiveJobs);
    console.log('- Red Team Tests:', updatedRedTeamTests);
    
    // Check if System Health shows actual data
    const systemHealthValue = await page.locator('[data-testid="system-health-card"]').textContent();
    console.log('System Health card:', systemHealthValue);
    
    // Check for any error alerts
    const errorAlerts = await page.locator('.MuiAlert-root[severity="error"]').count();
    console.log('Error alerts:', errorAlerts);
    
    // Check for success indicators
    const successAlerts = await page.locator('.MuiAlert-root[severity="success"]').count();
    console.log('Success alerts:', successAlerts);
    
    // Check API calls made
    console.log('=== API CALLS MADE ===');
    apiCalls.forEach(call => console.log(call));
    
    // Check console logs for any errors
    const errorLogs = consoleLogs.filter(log => 
      log.includes('error') || 
      log.includes('Error') || 
      log.includes('failed') ||
      log.includes('Failed')
    );
    
    if (errorLogs.length > 0) {
      console.log('=== ERROR LOGS ===');
      errorLogs.forEach(log => console.log(log));
    } else {
      console.log('No error logs found');
    }
    
    // Check for successful data loading indicators
    const hasServiceData = serviceChips > 0;
    const hasUpdatedMetrics = updatedLoadedModels !== loadedModels || 
                             updatedActiveJobs !== activeJobs || 
                             updatedRedTeamTests !== redTeamTests;
    
    console.log('=== LOADING RESULTS ===');
    console.log('Has service data:', hasServiceData);
    console.log('Has updated metrics:', hasUpdatedMetrics);
    console.log('API calls made:', apiCalls.length);
    
    // Verify that data loading was successful
    expect(hasServiceData).toBe(true);
    expect(apiCalls.length).toBeGreaterThan(0);
    
    console.log('=== TEST COMPLETED SUCCESSFULLY ===');
  });
});
