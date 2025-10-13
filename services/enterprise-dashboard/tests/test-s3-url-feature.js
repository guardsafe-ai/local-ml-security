const puppeteer = require('puppeteer');

async function testS3UrlFeature() {
  console.log('🚀 Testing S3 URL Feature...');
  
  const browser = await puppeteer.launch({ 
    headless: false, 
    defaultViewport: null,
    args: ['--start-maximized']
  });
  
  const page = await browser.newPage();
  
  try {
    console.log('📍 Navigating to Data Management page...');
    await page.goto('http://localhost:3000/data-management', { 
      waitUntil: 'networkidle2',
      timeout: 30000 
    });
    
    console.log('⏳ Waiting for page to load...');
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Test the API response to see if S3 URLs are included
    console.log('🔍 Testing API response for S3 URLs...');
    const apiTest = await page.evaluate(async () => {
      try {
        const response = await fetch('http://localhost:8002/data/efficient/staged-files');
        const data = await response.json();
        
        if (data.files && data.files.length > 0) {
          const firstFile = data.files[0];
          return {
            success: true,
            hasS3Url: !!firstFile.s3_url,
            s3Url: firstFile.s3_url,
            minioPath: firstFile.minio_path,
            fileName: firstFile.original_name
          };
        } else {
          return { success: false, error: 'No files found' };
        }
      } catch (error) {
        return { success: false, error: error.message };
      }
    });
    
    console.log('📊 API Test Results:');
    if (apiTest.success) {
      console.log(`   ✅ API Success: ${apiTest.success}`);
      console.log(`   📁 File Name: ${apiTest.fileName}`);
      console.log(`   📂 MinIO Path: ${apiTest.minioPath}`);
      console.log(`   🔗 S3 URL: ${apiTest.s3Url}`);
      console.log(`   ✅ Has S3 URL: ${apiTest.hasS3Url}`);
      
      if (apiTest.hasS3Url) {
        console.log('🎉 SUCCESS: S3 URLs are included in API response!');
      } else {
        console.log('❌ ISSUE: S3 URLs not found in API response');
      }
    } else {
      console.log(`   ❌ API Error: ${apiTest.error}`);
    }
    
    // Test the UI to see if S3 URLs are displayed
    console.log('🔍 Testing UI for S3 URL display...');
    const uiTest = await page.evaluate(() => {
      // Look for S3 URL column header
      const headers = Array.from(document.querySelectorAll('th, .MuiTableCell-head'));
      const s3UrlHeader = headers.find(header => 
        header.textContent?.toLowerCase().includes('s3') || 
        header.textContent?.toLowerCase().includes('url')
      );
      
      // Look for S3 URL content in table cells
      const cells = Array.from(document.querySelectorAll('td, .MuiTableCell-body'));
      const s3UrlCells = cells.filter(cell => 
        cell.textContent?.includes('s3://') || 
        cell.textContent?.includes('ml-security')
      );
      
      // Look for copy buttons
      const copyButtons = Array.from(document.querySelectorAll('button'));
      const copyButton = copyButtons.find(button => 
        button.getAttribute('title')?.toLowerCase().includes('copy') ||
        button.querySelector('[data-testid="ContentCopyIcon"]') ||
        button.querySelector('svg[data-testid="ContentCopyIcon"]')
      );
      
      return {
        hasS3UrlHeader: !!s3UrlHeader,
        s3UrlHeaderText: s3UrlHeader?.textContent,
        s3UrlCellsCount: s3UrlCells.length,
        s3UrlCells: s3UrlCells.map(cell => cell.textContent?.trim()).slice(0, 3),
        hasCopyButton: !!copyButton,
        copyButtonTitle: copyButton?.getAttribute('title')
      };
    });
    
    console.log('📊 UI Test Results:');
    console.log(`   ✅ Has S3 URL Header: ${uiTest.hasS3UrlHeader}`);
    console.log(`   📋 Header Text: ${uiTest.s3UrlHeaderText}`);
    console.log(`   📊 S3 URL Cells Found: ${uiTest.s3UrlCellsCount}`);
    console.log(`   🔗 Sample S3 URLs:`, uiTest.s3UrlCells);
    console.log(`   📋 Has Copy Button: ${uiTest.hasCopyButton}`);
    console.log(`   📋 Copy Button Title: ${uiTest.copyButtonTitle}`);
    
    if (uiTest.hasS3UrlHeader && uiTest.s3UrlCellsCount > 0) {
      console.log('🎉 SUCCESS: S3 URLs are displayed in the UI!');
    } else {
      console.log('❌ ISSUE: S3 URLs not displayed in UI');
    }
    
    if (uiTest.hasCopyButton) {
      console.log('🎉 SUCCESS: Copy buttons are present!');
    } else {
      console.log('❌ ISSUE: Copy buttons not found');
    }
    
    // Test copy functionality
    console.log('🔍 Testing copy functionality...');
    const copyTest = await page.evaluate(() => {
      // Find a copy button and simulate click
      const copyButtons = Array.from(document.querySelectorAll('button'));
      const copyButton = copyButtons.find(button => 
        button.getAttribute('title')?.toLowerCase().includes('copy')
      );
      
      if (copyButton) {
        // Simulate click
        copyButton.click();
        return { success: true, message: 'Copy button clicked' };
      } else {
        return { success: false, message: 'Copy button not found' };
      }
    });
    
    console.log('📊 Copy Test Results:');
    console.log(`   ${copyTest.success ? '✅' : '❌'} ${copyTest.message}`);
    
    // Take a screenshot
    console.log('📸 Taking screenshot...');
    await page.screenshot({ 
      path: 's3-url-feature-test.png', 
      fullPage: true 
    });
    
    console.log('✅ S3 URL feature testing completed');
    
  } catch (error) {
    console.error('❌ Error during testing:', error);
  } finally {
    await browser.close();
  }
}

testS3UrlFeature().catch(console.error);
