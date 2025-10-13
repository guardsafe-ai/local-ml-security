const puppeteer = require('puppeteer');

async function testChartDataDirect() {
  console.log('🔍 Testing Chart Data Direct...');
  
  const browser = await puppeteer.launch({ 
    headless: false, 
    defaultViewport: null,
    args: ['--start-maximized']
  });
  
  try {
    const page = await browser.newPage();
    
    // Enable all console logging
    page.on('console', msg => {
      const text = msg.text();
      console.log(`📊 [${msg.type().toUpperCase()}] ${text}`);
    });
    
    console.log('🌐 Navigating to training page...');
    await page.goto('http://localhost:3000/training', { 
      waitUntil: 'networkidle0',
      timeout: 30000 
    });
    
    // Wait for data to load
    console.log('⏳ Waiting for data to load...');
    await new Promise(resolve => setTimeout(resolve, 15000));
    
    // Click on Performance Analytics tab
    console.log('🎯 Clicking on Performance Analytics tab...');
    await page.click('button[role="tab"]:nth-child(3)');
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Check chart data directly
    const chartData = await page.evaluate(() => {
      const charts = document.querySelectorAll('.recharts-wrapper');
      const chartData = [];
      
      charts.forEach((chart, index) => {
        const svg = chart.querySelector('svg');
        if (svg) {
          // Check for data points
          const circles = svg.querySelectorAll('circle');
          const rects = svg.querySelectorAll('rect[class*="bar"]');
          const paths = svg.querySelectorAll('path[class*="line"]');
          const texts = svg.querySelectorAll('text');
          
          // Check for axis labels
          const xAxisLabels = svg.querySelectorAll('text[class*="tick"]');
          const yAxisLabels = svg.querySelectorAll('text[class*="tick"]');
          
          chartData.push({
            index,
            circles: circles.length,
            rects: rects.length,
            paths: paths.length,
            texts: texts.length,
            xAxisLabels: xAxisLabels.length,
            yAxisLabels: yAxisLabels.length,
            hasData: circles.length > 0 || rects.length > 0 || paths.length > 0,
            svgContent: svg.innerHTML.substring(0, 500)
          });
        }
      });
      
      return {
        chartCount: charts.length,
        chartData: chartData,
        allTexts: Array.from(document.querySelectorAll('text')).map(t => t.textContent).filter(t => t && t.trim().length > 0)
      };
    });
    
    console.log('📊 Chart Data Analysis:', JSON.stringify(chartData, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'test-chart-data-direct.png', 
      fullPage: true 
    });
    
    console.log('✅ Test completed! Check test-chart-data-direct.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testChartDataDirect().catch(console.error);
