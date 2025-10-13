const puppeteer = require('puppeteer');

async function debugCards() {
  console.log('🔍 Debugging Training Job Cards...');
  
  const browser = await puppeteer.launch({ 
    headless: false, 
    defaultViewport: null,
    args: ['--start-maximized']
  });
  
  try {
    const page = await browser.newPage();
    
    // Enable console logging
    page.on('console', msg => {
      if (msg.text().includes('jobs') || msg.text().includes('filter') || msg.text().includes('card')) {
        console.log(`📊 [${msg.type().toUpperCase()}] ${msg.text()}`);
      }
    });
    
    console.log('🌐 Navigating to training page...');
    await page.goto('http://localhost:3000/training', { 
      waitUntil: 'networkidle0',
      timeout: 30000 
    });
    
    // Wait for page to load
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    console.log('🔍 Checking for training job cards...');
    
    // Check for training job cards
    const cardsInfo = await page.evaluate(() => {
      // Look for training job cards
      const cards = document.querySelectorAll('[data-testid="training-job-card"], .MuiCard-root');
      const gridItems = document.querySelectorAll('.MuiGrid-item');
      
      // Check if there are any cards with job data
      const cardData = Array.from(cards).map((card, index) => ({
        index,
        text: card.textContent.substring(0, 200),
        hasJobId: card.textContent.includes('train_'),
        hasModelName: card.textContent.includes('bert') || card.textContent.includes('distilbert'),
        className: card.className
      }));
      
      // Check grid items
      const gridData = Array.from(gridItems).map((item, index) => ({
        index,
        text: item.textContent.substring(0, 100),
        hasCard: item.querySelector('.MuiCard-root') !== null
      }));
      
      return {
        cardCount: cards.length,
        gridItemCount: gridItems.length,
        cardData,
        gridData: gridData.filter(item => item.hasCard)
      };
    });
    
    console.log('📊 Cards Info:', JSON.stringify(cardsInfo, null, 2));
    
    // Check the actual data being used for rendering
    const renderingData = await page.evaluate(() => {
      // Try to find the React component state
      const reactRoot = document.querySelector('#root');
      
      // Check for any elements that might contain job data
      const allElements = document.querySelectorAll('*');
      const jobElements = Array.from(allElements).filter(el => 
        el.textContent && (
          el.textContent.includes('train_bert-base') || 
          el.textContent.includes('train_distilbert') ||
          el.textContent.includes('bert-base') ||
          el.textContent.includes('distilbert')
        )
      );
      
      return {
        totalElements: allElements.length,
        jobElements: jobElements.length,
        jobElementTexts: jobElements.map(el => el.textContent.substring(0, 100))
      };
    });
    
    console.log('📊 Rendering Data:', JSON.stringify(renderingData, null, 2));
    
    // Check if there are any loading states
    const loadingStates = await page.evaluate(() => {
      const loadingElements = document.querySelectorAll('[data-testid="loading"], .MuiCircularProgress-root, .MuiSkeleton-root');
      const loadingTexts = Array.from(loadingElements).map(el => el.textContent || el.className);
      
      return {
        loadingElementCount: loadingElements.length,
        loadingTexts
      };
    });
    
    console.log('📊 Loading States:', JSON.stringify(loadingStates, null, 2));
    
    // Take a screenshot
    await page.screenshot({ 
      path: 'training-cards-debug.png', 
      fullPage: true 
    });
    
    console.log('✅ Debug completed! Check training-cards-debug.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

debugCards().catch(console.error);
