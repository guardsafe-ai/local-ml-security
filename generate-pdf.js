const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

async function generatePDF() {
    try {
        console.log('🚀 Starting PDF generation...');
        
        // Launch browser
        const browser = await puppeteer.launch({
            headless: true,
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });
        
        const page = await browser.newPage();
        
        // Set viewport for consistent rendering
        await page.setViewport({
            width: 1200,
            height: 800,
            deviceScaleFactor: 2
        });
        
        // Read HTML file
        const htmlPath = path.join(__dirname, 'EMAIL_NEWSLETTER.html');
        const htmlContent = fs.readFileSync(htmlPath, 'utf8');
        
        // Set content
        await page.setContent(htmlContent, {
            waitUntil: 'networkidle0'
        });
        
        // Generate PDF
        const pdf = await page.pdf({
            path: 'Guardsafe_ML_Security_Newsletter.pdf',
            format: 'A4',
            printBackground: true,
            margin: {
                top: '20mm',
                right: '20mm',
                bottom: '20mm',
                left: '20mm'
            },
            displayHeaderFooter: true,
            headerTemplate: '<div style="font-size: 10px; text-align: center; width: 100%; color: #666;">Guardsafe.ai ML Security Newsletter</div>',
            footerTemplate: '<div style="font-size: 10px; text-align: center; width: 100%; color: #666;">Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>'
        });
        
        console.log('✅ PDF generated successfully: Guardsafe_ML_Security_Newsletter.pdf');
        
        await browser.close();
        
    } catch (error) {
        console.error('❌ Error generating PDF:', error);
    }
}

generatePDF();
