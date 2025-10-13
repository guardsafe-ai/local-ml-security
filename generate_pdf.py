#!/usr/bin/env python3
"""
Generate PDF from HTML newsletter using weasyprint
"""

import weasyprint
import os
from pathlib import Path

def generate_pdf():
    try:
        print("🚀 Starting PDF generation...")
        
        # Get current directory
        current_dir = Path(__file__).parent
        
        # HTML file path
        html_file = current_dir / "EMAIL_NEWSLETTER.html"
        
        # Output PDF path
        pdf_file = current_dir / "Guardsafe_ML_Security_Newsletter.pdf"
        
        # Check if HTML file exists
        if not html_file.exists():
            print(f"❌ HTML file not found: {html_file}")
            return
        
        # Generate PDF
        print("📄 Converting HTML to PDF...")
        weasyprint.HTML(filename=str(html_file)).write_pdf(str(pdf_file))
        
        print(f"✅ PDF generated successfully: {pdf_file}")
        print(f"📁 File size: {pdf_file.stat().st_size / 1024:.1f} KB")
        
    except ImportError:
        print("❌ weasyprint not installed. Install with: pip install weasyprint")
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")

if __name__ == "__main__":
    generate_pdf()
