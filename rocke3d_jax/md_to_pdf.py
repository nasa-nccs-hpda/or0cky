#!/usr/bin/env python
"""
Convert Markdown to PDF using markdown2 and weasyprint.
Falls back to a basic HTML approach if weasyprint fails.
"""

import os
import sys
from markdown2 import markdown

# Try using weasyprint first
try:
    from weasyprint import HTML
    use_weasyprint = True
except ImportError:
    use_weasyprint = False

def convert_md_to_pdf(md_file, pdf_file, use_weasyprint_flag=True):
    """Convert a Markdown file to PDF."""
    use_weasyprint = use_weasyprint_flag
    
    # Read the Markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert Markdown to HTML
    html_content = markdown(md_content)
    
    # Add basic CSS for styling
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 2em; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.5em; }}
            h2 {{ color: #3498db; }}
            h3 {{ color: #2980b9; }}
            table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            code {{ background-color: #f4f4f4; padding: 0.2em 0.4em; border-radius: 3px; }}
            pre {{ background-color: #f4f4f4; padding: 1em; border-radius: 5px; overflow-x: auto; }}
            blockquote {{ border-left: 4px solid #3498db; padding-left: 1em; color: #555; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    if use_weasyprint:
        try:
            # Use weasyprint to generate PDF
            HTML(string=styled_html).write_pdf(pdf_file)
            print(f"✅ PDF generated successfully using WeasyPrint: {pdf_file}")
            return True
        except Exception as e:
            print(f"❌ WeasyPrint failed: {e}")
            use_weasyprint = False
    
    if not use_weasyprint:
        # Fallback: Save as HTML (can be printed to PDF manually)
        html_file = pdf_file.replace('.pdf', '.html')
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(styled_html)
        print(f"⚠️  WeasyPrint failed. HTML saved to: {html_file}")
        print("   Open the HTML file in a browser and print to PDF manually.")
        return False

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python md_to_pdf.py <input.md> <output.pdf>")
        sys.exit(1)
    
    md_file = sys.argv[1]
    pdf_file = sys.argv[2]
    
    if not os.path.exists(md_file):
        print(f"❌ Error: File not found: {md_file}")
        sys.exit(1)
    
    success = convert_md_to_pdf(md_file, pdf_file, use_weasyprint)
    sys.exit(0 if success else 1)
