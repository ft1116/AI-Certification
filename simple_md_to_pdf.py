#!/usr/bin/env python3
"""
Simple markdown to PDF converter using markdown2 and basic HTML
"""

import markdown2
import os

def convert_md_to_pdf():
    """Convert markdown file to HTML that can be printed to PDF"""
    
    # Read the markdown file
    with open('MINERAL_INSIGHTS_DOCUMENTATION.md', 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown2.markdown(markdown_content, extras=['tables', 'fenced-code-blocks', 'toc'])
    
    # Create full HTML document
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Mineral Insights Documentation</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
            h2 {{ color: #34495e; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            code {{ background-color: #f4f4f4; padding: 2px 4px; }}
            pre {{ background-color: #f8f8f8; padding: 10px; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Save HTML file
    with open('MINERAL_INSIGHTS_DOCUMENTATION.html', 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print("✅ HTML file created: MINERAL_INSIGHTS_DOCUMENTATION.html")
    print("📄 To convert to PDF:")
    print("   1. Open the HTML file in your browser")
    print("   2. Press Ctrl+P (or Cmd+P on Mac)")
    print("   3. Select 'Save as PDF'")
    print("   4. Save the file")
    
    return True

if __name__ == "__main__":
    print("🔄 Converting MINERAL_INSIGHTS_DOCUMENTATION.md to HTML...")
    convert_md_to_pdf()
