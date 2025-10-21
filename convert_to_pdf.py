#!/usr/bin/env python3

import markdown
import weasyprint
import os

def convert_markdown_to_pdf():
    # Read the markdown file
    with open('MINERAL_INSIGHTS_DOCUMENTATION.md', 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    # Convert markdown to HTML
    html_content = markdown.markdown(markdown_content, extensions=['tables', 'fenced_code'])

    # Add CSS styling for better PDF appearance
    styled_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Mineral Insights Documentation</title>
        <style>
            @page {{
                size: A4;
                margin: 1in;
            }}
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                font-size: 12pt;
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: #2c3e50;
                margin-top: 20px;
                margin-bottom: 10px;
            }}
            h1 {{
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                font-size: 24pt;
            }}
            h2 {{
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 8px;
                font-size: 18pt;
            }}
            h3 {{
                font-size: 14pt;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
                font-size: 10pt;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: bold;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
            }}
            pre {{
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
                font-size: 10pt;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                margin: 15px 0;
                padding-left: 15px;
                color: #555;
            }}
            img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 15px auto;
            }}
            .page-break {{
                page-break-before: always;
            }}
            ul, ol {{
                margin: 10px 0;
                padding-left: 20px;
            }}
            li {{
                margin: 5px 0;
            }}
        </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    '''

    # Convert HTML to PDF using WeasyPrint
    pdf_file = 'Mineral_Insights_Documentation.pdf'
    try:
        weasyprint.HTML(string=styled_html).write_pdf(pdf_file)
        print(f'✅ PDF created successfully: {pdf_file}')
        print(f'📁 File location: {os.path.abspath(pdf_file)}')
        print(f'📊 File size: {os.path.getsize(pdf_file)} bytes')
        return pdf_file
    except Exception as e:
        print(f'❌ Error creating PDF: {e}')
        return None

if __name__ == "__main__":
    convert_markdown_to_pdf()