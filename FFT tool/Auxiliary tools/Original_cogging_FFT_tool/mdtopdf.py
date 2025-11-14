import markdown
from weasyprint import HTML
from pathlib import Path

# Path setup
md_path = Path("README_Cogging_Analysis.md").resolve()
base_dir = md_path.parent

# Read markdown file
with open(md_path, "r", encoding="utf-8") as f:
    md_text = f.read()

# Convert Markdown → HTML with useful extensions
html_body = markdown.markdown(
    md_text,
    extensions=['extra', 'tables', 'toc']
)

# Add improved styling with smaller margins
style = """
<style>
  @page {
    margin: 0.5in;   /* much smaller page margins */
  }

  body { 
    font-family: Arial, sans-serif; 
    margin: 0.25in;  /* smaller body margins */
    line-height: 1.5;
    color: #222;
  }
  h1, h2, h3 { 
    color: #222; 
    margin-top: 1.1em;
  }
  h1 { 
    border-bottom: 1px solid #ccc; 
    padding-bottom: 0.2em; 
  }
  img { 
    max-width: 95%; 
    height: auto; 
    display: block; 
    margin: 0.5em auto;
    border: 0.5px solid #ddd;
    border-radius: 4px;
  }
  table { 
    border-collapse: collapse; 
    width: 100%; 
    margin: 0.75em 0;
  }
  th, td { 
    border: 0.5px solid #ccc; 
    padding: 5px 7px; 
    text-align: left;
  }
  th { 
    background-color: #f7f7f7; 
    font-weight: bold;
  }
  code { 
    background: #f8f8f8; 
    padding: 2px 4px; 
    border-radius: 3px; 
    font-size: 0.95em;
  }
</style>
"""

# Combine style + body into full HTML doc
html_content = f"<html><head>{style}</head><body>{html_body}</body></html>"

# Convert HTML → PDF, giving WeasyPrint the correct base folder for relative image paths
HTML(string=html_content, base_url=str(base_dir)).write_pdf("README_Cogging_Analysis.pdf")

print(f"✅ PDF created successfully at: {base_dir / 'README_Cogging_Analysis.pdf'}")
