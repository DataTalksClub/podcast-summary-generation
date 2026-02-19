import pdfkit

def generate_pdf(html_content, output_file):
    """Generate a PDF file from HTML content."""
    pdfkit.from_string(html_content, output_file)