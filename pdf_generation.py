import pdfkit

def generate_pdf(input_html, output_pdf):
    """Generate a PDF from HTML content."""
    pdfkit.from_file(input_html, output_pdf)