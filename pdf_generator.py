import pdfkit

class PDFGenerator:
    def __init__(self, options=None):
        self.options = options or {}

    def generate_pdf(self, html_content, output_path):
        pdfkit.from_string(html_content, output_path, options=self.options)