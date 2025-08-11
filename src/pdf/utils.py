import pymupdf
import os

def extract_pages_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    """
    Extracts text from a PDF, keeping one page per item
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return []

    doc = pymupdf.open(pdf_path)
    pages_with_text = []
    print(f"Extracting text from {doc.page_count} pages...")
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():  # Add the page only if it contains text
            pages_with_text.append((page_num + 1, text))
    print("Extraction complete.")
    return pages_with_text