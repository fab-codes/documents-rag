import pypdf

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from all pages of a PDF file.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A string containing all the text from the PDF, or None if an error occurs.
    """
    try:
        reader = pypdf.PdfReader(pdf_path)
        full_text = ""
        # Iterate through each page to extract text
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text
        return full_text
    except FileNotFoundError:
        print(f"Error: The PDF file was not found at path '{pdf_path}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the PDF: {e}")
        return None