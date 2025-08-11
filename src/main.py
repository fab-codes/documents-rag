
from src.chunk.chunk import chunk_pages
from src.config.config import PDF_FILE_PATH
from src.embedding.embedding import store_in_qdrant
from src.pdf.utils import extract_pages_from_pdf

def main():
    """
    Main function that orchestrates the extraction and chunking process.
    """
    pdf_path = PDF_FILE_PATH  # read variable from .env file

    print(f"1. Extracting pages from file: {pdf_path}")
    extracted_pages = extract_pages_from_pdf(pdf_path)
    
    if not extracted_pages:
        print("Process interrupted due to an extraction error.")
        return

    print("2. Starting the chunking process...")

    docs = chunk_pages(extracted_pages, pdf_path)

    store_in_qdrant(docs)
    
    print(f"3. Process complete. Created {len(docs)} chunks.")
    
    # Let's display the first two chunks for inspection
    print("\n--- Content of the first chunk ---")
    print(docs[0].page_content)  # .page_content contains the chunk's text


if __name__ == "__main__":
    main()