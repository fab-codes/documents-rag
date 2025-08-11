
import os
from src.chunk.chunk import chunk_pages
from src.embedding.embedding import create_and_store_embeddings
from src.pdf.utils import extract_pages_from_pdf
from dotenv import load_dotenv

def main():
    """
    Main function that orchestrates the extraction and chunking process.
    """
    load_dotenv()  # Load environment variables from .env file
    pdf_path = os.getenv("PDF_FILE_PATH")  # read variable from .env file

    print(f"1. Extracting text from file: {pdf_path}")
    extracted_pages = extract_pages_from_pdf(pdf_path)
    
    if not extracted_pages:
        print("Process interrupted due to an extraction error.")
        return

    print("2. Starting the chunking process...")

    docs = chunk_pages(extracted_pages)

    print(f"3. Process complete. Created {len(docs)} chunks.")
    
    # Let's display the first two chunks for inspection
    print("\n--- Content of the first chunk ---")
    print(docs[0].page_content)  # .page_content contains the chunk's text


if __name__ == "__main__":
    main()