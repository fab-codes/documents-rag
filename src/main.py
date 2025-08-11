
import os
from src.chunk.chunker import chunk_text
from src.pdf.pdf_utils import extract_text_from_pdf
from dotenv import load_dotenv

def main():
    """
    Main function that orchestrates the extraction and chunking process.
    """
    load_dotenv()  # Load environment variables from .env file
    pdf_path = os.getenv("PDF_FILE_PATH")  # read variable from .env file

    print(f"1. Extracting text from file: {pdf_path}")
    extracted_text = extract_text_from_pdf(pdf_path)
    
    if not extracted_text:
        print("Process interrupted due to an extraction error.")
        return

    print("2. Starting the chunking process...")

    docs = chunk_text(extracted_text)
    
    print(f"3. Process complete. Created {len(docs)} chunks.")
    
    # Let's display the first two chunks for inspection
    print("\n--- Content of the first chunk ---")
    print(docs[0].page_content)  # .page_content contains the chunk's text


if __name__ == "__main__":
    main()