
from src.chunk.chunk import chunk_pages
from src.config.config import PDF_FILE_PATH
from src.pdf.utils import extract_pages_from_pdf
from src.qdrant.qdrant import store
from src.rag.rag import create_rag_chain

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

    vector_store = store(docs)

    rag_chain = create_rag_chain(vector_store)
    
    print(f"3. Process complete. Created {len(docs)} chunks.")
    
    print("\n\n✅ RAG ready. Ask your questions! (write 'exit' to quit)\n")

    while True:
        question = input("> ")
        if question.lower() == 'exit':
            break
        
        # Invoke the RAG chain with the user's question
        answer = rag_chain.invoke(question)
        print("Answer:", answer, "\n")


if __name__ == "__main__":
    main()