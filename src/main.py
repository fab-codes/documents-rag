from src.chunk.chunk import chunk_pages
from src.config.config import PDF_FILE_PATH
from src.pdf.utils import extract_pages_from_pdf
from src.qdrant.qdrant import get_existing_vector_store, store
from src.rag.rag import create_rag_chain
from langchain_core.messages import HumanMessage, AIMessage

def main():
    """
    Main function that orchestrates the entire process.
    """
    embed_file = input("Would you like to embed the file? y/Y/n/N ")

    vector_store = None

    if(embed_file.lower() == "y"):
        # --- SETUP PHASE (runs only once) ---
        pdf_path = PDF_FILE_PATH

        print(f"1. Extracting pages from file: {pdf_path}")
        extracted_pages = extract_pages_from_pdf(pdf_path)
        
        if not extracted_pages:
            print("Process interrupted due to an extraction error.")
            return

        print("2. Starting the chunking process...")
        docs = chunk_pages(extracted_pages, pdf_path)

        print("3. Storing chunks in the Vector Store (Qdrant)...")
        vector_store = store(docs)

        print(f"\n✅ Setup complete. Created {len(docs)} chunks.")

    vector_store = vector_store or get_existing_vector_store()

    print("Creating the conversational RAG chain...")
    rag_chain = create_rag_chain(vector_store)
        
    print("🤖 You can now ask your questions! (type 'exit' to quit)\n")

    # --- INTERACTIVE CHAT PHASE (loops) ---
    
    # Empty list to store the conversation history.
    chat_history = []

    # Infinite loop for the chat
    while True:
        # Ask the user for a question
        question = input("You: ")
        
        # If the user types 'exit', break the loop and end the program
        if question.lower() == 'exit':
            print("🤖 See you soon!")
            break
        
        # Call the RAG chain by passing a dictionary containing:
        # 1. The current question
        # 2. The entire past conversation
        result = rag_chain.invoke({
            "input": question,
            "chat_history": chat_history
        })
        
        answer = result["answer"]
        print(f"AI: {answer}\n")
        
        # Update history with last question and answer.
        chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer)
        ])

if __name__ == "__main__":
    main()