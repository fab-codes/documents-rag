import os
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def create_and_store_embeddings(chunks: list[Document]):
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY non impostata")

    print("Initializing Cohere embedding model...")
    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=api_key,
    )

    print("Creating vector store with FAISS...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Vector store created successfully.")

    embeddings_matrix = vector_store.index.reconstruct_n(0, len(vector_store.docstore._dict))

    print("\n--- Vector Store Preview ---")
    print(f"Totale documenti: {len(vector_store.docstore._dict)}\n")
    for i, (doc_id, doc) in enumerate(vector_store.docstore._dict.items()):
        print(f"ID: {doc_id}")
        print(f"Pagina: {doc.metadata.get('page')}")
        print(f"Sorgente: {doc.metadata.get('source')}")
        print(f"Testo: {doc.page_content[:200]}...\n")
        print(f"Embedding (prime 5 dimensioni): {embeddings_matrix[i][:5]}\n")
        if i >= 2:
            break
    print("----------------------------\n")

    return vector_store
