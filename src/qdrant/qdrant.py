from langchain_qdrant import Qdrant
from langchain_core.documents import Document
from src.config.config import QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL
from src.embedding.embedding_setup import embeddings
from qdrant_client import QdrantClient 

def store(chunks: list[Document]):
    collection = QDRANT_COLLECTION

    # Will create the collection if it doesn't exist
    vector_store = Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        prefer_grpc=False,  # True if endpoint supports gRPC
        api_key=QDRANT_API_KEY,
        collection_name=collection,
    )
    return vector_store

def get_existing_vector_store():
    """Connects to an existing Qdrant collection."""
    print("Connecting to existing Vector Store...")
    
    client = QdrantClient(
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY,
        prefer_grpc=False
    )
    
    vector_store = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embeddings=embeddings,
    )
    print("✅ Connection successful.")
    return vector_store