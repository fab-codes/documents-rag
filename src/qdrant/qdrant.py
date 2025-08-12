from langchain_qdrant import Qdrant
from langchain_core.documents import Document
from src.config.config import QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL
from src.embedding.embedding_setup import embeddings

def store(chunks: list[Document]):
    collection = QDRANT_COLLECTION

    # Will create the collection if it doesn't exist
    vs = Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        prefer_grpc=False,  # True if endpoint supports gRPC
        api_key=QDRANT_API_KEY,
        collection_name=collection,
    )
    return vs
