# vector_store.py
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from src.config.config import QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL
from src.services.remote_embedding.remote_embedding import get_embeddings, get_vector_size

def ensure_collection(client: QdrantClient, collection: str):
    if not client.collection_exists(collection):
        size = get_vector_size()
        client.recreate_collection(
            collection_name=collection,
            vectors_config=rest.VectorParams(size=size, distance=rest.Distance.COSINE),
        )

def get_vector_store(client: QdrantClient) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=get_embeddings(),
    )

def store(chunks: list[Document]) -> QdrantVectorStore:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)
    ensure_collection(client, QDRANT_COLLECTION)

    vector_store = get_vector_store(client)
    vector_store.add_documents(chunks)

    return vector_store

def get_existing_vector_store() -> QdrantVectorStore:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)
    ensure_collection(client, QDRANT_COLLECTION)

    return get_vector_store(client)
