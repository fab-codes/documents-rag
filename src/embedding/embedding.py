from langchain_cohere import CohereEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from src.config.config import COHERE_API_KEY, QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL

# TODO: Update name and split code
def store_in_qdrant(chunks: list[Document]):
    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=COHERE_API_KEY,
    )

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

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
