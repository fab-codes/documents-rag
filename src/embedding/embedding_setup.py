from langchain_cohere import CohereEmbeddings
from src.config.config import COHERE_API_KEY

embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=COHERE_API_KEY,
)