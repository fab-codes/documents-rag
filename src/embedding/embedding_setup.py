from langchain_huggingface import HuggingFaceEmbeddings
from src.config.config import CPU_OR_CUDA, EMBEDDING_MODEL_ID

print("Loading local Sentence Transformer model (Qwen/Qwen3-Embedding-0.6B)...")
print("This may take a moment, as the model needs to be downloaded on the first run.")

# Define where the model will run
model_kwargs = {'device': CPU_OR_CUDA}

# Define the way how the text should be processed
encode_kwargs = {'normalize_embeddings': True}

# Initialize the embedding model using the HuggingFaceEmbeddings wrapper
embeddings = HuggingFaceEmbeddings(
    # The name of the model
    model_name=EMBEDDING_MODEL_ID,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("✅ Local embedding model loaded successfully.")