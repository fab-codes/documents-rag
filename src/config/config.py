from dotenv import load_dotenv
import os

load_dotenv()

# FILES
PDF_FILE_PATH = os.getenv("PDF_FILE_PATH")

# EMBEDDING
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID")
CPU_OR_CUDA= os.getenv("CPU_OR_CUDA")

# QDRANT
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

# GOOGLE
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL_ID = os.getenv("GOOGLE_MODEL_ID")