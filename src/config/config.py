from dotenv import load_dotenv
import os

load_dotenv()

# FILES
PDF_FILE_PATH = os.getenv("PDF_FILE_PATH")

# COHERE
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# QDRANT
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")