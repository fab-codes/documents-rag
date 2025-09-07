from typing import List, Optional
import time, requests
from langchain_core.embeddings import Embeddings
from src.config.config import EMBEDDING_SERVICE_URL

class RemoteEmbeddings(Embeddings):
    def __init__(self, timeout: int = 10, retries: int = 2, session: Optional[requests.Session] = None):
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()

    def _post(self, texts: List[str]) -> dict:
        last_err = None
        url = f"{EMBEDDING_SERVICE_URL}/embed"
        for attempt in range(self.retries + 1):
            try:
                r = self.session.post(url, json={"texts": texts}, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                if "vectors" not in data:
                    raise KeyError(f"Missing 'vectors' in response: {data}")
                return data
            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Embedding call failed at {url}: {type(last_err).__name__}: {last_err}") from last_err

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._post(texts)["vectors"]

    def embed_query(self, text: str) -> List[float]:
        return self._post([text])["vectors"][0]


# singleton/factory
_emb_instance: Optional[RemoteEmbeddings] = None
_vector_size_cache: Optional[int] = None

def get_embeddings() -> RemoteEmbeddings:
    global _emb_instance
    if _emb_instance is None:
        _emb_instance = RemoteEmbeddings(timeout=10, retries=2)
    return _emb_instance

def get_vector_size() -> int:
    global _vector_size_cache
    if _vector_size_cache is None:
        _vector_size_cache = len(get_embeddings().embed_query("__dim__"))
    return _vector_size_cache
