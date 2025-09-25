# utils.py
from sentence_transformers import SentenceTransformer

_EMBEDDER = None

def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER
