from sentence_transformers import SentenceTransformer
import numpy as np
import os
# --- Fix: redirect model cache to writable directory ---
CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE", "/tmp/hf_cache")
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-MiniLM-L6-v2", cache_folder=CACHE_DIR)
    return _model

def embed(text: str):
    model = get_model()
    emb = model.encode(text, normalize_embeddings=True)
    return emb.tolist()

def cosine(a: list[float], b: list[float]) -> float:
    return float(np.dot(a, b))  # normalized
def embed_batch(texts: list[str]) -> list[list[float]]:
    model = get_model()
    return model.encode(texts, normalize_embeddings=True).tolist()
