import numpy as np
from model2vec import StaticModel

_MODEL = None

# minishlab/potion-multilingual-128M
# sentence-transformers/static-similarity-mrl-multilingual-v1
# ../../models/snowflake_custom
def init_model(path: str):
    global _MODEL
    if _MODEL is None:
        _MODEL = StaticModel.from_pretrained(path)

def get_token_level_embeddings(text: str, path: str) -> np.ndarray:
    """
    Return token-level embeddings for one text.

    Expected:
        np.ndarray with shape [num_tokens, dim], dtype float32/float16.
    """
    init_model(path)
    
    # Handle empty text just in case
    if not text or not text.strip():
        dim = _MODEL.dim
        return np.zeros((1, dim), dtype=np.float32)

    token_embeddings = _MODEL.encode_as_sequence(text)
    
    if token_embeddings.shape[0] == 0:
        dim = _MODEL.dim
        return np.zeros((1, dim), dtype=np.float32)
    
    return token_embeddings.astype(np.float32)
