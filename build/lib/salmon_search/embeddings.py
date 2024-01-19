from typing import Optional

from numpy import ndarray
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'all-MiniLM-L6-v2'
VECTOR_SIZE = 384

model = None


def get_model() -> Optional[SentenceTransformer]:
    global model
    if not model:
        model = SentenceTransformer(MODEL_NAME)
    return model


def encode(sentences: str | list[str], show_progress_bar: bool = False) -> ndarray:
    return get_model().encode(sentences, show_progress_bar=show_progress_bar)
