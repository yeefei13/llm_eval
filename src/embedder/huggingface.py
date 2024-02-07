import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .base import BaseEmbedder
from sentence_transformers import SentenceTransformer
# import constants
from constants import EMBEDDINGS_DIR


class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(self, model_name, device="cuda:0"):
        super().__init__()
        self.model_name = model_name
        self.model = SentenceTransformer(model_name).to(device)

    def embed(self, docs):
        if isinstance(docs, str):
            return self.model.encode(docs)
        else:
            return self.model.encode(docs, show_progress_bar=True)

    def save(self, docs, path):
        raise NotImplementedError("Save method is not implemented")

    def load(self, path):
        raise NotImplementedError("Load method is not implemented")
