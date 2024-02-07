import sys
import os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from .base import BaseRetriever
import faiss
from constants import VECTOR_STORE_DIR, EMBEDDING_DIM_DICT


class FaissRetriever(BaseRetriever):
    """Retriever using Faiss index"""

    def __init__(self, embedder_name, gpu=False):
        super().__init__()
        self.index = faiss.IndexIDMap(
            faiss.IndexFlatL2(EMBEDDING_DIM_DICT[embedder_name])
        )
        if gpu:
            ngpus = faiss.get_num_gpus()
            res = [faiss.StandardGpuResources() for _ in range(ngpus)]
            self.index = faiss.index_cpu_to_gpu_multiple_py(res, self.index)
        self.embedder_name = embedder_name

    def create_index(self, dataset, vectors):
        if os.path.exists(
            f"{VECTOR_STORE_DIR}/{dataset}/faiss/{self.embedder_name}.index"
        ):
            self.index = faiss.read_index(
                f"{VECTOR_STORE_DIR}/{dataset}/faiss/{self.embedder_name}.index"
            )
        self.vectors = vectors
        self.index.add_with_ids(vectors, ids=np.arange(vectors.shape[0]))

    def retrieve(self, query, k=1):
        # if self.index is None:
        #     self.index = faiss.read_index(self.vector_store.index_path)

        query = query.reshape(1, -1)
        _, indices = self.index.search(query, k)
        return indices

    def save(self, path):
        faiss.write_index(self.index, path)

    def load(self, path):
        self.index = faiss.read_index(path)
