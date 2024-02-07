import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.embedder.huggingface import HuggingFaceEmbedder
from src.retriever.faiss import FaissRetriever
from src.generator.huggingface import HuggingFaceGenerator

from constants import DATASETS_DIR

if __name__ == "__main__":

    dataset = "natural-questions"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # load dataset
    dataset_path = os.path.join(DATASETS_DIR, dataset, "wikipedia_documents.tsv")
    documents = pd.read_csv(dataset_path, sep="\t")

    # Embedder
    embedder_name = "all-MiniLM-L6-v2"
    embedder = HuggingFaceEmbedder(embedder_name, device)
    embedded_docs = embedder.embed(documents["text"].tolist()[:1000])

    # Retriever
    retriever = FaissRetriever(embedder_name, gpu=True)
    retriever.create_index(dataset, embedded_docs)

    # Generator
    generator = HuggingFaceGenerator("mistralai/Mistral-7B-Instruct-v0.1", device)

    while True:
        query = input("Ask a question: ")
        if query == "exit":
            break
        q_embedding = embedder.embed(query)

        retrieved_indices = retriever.retrieve(q_embedding, k=5)
        retrieved_context = documents.iloc[retrieved_indices.flatten()]["text"].tolist()
        from pprint import pprint

        pprint(retrieved_context)
        context = "".join(retrieved_context)
        RAG_prompt = f"""You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.
            Only output the keywords that answer the question, not a full sentence.
            Question: {query}
            Context: {context}
            Answer:
        """
        answer = generator.generate(RAG_prompt)
        print(answer)
