import os


MAX_NEW_TOKENS = 32

MODEL_FAMILY_DICT = {
    # embeddings
    "text-embedding-ada-002": "openai",
    "all-MiniLM-L6-v2": "sentence-transformers",
    "BAAI/bge-base-en-v1.5": "sentence-transformers",
    "BAAI/bge-large-en-v1.5": "sentence-transformers",
    # generators
    "gpt-35-turbo": "openai",
    "gpt-35-turbo-16k": "openai",
    "gpt-4": "openai",
    "gpt-4-0613": "openai",
    "mistralai/Mistral-7B-v0.1": "transformers",
    "mistralai/Mistral-7B-Instruct-v0.1": "transformers",
    "mistralai/Mistral-7B-Instruct-v0.2": "transformers",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "transformers",
    "microsoft/phi-2": "transformers",
    "gemini-pro": "google",
}

EMBEDDING_DIM_DICT = {
    "text-embedding-ada-002": 1536,
    "all-MiniLM-L6-v2": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}

DATA_DIR = "data"
RESULTS_DIR = os.path.join(DATA_DIR, "results")
EMBEDDINGS_DIR = os.path.join(RESULTS_DIR, "embeddings")
VECTOR_STORE_DIR = os.path.join(RESULTS_DIR, "vector_stores")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
