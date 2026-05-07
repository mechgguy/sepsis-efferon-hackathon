import os
import weaviate

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-m3")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

COLLECTION_NAME = "RagDocumentChunk"


def get_client() -> weaviate.WeaviateClient:
    return weaviate.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT)
