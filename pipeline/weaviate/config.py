import weaviate
from pipeline.config import WEAVIATE_HOST, WEAVIATE_PORT


def get_client() -> weaviate.WeaviateClient:
    return weaviate.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT)
