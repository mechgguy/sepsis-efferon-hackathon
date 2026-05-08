"""BM25 / vector / hybrid retrieval with optional reranking."""
import argparse

from weaviate.classes.query import MetadataQuery

from pipeline.config import COLLECTION_NAME
from pipeline.weaviate.config import get_client
from pipeline.weaviate.embedder import Embedder
from pipeline.weaviate.reranker import Reranker

_PROPS = [
    "chunkId", "title", "chunkIndex", "chapterIndex",
    "compressedContent", "pageNumber", "shortSummary", "fullSummary",
]


def _to_record(obj) -> dict:
    p = obj.properties
    return {
        "compressedContent": p.get("compressedContent", ""),
        "title":             p.get("title", ""),
        "chunkIndex":        p.get("chunkIndex", 0),
        "chapterIndex":      p.get("chapterIndex", 0),
        "page_number":       p.get("pageNumber", 0),
        "shortSummary":      p.get("shortSummary", ""),
        "fullSummary":       p.get("fullSummary", ""),
        "chunkId":           p.get("chunkId", ""),
        "_id":               str(obj.uuid),
        "_score":            obj.metadata.score if obj.metadata else None,
    }


def _bm25(collection, query: str, limit: int) -> list[dict]:
    resp = collection.query.bm25(
        query=query, limit=limit,
        return_properties=_PROPS,
        return_metadata=MetadataQuery(score=True),
    )
    return [_to_record(o) for o in resp.objects]


def _vector(collection, vector: list[float], limit: int) -> list[dict]:
    resp = collection.query.near_vector(
        near_vector=vector, limit=limit,
        return_properties=_PROPS,
        return_metadata=MetadataQuery(distance=True),
    )
    return [_to_record(o) for o in resp.objects]


def _hybrid(collection, query: str, vector: list[float], limit: int) -> list[dict]:
    resp = collection.query.hybrid(
        query=query, vector=vector, limit=limit,
        return_properties=_PROPS,
        return_metadata=MetadataQuery(score=True),
    )
    return [_to_record(o) for o in resp.objects]


def retrieve(
    query: str,
    mode: str = "hybrid",
    top_k: int = 5,
    candidates: int = 20,
    rerank: bool = True,
) -> list[dict]:
    """Retrieve relevant chunks from Weaviate. Returns list of chunk dicts."""
    embedder = Embedder()
    vector = embedder.embed(query)

    client = get_client()
    try:
        collection = client.collections.get(COLLECTION_NAME)
        if mode == "bm25":
            results = _bm25(collection, query, candidates)
        elif mode == "vector":
            results = _vector(collection, vector, candidates)
        else:
            results = _hybrid(collection, query, vector, candidates)
    finally:
        client.close()

    if rerank and results:
        results = Reranker().rerank(query, results, top_k)
    else:
        results = results[:top_k]

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    results = retrieve(args.query, args.mode, args.top_k, args.candidates, not args.no_rerank)
    for i, chunk in enumerate(results, 1):
        score = chunk.get("_rerank_score") or chunk.get("_score")
        score_str = f" (score={score:.4f})" if score is not None else ""
        print(f"[{i}] {chunk['title']}{score_str}")
        print(f"    {chunk['compressedContent'][:200]}...")
        print()
