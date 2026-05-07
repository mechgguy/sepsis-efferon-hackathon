import argparse
import json
import time

from weaviate.classes.config import Configure, DataType, Property
from weaviate.util import generate_uuid5

from config import get_client, COLLECTION_NAME
from embedder import Embedder


def ensure_collection(client):
    if client.collections.exists(COLLECTION_NAME):
        return
    client.collections.create(
        name=COLLECTION_NAME,
        properties=[
            Property(name="chunkId", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="chunkIndex", data_type=DataType.INT),
            Property(name="chapterIndex", data_type=DataType.INT),
            Property(name="compressedContent", data_type=DataType.TEXT),
            Property(name="shortSummary", data_type=DataType.TEXT),
            Property(name="fullSummary", data_type=DataType.TEXT),
        ],
        vector_config=Configure.Vectors.self_provided(),
    )


def ingest(chunks_path: str):
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Embedding {len(chunks)} chunks with BGE-M3...")
    t0 = time.time()
    embedder = Embedder()
    vectors = embedder.embed_batch([c["text"] for c in chunks])

    client = get_client()
    try:
        ensure_collection(client)
        collection = client.collections.get(COLLECTION_NAME)

        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, vectors):
                meta = chunk.get("metadata", {})
                batch.add_object(
                    uuid=generate_uuid5(chunk["id"]),
                    properties={
                        "chunkId": chunk["id"],
                        "title": meta.get("section", ""),
                        "chunkIndex": meta.get("section_index", 0),
                        "chapterIndex": meta.get("part_index", 0),
                        "compressedContent": chunk["text"],
                        "shortSummary": "",
                        "fullSummary": "",
                    },
                    vector=vector,
                )
    finally:
        client.close()

    print(f"Ingested {len(chunks)} chunks in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="Path to chunks JSON file")
    args = parser.parse_args()
    ingest(args.chunks)
