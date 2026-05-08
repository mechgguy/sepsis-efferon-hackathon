"""Embed chunks and write them to Weaviate."""
import argparse
import json
import time
from pathlib import Path

from weaviate.classes.config import Configure, Property
from weaviate.util import generate_uuid5

from pipeline.config import COLLECTION_NAME, PARSED_CACHE_DIR
from pipeline.weaviate.config import get_client
from pipeline.weaviate.embedder import Embedder
from pipeline.weaviate.schema import CHUNK_SCHEMA


def ensure_collection(client):
    if client.collections.exists(COLLECTION_NAME):
        return
    client.collections.create(
        name=COLLECTION_NAME,
        properties=[Property(name=k, data_type=v[0]) for k, v in CHUNK_SCHEMA.items()],
        vector_config=Configure.Vectors.self_provided(),
    )


def ingest_chunks(chunks: list[dict]) -> None:
    """Embed and write a list of chunk dicts to Weaviate."""
    print(f"Embedding {len(chunks)} chunks...")
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
                props = {k: fn(meta, chunk) for k, (_, fn) in CHUNK_SCHEMA.items()}
                batch.add_object(
                    uuid=generate_uuid5(chunk["id"]),
                    properties=props,
                    vector=vector,
                )
    finally:
        client.close()
    print(f"Ingested {len(chunks)} chunks in {time.time() - t0:.1f}s")


def ingest_file(chunks_path: str | Path) -> None:
    """Load a single *_chunks.json file and ingest it."""
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {Path(chunks_path).name}")
    ingest_chunks(chunks)


def ingest_folder(folder: str | Path | None = None) -> None:
    """Ingest all *_chunks.json files from folder (default: data/parsed_papers/)."""
    folder = Path(folder) if folder else PARSED_CACHE_DIR
    paths = sorted(folder.glob("*_chunks.json"))
    if not paths:
        raise FileNotFoundError(f"No *_chunks.json found in {folder}")
    all_chunks: list[dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            all_chunks.extend(json.load(f))
        print(f"  Loaded {p.name}")
    print(f"Total chunks: {len(all_chunks)}")
    ingest_chunks(all_chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest chunks into Weaviate")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--chunks", help="Path to a single *_chunks.json file")
    group.add_argument("--folder", help="Folder containing *_chunks.json files")
    group.add_argument("--all", action="store_true",
                       help="Ingest all *_chunks.json from data/parsed_papers/")
    args = parser.parse_args()

    if args.all:
        ingest_folder()
    elif args.folder:
        ingest_folder(args.folder)
    else:
        ingest_file(args.chunks)
