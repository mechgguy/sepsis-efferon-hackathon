import argparse
import json
import time

from weaviate.classes.config import Configure, DataType, Property
from weaviate.util import generate_uuid5

from config import get_client, COLLECTION_NAME
from embedder import Embedder

from schema import CHUNK_SCHEMA
import os
from pathlib import Path 
def ensure_collection(client):
    if client.collections.exists(COLLECTION_NAME):
        return
    client.collections.create(
        name=COLLECTION_NAME,
        properties=[Property(name=k, data_type=v[0]) for k, v in CHUNK_SCHEMA.items()],
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
                props = {k: fn(meta, chunk) for k, (_, fn) in CHUNK_SCHEMA.items()}
                batch.add_object(uuid=generate_uuid5(chunk["id"]), properties=props, vector=vector)
    finally:
        client.close()

    print(f"Ingested {len(chunks)} chunks in {time.time() - t0:.1f}s")

def ingest_folder(folder: str):
    paths = sorted(Path(folder).glob("*_chunks.json"))
    if not paths:
        raise FileNotFoundError(f"No *_chunks.json found in {folder}")
    all_chunks = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            all_chunks.extend(json.load(f))
        print(f"  Loaded {p.name}")
    print(f"Total chunks: {len(all_chunks)}")
    ingest_chunks(all_chunks)

def ingest_chunks(chunks: list[dict]):
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
                batch.add_object(uuid=generate_uuid5(chunk["id"]), properties=props, vector=vector)
    finally:
        client.close()
    print(f"Ingested {len(chunks)} chunks in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--chunks", help="Path to single chunks JSON file")
    group.add_argument("--folder", help="Folder containing *_chunks.json files")
    args = parser.parse_args()

    if args.folder:
        ingest_folder(args.folder)
    else:
        with open(args.chunks, encoding="utf-8") as f:
            chunks = json.load(f)
        ingest_chunks(chunks)