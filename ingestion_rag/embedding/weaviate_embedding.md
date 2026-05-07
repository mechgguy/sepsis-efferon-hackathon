<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# ok now is running and i have chunks  how to ingest them and store them using weaviate, also add an option so i can call weavaite  and check that my chunks are there, modify my python script:

CHUNKS (from Baloch_2022_markdown_chunks.json):
[
{
"id": "Baloch_2022_section_0_0",
"text": "\#\# preamble\nReview began 11/26/2021 Review ended 01/05/2022 Published 01/09/2022 © Copyright 2022 Baloch et al. This is an open access article distributed under the terms of the Creative Commons Attribution License CC-BY 4.0., which permits unrestricted use, distribution, and reproduction in any medium, provided the original author and source are credited.",
"metadata": {
"document_id": "Baloch_2022",
"paper_id": "Baloch_2022",
"title": "Comparison of Pediatric Sequential Organ Failure Assessment and Pediatric Risk of Mortality III Score as Mortality Prediction in Pediatric Intensive Care Unit",
"year": 2022,
"source_markdown": "data/parsed_papers/Baloch_2022.md",
"chunk_type": "section",
"section": "preamble",
"level": 2,
"section_index": 0,
"part_index": 0,
"split_strategy": "heading",
"paragraph_id": null,
"fallback": false
}
},
{
"id": "Baloch_2022_section_1_0",
"text": "\#\# Comparison of Pediatric Sequential Organ Failure Assessment and Pediatric Risk of Mortality III Score as Mortality Prediction in Pediatric Intensive Care Unit\nSadam H. Baloch   , Ikramullah Shaikh   , Murtaza A. Gowa   , Pooja D. Lohano   , Mohsina N. Ibrahim 1 1 2 1 3 Pediatric Medicine, National Institute of Child Health, Karachi, PAK  2. Pediatric Critical Care, National Institute of Child Health, Karachi, PAK 3. Pediatrics and Endocrinology, National Institute of Child Health, Karachi, PAK Corresponding author: Sadam H. Baloch, balochsadam126@gmail.com",
"metadata": {
"document_id": "Baloch_2022",
"paper_id": "Baloch_2022",
"title": "Comparison of Pediatric Sequential Organ Failure Assessment and Pediatric Risk of Mortality III Score as Mortality Prediction in Pediatric Intensive Care Unit",
"year": 2022,
"source_markdown": "data/parsed_papers/Baloch_2022.md",
"chunk_type": "section",
"section": "Comparison of Pediatric Sequential Organ Failure Assessment and Pediatric Risk of Mortality III Score as Mortality Prediction in Pediatric Intensive Care Unit",
"level": 2,
"section_index": 1,
"part_index": 0,
"split_strategy": "heading",
"paragraph_id": null,
"fallback": false
}
},

PYTHON SCRIPT:

import json
import weaviate
from weaviate.classes.config import Configure, DataType, Property

client = weaviate.connect_to_local()

collection_name = "PaperChunk"

if not client.collections.exists(collection_name):
client.collections.create(
name=collection_name,
properties=[
Property(name="text", data_type=DataType.TEXT),
Property(name="document_id", data_type=DataType.TEXT),
Property(name="paper_id", data_type=DataType.TEXT),
Property(name="title", data_type=DataType.TEXT),
Property(name="year", data_type=DataType.INT),
Property(name="source_markdown", data_type=DataType.TEXT),
Property(name="chunk_type", data_type=DataType.TEXT),
Property(name="section", data_type=DataType.TEXT),
Property(name="level", data_type=DataType.INT),
Property(name="section_index", data_type=DataType.INT),
Property(name="part_index", data_type=DataType.INT),
Property(name="split_strategy", data_type=DataType.TEXT),
Property(name="fallback", data_type=DataType.BOOL),
],
vector_config=Configure.Vectors.self_provided()
)

chunks = json.load(open("chunks.json", "r", encoding="utf-8"))
collection = client.collections.get(collection_name)

with collection.batch.dynamic() as batch:
for chunk in chunks:
metadata = chunk["metadata"]
batch.add_object(
uuid=chunk["id"],
properties={
"text": chunk["text"],
"document_id": metadata.get("document_id"),
"paper_id": metadata.get("paper_id"),
"title": metadata.get("title"),
"year": metadata.get("year"),
"source_markdown": metadata.get("source_markdown"),
"chunk_type": metadata.get("chunk_type"),
"section": metadata.get("section"),
"level": metadata.get("level"),
"section_index": metadata.get("section_index"),
"part_index": metadata.get("part_index"),
"split_strategy": metadata.get("split_strategy"),
"fallback": metadata.get("fallback"),
},
vector=[0.01, 0.02, 0.03]  \# replace with your real embedding
)

client.close()

Yes — your script already has the basic ingestion flow, but it needs three practical improvements:

1. generate or pass a **real embedding** for each chunk,
2. make the UUID valid for Weaviate,
3. add a simple **check/query mode** so you can confirm your chunks were stored.[^1][^2][^3]

## Updated script

Use this version as a drop-in starting point:

```python
import json
import argparse
import uuid
import weaviate
from weaviate.classes.config import Configure, DataType, Property


COLLECTION_NAME = "PaperChunk"


def make_valid_uuid(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, value))


def connect():
    return weaviate.connect_to_local()


def ensure_collection(client):
    if not client.collections.exists(COLLECTION_NAME):
        client.collections.create(
            name=COLLECTION_NAME,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="document_id", data_type=DataType.TEXT),
                Property(name="paper_id", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="year", data_type=DataType.INT),
                Property(name="source_markdown", data_type=DataType.TEXT),
                Property(name="chunk_type", data_type=DataType.TEXT),
                Property(name="section", data_type=DataType.TEXT),
                Property(name="level", data_type=DataType.INT),
                Property(name="section_index", data_type=DataType.INT),
                Property(name="part_index", data_type=DataType.INT),
                Property(name="split_strategy", data_type=DataType.TEXT),
                Property(name="fallback", data_type=DataType.BOOL),
            ],
            vector_config=Configure.Vectors.self_provided()
        )


def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_embedding(text: str):
    # Replace this with your real embedding function.
    # Example: OpenAI, SentenceTransformers, BGE, Instructor, etc.
    return [0.01, 0.02, 0.03]


def ingest(chunks_path):
    client = connect()
    try:
        ensure_collection(client)
        collection = client.collections.get(COLLECTION_NAME)
        chunks = load_chunks(chunks_path)

        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                metadata = chunk.get("metadata", {})
                batch.add_object(
                    uuid=make_valid_uuid(chunk["id"]),
                    properties={
                        "text": chunk.get("text"),
                        "document_id": metadata.get("document_id"),
                        "paper_id": metadata.get("paper_id"),
                        "title": metadata.get("title"),
                        "year": metadata.get("year"),
                        "source_markdown": metadata.get("source_markdown"),
                        "chunk_type": metadata.get("chunk_type"),
                        "section": metadata.get("section"),
                        "level": metadata.get("level"),
                        "section_index": metadata.get("section_index"),
                        "part_index": metadata.get("part_index"),
                        "split_strategy": metadata.get("split_strategy"),
                        "fallback": metadata.get("fallback"),
                    },
                    vector=get_embedding(chunk["text"])
                )

        print(f"Ingested {len(chunks)} chunks into {COLLECTION_NAME}")

    finally:
        client.close()


def check_chunks(limit=10):
    client = connect()
    try:
        collection = client.collections.get(COLLECTION_NAME)
        response = collection.query.fetch_objects(limit=limit)

        print(f"Found {len(response.objects)} objects")
        for i, obj in enumerate(response.objects, start=1):
            props = obj.properties
            print(f"\n--- Chunk {i} ---")
            print("document_id:", props.get("document_id"))
            print("title:", props.get("title"))
            print("section:", props.get("section"))
            print("text preview:", (props.get("text", "")[:200] + "..."))
    finally:
        client.close()


def check_by_document(document_id, limit=20):
    client = connect()
    try:
        collection = client.collections.get(COLLECTION_NAME)

        response = collection.query.fetch_objects(
            limit=limit
        )

        filtered = [
            obj for obj in response.objects
            if obj.properties.get("document_id") == document_id
        ]

        print(f"Found {len(filtered)} objects for document_id={document_id}")
        for i, obj in enumerate(filtered, start=1):
            props = obj.properties
            print(f"\n--- Match {i} ---")
            print("section:", props.get("section"))
            print("part_index:", props.get("part_index"))
            print("text preview:", (props.get("text", "")[:200] + "..."))
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ingest", "check", "check-doc"], required=True)
    parser.add_argument("--chunks", default="chunks.json")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--document-id", default=None)

    args = parser.parse_args()

    if args.mode == "ingest":
        ingest(args.chunks)
    elif args.mode == "check":
        check_chunks(limit=args.limit)
    elif args.mode == "check-doc":
        if not args.document_id:
            raise ValueError("--document-id is required for --mode check-doc")
        check_by_document(document_id=args.document_id, limit=args.limit)
```

This follows the standard Weaviate Python client pattern of connecting to a collection, batch inserting objects, and using `fetch_objects(limit=...)` to retrieve stored items for verification.[^4][^3][^1]

## How to use it

Run ingestion:

```bash
python ingest_weaviate.py --mode ingest --chunks Baloch_2022_markdown_chunks.json
```

Check whether chunks exist:

```bash
python ingest_weaviate.py --mode check --limit 5
```

Check only one paper:

```bash
python ingest_weaviate.py --mode check-doc --document-id Baloch_2022 --limit 20
```

`fetch_objects()` returns a limited number of objects unless you explicitly set `limit`, so adding `--limit` is useful when verifying larger ingestions.[^5][^4]

## Important fixes

Your current script uses `Configure.Vectors.self_provided()`, which means Weaviate expects **you** to send vectors during insert; the placeholder `[0.01, 0.02, 0.03]` will only work as a stub and is not useful for real retrieval.[^2][^3]

Also, `uuid=chunk["id"]` is risky because strings like `Baloch_2022_section_0_0` are not standard UUIDs, so converting them deterministically with `uuid.uuid5(...)` is safer. Weaviate examples and docs use proper UUIDs or deterministic UUID generation for object IDs.[^6][^7]

## Better query option

If you want to verify semantic retrieval, add a search function once you have real embeddings. With self-provided vectors, semantic quality depends entirely on the embeddings you insert, because Weaviate will not create them for you automatically.[^3][^2]

A simple next step would be to add:

```python
def semantic_search(query_text, limit=5):
    client = connect()
    try:
        collection = client.collections.get(COLLECTION_NAME)
        query_vector = get_embedding(query_text)

        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit
        )

        print(f"Top {len(response.objects)} matches")
        for i, obj in enumerate(response.objects, start=1):
            props = obj.properties
            print(f"\n--- Result {i} ---")
            print("document_id:", props.get("document_id"))
            print("section:", props.get("section"))
            print("text preview:", props.get("text", "")[:250], "...")
    finally:
        client.close()
```

Then call it with a mode like `search` after you wire in a real embedding model.[^2][^3]

## One cleaner alternative

If you do **not** want to generate embeddings yourself, you can configure a Weaviate vectorizer instead of `self_provided()`, and then insert only the properties without passing `vector=...`. Weaviate’s docs show this pattern as an alternative to bring-your-own vectors.[^3][^2]

A practical example is:

- keep `self_provided()` if you already have an embedding pipeline,
- switch to a built-in or integrated vectorizer if you want simpler ingestion.[^2][^3]

Paste your current file path and embedding method next, and I’ll rewrite the script exactly for your project structure.
<span style="display:none">[^10][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://docs.weaviate.io/weaviate/manage-objects/import

[^2]: https://github.com/weaviate/weaviate

[^3]: https://docs.weaviate.io/weaviate/client-libraries/python

[^4]: https://forum.weaviate.io/t/fetch-objects-only-returns-10-items-is-limit-is-no-specified/3026

[^5]: https://docs.weaviate.io/weaviate/tutorials/query

[^6]: https://docs.weaviate.io/weaviate/manage-objects/create

[^7]: https://forum.weaviate.io/t/subject-unable-to-modify-default-query-limits-in-weaviate-deployment-retreving-only-100-results/1576

[^8]: https://docs.langchain.com/oss/python/integrations/vectorstores/weaviate

[^9]: https://libraries.io/go/github.com%2Fweaviate%2Fweaviate

[^10]: https://forum.weaviate.io/t/how-to-add-new-property-to-an-existing-collection-with-v4-client/1850

