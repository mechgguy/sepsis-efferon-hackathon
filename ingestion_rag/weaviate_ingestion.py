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
            vector=[0.01, 0.02, 0.03]  # replace with your real embedding
        )

client.close()