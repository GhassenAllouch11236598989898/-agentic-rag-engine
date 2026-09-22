"""Bounded, transactional ingestion preserving PDF page provenance."""

import hashlib
import json
from ..db_utils import db_pool
from ..providers import get_embedding_provider


async def ingest_pages(title, pages):
    content = "\n\n".join(text for _, text in pages)
    checksum = hashlib.sha256(content.encode()).hexdigest()
    provider = get_embedding_provider()
    chunks = []
    for page, text in pages:
        for start in range(0, len(text), 960):
            part = text[start : start + 1100].strip()
            if part:
                chunks.append((part, page))
    embeddings = []
    for offset in range(0, len(chunks), 32):
        embeddings.extend(
            await provider.embed_batch(
                [text for text, _ in chunks[offset : offset + 32]]
            )
        )
    if len(embeddings) != len(chunks) or any(
        len(v) != provider.dimension for v in embeddings
    ):
        raise ValueError(
            "Embedding dimensions do not match configuration; no document was saved."
        )
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", checksum)
            existing = await conn.fetchval(
                "SELECT id FROM documents WHERE metadata->>'checksum'=$1", checksum
            )
            if existing:
                return {"document_id": str(existing), "duplicate": True, "title": title}
            doc_id = await conn.fetchval(
                "INSERT INTO documents(title,source,content,metadata) VALUES($1,$2,$3,$4) RETURNING id",
                title,
                title,
                content,
                json.dumps(
                    {"checksum": checksum, "embedding_model": provider.model_name}
                ),
            )
            for index, ((text, page), vector) in enumerate(zip(chunks, embeddings)):
                await conn.execute(
                    "INSERT INTO chunks(document_id,content,embedding,chunk_index,metadata,token_count) VALUES($1,$2,$3::vector,$4,$5,$6)",
                    doc_id,
                    text,
                    json.dumps(vector),
                    index,
                    json.dumps({"page": page}),
                    len(text) // 4,
                )
    return {
        "document_id": str(doc_id),
        "title": title,
        "chunks_created": len(chunks),
        "duplicate": False,
    }
