"""
Document ingestion pipeline.

Reads PDF files from a directory, extracts content with Docling,
chunks the text, generates embeddings via the configured provider,
and stores everything in PostgreSQL + pgvector.
"""

import argparse
import asyncio
from datetime import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional
from pathlib import Path

from dotenv import load_dotenv
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning)

from app.db_utils import close_database, initialize_database, db_pool, execute_init_sql
from app.models import IngestionConfig, IngestionResult
from app.providers import get_embedding_provider

from .extract_files import create_pdf_extractor, PDFExtractionConfig
from .chunker import ChunkingConfig, DocumentChunk, create_chunker

load_dotenv()

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """End-to-end pipeline for ingesting PDF documents into the vector store."""

    def __init__(
        self,
        config: IngestionConfig,
        documents_folder: str = "documents",
        clean_before_ingest: bool = False,
        sql_schema_path: str = "sql/schema.sql",
    ):
        self.config = config
        self.documents_folder = documents_folder
        self.clean_before_ingest = clean_before_ingest
        self.sql_schema_path = sql_schema_path

        # PDF extraction config
        self.extractor_config = PDFExtractionConfig(
            enable_ocr=False,
            images_scale=1.0,
            include_images=False,
            include_tables=False,
        )

        # Chunking config
        self.chunker_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_chunk_size=config.max_chunk_size,
            use_semantic_splitting=config.use_semantic_chunking,
        )

        self.extractor = create_pdf_extractor(self.extractor_config)
        self.chunker = create_chunker(self.chunker_config)
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self):
        """Initialise database connections."""
        if self._initialized:
            return
        logger.info("Initialising ingestion pipeline …")
        await initialize_database()
        await execute_init_sql(self.sql_schema_path)
        self._initialized = True
        logger.info("Ingestion pipeline ready")

    async def close(self):
        """Close database connections."""
        if self._initialized:
            await close_database()
            self._initialized = False

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def _clean_databases(self):
        """Truncate all data tables."""
        logger.warning("Cleaning existing data …")
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM messages")
                await conn.execute("DELETE FROM sessions")
                await conn.execute("DELETE FROM chunks")
                await conn.execute("DELETE FROM documents")
        logger.info("Database cleaned")

    async def _ingest_single_document(self, file_path: str) -> IngestionResult:
        """Process and store a single PDF document."""
        start_time = datetime.now()

        document_content, document_metadata = self.extractor.extract_pdf_content(
            file_path
        )
        document_source = os.path.relpath(file_path, self.documents_folder)
        document_title = document_metadata.get("title", document_source)

        logger.info("Processing: %s", document_title)
        logger.info(
            "Found %d images, %d tables",
            document_metadata.get("pictures", 0),
            document_metadata.get("tables", 0),
        )

        # Chunk
        main_chunks = self.chunker.chunk_content(
            content=document_content,
            title=document_title,
            source=document_source,
            metadata=document_metadata,
        )

        if not main_chunks:
            logger.warning("No chunks created for %s", document_title)
            return IngestionResult(
                document_id="",
                title=document_title,
                chunks_created=0,
                processing_time_ms=(
                    (datetime.now() - start_time).total_seconds() * 1000
                ),
            )

        logger.info("Chunks created: %d", len(main_chunks))

        # Embed
        embedded_chunks = await self._embed_chunks(main_chunks)
        logger.info("Embeddings generated: %d", len(embedded_chunks))

        # Persist
        document_id = await self._save_to_postgres(
            document_title,
            document_source,
            document_content,
            embedded_chunks,
            document_metadata,
        )
        logger.info("Saved to PostgreSQL: %s", document_id)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        return IngestionResult(
            document_id=document_id,
            title=document_title,
            chunks_created=len(main_chunks),
            processing_time_ms=processing_time,
        )

    async def ingest_documents(
        self, progress_callback: Optional[callable] = None
    ) -> List[IngestionResult]:
        """Ingest all PDFs from the configured documents folder."""
        if not self._initialized:
            await self.initialize()

        if self.clean_before_ingest:
            await self._clean_databases()

        pdf_files = self._find_pdfs_in_directory(self.documents_folder)
        if not pdf_files:
            logger.warning("No PDF files found in %s", self.documents_folder)
            return []

        logger.info("Found %d PDF files to process", len(pdf_files))
        results: List[IngestionResult] = []

        for i, file_path in enumerate(pdf_files):
            try:
                logger.info(
                    "Processing file %d/%d: %s", i + 1, len(pdf_files), file_path
                )
                result = await self._ingest_single_document(file_path)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(pdf_files))
            except Exception as exc:
                logger.error("Failed to process %s: %s", file_path, exc)
                results.append(
                    IngestionResult(
                        document_id="",
                        title=os.path.basename(file_path),
                        chunks_created=0,
                        processing_time_ms=0,
                    )
                )

        total_chunks = sum(r.chunks_created for r in results)
        logger.info(
            "Ingestion complete: %d documents, %d chunks", len(results), total_chunks
        )
        return results

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    async def _embed_chunks(
        self, chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """Generate embeddings for all chunks using the configured provider."""
        provider = get_embedding_provider()
        texts = [chunk.content for chunk in chunks]
        vectors = await provider.embed_batch(texts)

        embedded: List[DocumentChunk] = []
        for chunk, vector in zip(chunks, vectors):
            new_chunk = DocumentChunk(
                content=chunk.content,
                index=chunk.index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata={
                    **chunk.metadata,
                    "embedding_model": provider.model_name,
                    "embedding_generated_at": datetime.now().isoformat(),
                },
            )
            new_chunk.embedding = vector
            embedded.append(new_chunk)

        return embedded

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _find_pdfs_in_directory(
        directory: str, recursive: bool = True
    ) -> List[str]:
        """Find all PDF files in *directory*."""
        directory_path = Path(directory)
        if not directory_path.exists() or not directory_path.is_dir():
            raise FileNotFoundError(
                f"Directory not found: {directory_path}"
            )

        if recursive:
            pdf_files = list(directory_path.rglob("*.pdf"))
        else:
            pdf_files = list(directory_path.glob("*.pdf"))

        pdf_paths = [str(pdf.resolve()) for pdf in pdf_files if pdf.is_file()]
        logger.info("Found %d PDF files in %s", len(pdf_paths), directory_path)
        return pdf_paths

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _save_to_postgres(
        self,
        title: str,
        source: str,
        content: str,
        chunks: List[DocumentChunk],
        metadata: Dict[str, Any],
    ) -> str:
        """Save document and its chunks to PostgreSQL."""
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                doc_result = await conn.fetchrow(
                    """
                    INSERT INTO documents (title, source, content, metadata)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id::text
                    """,
                    title,
                    source,
                    content,
                    json.dumps(metadata),
                )
                document_id = doc_result["id"]

                for chunk in chunks:
                    embedding_data = None
                    if hasattr(chunk, "embedding") and chunk.embedding:
                        embedding_data = (
                            "[" + ",".join(map(str, chunk.embedding)) + "]"
                        )

                    chunk_meta = {
                        **chunk.metadata,
                        "chunk_type": chunk.metadata.get("content_type", "text"),
                    }

                    await conn.execute(
                        """
                        INSERT INTO chunks
                            (document_id, content, embedding, chunk_index, metadata, token_count)
                        VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
                        """,
                        document_id,
                        chunk.content,
                        embedding_data,
                        chunk.index,
                        json.dumps(chunk_meta),
                        (
                            chunk.token_count
                            if hasattr(chunk, "token_count")
                            else len(chunk.content.split())
                        ),
                    )

                return document_id


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

async def main():
    """Run the ingestion pipeline from the command line."""
    parser = argparse.ArgumentParser(
        description="Agentic RAG Engine — Document Ingestion"
    )
    parser.add_argument(
        "--documents", "-d", default="documents", help="Documents folder path"
    )
    parser.add_argument(
        "--clean", "-c", action="store_true", help="Clean DB before ingestion"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=850, help="Chunk size"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=150, help="Chunk overlap"
    )
    parser.add_argument(
        "--no-semantic", action="store_true", help="Disable semantic chunking"
    )
    parser.add_argument(
        "--no-images", action="store_true", help="Skip image extraction"
    )
    parser.add_argument(
        "--no-tables", action="store_true", help="Skip table extraction"
    )
    parser.add_argument(
        "--sql-schema-path", "-sql", default="sql/schema.sql", help="Schema SQL path"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = IngestionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_semantic_chunking=not args.no_semantic,
    )

    pipeline = DocumentIngestionPipeline(
        config=config,
        documents_folder=args.documents,
        clean_before_ingest=args.clean,
        sql_schema_path=args.sql_schema_path,
    )

    if args.no_images:
        pipeline.extractor_config.include_images = False
    if args.no_tables:
        pipeline.extractor_config.include_tables = False

    def progress_callback(current: int, total: int):
        print(f"Progress: {current}/{total} documents processed")

    try:
        start_time = datetime.now()
        results = await pipeline.ingest_documents(progress_callback)
        total_time = (datetime.now() - start_time).total_seconds()

        print("\n" + "=" * 60)
        print("INGESTION SUMMARY")
        print("=" * 60)
        print(f"Documents processed: {len(results)}")
        print(f"Total chunks created: {sum(r.chunks_created for r in results)}")
        print(f"Total processing time: {total_time:.2f}s")
        print("=" * 60)

        for result in results:
            if result.chunks_created > 0:
                logger.info(
                    "%s: %d chunks (%.1fs)",
                    result.title,
                    result.chunks_created,
                    result.processing_time_ms / 1000,
                )
            else:
                logger.warning("%s: Failed to process", result.title)

    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user")
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        raise
    finally:
        await pipeline.close()


async def ingest_single_file(file_path: str, filename: Optional[str] = None) -> IngestionResult:
    """Ingest any file (PDF, code, markdown, text) into the vector database."""
    from .extract_files import extract_file_content
    from .chunker import ChunkingConfig, DocumentChunk, create_chunker

    start_time = datetime.now()
    path = Path(file_path)
    display_title = filename or path.name

    document_content, document_metadata = extract_file_content(str(path))
    document_metadata["title"] = display_title

    # For code/text, recursive splitting works best and prevents huge chunks
    chunker = create_chunker(
        ChunkingConfig(
            chunk_size=700,
            chunk_overlap=120,
            max_chunk_size=1200,
            use_semantic_splitting=False,
        )
    )

    chunks = chunker.chunk_content(
        content=document_content,
        title=display_title,
        source=display_title,
        metadata=document_metadata,
    )

    if not chunks and document_content.strip():
        chunks = [
            DocumentChunk(
                content=document_content[:1500],
                index=0,
                start_char=0,
                end_char=min(len(document_content), 1500),
                metadata=document_metadata,
                token_count=len(document_content[:1500]) // 4,
            )
        ]

    # Generate embeddings
    provider = get_embedding_provider()
    texts = [c.content for c in chunks]
    vectors = await provider.embed_batch(texts)
    for chunk, vec in zip(chunks, vectors):
        chunk.embedding = vec

    # Save to PostgreSQL
    pipeline = DocumentIngestionPipeline(IngestionConfig())
    document_id = await pipeline._save_to_postgres(
        title=display_title,
        source=display_title,
        content=document_content,
        chunks=chunks,
        metadata=document_metadata,
    )

    elapsed = (datetime.now() - start_time).total_seconds() * 1000
    return IngestionResult(
        document_id=document_id,
        title=display_title,
        chunks_created=len(chunks),
        processing_time_ms=elapsed,
    )


if __name__ == "__main__":
    asyncio.run(main())
