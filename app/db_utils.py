"""
PostgreSQL + pgvector database utilities.

Manages connection pooling, session/message CRUD, document storage,
and vector / hybrid search via stored SQL functions.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
import logging

import asyncpg
from asyncpg.pool import Pool
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------

class DatabasePool:
    """Manages a PostgreSQL async connection pool."""

    def __init__(self, database_url: Optional[str] = None):
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "postgres")
        host = os.getenv("DB_HOST", "postgres")
        port = os.getenv("DB_PORT", "5432")
        dbname = os.getenv("DB_NAME", "vector_db")

        self.database_url = (
            database_url
            or f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        )
        self.pool: Optional[Pool] = None

    async def initialize(self):
        """Create the connection pool."""
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                max_inactive_connection_lifetime=300,
                command_timeout=60,
            )
            logger.info("Database connection pool initialised")

    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database connection pool closed")

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        if not self.pool:
            await self.initialize()
        async with self.pool.acquire() as connection:
            yield connection


# Global singleton
db_pool = DatabasePool()


async def initialize_database():
    """Initialise the database connection pool."""
    await db_pool.initialize()


async def close_database():
    """Close the database connection pool."""
    await db_pool.close()


async def execute_init_sql(sql_path: str):
    """Run the schema SQL if tables do not exist yet."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_name = 'documents'
            ) AS exists
        """)

        if row["exists"]:
            logger.info("Schema already initialised — skipping.")
            return

        with open(sql_path, "r") as fh:
            sql = fh.read()
            await conn.execute(sql)
            logger.info("Schema created successfully.")


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

async def create_session(
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    timeout_minutes: int = 60,
) -> str:
    """Create a new chat session and return its UUID."""
    async with db_pool.acquire() as conn:
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=timeout_minutes)
        result = await conn.fetchrow(
            """
            INSERT INTO sessions (user_id, metadata, expires_at)
            VALUES ($1, $2, $3)
            RETURNING id::text
            """,
            user_id,
            json.dumps(metadata or {}),
            expires_at,
        )
        return result["id"]


async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session by ID (returns ``None`` if expired or missing)."""
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT
                id::text,
                user_id,
                metadata,
                created_at,
                updated_at,
                expires_at
            FROM sessions
            WHERE id = $1::uuid
            AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """,
            session_id,
        )

        if result:
            return {
                "id": result["id"],
                "user_id": result["user_id"],
                "metadata": json.loads(result["metadata"]),
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat(),
                "expires_at": (
                    result["expires_at"].isoformat() if result["expires_at"] else None
                ),
            }
        return None


# ---------------------------------------------------------------------------
# Message management
# ---------------------------------------------------------------------------

async def add_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Append a message to a session. Returns the message UUID."""
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            INSERT INTO messages (session_id, role, content, metadata)
            VALUES ($1::uuid, $2, $3, $4)
            RETURNING id::text
            """,
            session_id,
            role,
            content,
            json.dumps(metadata or {}),
        )
        return result["id"]


async def get_session_messages(
    session_id: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return messages for a session ordered chronologically."""
    async with db_pool.acquire() as conn:
        query = """
            SELECT
                id::text,
                role,
                content,
                metadata,
                created_at
            FROM messages
            WHERE session_id = $1::uuid
            ORDER BY created_at
        """
        if limit:
            query += f" LIMIT {limit}"

        results = await conn.fetch(query, session_id)

        return [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"].isoformat(),
            }
            for row in results
        ]


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------

async def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    """Get a single document by UUID."""
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT
                id::text,
                title,
                source,
                content,
                metadata,
                created_at,
                updated_at
            FROM documents
            WHERE id = $1::uuid
            """,
            document_id,
        )

        if result:
            return {
                "id": result["id"],
                "title": result["title"],
                "source": result["source"],
                "content": result["content"],
                "metadata": json.loads(result["metadata"]),
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat(),
            }
        return None


async def list_documents(
    limit: int = 100,
    offset: int = 0,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """List documents with optional JSONB metadata filtering."""
    async with db_pool.acquire() as conn:
        query = """
            SELECT
                d.id::text,
                d.title,
                d.source,
                d.metadata,
                d.created_at,
                d.updated_at,
                COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
        """
        params: list = []
        conditions: list = []

        if metadata_filter:
            conditions.append(f"d.metadata @> ${len(params) + 1}::jsonb")
            params.append(json.dumps(metadata_filter))

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += """
            GROUP BY d.id, d.title, d.source, d.metadata, d.created_at, d.updated_at
            ORDER BY d.created_at DESC
            LIMIT $%d OFFSET $%d
        """ % (len(params) + 1, len(params) + 2)

        params.extend([limit, offset])
        results = await conn.fetch(query, *params)

        return [
            {
                "id": row["id"],
                "title": row["title"],
                "source": row["source"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
                "chunk_count": row["chunk_count"],
            }
            for row in results
        ]


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

async def vector_search(
    embedding: List[float],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Perform vector cosine-similarity search via the ``match_chunks`` function."""
    async with db_pool.acquire() as conn:
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        results = await conn.fetch(
            "SELECT * FROM match_chunks($1::vector, $2)",
            embedding_str,
            limit,
        )
        return [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "content": row["content"],
                "similarity": row["similarity"],
                "metadata": json.loads(row["metadata"]),
                "document_title": row["document_title"],
                "document_source": row["document_source"],
            }
            for row in results
        ]


async def hybrid_search(
    embedding: List[float],
    query_text: str,
    limit: int = 10,
    text_weight: float = 0.3,
) -> List[Dict[str, Any]]:
    """Perform hybrid (vector + keyword) search via the ``hybrid_search`` function."""
    async with db_pool.acquire() as conn:
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        results = await conn.fetch(
            "SELECT * FROM hybrid_search($1::vector, $2, $3, $4)",
            embedding_str,
            query_text,
            limit,
            text_weight,
        )
        return [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "content": row["content"],
                "combined_score": row["combined_score"],
                "vector_similarity": row["vector_similarity"],
                "text_similarity": row["text_similarity"],
                "metadata": json.loads(row["metadata"]),
                "document_title": row["document_title"],
                "document_source": row["document_source"],
            }
            for row in results
        ]


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------

async def get_document_chunks(document_id: str) -> List[Dict[str, Any]]:
    """Return all chunks for a given document, ordered by index."""
    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            "SELECT * FROM get_document_chunks($1::uuid)",
            document_id,
        )
        return [
            {
                "chunk_id": row["chunk_id"],
                "content": row["content"],
                "chunk_index": row["chunk_index"],
                "metadata": json.loads(row["metadata"]),
            }
            for row in results
        ]


async def test_connection() -> bool:
    """Return ``True`` if the database is reachable."""
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as exc:
        logger.error("Database connection test failed: %s", exc)
        return False
