"""Index the fictional billing policies in a configured connected workspace."""

import asyncio
from pathlib import Path
from app.db_utils import initialize_database, execute_init_sql, close_database
from app.ingestion.live import ingest_pages


async def main():
    root = Path(__file__).resolve().parents[1]
    await initialize_database()
    try:
        await execute_init_sql(str(root / "sql/schema.sql"))
        for path in sorted((root / "sample_data/billing").glob("*.md")):
            result = await ingest_pages(
                path.stem.replace("-", " ").title(),
                [(None, path.read_text(encoding="utf-8"))],
            )
            print(
                f"{result['title']}: {'already indexed' if result['duplicate'] else 'indexed'}"
            )
    finally:
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
