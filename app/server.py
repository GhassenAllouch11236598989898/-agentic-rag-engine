"""Billing Copilot: a support team's evidence-backed reply workspace."""

import asyncio
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .security import Principal, RateLimiter, authenticate, parse_keys
from .support import SupportService
from .workspace import Workspace

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=6000)
    session_id: str | None = Field(None, max_length=80)
    search_type: Literal["hybrid", "vector"] = "hybrid"


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    limit: int = Field(4, ge=1, le=20)


class Feedback(BaseModel):
    value: Literal["helpful", "needs_work"]


def extract_upload(path, filename):
    if Path(filename).suffix.lower() == ".pdf":
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        if Path(path).read_bytes()[:5] != b"%PDF-":
            raise ValueError("This file is not a valid PDF.")
        try:
            reader = PdfReader(path)
            if reader.is_encrypted:
                raise ValueError("Upload an unencrypted PDF.")
            if len(reader.pages) > 200:
                raise ValueError("Please split PDFs longer than 200 pages.")
            pages = [
                (i + 1, page.extract_text() or "")
                for i, page in enumerate(reader.pages)
            ]
        except PdfReadError as exc:
            raise ValueError("This PDF is damaged or could not be read.") from exc
    else:
        try:
            text = Path(path).read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "Text and Markdown files must use UTF-8 encoding."
            ) from exc
        if "\x00" in text:
            raise ValueError("Binary files are not supported.")
        pages = [(None, text)]
    if sum(len(text) for _, text in pages) > 2_000_000:
        raise ValueError("Extracted text exceeds the 2 million character limit.")
    if sum(len(text.strip()) for _, text in pages) < 20:
        raise ValueError("No usable text found. Scanned PDFs need OCR before upload.")
    return pages


def create_app(mode=None, data_path=None, keys=None, seed=True):
    mode = mode or os.getenv("APP_MODE", "demo")
    if mode not in {"demo", "live"}:
        raise ValueError("APP_MODE must be demo or live.")
    key_config = parse_keys(
        keys if keys is not None else os.getenv("RELAY_API_KEYS", "{}")
    )
    if (mode == "live" or os.getenv("APP_ENV") == "production") and not key_config:
        raise ValueError("Connected and production deployments require RELAY_API_KEYS.")
    store = Workspace(data_path or ROOT / ".data" / f"billing-{mode}.sqlite3")
    service = SupportService(store, mode)
    limiter = RateLimiter(int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")))
    expensive = asyncio.Semaphore(2)

    @asynccontextmanager
    async def lifespan(app):
        await asyncio.to_thread(store.initialize)
        if mode == "demo" and seed and not await asyncio.to_thread(store.is_seeded):
            for file in sorted((ROOT / "sample_data" / "billing").glob("*.md")):
                await asyncio.to_thread(
                    store.ingest,
                    file.stem.replace("-", " ").title(),
                    [(None, file.read_text(encoding="utf-8"))],
                    "sample",
                )
            await asyncio.to_thread(store.mark_seeded)
        if mode == "live":
            from .db_utils import initialize_database, execute_init_sql, close_database

            await initialize_database()
            await execute_init_sql(str(ROOT / "sql" / "schema.sql"))
        yield
        if mode == "live":
            await close_database()

    app = FastAPI(
        title="Billing Copilot",
        version="3.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.workspace, app.state.service = store, service

    @app.middleware("http")
    async def protect(request, call_next):
        if request.url.path.startswith("/api/"):
            if not limiter.allow(request.client.host if request.client else "unknown"):
                return JSONResponse(
                    {"detail": "Too many requests. Please try again in a minute."},
                    status_code=429,
                    headers={"Retry-After": "60"},
                )
            if key_config:
                principal = authenticate(
                    request.headers.get("authorization", ""), key_config
                )
                if not principal:
                    return JSONResponse(
                        {"detail": "Enter your workspace access key."},
                        status_code=401,
                        headers={"WWW-Authenticate": "Bearer"},
                    )
            else:
                if request.client and request.client.host not in {
                    "127.0.0.1",
                    "::1",
                    "testclient",
                }:
                    return JSONResponse(
                        {"detail": "Remote access requires workspace keys."},
                        status_code=403,
                    )
                principal = Principal("local-demo", "admin")
            request.state.principal = principal
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        elif request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.exception_handler(Exception)
    async def unexpected(request, exc):
        logger.error(
            "Request failed: %s",
            request.url.path,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(
            {
                "detail": "The service could not complete this request. Check the server logs and try again."
            },
            status_code=503,
        )

    @app.get("/health")
    async def health():
        healthy = True
        if mode == "live":
            from .db_utils import test_connection

            healthy = await test_connection()
        return JSONResponse(
            {
                "status": "healthy" if healthy else "unhealthy",
                "mode": mode,
                "checks": {"database": healthy},
                "note": "Provider inference is checked when generating a draft.",
            },
            status_code=200 if healthy else 503,
        )

    @app.get("/api/workspace")
    async def workspace(request: Request):
        return {
            "name": "Northstar Cloud Billing"
            if mode == "demo"
            else os.getenv("WORKSPACE_NAME", "Billing workspace"),
            "product": "Billing Copilot",
            "mode": mode,
            "role": request.state.principal.role,
            "user": request.state.principal.name,
            "description": "Fictional B2B software company"
            if mode == "demo"
            else "Connected support workspace",
            "tickets": json.loads(
                (ROOT / "sample_data" / "tickets.json").read_text(encoding="utf-8")
            )
            if mode == "demo"
            else [],
        }

    async def draft(request, body, persist=True):
        try:
            async with expensive:
                return await asyncio.wait_for(
                    service.draft(
                        request.state.principal.name,
                        body.message,
                        body.session_id,
                        body.search_type,
                        persist,
                    ),
                    timeout=120,
                )
        except LookupError as exc:
            raise HTTPException(404, "Conversation not found.") from exc
        except TimeoutError as exc:
            raise HTTPException(
                504, "Draft generation timed out. Try again or check the AI provider."
            ) from exc

    @app.post("/api/chat")
    async def chat(body: ChatRequest, request: Request):
        return await draft(request, body)

    @app.post("/api/chat/stream")
    async def chat_stream(body: ChatRequest, request: Request):
        if body.session_id:
            try:
                await asyncio.to_thread(
                    store.session, request.state.principal.name, body.session_id
                )
            except LookupError as exc:
                raise HTTPException(404, "Conversation not found.") from exc

        async def events():
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "progress",
                        "stage": "Searching knowledge and preparing a draft",
                    }
                )
                + "\n\n"
            )
            try:
                result = await draft(request, body)
                yield "data: " + json.dumps({"type": "result", **result}) + "\n\n"
                yield 'data: {"type":"end"}\n\n'
            except Exception:
                logger.exception("Draft stream failed")
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "error",
                            "message": "Draft unavailable. Check your connection or provider and try again.",
                        }
                    )
                    + "\n\n"
                )

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/sessions/{session_id}")
    async def session(session_id: str, request: Request):
        try:
            return {
                "messages": await asyncio.to_thread(
                    store.history, request.state.principal.name, session_id
                )
            }
        except LookupError as exc:
            raise HTTPException(404, "Conversation not found.") from exc

    @app.post("/api/search/{strategy}")
    async def search(strategy: Literal["hybrid", "vector"], body: SearchRequest):
        return {"results": await service.retrieve(body.query, strategy, body.limit)}

    @app.get("/api/documents")
    async def documents():
        if mode == "demo":
            return {"documents": await asyncio.to_thread(store.documents)}
        from .db_utils import list_documents

        return {"documents": await list_documents(limit=100)}

    @app.get("/api/documents/{document_id}")
    async def document(document_id: str):
        if mode == "demo":
            result = await asyncio.to_thread(store.document, document_id)
        else:
            from .db_utils import get_document
            import uuid

            try:
                uuid.UUID(document_id)
            except ValueError as exc:
                raise HTTPException(404, "Document not found.") from exc
            result = await get_document(document_id)
        if not result:
            raise HTTPException(404, "Document not found.")
        return result

    @app.post("/api/documents", status_code=201)
    async def upload(request: Request, file: UploadFile = File(...)):
        if request.state.principal.role != "admin":
            raise HTTPException(
                403, "An administrator must upload knowledge documents."
            )
        filename = (file.filename or "").replace("\\", "/").split("/")[-1][:150]
        if Path(filename).suffix.lower() not in {".pdf", ".txt", ".md"}:
            await file.close()
            raise HTTPException(
                415, "Supported formats: PDF, Markdown, and UTF-8 text."
            )
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(filename).suffix
            ) as output:
                temporary = output.name
                total = 0
                while block := await file.read(65536):
                    total += len(block)
                    if total > 10 * 1024 * 1024:
                        raise HTTPException(413, "Files must be 10 MB or smaller.")
                    output.write(block)
            async with expensive:
                pages = await asyncio.to_thread(extract_upload, temporary, filename)
                if mode == "demo":
                    return await asyncio.to_thread(store.ingest, filename, pages)
                from .ingestion.live import ingest_pages

                return await ingest_pages(filename, pages)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            await file.close()
            if temporary:
                Path(temporary).unlink(missing_ok=True)

    @app.delete("/api/documents/{document_id}")
    async def delete_document(document_id: str, request: Request):
        if request.state.principal.role != "admin":
            raise HTTPException(
                403, "An administrator must remove knowledge documents."
            )
        if mode == "demo":
            deleted = await asyncio.to_thread(store.delete_document, document_id)
        else:
            from .db_utils import db_pool
            import uuid

            try:
                uuid.UUID(document_id)
            except ValueError as exc:
                raise HTTPException(404, "Document not found.") from exc
            async with db_pool.acquire() as conn:
                deleted = await conn.fetchval(
                    "DELETE FROM documents WHERE id=$1::uuid RETURNING id", document_id
                )
        if not deleted:
            raise HTTPException(404, "Document not found.")
        return {"deleted": True}

    @app.get("/api/activity")
    async def activity(request: Request):
        return {
            "drafts": await asyncio.to_thread(
                store.activity, request.state.principal.name
            )
        }

    @app.post("/api/drafts/{draft_id}/feedback")
    async def feedback(draft_id: str, body: Feedback, request: Request):
        if not await asyncio.to_thread(
            store.feedback, request.state.principal.name, draft_id, body.value
        ):
            raise HTTPException(404, "Draft not found.")
        return {"saved": True}

    @app.post("/api/benchmark/run")
    async def benchmark(request: Request):
        from .benchmark import run_checks

        if mode != "demo":
            raise HTTPException(
                409,
                "These checks target the sample policies. Use a client-specific evaluation dataset for connected knowledge.",
            )
        async with expensive:
            return await run_checks(service, request.state.principal.name)

    app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")

    @app.get("/")
    async def frontend():
        return FileResponse(ROOT / "frontend" / "index.html")

    return app
