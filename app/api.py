"""
FastAPI application — Agentic RAG Engine REST API.

Exposes chat (streaming + non-streaming), search, document, session,
and health-check endpoints.
"""

import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
from dotenv import load_dotenv
from pydantic_ai.messages import PartStartEvent, PartDeltaEvent, TextPartDelta

from .agent import rag_agent, AgentDependencies
from .db_utils import (
    execute_init_sql,
    initialize_database,
    close_database,
    create_session,
    get_session,
    add_message,
    get_session_messages,
    test_connection,
)
from .models import (
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResponse,
    ErrorResponse,
    HealthStatus,
    ToolCall,
    EvaluationRequest,
    EvaluationResult,
)
from .evaluator import get_evaluator
from .tools import (
    vector_search_tool,
    hybrid_search_tool,
    list_documents_tool,
    VectorSearchInput,
    HybridSearchInput,
    DocumentListInput,
)

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_ENV = os.getenv("APP_ENV", "development")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

if APP_ENV == "development":
    logger.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("Starting Agentic RAG Engine …")

    try:
        await initialize_database()
        await execute_init_sql("sql/schema.sql")
        logger.info("Database initialised")

        db_ok = await test_connection()
        if not db_ok:
            logger.error("Database connection check failed")

        logger.info("System startup complete")
    except Exception as exc:
        logger.error("Startup failed: %s", exc)
        raise

    yield

    logger.info("Shutting down …")
    try:
        await close_database()
        logger.info("Connections closed")
    except Exception as exc:
        logger.error("Shutdown error: %s", exc)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agentic RAG Engine",
    description=(
        "Enterprise-grade Retrieval-Augmented Generation API with "
        "agentic workflows, hybrid search, and local LLM support."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount frontend static files
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

async def get_or_create_session(request: ChatRequest) -> str:
    """Return an existing session or create a new one."""
    if request.session_id and request.session_id.lower() != "string":
        session = await get_session(request.session_id)
        if session:
            return request.session_id

    user_id = request.user_id if request.user_id and request.user_id.lower() != "string" else None
    metadata = request.metadata if request.metadata != {"additionalProp1": {}} else {}
    return await create_session(
        user_id=user_id, metadata=metadata
    )


async def get_conversation_context(
    session_id: str, max_messages: int = 10
) -> List[Dict[str, str]]:
    """Retrieve recent conversation turns for context injection."""
    messages = await get_session_messages(session_id, limit=max_messages)
    return [{"role": msg["role"], "content": msg["content"]} for msg in messages]


def extract_tool_calls(result) -> List[ToolCall]:
    """Extract tool-call metadata from a Pydantic AI result."""
    tools_used: List[ToolCall] = []

    try:
        if hasattr(result, "all_messages") and callable(result.all_messages):
            messages = result.all_messages()
        elif hasattr(result, "all_messages"):
            messages = result.all_messages
        elif hasattr(result, "messages") and callable(result.messages):
            messages = result.messages()
        elif hasattr(result, "messages"):
            messages = result.messages
        else:
            messages = []

        for message in messages:
            if hasattr(message, "parts"):
                for part in message.parts:
                    part_type = part.__class__.__name__
                    if "ToolCall" in part_type or "tool_call" in getattr(part, "part_kind", ""):
                        try:
                            tool_name = (
                                str(part.tool_name)
                                if hasattr(part, "tool_name")
                                else "unknown"
                            )

                            tool_args: dict = {}
                            if hasattr(part, "args") and part.args is not None:
                                if isinstance(part.args, str):
                                    try:
                                        tool_args = json.loads(part.args)
                                    except json.JSONDecodeError:
                                        tool_args = {}
                                elif isinstance(part.args, dict):
                                    tool_args = part.args

                            tool_call_id = None
                            if hasattr(part, "tool_call_id") and part.tool_call_id:
                                tool_call_id = str(part.tool_call_id)

                            tools_used.append(
                                ToolCall(
                                    tool_name=tool_name,
                                    args=tool_args,
                                    tool_call_id=tool_call_id,
                                )
                            )
                        except Exception:
                            continue
    except Exception as exc:
        logger.warning("Failed to extract tool calls: %s", exc)

    return tools_used


async def save_conversation_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Persist a user ↔ assistant exchange."""
    await add_message(
        session_id=session_id,
        role="user",
        content=user_message,
        metadata=metadata or {},
    )
    await add_message(
        session_id=session_id,
        role="assistant",
        content=assistant_message,
        metadata=metadata or {},
    )


async def execute_agent(
    message: str,
    session_id: str,
    user_id: Optional[str] = None,
    save_conversation: bool = True,
) -> tuple[str, List[ToolCall], List[str]]:
    """Run the agent with context and return (response, tool_calls, retrieved_contexts)."""
    try:
        deps = AgentDependencies(session_id=session_id, user_id=user_id)
        context = await get_conversation_context(session_id)

        full_prompt = message
        if context:
            context_str = "\n".join(
                f"{msg['role']}: {msg['content']}" for msg in context[-6:]
            )
            full_prompt = (
                f"Previous conversation:\n{context_str}\n\n"
                f"Current question: {message}"
            )

        result = await rag_agent.run(full_prompt, deps=deps)
        
        # Handle pydantic_ai result attributes across versions
        if hasattr(result, "output"):
            response = str(result.output)
        elif hasattr(result, "data"):
            response = str(result.data)
        else:
            response = str(result)

        tools_used = extract_tool_calls(result)

        # Collect retrieved chunk texts for evaluation
        retrieved_contexts = [c["content"] for c in deps.retrieved_chunks if "content" in c]

        if save_conversation:
            await save_conversation_turn(
                session_id=session_id,
                user_message=message,
                assistant_message=response,
                metadata={"user_id": user_id, "tool_calls": len(tools_used)},
            )

        return response, tools_used, retrieved_contexts

    except Exception as exc:
        logger.error("Agent execution failed: %s", exc)
        error_response = (
            f"I encountered an error while processing your request: {exc}"
        )
        if save_conversation:
            await save_conversation_turn(
                session_id=session_id,
                user_message=message,
                assistant_message=error_response,
                metadata={"error": str(exc)},
            )
        return error_response, [], []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """System health probe."""
    try:
        db_status = await test_connection()
        status = "healthy" if db_status else "unhealthy"

        return HealthStatus(
            status=status,
            database=db_status,
            llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local"),
            version="2.0.0",
            timestamp=datetime.now(),
        )
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint with automatic RAG evaluation."""
    try:
        session_id = await get_or_create_session(request)

        response, tools_used, retrieved_contexts = await execute_agent(
            message=request.message,
            session_id=session_id,
            user_id=request.user_id,
        )

        # Run RAG evaluation
        evaluation = None
        try:
            evaluator = get_evaluator()
            evaluation = await evaluator.evaluate(
                query=request.message,
                response=response,
                contexts=retrieved_contexts,
            )
        except Exception as eval_exc:
            logger.warning("Evaluation failed (non-blocking): %s", eval_exc)

        return ChatResponse(
            message=response,
            session_id=session_id,
            tools_used=tools_used,
            metadata={"search_type": str(request.search_type)},
            evaluation=evaluation,
        )
    except Exception as exc:
        logger.error("Chat endpoint failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint (Server-Sent Events)."""
    try:
        session_id = await get_or_create_session(request)

        async def generate_stream():
            try:
                yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

                deps = AgentDependencies(
                    session_id=session_id, user_id=request.user_id
                )

                context = await get_conversation_context(session_id)
                full_prompt = request.message
                if context:
                    context_str = "\n".join(
                        f"{msg['role']}: {msg['content']}"
                        for msg in context[-6:]
                    )
                    full_prompt = (
                        f"Previous conversation:\n{context_str}\n\n"
                        f"Current question: {request.message}"
                    )

                await add_message(
                    session_id=session_id,
                    role="user",
                    content=request.message,
                    metadata={"user_id": request.user_id},
                )

                full_response = ""

                async with rag_agent.iter(full_prompt, deps=deps) as run:
                    async for node in run:
                        if rag_agent.is_model_request_node(node):
                            async with node.stream(run.ctx) as request_stream:
                                async for event in request_stream:
                                    if (
                                        isinstance(event, PartStartEvent)
                                        and event.part.part_kind == "text"
                                    ):
                                        delta = event.part.content
                                        yield f"data: {json.dumps({'type': 'text', 'content': delta})}\n\n"
                                        full_response += delta
                                    elif isinstance(
                                        event, PartDeltaEvent
                                    ) and isinstance(event.delta, TextPartDelta):
                                        delta = event.delta.content_delta
                                        yield f"data: {json.dumps({'type': 'text', 'content': delta})}\n\n"
                                        full_response += delta

                result = run.result
                tools_used = extract_tool_calls(result)

                if tools_used:
                    tools_data = [
                        {
                            "tool_name": t.tool_name,
                            "args": t.args,
                            "tool_call_id": t.tool_call_id,
                        }
                        for t in tools_used
                    ]
                    yield f"data: {json.dumps({'type': 'tools', 'tools': tools_data})}\n\n"

                await add_message(
                    session_id=session_id,
                    role="assistant",
                    content=full_response,
                    metadata={
                        "streamed": True,
                        "tool_calls": len(tools_used),
                    },
                )

                # Run RAG evaluation and emit as SSE event
                try:
                    retrieved_contexts = [c["content"] for c in deps.retrieved_chunks if "content" in c]
                    evaluator = get_evaluator()
                    evaluation = await evaluator.evaluate(
                        query=request.message,
                        response=full_response,
                        contexts=retrieved_contexts,
                    )
                    eval_data = evaluation.model_dump(mode="json")
                    yield f"data: {json.dumps({'type': 'evaluation', 'evaluation': eval_data})}\n\n"
                except Exception as eval_exc:
                    logger.warning("Stream evaluation failed (non-blocking): %s", eval_exc)

                yield f"data: {json.dumps({'type': 'end'})}\n\n"

            except Exception as exc:
                logger.error("Stream error: %s", exc)
                yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            },
        )
    except Exception as exc:
        logger.error("Streaming chat failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/search/vector")
async def search_vector(request: SearchRequest):
    """Standalone vector similarity search."""
    try:
        input_data = VectorSearchInput(query=request.query, limit=request.limit)
        start = datetime.now()
        results = await vector_search_tool(input_data)
        elapsed = (datetime.now() - start).total_seconds() * 1000

        return SearchResponse(
            results=results,
            total_results=len(results),
            search_type="vector",
            query_time_ms=elapsed,
        )
    except Exception as exc:
        logger.error("Vector search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/search/hybrid")
async def search_hybrid(request: SearchRequest):
    """Standalone hybrid (vector + keyword) search."""
    try:
        input_data = HybridSearchInput(
            query=request.query, limit=request.limit, text_weight=0.3
        )
        start = datetime.now()
        results = await hybrid_search_tool(input_data)
        elapsed = (datetime.now() - start).total_seconds() * 1000

        return SearchResponse(
            results=results,
            total_results=len(results),
            search_type="hybrid",
            query_time_ms=elapsed,
        )
    except Exception as exc:
        logger.error("Hybrid search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Evaluation endpoint
# ---------------------------------------------------------------------------

@app.post("/evaluate", response_model=EvaluationResult)
async def evaluate_response(request: EvaluationRequest):
    """Standalone RAG evaluation endpoint.

    Evaluate any query-response-contexts triplet for confidence,
    faithfulness, relevance, and hallucination risk.
    """
    try:
        evaluator = get_evaluator()
        result = await evaluator.evaluate(
            query=request.query,
            response=request.response,
            contexts=request.contexts,
        )
        return result
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/documents")
async def list_documents_endpoint(limit: int = 20, offset: int = 0):
    """List ingested documents."""
    try:
        input_data = DocumentListInput(limit=limit, offset=offset)
        documents = await list_documents_tool(input_data)
        return {
            "documents": documents,
            "total": len(documents),
            "limit": limit,
            "offset": offset,
        }
    except Exception as exc:
        logger.error("Document listing failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Retrieve session details."""
    try:
        session = await get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Session retrieval failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions."""
    logger.error("Unhandled exception: %s", exc)
    return ErrorResponse(
        error=str(exc),
        error_type=type(exc).__name__,
        request_id=str(uuid.uuid4()),
    )


# ---------------------------------------------------------------------------
# Frontend route
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def serve_frontend():
    """Serve the web UI."""
    index_path = _frontend_dir / "index.html"
    if index_path.is_file():
        return FileResponse(str(index_path), media_type="text/html")
    raise HTTPException(status_code=404, detail="Frontend not found")


@app.get("/style.css", include_in_schema=False)
async def serve_style():
    style_path = _frontend_dir / "style.css"
    if style_path.is_file():
        return FileResponse(str(style_path), media_type="text/css")
    raise HTTPException(status_code=404, detail="Style not found")


@app.get("/app.js", include_in_schema=False)
async def serve_script():
    script_path = _frontend_dir / "app.js"
    if script_path.is_file():
        return FileResponse(str(script_path), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="Script not found")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.api:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=APP_ENV == "development",
        log_level=LOG_LEVEL.lower(),
    )
