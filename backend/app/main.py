from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Callable
from threading import Lock
from time import monotonic
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import Settings, get_async_openai_client, get_openai_client, get_qdrant_client, get_settings
from app.rag import (
    build_augmented_messages,
    extract_latest_user_message,
    retrieve_context,
    stream_chat_completion,
)

app = FastAPI(title="ElevenLabs Knowledge Agent", version="0.1.0")
logger = logging.getLogger(__name__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="custom")
    messages: list[ChatMessage]
    stream: bool = True


class ConversationTokenResponse(BaseModel):
    token: str


class RateLimiter:
    def __init__(self) -> None:
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str, *, limit: int, window_seconds: int) -> bool:
        now = monotonic()
        with self._lock:
            timestamps = self._requests[key]
            while timestamps and now - timestamps[0] >= window_seconds:
                timestamps.popleft()
            if len(timestamps) >= limit:
                return False
            timestamps.append(now)
            return True


token_rate_limiter = RateLimiter()


def is_missing_qdrant_collection_error(exc: UnexpectedResponse, collection_name: str) -> bool:
    status_code = getattr(exc, "status_code", None)
    detail = " ".join(
        str(part)
        for part in (
            getattr(exc, "reason_phrase", ""),
            getattr(exc, "content", b"").decode("utf-8", errors="ignore")
            if isinstance(getattr(exc, "content", b""), bytes)
            else str(getattr(exc, "content", "")),
            str(exc),
        )
        if part
    ).lower()
    return status_code == 404 and "collection" in detail and collection_name.lower() in detail


def get_qdrant_collection_stats(settings: Settings) -> dict[str, int | bool | None]:
    qdrant_client = get_qdrant_client()
    try:
        collection_info = qdrant_client.get_collection(settings.qdrant_collection_name)
    except UnexpectedResponse as exc:
        if is_missing_qdrant_collection_error(exc, settings.qdrant_collection_name):
            return {"qdrant_collection_exists": False, "qdrant_points_count": 0}
        raise

    points_count = getattr(collection_info, "points_count", None)
    if points_count is None:
        result = getattr(collection_info, "result", None)
        points_count = getattr(result, "points_count", None)

    return {
        "qdrant_collection_exists": True,
        "qdrant_points_count": int(points_count) if points_count is not None else None,
    }


def get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        first_ip = forwarded_for.split(",")[0].strip()
        if first_ip:
            return first_ip
    return request.client.host if request.client else "unknown"


def is_origin_allowed(origin: str | None, settings: Settings) -> bool:
    if not origin:
        return True
    return origin in settings.allowed_origin_list


async def fetch_conversation_token(settings: Settings) -> str:
    if not settings.elevenlabs_api_key or not settings.elevenlabs_agent_id:
        raise HTTPException(
            status_code=500,
            detail="Missing ELEVENLABS_API_KEY or ELEVENLABS_AGENT_ID in the backend environment.",
        )

    url = "https://api.elevenlabs.io/v1/convai/conversation/token"
    params = {"agent_id": settings.elevenlabs_agent_id}
    headers = {"xi-api-key": settings.elevenlabs_api_key}

    try:
        async with httpx.AsyncClient(timeout=settings.conversation_token_timeout_seconds) as client:
            response = await client.get(url, params=params, headers=headers)
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Timed out while requesting an ElevenLabs conversation token.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Failed to reach ElevenLabs while requesting a conversation token.") from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail="ElevenLabs rejected the conversation token request.")

    token = response.json().get("token")
    if not token:
        raise HTTPException(status_code=502, detail="ElevenLabs returned no conversation token.")
    return str(token)


def require_debug_retrieval_enabled(settings: Settings) -> None:
    if not settings.enable_debug_retrieval:
        raise HTTPException(status_code=404, detail="Not found")


@app.middleware("http")
async def reject_disallowed_cross_origin_token_requests(request: Request, call_next: Callable):
    if request.url.path == "/v1/elevenlabs/conversation-token" and request.method == "POST":
        origin = request.headers.get("origin")
        if not is_origin_allowed(origin, get_settings()):
            return JSONResponse(status_code=403, content={"detail": "Origin is not allowed."})
    return await call_next(request)


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, str | bool | int | None]:
    payload: dict[str, str | bool | int | None] = {
        "status": "ok",
        "qdrant_in_memory": settings.qdrant_in_memory,
        "collection": settings.qdrant_collection_name,
    }
    payload.update(get_qdrant_collection_stats(settings))
    return payload


@app.get("/debug/retrieval")
def debug_retrieval(
    q: str,
    limit: int = 3,
    settings: Settings = Depends(get_settings),
):
    require_debug_retrieval_enabled(settings)
    openai_client = get_openai_client()
    qdrant_client = get_qdrant_client()
    chunks = retrieve_context(
        query=q,
        qdrant_client=qdrant_client,
        openai_client=openai_client,
        settings=settings,
        limit=limit,
    )
    return {
        "query": q,
        "collection": settings.qdrant_collection_name,
        "limit": limit,
        "matches": [chunk.model_dump(mode="json") for chunk in chunks],
    }


@app.post("/v1/elevenlabs/conversation-token", response_model=ConversationTokenResponse)
async def create_conversation_token(
    request: Request,
    settings: Settings = Depends(get_settings),
):
    client_ip = get_client_ip(request)
    allowed = token_rate_limiter.allow(
        client_ip,
        limit=settings.conversation_token_rate_limit,
        window_seconds=settings.conversation_token_rate_limit_window_seconds,
    )
    if not allowed:
        raise HTTPException(status_code=429, detail="Too many conversation token requests. Please retry shortly.")

    token = await fetch_conversation_token(settings)
    return ConversationTokenResponse(token=token)


async def handle_chat_completion(
    payload: ChatCompletionRequest,
    settings: Settings = Depends(get_settings),
):
    if not payload.stream:
        raise HTTPException(status_code=400, detail="Only stream=true is supported for the ElevenLabs webhook.")

    openai_client = get_openai_client()
    async_openai_client = get_async_openai_client()
    qdrant_client = get_qdrant_client()

    messages = [message.model_dump(mode="json") for message in payload.messages]
    user_query = extract_latest_user_message(messages)

    try:
        chunks = retrieve_context(
            query=user_query,
            qdrant_client=qdrant_client,
            openai_client=openai_client,
            settings=settings,
        )
    except UnexpectedResponse as exc:
        if is_missing_qdrant_collection_error(exc, settings.qdrant_collection_name):
            logger.warning("Qdrant collection '%s' not found. Falling back to base model.", settings.qdrant_collection_name)
            chunks = []
        else:
            raise HTTPException(status_code=500, detail=f"Qdrant request failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        detail = str(exc) or exc.__class__.__name__
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {detail}") from exc

    logger.info(
        "Retrieved %s chunks from collection=%s",
        len(chunks),
        settings.qdrant_collection_name,
    )
    if settings.enable_debug_retrieval:
        logger.info(
            "Retrieval details sources=%s",
            [f"{chunk.source}:{chunk.chunk_index}@{chunk.score:.3f}" for chunk in chunks],
        )

    augmented_messages = build_augmented_messages(messages, chunks)
    stream = stream_chat_completion(
        messages=augmented_messages,
        async_openai_client=async_openai_client,
        settings=settings,
    )
    return StreamingResponse(stream, media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def create_chat_completion_v1(
    payload: ChatCompletionRequest,
    settings: Settings = Depends(get_settings),
):
    return await handle_chat_completion(payload, settings)


@app.post("/chat/completions")
async def create_chat_completion_compat(
    payload: ChatCompletionRequest,
    settings: Settings = Depends(get_settings),
):
    return await handle_chat_completion(payload, settings)
