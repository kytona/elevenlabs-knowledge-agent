from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="custom")
    messages: list[ChatMessage]
    stream: bool = True


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


def require_debug_retrieval_enabled(settings: Settings) -> None:
    if not settings.enable_debug_retrieval:
        raise HTTPException(status_code=404, detail="Not found")


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
async def chat_completions(
    payload: ChatCompletionRequest,
    settings: Settings = Depends(get_settings),
):
    return await handle_chat_completion(payload, settings)


@app.post("/chat/completions")
async def chat_completions_compat(
    payload: ChatCompletionRequest,
    settings: Settings = Depends(get_settings),
):
    return await handle_chat_completion(payload, settings)
