from __future__ import annotations

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


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, str | bool]:
    return {
        "status": "ok",
        "qdrant_in_memory": settings.qdrant_in_memory,
        "collection": settings.qdrant_collection_name,
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(
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
            chunks = []
        else:
            raise HTTPException(status_code=500, detail=f"Qdrant request failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        detail = str(exc) or exc.__class__.__name__
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {detail}") from exc

    augmented_messages = build_augmented_messages(messages, chunks)
    stream = stream_chat_completion(
        messages=augmented_messages,
        async_openai_client=async_openai_client,
        settings=settings,
    )
    return StreamingResponse(stream, media_type="text/event-stream")
