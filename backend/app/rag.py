from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import HTTPException
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from app.config import Settings


class RetrievedChunk(BaseModel):
    source: str
    chunk_index: int
    text: str
    score: float


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        primary_parts: list[str] = []
        fallback_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text", "output_text"}:
                value = item.get("text", "")
                if value:
                    target = primary_parts if item_type == "text" else fallback_parts
                    target.append(value)
        parts = primary_parts or fallback_parts
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return ""


def extract_latest_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = extract_text_content(message.get("content"))
            if content:
                return content
    raise HTTPException(status_code=400, detail="Request must include a user message with text content.")


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    length = len(normalized)
    while start < length:
        end = min(length, start + chunk_size)
        chunks.append(normalized[start:end].strip())
        if end >= length:
            break
        start = max(0, end - chunk_overlap)
    return [chunk for chunk in chunks if chunk]


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    collections = client.get_collections().collections
    if any(collection.name == collection_name for collection in collections):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
    )


def embed_texts(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def ingest_chunks(
    *,
    client: QdrantClient,
    openai_client: OpenAI,
    settings: Settings,
    source: str,
    chunks: list[str],
    recreate_collection: bool = False,
) -> int:
    if not chunks:
        return 0

    embeddings = embed_texts(openai_client, settings.openai_embedding_model, chunks)
    if recreate_collection:
        client.recreate_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=rest.VectorParams(
                size=len(embeddings[0]),
                distance=rest.Distance.COSINE,
            ),
        )
    else:
        ensure_collection(client, settings.qdrant_collection_name, len(embeddings[0]))

    points = [
        rest.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"source": source, "chunk_index": index, "text": chunk},
        )
        for index, (chunk, vector) in enumerate(zip(chunks, embeddings, strict=True))
    ]

    client.upsert(collection_name=settings.qdrant_collection_name, points=points)
    return len(points)


def retrieve_context(
    *,
    query: str,
    qdrant_client: QdrantClient,
    openai_client: OpenAI,
    settings: Settings,
    limit: int = 3,
) -> list[RetrievedChunk]:
    vector = embed_texts(openai_client, settings.openai_embedding_model, [query])[0]
    search_result = qdrant_client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=vector,
        limit=limit,
        with_payload=True,
    )
    points = getattr(search_result, "points", search_result)
    chunks: list[RetrievedChunk] = []
    for point in points:
        payload = point.payload or {}
        chunks.append(
            RetrievedChunk(
                source=str(payload.get("source", "unknown")),
                chunk_index=int(payload.get("chunk_index", 0)),
                text=str(payload.get("text", "")),
                score=float(getattr(point, "score", 0.0)),
            )
        )
    return [chunk for chunk in chunks if chunk.text]


def build_augmented_messages(
    messages: list[dict[str, Any]],
    chunks: list[RetrievedChunk | dict[str, Any]],
) -> list[dict[str, Any]]:
    if not chunks:
        return messages

    normalized_chunks = [
        chunk if isinstance(chunk, RetrievedChunk) else RetrievedChunk.model_validate(chunk)
        for chunk in chunks
    ]

    context_lines = [
        "Use the following retrieved context to answer the user's question.",
        "If the answer is not supported by the context, say that clearly.",
        "",
    ]
    for chunk in normalized_chunks:
        context_lines.append(
            f"[Source: {chunk.source} | Chunk: {chunk.chunk_index} | Score: {chunk.score:.3f}] {chunk.text}"
        )
    context_block = "\n".join(context_lines)

    updated_messages = [dict(message) for message in messages]
    for message in updated_messages:
        if message.get("role") == "system":
            existing = extract_text_content(message.get("content"))
            message["content"] = f"{existing}\n\n{context_block}".strip()
            return updated_messages

    return [{"role": "system", "content": context_block}, *updated_messages]


def format_sse(data: str) -> str:
    return f"data: {data}\n\n"


async def stream_chat_completion(
    *,
    messages: list[dict[str, Any]],
    async_openai_client: AsyncOpenAI,
    settings: Settings,
) -> AsyncIterator[str]:
    stream = await async_openai_client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=messages,
        stream=True,
    )

    saw_terminal_chunk = False

    async for chunk in stream:
        payload = chunk.model_dump(mode="json")
        if not payload.get("model"):
            payload["model"] = settings.openai_chat_model
        if not payload.get("created"):
            payload["created"] = int(time.time())
        if not payload.get("object"):
            payload["object"] = "chat.completion.chunk"
        if not payload.get("id"):
            payload["id"] = f"chatcmpl-{uuid.uuid4().hex}"
        if any(choice.get("finish_reason") for choice in payload.get("choices", [])):
            saw_terminal_chunk = True
        yield format_sse(json.dumps(payload))

    if not saw_terminal_chunk:
        terminal_payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": settings.openai_chat_model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield format_sse(json.dumps(terminal_payload))
    yield format_sse("[DONE]")
