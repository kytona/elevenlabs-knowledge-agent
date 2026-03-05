import asyncio
import json
from types import SimpleNamespace

from fastapi import HTTPException

from app.config import Settings
from app.rag import build_augmented_messages, chunk_text, extract_latest_user_message, stream_chat_completion


class FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._iterator = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration as exc:  # pragma: no cover
            raise StopAsyncIteration from exc


class FakeAsyncOpenAI:
    def __init__(self, chunks):
        async def create(**_kwargs):
            return FakeAsyncStream(chunks)

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def test_extract_latest_user_message_from_string_content() -> None:
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What does the refund policy say?"},
    ]
    assert extract_latest_user_message(messages) == "What does the refund policy say?"


def test_extract_latest_user_message_from_content_parts() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "ignore me"},
                {"type": "text", "text": "Tell me about custom LLM webhooks."},
            ],
        }
    ]
    assert extract_latest_user_message(messages) == "Tell me about custom LLM webhooks."


def test_extract_latest_user_message_raises_for_missing_user_text() -> None:
    try:
        extract_latest_user_message([{"role": "assistant", "content": "No user content"}])
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("Expected HTTPException for missing user content")


def test_build_augmented_messages_appends_to_existing_system_prompt() -> None:
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Question"},
    ]
    chunks = [
        {
            "source": "docs.md",
            "chunk_index": 0,
            "text": "Custom LLM lets you route requests to your own backend.",
            "score": 0.99,
        }
    ]

    augmented = build_augmented_messages(messages, chunks)
    assert augmented[0]["role"] == "system"
    assert "Be concise." in augmented[0]["content"]
    assert "route requests to your own backend" in augmented[0]["content"]


def test_build_augmented_messages_inserts_system_prompt_if_missing() -> None:
    messages = [{"role": "user", "content": "Question"}]
    chunks = [{"source": "docs.md", "chunk_index": 0, "text": "Answer text", "score": 0.99}]
    augmented = build_augmented_messages(messages, chunks)
    assert augmented[0]["role"] == "system"
    assert "Answer text" in augmented[0]["content"]


def test_chunk_text_uses_overlap_without_duplicates() -> None:
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=400, chunk_overlap=100)
    assert len(chunks) == 3
    assert all(chunks)


def test_stream_chat_completion_does_not_append_duplicate_stop_chunk() -> None:
    upstream_chunks = [
        SimpleNamespace(
            model_dump=lambda mode="json": {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
            }
        ),
        SimpleNamespace(
            model_dump=lambda mode="json": {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        ),
    ]
    settings = Settings(OPENAI_API_KEY="test-key")

    async def collect():
        return [
            item
            async for item in stream_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                async_openai_client=FakeAsyncOpenAI(upstream_chunks),
                settings=settings,
            )
        ]

    events = asyncio.run(collect())
    assert events[-1] == "data: [DONE]\n\n"
    chunk_events = [event for event in events[:-1]]
    assert len(chunk_events) == 2
    last_payload = json.loads(chunk_events[-1].removeprefix("data: ").strip())
    assert last_payload["choices"][0]["finish_reason"] == "stop"
    assert last_payload["model"] == "gpt-4o-mini"


def test_stream_chat_completion_synthesizes_single_stop_chunk_with_provider_model() -> None:
    upstream_chunks = [
        SimpleNamespace(
            model_dump=lambda mode="json": {
                "id": "chatcmpl-2",
                "object": "chat.completion.chunk",
                "created": 1,
                "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
            }
        )
    ]
    settings = Settings(OPENAI_API_KEY="test-key", OPENAI_CHAT_MODEL="gpt-4o-mini")

    async def collect():
        return [
            item
            async for item in stream_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                async_openai_client=FakeAsyncOpenAI(upstream_chunks),
                settings=settings,
            )
        ]

    events = asyncio.run(collect())
    assert events[-1] == "data: [DONE]\n\n"
    chunk_events = [event for event in events[:-1]]
    assert len(chunk_events) == 2
    first_payload = json.loads(chunk_events[0].removeprefix("data: ").strip())
    second_payload = json.loads(chunk_events[1].removeprefix("data: ").strip())
    assert first_payload["model"] == "gpt-4o-mini"
    assert second_payload["model"] == "gpt-4o-mini"
    assert second_payload["choices"][0]["finish_reason"] == "stop"
