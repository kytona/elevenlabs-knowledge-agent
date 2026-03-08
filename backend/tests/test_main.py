from types import SimpleNamespace

from fastapi.testclient import TestClient
from qdrant_client.http.exceptions import UnexpectedResponse

import app.main as main_module
from app.config import Settings


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
    class _ChatCompletions:
        async def create(self, **_kwargs):
            chunk = SimpleNamespace(
                model_dump=lambda mode="json": {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
                }
            )
            return FakeAsyncStream([chunk])

    def __init__(self):
        self.chat = SimpleNamespace(completions=self._ChatCompletions())


class FakeUnexpectedResponse(UnexpectedResponse):
    def __init__(self, *, status_code: int, content: str, reason_phrase: str = "error"):
        self.status_code = status_code
        self.content = content.encode("utf-8")
        self.reason_phrase = reason_phrase

    def __str__(self) -> str:
        return self.content.decode("utf-8")


def reset_token_rate_limiter() -> None:
    main_module.token_rate_limiter = main_module.RateLimiter()


def test_health_endpoint(monkeypatch) -> None:
    reset_token_rate_limiter()
    main_module.app.dependency_overrides[main_module.get_settings] = lambda: Settings(OPENAI_API_KEY="")
    monkeypatch.setattr(main_module, "get_qdrant_collection_stats", lambda _settings: {
        "qdrant_collection_exists": False,
        "qdrant_points_count": 0,
    })
    client = TestClient(main_module.app)
    response = client.get("/health")
    main_module.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["qdrant_collection_exists"] is False
    assert response.json()["qdrant_points_count"] == 0


def test_chat_completion_stream(monkeypatch) -> None:
    reset_token_rate_limiter()
    test_settings = Settings(
        OPENAI_API_KEY="test-key",
        QDRANT_IN_MEMORY=True,
    )

    main_module.app.dependency_overrides[main_module.get_settings] = lambda: test_settings
    monkeypatch.setattr(main_module, "get_openai_client", lambda: object())
    monkeypatch.setattr(main_module, "get_async_openai_client", lambda: FakeAsyncOpenAI())
    monkeypatch.setattr(main_module, "get_qdrant_client", lambda: object())
    monkeypatch.setattr(
        main_module,
        "retrieve_context",
        lambda **_kwargs: [],
    )

    client = TestClient(main_module.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "custom",
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    main_module.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "data: " in response.text
    assert "[DONE]" in response.text


def test_chat_completion_stream_compat_route(monkeypatch) -> None:
    reset_token_rate_limiter()
    test_settings = Settings(
        OPENAI_API_KEY="test-key",
        QDRANT_IN_MEMORY=True,
    )

    main_module.app.dependency_overrides[main_module.get_settings] = lambda: test_settings
    monkeypatch.setattr(main_module, "get_openai_client", lambda: object())
    monkeypatch.setattr(main_module, "get_async_openai_client", lambda: FakeAsyncOpenAI())
    monkeypatch.setattr(main_module, "get_qdrant_client", lambda: object())
    monkeypatch.setattr(
        main_module,
        "retrieve_context",
        lambda **_kwargs: [],
    )

    client = TestClient(main_module.app)
    response = client.post(
        "/chat/completions",
        json={
            "model": "custom",
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    main_module.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "[DONE]" in response.text


def test_chat_completion_requires_stream() -> None:
    reset_token_rate_limiter()
    main_module.app.dependency_overrides[main_module.get_settings] = lambda: Settings(OPENAI_API_KEY="")
    client = TestClient(main_module.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "custom",
            "stream": False,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    main_module.app.dependency_overrides.clear()
    assert response.status_code == 400


def test_debug_retrieval(monkeypatch) -> None:
    test_settings = Settings(
        OPENAI_API_KEY="test-key",
        QDRANT_IN_MEMORY=True,
        QDRANT_COLLECTION_NAME="knowledge_base",
        ENABLE_DEBUG_RETRIEVAL=True,
    )

    reset_token_rate_limiter()
    main_module.app.dependency_overrides[main_module.get_settings] = lambda: test_settings
    monkeypatch.setattr(main_module, "get_openai_client", lambda: object())
    monkeypatch.setattr(main_module, "get_qdrant_client", lambda: object())
    monkeypatch.setattr(
        main_module,
        "retrieve_context",
        lambda **_kwargs: [
            {
                "source": "../data/sample_docs/the-adventure-of-the-speckled-band.md",
                "chunk_index": 0,
                "text": "Helen Stoner came to Baker Street in great agitation to consult Sherlock Holmes about her stepfather, Dr. Grimesby Roylott.",
                "score": 0.91,
            }
        ],
    )

    client = TestClient(main_module.app)
    response = client.get("/debug/retrieval", params={"q": "Who hired Sherlock Holmes?"})

    main_module.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["collection"] == "knowledge_base"
    assert response.json()["matches"][0]["source"].endswith("the-adventure-of-the-speckled-band.md")


def test_debug_retrieval_returns_404_when_disabled() -> None:
    reset_token_rate_limiter()
    main_module.app.dependency_overrides[main_module.get_settings] = lambda: Settings(OPENAI_API_KEY="")
    client = TestClient(main_module.app)
    response = client.get("/debug/retrieval", params={"q": "Who hired Sherlock Holmes?"})
    main_module.app.dependency_overrides.clear()
    assert response.status_code == 404


def test_conversation_token_endpoint_returns_token_only(monkeypatch) -> None:
    reset_token_rate_limiter()
    test_settings = Settings(
        OPENAI_API_KEY="test-key",
        ELEVENLABS_API_KEY="elevenlabs-key",
        ELEVENLABS_AGENT_ID="agent_123",
        ALLOWED_ORIGINS="http://localhost:3000",
    )
    main_module.app.dependency_overrides[main_module.get_settings] = lambda: test_settings
    monkeypatch.setattr(main_module, "get_settings", lambda: test_settings)

    async def fake_fetch_conversation_token(_settings):
        return "conv-token-123"

    monkeypatch.setattr(main_module, "fetch_conversation_token", fake_fetch_conversation_token)

    client = TestClient(main_module.app)
    response = client.post(
        "/v1/elevenlabs/conversation-token",
        headers={"Origin": "http://localhost:3000"},
    )

    main_module.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {"token": "conv-token-123"}
    assert "agent_id" not in response.text
    assert "elevenlabs-key" not in response.text


def test_conversation_token_endpoint_uses_x_forwarded_for_for_rate_limiting(monkeypatch) -> None:
    reset_token_rate_limiter()
    test_settings = Settings(
        OPENAI_API_KEY="test-key",
        ELEVENLABS_API_KEY="elevenlabs-key",
        ELEVENLABS_AGENT_ID="agent_123",
        ALLOWED_ORIGINS="http://localhost:3000",
        CONVERSATION_TOKEN_RATE_LIMIT=1,
        CONVERSATION_TOKEN_RATE_LIMIT_WINDOW_SECONDS=60,
    )
    main_module.app.dependency_overrides[main_module.get_settings] = lambda: test_settings
    monkeypatch.setattr(main_module, "get_settings", lambda: test_settings)

    async def fake_fetch_conversation_token(_settings):
        return "conv-token-123"

    monkeypatch.setattr(main_module, "fetch_conversation_token", fake_fetch_conversation_token)

    client = TestClient(main_module.app)
    headers = {
        "Origin": "http://localhost:3000",
        "X-Forwarded-For": "198.51.100.10, 10.0.0.1",
    }
    first = client.post("/v1/elevenlabs/conversation-token", headers=headers)
    second = client.post("/v1/elevenlabs/conversation-token", headers=headers)

    main_module.app.dependency_overrides.clear()
    assert first.status_code == 200
    assert second.status_code == 429


def test_conversation_token_endpoint_rejects_unlisted_origin(monkeypatch) -> None:
    reset_token_rate_limiter()
    test_settings = Settings(
        OPENAI_API_KEY="test-key",
        ELEVENLABS_API_KEY="elevenlabs-key",
        ELEVENLABS_AGENT_ID="agent_123",
        ALLOWED_ORIGINS="http://localhost:3000",
    )
    main_module.app.dependency_overrides[main_module.get_settings] = lambda: test_settings
    monkeypatch.setattr(main_module, "get_settings", lambda: test_settings)

    async def fake_fetch_conversation_token(_settings):
        return "conv-token-123"

    monkeypatch.setattr(main_module, "fetch_conversation_token", fake_fetch_conversation_token)

    client = TestClient(main_module.app)
    response = client.post(
        "/v1/elevenlabs/conversation-token",
        headers={"Origin": "https://evil.example"},
    )

    main_module.app.dependency_overrides.clear()
    assert response.status_code == 403


def test_conversation_token_endpoint_sets_cors_for_allowed_origin(monkeypatch) -> None:
    reset_token_rate_limiter()
    test_settings = Settings(
        OPENAI_API_KEY="test-key",
        ELEVENLABS_API_KEY="elevenlabs-key",
        ELEVENLABS_AGENT_ID="agent_123",
        ALLOWED_ORIGINS="http://localhost:3000",
    )
    main_module.app.dependency_overrides[main_module.get_settings] = lambda: test_settings
    monkeypatch.setattr(main_module, "get_settings", lambda: test_settings)

    async def fake_fetch_conversation_token(_settings):
        return "conv-token-123"

    monkeypatch.setattr(main_module, "fetch_conversation_token", fake_fetch_conversation_token)

    client = TestClient(main_module.app)
    response = client.post(
        "/v1/elevenlabs/conversation-token",
        headers={"Origin": "http://localhost:3000"},
    )

    main_module.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_chat_completion_falls_back_when_qdrant_collection_is_missing(monkeypatch) -> None:
    reset_token_rate_limiter()
    test_settings = Settings(
        OPENAI_API_KEY="test-key",
        QDRANT_IN_MEMORY=True,
        QDRANT_COLLECTION_NAME="knowledge_base",
    )

    main_module.app.dependency_overrides[main_module.get_settings] = lambda: test_settings
    monkeypatch.setattr(main_module, "get_openai_client", lambda: object())
    monkeypatch.setattr(main_module, "get_async_openai_client", lambda: FakeAsyncOpenAI())
    monkeypatch.setattr(main_module, "get_qdrant_client", lambda: object())
    monkeypatch.setattr(
        main_module,
        "retrieve_context",
        lambda **_kwargs: (_ for _ in ()).throw(
            FakeUnexpectedResponse(
                status_code=404,
                reason_phrase="Not Found",
                content="Collection `knowledge_base` not found",
            )
        ),
    )

    client = TestClient(main_module.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "custom",
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    main_module.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert "[DONE]" in response.text


def test_chat_completion_returns_500_for_non_collection_404(monkeypatch) -> None:
    reset_token_rate_limiter()
    test_settings = Settings(
        OPENAI_API_KEY="test-key",
        QDRANT_IN_MEMORY=True,
        QDRANT_COLLECTION_NAME="knowledge_base",
    )

    main_module.app.dependency_overrides[main_module.get_settings] = lambda: test_settings
    monkeypatch.setattr(main_module, "get_openai_client", lambda: object())
    monkeypatch.setattr(main_module, "get_async_openai_client", lambda: FakeAsyncOpenAI())
    monkeypatch.setattr(main_module, "get_qdrant_client", lambda: object())
    monkeypatch.setattr(
        main_module,
        "retrieve_context",
        lambda **_kwargs: (_ for _ in ()).throw(
            FakeUnexpectedResponse(
                status_code=404,
                reason_phrase="Not Found",
                content="404 page not found",
            )
        ),
    )

    client = TestClient(main_module.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "custom",
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    main_module.app.dependency_overrides.clear()
    assert response.status_code == 500
    assert "Qdrant request failed" in response.json()["detail"]
