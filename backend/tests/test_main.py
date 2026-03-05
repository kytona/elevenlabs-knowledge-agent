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


def test_health_endpoint() -> None:
    main_module.app.dependency_overrides[main_module.get_settings] = lambda: Settings(OPENAI_API_KEY="")
    client = TestClient(main_module.app)
    response = client.get("/health")
    main_module.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_completion_stream(monkeypatch) -> None:
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


def test_chat_completion_requires_stream() -> None:
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


def test_chat_completion_falls_back_when_qdrant_collection_is_missing(monkeypatch) -> None:
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
