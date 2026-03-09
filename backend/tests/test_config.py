from pathlib import Path

import app.config as config_module


def test_settings_load_repo_root_env_file(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    backend_root = repo_root / "backend"
    backend_app_root = backend_root / "app"
    backend_app_root.mkdir(parents=True)

    env_file = repo_root / ".env"
    env_file.write_text("OPENAI_API_KEY=repo-root-key\nQDRANT_IN_MEMORY=true\n", encoding="utf-8")

    monkeypatch.setattr(config_module, "REPO_ROOT", repo_root)
    monkeypatch.setattr(config_module, "BACKEND_ROOT", backend_root)

    class TestSettings(config_module.Settings):
        model_config = config_module.SettingsConfigDict(
            env_file=(str(repo_root / ".env"), str(backend_root / ".env")),
            env_file_encoding="utf-8",
            populate_by_name=True,
            extra="ignore",
        )

    settings = TestSettings()
    assert settings.openai_api_key == "repo-root-key"
    assert settings.qdrant_in_memory is True


def test_get_qdrant_client_uses_persistent_local_path(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    test_settings = config_module.Settings(
        OPENAI_API_KEY="test-key",
        QDRANT_IN_MEMORY=True,
        QDRANT_LOCAL_PATH=str(tmp_path / "qdrant-data"),
    )

    def fake_qdrant_client(**kwargs):
        captured.update(kwargs)
        return object()

    config_module.get_settings.cache_clear()
    config_module.get_qdrant_client.cache_clear()
    monkeypatch.setattr(config_module, "get_settings", lambda: test_settings)
    monkeypatch.setattr(config_module, "QdrantClient", fake_qdrant_client)

    config_module.get_qdrant_client()

    assert captured == {"path": str(tmp_path / "qdrant-data")}
