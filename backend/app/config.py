from functools import lru_cache
from pathlib import Path

from openai import AsyncOpenAI, OpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection_name: str = Field(default="knowledge_base", alias="QDRANT_COLLECTION_NAME")
    qdrant_in_memory: bool = Field(default=False, alias="QDRANT_IN_MEMORY")
    qdrant_local_path: str = Field(default=str(REPO_ROOT / ".qdrant"), alias="QDRANT_LOCAL_PATH")
    backend_public_url: str = Field(default="http://localhost:8000", alias="BACKEND_PUBLIC_URL")

    model_config = SettingsConfigDict(
        env_file=(str(REPO_ROOT / ".env"), str(BACKEND_ROOT / ".env")),
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_openai_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


@lru_cache
def get_async_openai_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


@lru_cache
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    if settings.qdrant_in_memory:
        return QdrantClient(path=settings.qdrant_local_path)
    return QdrantClient(url=settings.qdrant_url)
