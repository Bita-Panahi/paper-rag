"""Central configuration. All settings come from environment variables or a local .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "papers"

    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5

    embedding_batch_size: int = 64
    # List prices, used only for the cost estimate printed after indexing.
    embedding_price_per_million_usd: float = 0.02
    usd_to_nok: float = 10.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
