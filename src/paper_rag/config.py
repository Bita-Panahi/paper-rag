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

    pdf_dir: str = "data/pdfs"
    manifest_path: str = "data/papers.csv"

    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5
    # Hits scoring below this are ignored. Tune it with the evaluation set (Step 6).
    min_score: float = 0.3
    max_answer_tokens: int = 500
    # If the question names a paper ("in the CIC 2024 paper"), search only that paper.
    scope_to_named_papers: bool = True

    # If set, /ask and /ingest require the header X-API-Key with this value.
    api_key: str = ""
    log_level: str = "INFO"

    embedding_batch_size: int = 64
    # List prices, used only for the cost estimate printed after indexing.
    embedding_price_per_million_usd: float = 0.02
    chat_input_price_per_million_usd: float = 0.15
    chat_output_price_per_million_usd: float = 0.60
    usd_to_nok: float = 10.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
