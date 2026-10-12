"""Request and response shapes of the HTTP API. FastAPI validates against these automatically."""

from pydantic import BaseModel, Field, field_validator, model_validator


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    year_min: int | None = Field(default=None, ge=1900, le=2100)
    year_max: int | None = Field(default=None, ge=1900, le=2100)
    own_paper: bool | None = None
    file_name: str | None = None

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("question is too short")
        return value

    @model_validator(mode="after")
    def _years_in_order(self) -> "AskRequest":
        if self.year_min and self.year_max and self.year_min > self.year_max:
            raise ValueError("year_min must be <= year_max")
        return self


class SourceOut(BaseModel):
    number: int
    title: str
    file_name: str
    page_start: int
    page_end: int
    year: int | None
    own_paper: bool
    score: float
    cited: bool
    snippet: str


class UsageOut(BaseModel):
    embedding_tokens: int
    input_tokens: int
    output_tokens: int
    cost_nok: float
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class AskResponse(BaseModel):
    question: str
    answer: str
    answered: bool
    sources: list[SourceOut]
    usage: UsageOut
    scoped_files: list[str]


class IngestResponse(BaseModel):
    indexed: list[str]
    updated: list[str]
    skipped: list[str]
    failed: dict[str, str]
    chunks_indexed: int
    embedding_tokens: int
    cost_nok: float
    seconds: float


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
