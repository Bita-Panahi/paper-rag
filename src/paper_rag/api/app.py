"""The HTTP backend: /health, /ask and /ingest, with request logging and clean errors."""

import logging
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict

import openai
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from paper_rag.api.schemas import AskRequest, AskResponse, HealthResponse, IngestResponse
from paper_rag.config import Settings, get_settings
from paper_rag.ingest.manifest import ManifestError, load_manifest
from paper_rag.ingest.pipeline import index_directory
from paper_rag.llm.embeddings import embedding_cost_nok
from paper_rag.logging_config import configure_logging
from paper_rag.retrieval.service import RagService
from paper_rag.store.models import SearchFilters

logger = logging.getLogger("paper_rag.api")


def create_app(service: RagService | None = None) -> FastAPI:
    """Build the app. Tests pass a ready-made service; in production it is built at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            settings = get_settings()
            configure_logging(settings.log_level)
            app.state.service = RagService.from_settings(settings)  # fails fast if misconfigured
            logger.info("service started")
        else:
            app.state.service = service
        yield

    app = FastAPI(title="Paper RAG", version="0.1.0", lifespan=lifespan)
    ingest_lock = threading.Lock()

    # ---- cross-cutting: request id + access log ------------------------------------------
    @app.middleware("http")
    async def access_log(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %d",
            request.method,
            request.url.path,
            response.status_code,
            extra={"request_id": request_id, "status": response.status_code, "ms": round(ms, 1)},
        )
        return response

    # ---- errors: log the details, return a short safe message -----------------------------
    def error_response(request: Request, status: int, message: str) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=status, content={"detail": message, "request_id": request_id}
        )

    @app.exception_handler(openai.OpenAIError)
    async def llm_error(request: Request, exc: openai.OpenAIError):
        logger.error("LLM provider error: %s", exc, exc_info=exc)
        return error_response(request, 502, "The language model provider failed. Try again.")

    @app.exception_handler(ResponseHandlingException)
    @app.exception_handler(UnexpectedResponse)
    async def store_error(request: Request, exc: Exception):
        logger.error("Vector store error: %s", exc, exc_info=exc)
        return error_response(request, 503, "The vector store is unavailable.")

    @app.exception_handler(ManifestError)
    async def manifest_error(request: Request, exc: ManifestError):
        logger.error("Bad papers.csv: %s", exc)
        return error_response(request, 500, f"papers.csv is invalid: {exc}")

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        logger.error("Unhandled error: %s", exc, exc_info=exc)
        return error_response(request, 500, "Internal server error.")

    # ---- dependencies --------------------------------------------------------------------
    def get_service(request: Request) -> RagService:
        return request.app.state.service

    def require_api_key(
        service: RagService = Depends(get_service),
        x_api_key: str | None = Header(default=None),
    ) -> None:
        expected = service.settings.api_key
        if expected and not secrets.compare_digest(x_api_key or "", expected):
            raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key header.")

    # ---- endpoints (plain `def` so FastAPI runs the blocking clients in a thread pool) -----
    @app.get("/health", response_model=HealthResponse)
    def health(service: RagService = Depends(get_service)) -> HealthResponse:
        # count() talks to Qdrant, so a failure here means the service is not healthy.
        return HealthResponse(status="ok", indexed_chunks=service.store.count())

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
    def ask(body: AskRequest, service: RagService = Depends(get_service)) -> AskResponse:
        filters = SearchFilters(
            year_min=body.year_min,
            year_max=body.year_max,
            own_paper=body.own_paper,
            file_name=body.file_name,
        )
        answer = service.ask(body.question, body.top_k, filters)
        return AskResponse(**asdict(answer))

    @app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
    def ingest(force: bool = False, service: RagService = Depends(get_service)) -> IngestResponse:
        settings: Settings = service.settings
        if not ingest_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="An ingest run is already in progress.")
        try:
            report = index_directory(
                settings.pdf_dir,
                service.store,
                service.embedder,
                manifest=load_manifest(settings.manifest_path),
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
                force=force,
            )
        finally:
            ingest_lock.release()
        return IngestResponse(
            **asdict(report),
            cost_nok=embedding_cost_nok(report.embedding_tokens, settings),
        )

    return app


app = create_app()
