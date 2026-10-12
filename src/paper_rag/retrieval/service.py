"""Question answering: embed the question -> retrieve -> build the prompt -> ask the LLM.

Every call returns the answer, its sources, and what it cost in tokens, NOK and milliseconds.
"""

import logging
import re
import time
from dataclasses import dataclass, replace

from paper_rag.config import Settings
from paper_rag.ingest.manifest import load_manifest
from paper_rag.llm.chat import LLMClient, OpenAIChat, chat_cost_nok
from paper_rag.llm.embeddings import Embedder, OpenAIEmbedder, embedding_cost_nok
from paper_rag.retrieval.prompts import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_user_prompt
from paper_rag.retrieval.scope import PaperResolver
from paper_rag.store.models import SearchFilters, SearchHit
from paper_rag.store.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

_CITATION = re.compile(r"\[(\d+)\]")
_SNIPPET_CHARS = 300


@dataclass(frozen=True)
class Source:
    number: int  # matches [number] in the answer text
    title: str
    file_name: str
    page_start: int
    page_end: int
    year: int | None
    own_paper: bool
    score: float
    cited: bool  # did the answer actually use [number]?
    snippet: str


@dataclass(frozen=True)
class Usage:
    embedding_tokens: int
    input_tokens: int
    output_tokens: int
    cost_nok: float
    retrieval_ms: float
    generation_ms: float
    total_ms: float


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    answered: bool  # False when the papers did not contain the answer
    sources: list[Source]
    usage: Usage
    scoped_files: list[str]  # papers the search was limited to (named in the question)


class RagService:
    def __init__(
        self,
        store: QdrantStore,
        embedder: Embedder,
        llm: LLMClient,
        settings: Settings,
        resolver: PaperResolver | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.settings = settings
        self.resolver = resolver

    @classmethod
    def from_settings(cls, settings: Settings) -> "RagService":
        return cls(
            QdrantStore.from_settings(settings),
            OpenAIEmbedder.from_settings(settings),
            OpenAIChat.from_settings(settings),
            settings,
            PaperResolver(load_manifest(settings.manifest_path)),
        )

    def scope(
        self, question: str, filters: SearchFilters | None
    ) -> tuple[SearchFilters | None, list[str]]:
        """If the question names papers, limit the search to them (unless the caller chose)."""
        if not self.settings.scope_to_named_papers or self.resolver is None:
            return filters, []
        if filters is not None and (filters.file_name or filters.file_names):
            return filters, []  # an explicit choice always wins
        names = self.resolver.resolve(question)
        if not names:
            return filters, []
        return replace(filters or SearchFilters(), file_names=names), list(names)

    def retrieve(
        self, question: str, top_k: int | None = None, filters: SearchFilters | None = None
    ) -> tuple[list[SearchHit], int]:
        """Return hits above the score threshold, and the tokens spent embedding the question."""
        filters, _ = self.scope(question, filters)
        return self._search(question, top_k, filters)

    def _search(
        self, question: str, top_k: int | None, filters: SearchFilters | None
    ) -> tuple[list[SearchHit], int]:
        embedded = self.embedder.embed([question])
        hits = self.store.search(embedded.vectors[0], top_k or self.settings.top_k, filters)
        return [h for h in hits if h.score >= self.settings.min_score], embedded.total_tokens

    def ask(
        self, question: str, top_k: int | None = None, filters: SearchFilters | None = None
    ) -> Answer:
        started = time.perf_counter()
        filters, scoped_files = self.scope(question, filters)
        hits, embedding_tokens = self._search(question, top_k, filters)
        retrieval_ms = (time.perf_counter() - started) * 1000

        if not hits:
            # Nothing relevant was found: do not pay the LLM to say "I don't know".
            total_ms = (time.perf_counter() - started) * 1000
            usage = Usage(
                embedding_tokens=embedding_tokens,
                input_tokens=0,
                output_tokens=0,
                cost_nok=embedding_cost_nok(embedding_tokens, self.settings),
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
                total_ms=total_ms,
            )
            result = Answer(question, NOT_FOUND_MESSAGE, False, [], usage, scoped_files)
            self._log(result)
            return result

        generation_started = time.perf_counter()
        reply = self.llm.complete(SYSTEM_PROMPT, build_user_prompt(question, hits))
        generation_ms = (time.perf_counter() - generation_started) * 1000

        cited = {int(n) for n in _CITATION.findall(reply.text)}
        sources = [
            Source(
                number=i,
                title=h.title,
                file_name=h.file_name,
                page_start=h.page_start,
                page_end=h.page_end,
                year=h.year,
                own_paper=h.own_paper,
                score=round(h.score, 4),
                cited=i in cited,
                snippet=h.text[:_SNIPPET_CHARS],
            )
            for i, h in enumerate(hits, start=1)
        ]
        usage = Usage(
            embedding_tokens=embedding_tokens,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            cost_nok=embedding_cost_nok(embedding_tokens, self.settings)
            + chat_cost_nok(reply.input_tokens, reply.output_tokens, self.settings),
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=(time.perf_counter() - started) * 1000,
        )
        result = Answer(
            question=question,
            answer=reply.text,
            answered=NOT_FOUND_MESSAGE not in reply.text,
            sources=sources,
            usage=usage,
            scoped_files=scoped_files,
        )
        self._log(result)
        return result

    @staticmethod
    def _log(result: Answer) -> None:
        u = result.usage
        logger.info(
            "ask answered=%s sources=%d tokens_in=%d tokens_out=%d cost_nok=%.5f "
            "retrieval_ms=%.0f generation_ms=%.0f total_ms=%.0f",
            result.answered,
            len(result.sources),
            u.input_tokens,
            u.output_tokens,
            u.cost_nok,
            u.retrieval_ms,
            u.generation_ms,
            u.total_ms,
        )
