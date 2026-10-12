"""Chat (answer-writing) providers behind one small interface, like embeddings.py."""

from dataclasses import dataclass
from typing import Protocol

from paper_rag.config import Settings


@dataclass(frozen=True)
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str) -> ChatResult: ...


class OpenAIChat:
    """Chat completions from OpenAI (or any OpenAI-compatible client, e.g. AzureOpenAI)."""

    def __init__(self, client, model: str, max_tokens: int = 500) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAIChat":
        from openai import OpenAI

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is empty. Put it in your .env file.")
        client = OpenAI(api_key=settings.openai_api_key, max_retries=3, timeout=60)
        return cls(client, settings.chat_model, settings.max_answer_tokens)

    def complete(self, system: str, user: str) -> ChatResult:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,  # same question -> (nearly) same answer; easier to evaluate
            max_tokens=self.max_tokens,
        )
        return ChatResult(
            text=(response.choices[0].message.content or "").strip(),
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )


class FakeChat:
    """Offline stand-in for tests: returns a fixed reply and records what it was asked."""

    def __init__(self, reply: str = "An answer [1].", model: str = "fake-chat") -> None:
        self.reply = reply
        self.model = model
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> ChatResult:
        self.calls.append((system, user))
        return ChatResult(
            text=self.reply, input_tokens=len(user.split()), output_tokens=len(self.reply.split())
        )


def chat_cost_nok(input_tokens: int, output_tokens: int, settings: Settings) -> float:
    """Estimated cost in NOK from list prices (an estimate, not an invoice)."""
    usd = (
        input_tokens / 1_000_000 * settings.chat_input_price_per_million_usd
        + output_tokens / 1_000_000 * settings.chat_output_price_per_million_usd
    )
    return usd * settings.usd_to_nok
