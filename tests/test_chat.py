from types import SimpleNamespace

from paper_rag.config import Settings
from paper_rag.llm.chat import FakeChat, OpenAIChat, chat_cost_nok


def test_openai_chat_sends_both_messages_and_reads_usage():
    seen = {}

    class Completions:
        def create(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="  Hello [1].  "))],
                usage=SimpleNamespace(prompt_tokens=120, completion_tokens=7),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    result = OpenAIChat(client, "m", max_tokens=99).complete("be careful", "what is X?")

    assert result.text == "Hello [1]."  # whitespace trimmed
    assert (result.input_tokens, result.output_tokens) == (120, 7)
    assert seen["model"] == "m" and seen["temperature"] == 0 and seen["max_tokens"] == 99
    assert [m["role"] for m in seen["messages"]] == ["system", "user"]


def test_openai_chat_handles_empty_reply():
    class Completions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    assert OpenAIChat(client, "m").complete("s", "u").text == ""


def test_fake_chat_records_calls():
    fake = FakeChat("ok")
    fake.complete("sys", "usr")
    assert fake.calls == [("sys", "usr")]


def test_chat_cost_uses_input_and_output_prices_in_nok():
    s = Settings(
        _env_file=None,
        chat_input_price_per_million_usd=1.0,
        chat_output_price_per_million_usd=4.0,
        usd_to_nok=10.0,
    )
    # 1M input = 1 USD, 0.5M output = 2 USD -> 3 USD -> 30 NOK
    assert chat_cost_nok(1_000_000, 500_000, s) == 30.0
