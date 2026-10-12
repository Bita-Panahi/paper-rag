from types import SimpleNamespace

import pytest

from paper_rag.config import Settings
from paper_rag.llm.embeddings import HashingEmbedder, OpenAIEmbedder, embedding_cost_nok


class FakeOpenAIClient:
    """Mimics client.embeddings.create(...) and returns items in scrambled order."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, model, input):
        self.calls.append(list(input))
        data = [
            SimpleNamespace(index=i, embedding=[float(len(t)), 0.0]) for i, t in enumerate(input)
        ]
        return SimpleNamespace(data=list(reversed(data)), usage=SimpleNamespace(total_tokens=10))


def test_openai_embedder_batches_orders_and_sums_tokens():
    client = FakeOpenAIClient()
    embedder = OpenAIEmbedder(client, "m", batch_size=2)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    result = embedder.embed(texts)
    assert len(client.calls) == 3  # 2 + 2 + 1
    assert [v[0] for v in result.vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]  # original order kept
    assert result.total_tokens == 30


def test_from_settings_requires_api_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbedder.from_settings(Settings(_env_file=None, openai_api_key=""))


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(dim=32)
    v1 = e.embed(["goniochromatic coatings"]).vectors[0]
    v2 = e.embed(["goniochromatic coatings"]).vectors[0]
    assert v1 == v2
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-9


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hashing_embedder_puts_similar_texts_closer():
    e = HashingEmbedder(dim=128)
    query, near, far = e.embed(
        [
            "goniochromatic coating angle",
            "coating angle of goniochromatic pigments",
            "neural network",
        ]
    ).vectors
    assert dot(query, near) > dot(query, far)


def test_cost_estimate_in_nok():
    settings = Settings(_env_file=None, embedding_price_per_million_usd=0.02, usd_to_nok=10.0)
    assert embedding_cost_nok(1_000_000, settings) == pytest.approx(0.2)
