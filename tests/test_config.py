from paper_rag.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.embedding_model == "text-embedding-3-small"
    assert s.top_k == 5
    assert s.chunk_overlap < s.chunk_size
