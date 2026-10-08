from paper_rag.ingest.cleaning import build_vocabulary, clean_block, dehyphenate


def test_hyphenation_is_repaired_when_joined_word_appears_elsewhere():
    text = "We learn func-\ntions of BRDFs. Such functions are smooth."
    vocab = build_vocabulary(text)
    assert "functions" in dehyphenate("func-\ntions", vocab)


def test_real_compound_keeps_its_hyphen():
    text = "a low-\ndimensional latent space"
    vocab = build_vocabulary(text)
    assert dehyphenate("low-\ndimensional", vocab) == "low-dimensional"


def test_clean_block_joins_lines_and_fixes_ligatures():
    vocab = set()
    raw = "The ﬁnal   result\nis shown  here.\n"
    assert clean_block(raw, vocab) == "The final result is shown here."
