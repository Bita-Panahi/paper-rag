from paper_rag.ingest.cleaning import build_vocabulary, clean_block, dehyphenate


def test_hyphenation_is_repaired_when_joined_word_appears_elsewhere():
    text = "We learn func-\ntions of BRDFs. Such functions are smooth."
    vocab = build_vocabulary(text)
    assert "functions" in dehyphenate("func-\ntions", vocab)


def test_real_compound_keeps_its_hyphen():
    # both halves occur as ordinary words elsewhere in the paper
    text = "a low-\ndimensional latent space. Low rank. A dimensional analysis."
    vocab = build_vocabulary(text)
    assert dehyphenate("low-\ndimensional", vocab) == "low-dimensional"


def test_single_occurrence_hyphenation_is_repaired():
    # words seen only once, split at a line end (taken from real papers)
    text = "the geo-\nmetrical mesh, a com-\npromise, parameter-\nising it"
    cleaned = clean_block(text, build_vocabulary(text))
    assert cleaned == "the geometrical mesh, a compromise, parameterising it"


def test_suspended_hyphen_is_kept_with_a_space():
    text = "interference-\nand diffraction-based coatings"
    assert (
        clean_block(text, build_vocabulary(text)) == "interference- and diffraction-based coatings"
    )


def test_hyphen_before_capital_is_a_real_hyphen():
    text = "the Cook-\nTorrance model"
    assert clean_block(text, build_vocabulary(text)) == "the Cook-Torrance model"


def test_clean_block_joins_lines_and_fixes_ligatures():
    vocab = set()
    raw = "The ﬁnal   result\nis shown  here.\n"
    assert clean_block(raw, vocab) == "The final result is shown here."
