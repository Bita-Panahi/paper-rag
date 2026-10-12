from paper_rag.ingest.manifest import PaperInfo
from paper_rag.ingest.pipeline import index_directory
from paper_rag.llm.embeddings import HashingEmbedder
from paper_rag.store import QdrantStore, SearchFilters

COATING = [
    f"Goniochromatic coating sample {i} shows strong angular colour shift." for i in range(8)
]
NEURAL = [f"Neural network layer {i} learns a compact latent representation." for i in range(8)]
MANIFEST = {"coating.pdf": PaperInfo(2020, True), "neural.pdf": PaperInfo(2024, False)}

SETTINGS = {"chunk_size": 300, "overlap": 50}


def run(pdf_dir, store, embedder, **kwargs):
    return index_directory(pdf_dir, store, embedder, manifest=MANIFEST, **SETTINGS, **kwargs)


def setup(tmp_path, make_pdf):
    make_pdf(tmp_path / "coating.pdf", COATING)
    make_pdf(tmp_path / "neural.pdf", NEURAL)
    return QdrantStore.in_memory(), HashingEmbedder(dim=64)


def test_first_run_indexes_everything(tmp_path, make_pdf):
    store, embedder = setup(tmp_path, make_pdf)
    report = run(tmp_path, store, embedder)
    assert sorted(report.indexed) == ["coating.pdf", "neural.pdf"]
    assert not report.failed and not report.skipped
    assert report.chunks_indexed == store.count() > 2
    assert report.embedding_tokens > 0 and report.seconds >= 0


def test_second_run_does_no_work(tmp_path, make_pdf):
    store, embedder = setup(tmp_path, make_pdf)
    run(tmp_path, store, embedder)
    calls_before, count_before = embedder.calls, store.count()

    report = run(tmp_path, store, embedder)

    assert sorted(report.skipped) == ["coating.pdf", "neural.pdf"]
    assert not report.indexed and not report.updated
    assert embedder.calls == calls_before  # no embedding requests, so no cost
    assert store.count() == count_before


def test_changed_file_is_reindexed_and_old_chunks_are_gone(tmp_path, make_pdf):
    store, embedder = setup(tmp_path, make_pdf)
    run(tmp_path, store, embedder)

    make_pdf(
        tmp_path / "coating.pdf",
        [f"Completely different spectral text number {i} here." for i in range(8)],
    )
    report = run(tmp_path, store, embedder)

    assert report.updated == ["coating.pdf"] and report.skipped == ["neural.pdf"]
    only_coating = SearchFilters(file_name="coating.pdf")
    texts = " ".join(
        h.text for h in store.search(embedder.embed(["x"]).vectors[0], 100, only_coating)
    )
    assert "Goniochromatic" not in texts and "spectral" in texts


def test_changing_chunk_settings_triggers_reindex(tmp_path, make_pdf):
    store, embedder = setup(tmp_path, make_pdf)
    run(tmp_path, store, embedder)
    report = index_directory(
        tmp_path, store, embedder, manifest=MANIFEST, chunk_size=500, overlap=50
    )
    assert sorted(report.updated) == ["coating.pdf", "neural.pdf"]


def test_force_reindexes_unchanged_files(tmp_path, make_pdf):
    store, embedder = setup(tmp_path, make_pdf)
    run(tmp_path, store, embedder)
    assert sorted(run(tmp_path, store, embedder, force=True).updated) == [
        "coating.pdf",
        "neural.pdf",
    ]


def test_metadata_from_manifest_drives_filters(tmp_path, make_pdf):
    store, embedder = setup(tmp_path, make_pdf)
    run(tmp_path, store, embedder)
    query = embedder.embed(["goniochromatic coating colour shift"]).vectors[0]

    assert store.search(query, 1)[0].file_name == "coating.pdf"
    recent = store.search(query, 20, SearchFilters(year_min=2022))
    assert recent and {h.file_name for h in recent} == {"neural.pdf"}
    mine = store.search(query, 20, SearchFilters(own_paper=True))
    assert mine and {h.file_name for h in mine} == {"coating.pdf"}


def test_broken_and_empty_pdfs_are_reported_without_stopping_the_run(tmp_path, make_pdf):
    import pymupdf

    store, embedder = setup(tmp_path, make_pdf)
    (tmp_path / "broken.pdf").write_bytes(b"this is not a pdf")
    blank = pymupdf.open()
    blank.new_page()
    blank.save(tmp_path / "blank.pdf")

    report = run(tmp_path, store, embedder)

    assert sorted(report.indexed) == ["coating.pdf", "neural.pdf"]
    assert set(report.failed) == {"broken.pdf", "blank.pdf"}
    assert "no extractable text" in report.failed["blank.pdf"]
