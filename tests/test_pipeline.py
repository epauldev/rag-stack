"""Tests de bout en bout : extraction → ingestion → recherche hybride → réponse (LM Studio simulé)."""
from __future__ import annotations

import shutil

import pytest

import make_samples
from app import answer, db, ingest, store
from app.chunk import chunk_document, window_spans
from app.config import load_settings, save_settings
from app.extract import Segment, extract
from app.llm import strip_think
from app.retrieve import fts_query, retrieve, search_chunks, to_passages


@pytest.fixture(scope="module")
def samples(tmp_root):
    return make_samples.make_all(tmp_root / "samples")


# ------------------------------------------------------------------ extraction & découpage


def test_extract_pdf_has_pages_and_headings(samples):
    ex = extract(samples["pdf"])
    assert ex.pages == 2 and ex.warning is None
    assert [s.page for s in ex.segments] == [1, 2]
    assert "chloroplastes" in ex.segments[0].text
    assert any(line.lstrip().startswith("#") for s in ex.segments for line in s.text.splitlines()), \
        "les titres du PDF doivent ressortir en Markdown"


def test_extract_scanned_pdf_is_flagged(samples):
    ex = extract(samples["scan"])
    assert ex.pages == 3
    assert ex.warning and "OCR" in ex.warning


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract non installé")
def test_image_only_pdf_is_ocrised_automatically(samples, tmp_root):
    """Un PDF qui n'est qu'une image doit être lu via l'OCR (Tesseract est inclus dans l'image Docker)."""
    import pymupdf

    src = pymupdf.open(samples["pdf"])
    img = pymupdf.open()
    for page in src:
        p = img.new_page(width=page.rect.width, height=page.rect.height)
        p.insert_image(p.rect, pixmap=page.get_pixmap(dpi=200))
    out = tmp_root / "samples" / "Image_seule.pdf"
    img.save(out)
    assert sum(len(p.get_text()) for p in pymupdf.open(out)) == 0, "précondition : aucune couche texte"
    ex = extract(out)
    assert ex.warning is None
    assert "chloroplastes" in " ".join(s.text for s in ex.segments).lower()


def test_extract_docx_epub_txt(samples):
    docx_text = "\n".join(s.text for s in extract(samples["docx"]).segments)
    assert "# Vector databases" in docx_text and "HNSW" in docx_text
    epub_text = "\n".join(s.text for s in extract(samples["epub"]).segments)
    assert "# Politique monétaire" in epub_text and "inflation" in epub_text
    assert "théorème central limite" in extract(samples["txt"]).segments[0].text


def test_window_spans_cover_text_with_overlap():
    text = ("Phrase numéro un. " * 200).strip()
    spans = window_spans(text, 500, 80)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
        assert a2 < b1, "recouvrement attendu"
        assert a2 > a1, "progression garantie"
    assert all(b - a <= 500 for a, b in spans)


def test_chunking_keeps_section_path_and_pages():
    segs = [
        Segment("# Livre\n\n## Chapitre A\n\n" + "Alpha beta gamma. " * 120, 1),
        Segment("## Chapitre B\n\n" + "Delta epsilon zêta. " * 120, 7),
    ]
    parents = chunk_document(segs, parent_chars=1800, child_chars=600, child_overlap=80)
    assert parents and all(p.children for p in parents)
    sections = {p.section for p in parents}
    assert "Livre > Chapitre A" in sections and "Livre > Chapitre B" in sections
    b = next(p for p in parents if p.section.endswith("Chapitre B"))
    assert b.page_start == 7 and all(c.page == 7 for c in b.children)
    assert all(len(p.text) <= 1800 + 50 for p in parents)


def test_strip_think_across_chunk_boundaries():
    pieces = ["Bonjour <thi", "nk>je réfl", "échis</th", "ink>\nRéponse [1]", "."]
    assert "".join(strip_think(pieces)) == "Bonjour Réponse [1]."
    assert "".join(strip_think(["Pas de balise ", "ici <b>gras</b>"])) == "Pas de balise ici <b>gras</b>"


def test_fts_query_drops_stopwords_and_quotes():
    q = fts_query("Quelle est la couleur des feuilles ?")
    assert q and '"couleur"' in q and '"feuilles"' in q and '"est"' not in q
    assert fts_query("le la les") is None


# ------------------------------------------------------------------ ingestion


@pytest.fixture(scope="module")
def indexed(samples):
    results = {
        "pdf": ingest.ingest_file(samples["pdf"], "Biologie"),
        "docx": ingest.ingest_file(samples["docx"], "Informatique"),
        "epub": ingest.ingest_file(samples["epub"], "Économie"),
        "txt": ingest.ingest_file(samples["txt"], "Statistiques"),
        "scan": ingest.ingest_file(samples["scan"], "Biologie"),
    }
    return results


def test_ingest_results(indexed):
    for key in ("pdf", "docx", "epub", "txt"):
        assert indexed[key].status == "ok", indexed[key].line()
        assert indexed[key].n_chunks >= 2
    assert indexed["scan"].status == "ocr_needed" and indexed["scan"].n_chunks == 0
    stats = db.stats()
    assert stats["documents"] == 5
    assert store.count() == stats["chunks"] > 0, "autant de vecteurs que de chunks"
    assert db.get_meta("embed_model") == "mock-embed-bge-m3"


def test_hybrid_search_finds_right_document(indexed):
    passages = retrieve("Comment fonctionne l'index HNSW pour la recherche approximative ?")
    assert passages[0].filename.endswith("Vector_Databases.docx")
    passages = retrieve("Où se déroule la photosynthèse ? chloroplastes")
    assert passages[0].filename.startswith("Campbell")
    assert passages[0].page_start in (1, 2)


def test_lexical_leg_finds_exact_identifier(indexed):
    hits = search_chunks("ZX-4471", top_k=3)
    assert hits and hits[0].lexical_rank == 1
    p = to_passages(hits)[0]
    assert "ZX-4471" in p.text and p.filename.endswith(".docx")


def test_collection_filter_is_applied_to_both_legs(indexed):
    allp = retrieve("inflation banque centrale taux directeurs", top_k=6)
    assert any(p.collection == "Économie" for p in allp)
    only_bio = retrieve("inflation banque centrale taux directeurs", collections=["Biologie"], top_k=6)
    assert only_bio and all(p.collection == "Biologie" for p in only_bio)
    only_eco = retrieve("inflation banque centrale taux directeurs", collections=["Économie"], top_k=6)
    assert only_eco and all(p.collection == "Économie" for p in only_eco)


def test_reingest_same_file_is_skipped_and_changed_file_replaced(indexed, samples):
    n_before = db.stats()["chunks"]
    r = ingest.ingest_file(samples["txt"], "Statistiques")
    assert r.status == "skipped" and db.stats()["chunks"] == n_before

    r = ingest.ingest_file(samples["txt"], "Probabilités")  # même contenu, autre espace
    assert r.status == "moved"
    assert retrieve("théorème central limite", collections=["Probabilités"])
    assert not retrieve("théorème central limite", collections=["Statistiques"])

    samples["txt"].write_text("Contenu entièrement nouveau sur les chaînes de Markov. " * 30, encoding="utf-8")
    r = ingest.ingest_file(samples["txt"], "Probabilités")
    assert r.status == "replaced"
    assert retrieve("chaînes de Markov", collections=["Probabilités"])
    assert store.count() == db.stats()["chunks"]


def test_unsupported_format_and_embedding_failure_leave_no_trace(tmp_root, lmstudio):
    bad = tmp_root / "samples" / "x.xyz"
    bad.write_text("peu importe")
    assert ingest.ingest_file(bad).status == "error"

    good = tmp_root / "samples" / "Panne.txt"
    good.write_text("Texte à indexer pendant une panne de LM Studio. " * 20)
    s = load_settings()
    save_settings(type(s)(**{**s.__dict__, "lmstudio_base_url": "http://127.0.0.1:9/v1"}))
    try:
        with pytest.raises(Exception):
            ingest.ingest_file(good)
    finally:
        save_settings(s)
    assert db.get_document_by_filename("Panne.txt") is None, "pas d'indexation à moitié faite"
    assert store.count() == db.stats()["chunks"]


# ------------------------------------------------------------------ réponse


def test_answer_pipeline_streams_without_think_and_lists_sources(indexed):
    prep = answer.prepare("Que produit la phase claire de la photosynthèse ?")
    assert prep.passages and "Extraits :" in prep.messages[-1]["content"]
    assert prep.messages[0]["role"] == "system"
    text = "".join(answer.stream_answer(prep))
    assert "Réponse simulée" in text and "<think>" not in text and "raisonnement interne" not in text
    sources = answer.format_sources(prep.passages, prep.standalone, prep.question, prep.retrieval_seconds)
    assert "[1]" in sources and "Campbell" in sources


def test_followup_question_is_condensed_with_history(indexed):
    history = [
        {"role": "user", "content": "Parle-moi de la photosynthèse"},
        {"role": "assistant", "content": "Elle se déroule dans les chloroplastes [1]."},
    ]
    prep = answer.prepare("Et la respiration ?", history)
    assert prep.standalone != ""  # reformulée par le (faux) LLM
    assert [m["role"] for m in prep.messages] == ["system", "user", "assistant", "user"]


def test_no_documents_gives_explicit_message(indexed):
    prep = answer.prepare("Question sans rapport", collections=["EspaceInexistant"])
    assert prep.passages == []
    assert "aucun passage" in "".join(answer.stream_answer(prep)).lower()


# ------------------------------------------------------------------ interface


def test_ui_respond_handles_structured_history_and_streams(indexed):
    from app import ui

    history = [
        {"role": "user", "content": [{"type": "text", "text": "Parle-moi de la photosynthèse"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Elle a lieu dans les chloroplastes [1]."}]},
    ]
    states = list(ui.respond("Et la respiration cellulaire ?", history, ["Biologie"], 4))
    final_chat, final_sources, _ = states[-1]
    assert [m["role"] for m in final_chat] == ["user", "assistant", "user", "assistant"]
    assert all(isinstance(m["content"], str) for m in final_chat)
    assert "Réponse simulée" in final_chat[-1]["content"]
    assert "Campbell" in final_sources
    assert len(states) >= 4, "la réponse doit arriver en streaming, pas d'un bloc"

    empty = list(ui.respond("   ", [], [], 4))
    assert len(empty) == 1


def test_ui_builds_and_library_helpers_work(indexed):
    from app import ui

    assert ui.build_app() is not None
    rows = ui._doc_rows()
    assert any(r[2].startswith("Campbell") for r in rows)
    assert "Qdrant" in ui.health_markdown()


# ------------------------------------------------------------------ évaluation & CLI


def test_eval_recall_and_cli(indexed, tmp_root, capsys):
    from app import cli, evaluate

    csv_path = tmp_root / "eval.csv"
    csv_path.write_text(
        "question,expected_doc,expected_page,collection\n"
        "Que produit la phase claire de la photosynthèse ?,Campbell,1,\n"
        "Où se déroule la respiration cellulaire ? mitochondries,Campbell,2,Biologie\n"
        "Qu'est-ce que l'index HNSW ?,Vector_Databases,,\n"
        "Question dont la réponse n'existe nulle part dans la bibliothèque zzqx,Inexistant,,\n",
        encoding="utf-8",
    )
    report = evaluate.evaluate(evaluate.read_rows(csv_path))
    assert report.n == 4
    assert report.recall[5] == 0.75 and report.recall[1] >= 0.5
    assert len(report.misses) == 1 and "Inexistant" in report.misses[0][1]

    capsys.readouterr()
    assert cli.main(["eval", str(csv_path)]) == 0
    assert "recall@5" in capsys.readouterr().out
    assert cli.main(["stats"]) == 0
    assert "Biologie" in capsys.readouterr().out
    assert cli.main(["ask", "Que produit la phase claire ?", "-c", "Biologie"]) == 0
    out = capsys.readouterr().out
    assert "Réponse simulée" in out and "Sources :" in out and "Campbell" in out


# ------------------------------------------------------------------ suppression, réindexation, changement de modèle


def test_embedding_model_mismatch_is_detected(indexed):
    s = load_settings()
    save_settings(type(s)(**{**s.__dict__, "embed_model": "autre-modele"}))
    try:
        with pytest.raises(store.EmbeddingMismatch):
            retrieve("photosynthèse")
    finally:
        save_settings(s)


def test_reindex_all_rebuilds_vectors_from_stored_text(indexed):
    n = db.stats()["chunks"]
    store.reset()
    assert store.count() == 0
    assert ingest.reindex_all() == n
    assert store.count() == n
    assert retrieve("chloroplastes", top_k=3)[0].filename.startswith("Campbell")


def test_delete_document_removes_everything(indexed):
    doc = db.get_document_by_filename("Mankiw_Economie.epub")
    assert retrieve("inflation banque centrale", collections=["Économie"])
    ingest.remove_document(doc["id"])
    assert db.get_document_by_filename("Mankiw_Economie.epub") is None
    assert not retrieve("inflation banque centrale", collections=["Économie"])
    assert store.count() == db.stats()["chunks"]
    with db.connect() as c:
        assert c.execute("SELECT COUNT(*) AS n FROM chunks_fts WHERE chunks_fts MATCH 'inflation'").fetchone()["n"] == 0


def test_inbox_subfolders_become_collections(tmp_root, samples):
    from app.config import cfg

    (cfg.inbox_dir / "Histoire").mkdir(parents=True, exist_ok=True)
    (cfg.inbox_dir / "Histoire" / "Rome.txt").write_text("La République romaine a précédé l'Empire. " * 30)
    (cfg.inbox_dir / "Vrac.txt").write_text("Document posé à la racine de l'inbox sur la marmotte. " * 30)
    results = ingest.scan_inbox("Divers")
    assert {r.status for r in results} == {"ok"}
    assert retrieve("République romaine", collections=["Histoire"])
    assert retrieve("marmotte", collections=["Divers"])
