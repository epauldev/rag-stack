"""Ingestion : fichier → extraction → découpage → SQLite → embeddings → Qdrant."""
from __future__ import annotations

import hashlib
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import db, store
from .chunk import chunk_document, embedding_text
from .config import cfg, load_settings
from .extract import SUPPORTED, ExtractionError, extract, title_from_filename
from .llm import embed

ProgressFn = Callable[[float, str], None]

# Une seule indexation à la fois (évite les conflits d'écriture et la saturation de LM Studio).
ingest_lock = threading.Lock()


@dataclass
class IngestResult:
    filename: str
    status: str  # ok | skipped | replaced | moved | ocr_needed | error
    message: str
    doc_id: int | None = None
    n_chunks: int = 0

    def line(self) -> str:
        icon = {"ok": "✅", "replaced": "🔁", "skipped": "⏭️", "moved": "📂", "ocr_needed": "⚠️", "error": "❌"}
        return f"{icon.get(self.status, '•')} {self.filename} — {self.message}"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def remove_document(doc_id: int) -> None:
    store.delete_doc(doc_id)
    db.delete_document_rows(doc_id)
    for f in cfg.extracted_dir.glob(f"{doc_id}_*.md"):
        f.unlink(missing_ok=True)


def move_document(doc_id: int, collection: str) -> None:
    collection = collection.strip() or "Général"
    with db.connect() as c:
        c.execute("UPDATE documents SET collection=? WHERE id=?", (collection, doc_id))
    store.set_doc_collection(doc_id, collection)


def ingest_file(
    path: str | Path,
    collection: str = "Général",
    filename: str | None = None,
    progress: ProgressFn | None = None,
    keep_copy: bool = False,
) -> IngestResult:
    """Indexe un fichier. `keep_copy` conserve l'original dans data/library (fichiers téléversés)."""
    path = Path(path)
    filename = filename or path.name
    collection = (collection or "Général").strip() or "Général"
    report = progress or (lambda _f, _m: None)

    if path.suffix.lower() not in SUPPORTED:
        return IngestResult(filename, "error", f"format non pris en charge ({path.suffix or 'sans extension'})")

    report(0.0, f"Lecture de {filename}")
    digest = sha256_of(path)
    existing = db.get_document_by_filename(filename)
    replaced = False
    if existing is not None:
        if existing["sha256"] == digest and existing["status"] == "ok":
            if existing["collection"] != collection:
                move_document(existing["id"], collection)
                return IngestResult(filename, "moved", f"déjà indexé, déplacé vers « {collection} »", existing["id"])
            return IngestResult(filename, "skipped", "déjà indexé, identique", existing["id"])
        remove_document(existing["id"])
        replaced = True

    report(0.05, f"Extraction du texte de {filename}")
    try:
        extracted = extract(path)
    except ExtractionError as exc:
        return IngestResult(filename, "error", str(exc))

    title = title_from_filename(filename)
    if extracted.n_chars == 0:
        doc_id = _insert_document(filename, title, collection, digest, extracted.pages, 0, 0, "ocr_needed",
                                  extracted.warning or "Aucun texte extrait.")
        return IngestResult(filename, "ocr_needed", extracted.warning or "aucun texte extrait", doc_id)

    parents = chunk_document(extracted.segments, cfg.parent_chars, cfg.child_chars, cfg.child_overlap)
    n_children = sum(len(p.children) for p in parents)
    if n_children == 0:
        return IngestResult(filename, "error", "aucun passage exploitable après découpage")

    # 1. Texte dans SQLite (source de vérité), pour obtenir les identifiants de chunks
    status = "ocr_needed" if extracted.warning else "ok"
    doc_id = _insert_document(filename, title, collection, digest, extracted.pages, n_children,
                              extracted.n_chars, status, extracted.warning)
    chunk_ids: list[int] = []
    texts: list[str] = []
    try:
        with db.connect() as c:
            for p in parents:
                cur = c.execute(
                    "INSERT INTO parents(doc_id, ord, section, page_start, page_end, text) VALUES (?,?,?,?,?,?)",
                    (doc_id, p.ord, p.section, p.page_start, p.page_end, p.text),
                )
                parent_id = cur.lastrowid
                for ch in p.children:
                    cur = c.execute(
                        "INSERT INTO chunks(doc_id, parent_id, ord, page, section, text) VALUES (?,?,?,?,?,?)",
                        (doc_id, parent_id, ch.ord, ch.page, p.section, ch.text),
                    )
                    chunk_ids.append(cur.lastrowid)
                    texts.append(embedding_text(title, p.section, ch.text))

        # 2. Embeddings par lots, puis écriture dans Qdrant
        settings = load_settings()
        vectors: list[list[float]] = []
        batch = 32
        for i in range(0, len(texts), batch):
            vectors.extend(embed(texts[i : i + batch], "doc", settings))
            report(0.1 + 0.85 * min(i + batch, len(texts)) / len(texts),
                   f"Embeddings {filename} : {min(i + batch, len(texts))}/{len(texts)} passages")
        store.ensure(len(vectors[0]))
        db.set_meta("embed_model", settings.embed_model)
        db.set_meta("embed_dim", str(len(vectors[0])))
        store.upsert(
            chunk_ids,
            vectors,
            [{"doc_id": doc_id, "collection": collection} for _ in chunk_ids],
        )
    except Exception:
        remove_document(doc_id)  # pas d'indexation à moitié faite
        raise

    _save_extracted_text(doc_id, filename, extracted)
    if keep_copy:
        cfg.library_dir.mkdir(parents=True, exist_ok=True)
        dest = cfg.library_dir / filename
        if path.resolve() != dest.resolve():
            shutil.copy2(path, dest)

    report(1.0, f"{filename} indexé")
    msg = f"{n_children} passages indexés"
    if extracted.pages:
        msg += f" ({extracted.pages} pages)"
    if extracted.warning:
        msg += f" — ⚠️ {extracted.warning}"
    return IngestResult(filename, "replaced" if replaced else ("ocr_needed" if extracted.warning else "ok"),
                        msg, doc_id, n_children)


def _insert_document(filename, title, collection, digest, pages, n_chunks, n_chars, status, note) -> int:
    with db.connect() as c:
        cur = c.execute(
            "INSERT INTO documents(filename, title, collection, sha256, pages, n_chunks, n_chars, status, note, added_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (filename, title, collection, digest, pages, n_chunks, n_chars, status, note, db.now_iso()),
        )
        return cur.lastrowid


def _save_extracted_text(doc_id: int, filename: str, extracted) -> None:
    """Garde le texte extrait hors des bases (inspection, réutilisation éventuelle)."""
    cfg.extracted_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.extracted_dir / f"{doc_id}_{Path(filename).stem}.md"
    parts = []
    for seg in extracted.segments:
        if seg.page is not None:
            parts.append(f"<!-- page {seg.page} -->")
        parts.append(seg.text)
    out.write_text("\n\n".join(parts), encoding="utf-8")


def scan_inbox(default_collection: str, progress: ProgressFn | None = None) -> list[IngestResult]:
    """Indexe le dossier inbox. Un sous-dossier de premier niveau = un espace (collection)."""
    files = sorted(p for p in cfg.inbox_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    results: list[IngestResult] = []
    for i, f in enumerate(files):
        rel = f.relative_to(cfg.inbox_dir)
        collection = rel.parts[0] if len(rel.parts) > 1 else default_collection
        sub = (lambda frac, msg, i=i: progress((i + frac) / max(len(files), 1), msg)) if progress else None
        try:
            results.append(ingest_file(f, collection, progress=sub))
        except Exception as exc:  # noqa: BLE001
            results.append(IngestResult(f.name, "error", str(exc)))
    return results


def reindex_all(progress: ProgressFn | None = None) -> int:
    """Recalcule tous les vecteurs depuis le texte stocké (après un changement de modèle d'embedding)."""
    settings = load_settings()
    report = progress or (lambda _f, _m: None)
    total = db.stats()["chunks"]
    if total == 0:
        return 0
    store.reset()
    done = 0
    dim_set = False
    for rows in db.all_chunks_for_embedding(batch=64):
        vectors = embed([embedding_text(r["title"], r["section"], r["text"]) for r in rows], "doc", settings)
        if not dim_set:
            store.ensure(len(vectors[0]))
            db.set_meta("embed_dim", str(len(vectors[0])))
            dim_set = True
        store.upsert([r["id"] for r in rows], vectors,
                     [{"doc_id": r["doc_id"], "collection": r["collection"]} for r in rows])
        done += len(rows)
        report(done / total, f"Réindexation : {done}/{total} passages")
    db.set_meta("embed_model", settings.embed_model)
    return done
