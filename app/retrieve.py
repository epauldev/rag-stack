"""Recherche hybride : vecteurs (Qdrant) + BM25 (SQLite FTS5), fusion RRF, reranking optionnel."""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

from . import db, store
from .chunk import embedding_text
from .config import Settings, cfg, load_settings
from .llm import embed

log = logging.getLogger(__name__)

RRF_K = 60

_STOP = set(
    """
    le la les un une des du de d l et ou à a au aux en dans sur sous par pour avec sans ce ces cet cette
    qui que quoi dont où est sont être été a ont avoir il elle ils elles on nous vous je tu me te se
    ne pas plus moins très comme mais donc or ni car si the a an and or of to in on at by for with
    without is are was were be been it its this that these those as from what which who how why
    """.split()
)


@dataclass
class Hit:
    """Un enfant retrouvé."""

    chunk_id: int
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None


@dataclass
class Passage:
    """Unité de contexte transmise au LLM : le parent d'un ou plusieurs enfants retrouvés."""

    parent_id: int
    doc_id: int
    title: str
    filename: str
    collection: str
    section: str
    page_start: int | None
    page_end: int | None
    text: str
    score: float

    def label(self) -> str:
        loc = self.title
        if self.section:
            loc += f" — {self.section}"
        if self.page_start is not None:
            loc += f" (p. {self.page_start}" + (f"–{self.page_end}" if self.page_end != self.page_start else "") + ")"
        return loc


# ------------------------------------------------------------------ lexical


def fts_query(question: str) -> str | None:
    tokens = [t for t in re.findall(r"\w+", question.lower()) if len(t) > 1 and t not in _STOP]
    if not tokens:
        return None
    seen: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.append(t)
    return " OR ".join(f'"{t}"' for t in seen)


def lexical_search(question: str, limit: int, collections: list[str] | None) -> list[int]:
    q = fts_query(question)
    if q is None:
        return []
    sql = (
        "SELECT c.id FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
        "JOIN documents d ON d.id = c.doc_id WHERE chunks_fts MATCH ?"
    )
    args: list = [q]
    if collections:
        sql += f" AND d.collection IN ({','.join('?' * len(collections))})"
        args += list(collections)
    sql += " ORDER BY bm25(chunks_fts) LIMIT ?"
    args.append(limit)
    with db.connect() as c:
        return [r["id"] for r in c.execute(sql, args).fetchall()]


# ------------------------------------------------------------------ reranker

_reranker = None
_reranker_failed = False
_reranker_lock = threading.Lock()


def _get_reranker():
    global _reranker, _reranker_failed
    with _reranker_lock:
        if _reranker is None and not _reranker_failed:
            try:
                from sentence_transformers import CrossEncoder

                _reranker = CrossEncoder(cfg.reranker_model, max_length=512)
            except Exception as exc:  # noqa: BLE001
                log.warning("Reranker indisponible (%s) : recherche sans reranking.", exc)
                _reranker_failed = True
        return _reranker


def reranker_available() -> bool:
    return _get_reranker() is not None


# ------------------------------------------------------------------ pipeline


def search_chunks(
    question: str,
    collections: list[str] | None = None,
    top_k: int = 6,
    settings: Settings | None = None,
) -> list[Hit]:
    """Meilleurs enfants pour la question (avant regroupement par parent)."""
    s = settings or load_settings()
    collections = collections or None

    indexed_model = db.get_meta("embed_model")
    if indexed_model and s.embed_model and indexed_model != s.embed_model:
        raise store.EmbeddingMismatch(
            f"L'index a été construit avec « {indexed_model} » mais le modèle d'embedding actuel est "
            f"« {s.embed_model} ». Revenez au premier modèle ou lancez « Réindexer tout » (Bibliothèque)."
        )

    dense: list[tuple[int, float]] = []
    if store.collection_dim() is not None:
        qvec = embed([question], "query", s)[0]
        store.ensure(len(qvec))
        dense = store.search(qvec, cfg.dense_k, collections)
    lexical = lexical_search(question, cfg.lexical_k, collections)

    hits: dict[int, Hit] = {}
    for rank, (cid, _score) in enumerate(dense, start=1):
        h = hits.setdefault(cid, Hit(cid, 0.0))
        h.dense_rank = rank
        h.score += 1.0 / (RRF_K + rank)
    for rank, cid in enumerate(lexical, start=1):
        h = hits.setdefault(cid, Hit(cid, 0.0))
        h.lexical_rank = rank
        h.score += 1.0 / (RRF_K + rank)

    ranked = sorted(hits.values(), key=lambda h: h.score, reverse=True)[: cfg.rerank_candidates]

    if s.use_reranker and ranked and (rr := _get_reranker()) is not None:
        rows = _fetch_chunks([h.chunk_id for h in ranked])
        pairs = [
            (question, embedding_text(rows[h.chunk_id]["title"], rows[h.chunk_id]["section"], rows[h.chunk_id]["text"]))
            for h in ranked
        ]
        for h, sc in zip(ranked, rr.predict(pairs)):
            h.rerank_score = float(sc)
        ranked.sort(key=lambda h: h.rerank_score, reverse=True)

    return ranked[:top_k]


def _fetch_chunks(ids: list[int]) -> dict[int, dict]:
    if not ids:
        return {}
    with db.connect() as c:
        rows = c.execute(
            "SELECT c.id, c.parent_id, c.section, c.text, c.page, d.title, d.filename, d.collection, d.id AS doc_id "
            f"FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.id IN ({','.join('?' * len(ids))})",
            ids,
        ).fetchall()
    return {r["id"]: dict(r) for r in rows}


def to_passages(hits: list[Hit]) -> list[Passage]:
    """Regroupe les enfants par parent (ordre de pertinence conservé) et charge le texte parent."""
    chunks = _fetch_chunks([h.chunk_id for h in hits])
    order: list[int] = []
    best: dict[int, float] = {}
    for h in hits:
        ch = chunks.get(h.chunk_id)
        if ch is None:  # chunk supprimé entre-temps
            continue
        pid = ch["parent_id"]
        if pid not in best:
            order.append(pid)
            best[pid] = h.score if h.rerank_score is None else h.rerank_score
    if not order:
        return []
    with db.connect() as c:
        prows = c.execute(
            "SELECT p.id, p.doc_id, p.section, p.page_start, p.page_end, p.text, d.title, d.filename, d.collection "
            f"FROM parents p JOIN documents d ON d.id = p.doc_id WHERE p.id IN ({','.join('?' * len(order))})",
            order,
        ).fetchall()
    by_id = {r["id"]: r for r in prows}
    out: list[Passage] = []
    for pid in order:
        r = by_id.get(pid)
        if r is None:
            continue
        out.append(
            Passage(pid, r["doc_id"], r["title"], r["filename"], r["collection"], r["section"],
                    r["page_start"], r["page_end"], r["text"], best[pid])
        )
    return out


def retrieve(
    question: str,
    collections: list[str] | None = None,
    top_k: int | None = None,
    settings: Settings | None = None,
) -> list[Passage]:
    s = settings or load_settings()
    return to_passages(search_chunks(question, collections, top_k or s.top_k, s))
