"""SQLite : source de vérité du texte (documents, parents, chunks) + index lexical FTS5.

Qdrant ne stocke que les vecteurs et quelques métadonnées de filtrage. Comme tout le texte
reste ici, on peut changer de modèle d'embedding et réindexer sans ré-extraire les livres.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from .config import cfg

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    collection  TEXT NOT NULL DEFAULT 'Général',
    sha256      TEXT NOT NULL,
    pages       INTEGER,
    n_chunks    INTEGER NOT NULL DEFAULT 0,
    n_chars     INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'ok',
    note        TEXT,
    added_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS parents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL,
    ord         INTEGER NOT NULL,
    section     TEXT NOT NULL DEFAULT '',
    page_start  INTEGER,
    page_end    INTEGER,
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_parents_doc ON parents(doc_id);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL,
    parent_id   INTEGER NOT NULL,
    ord         INTEGER NOT NULL,
    page        INTEGER,
    section     TEXT NOT NULL DEFAULT '',
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS ix_chunks_parent ON chunks(parent_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, section,
    content='chunks', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, section) VALUES (new.id, new.text, new.section);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, section)
    VALUES ('delete', old.id, old.text, old.section);
END;
"""

_initialised = False


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Connexion courte, validée à la sortie (rollback en cas d'exception)."""
    global _initialised
    cfg.ensure_dirs()
    conn = sqlite3.connect(cfg.db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        if not _initialised:
            conn.executescript("PRAGMA journal_mode=WAL;")
            conn.executescript(SCHEMA)
            _initialised = True
        conn.execute("PRAGMA synchronous=NORMAL")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def reset_for_tests() -> None:
    global _initialised
    _initialised = False


# ------------------------------------------------------------------ meta


def get_meta(key: str, default: str | None = None) -> str | None:
    with connect() as c:
        row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(key: str, value: str) -> None:
    with connect() as c:
        c.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


# ------------------------------------------------------------------ documents


def get_document_by_filename(filename: str) -> sqlite3.Row | None:
    with connect() as c:
        return c.execute("SELECT * FROM documents WHERE filename=?", (filename,)).fetchone()


def get_document(doc_id: int) -> sqlite3.Row | None:
    with connect() as c:
        return c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()


def list_documents() -> list[sqlite3.Row]:
    with connect() as c:
        return c.execute(
            "SELECT * FROM documents ORDER BY collection COLLATE NOCASE, title COLLATE NOCASE"
        ).fetchall()


def list_collections() -> list[str]:
    with connect() as c:
        rows = c.execute(
            "SELECT DISTINCT collection FROM documents ORDER BY collection COLLATE NOCASE"
        ).fetchall()
    return [r["collection"] for r in rows]


def delete_document_rows(doc_id: int) -> None:
    """Supprime document, parents et chunks (le trigger tient FTS5 à jour)."""
    with connect() as c:
        c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        c.execute("DELETE FROM parents WHERE doc_id=?", (doc_id,))
        c.execute("DELETE FROM documents WHERE id=?", (doc_id,))


def all_chunks_for_embedding(batch: int = 512) -> Iterator[list[sqlite3.Row]]:
    """Itère sur tous les chunks (avec titre du document) par lots, pour la réindexation."""
    last_id = 0
    while True:
        with connect() as c:
            rows = c.execute(
                "SELECT c.id, c.doc_id, c.section, c.text, d.title, d.collection "
                "FROM chunks c JOIN documents d ON d.id = c.doc_id "
                "WHERE c.id > ? ORDER BY c.id LIMIT ?",
                (last_id, batch),
            ).fetchall()
        if not rows:
            return
        last_id = rows[-1]["id"]
        yield rows


def stats() -> dict:
    with connect() as c:
        docs = c.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunks = c.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        chars = c.execute("SELECT COALESCE(SUM(n_chars),0) AS n FROM documents").fetchone()["n"]
    return {"documents": docs, "chunks": chunks, "chars": chars}
