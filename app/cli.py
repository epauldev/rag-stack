"""Ligne de commande : ingestion en lot, requêtes, réindexation, évaluation.

Exemples (dans le conteneur : `docker compose exec app python -m app.cli ...`)
    python -m app.cli stats
    python -m app.cli ingest /inbox --collection "IA"
    python -m app.cli inbox
    python -m app.cli ask "Qu'est-ce que la recherche hybride ?" -c IA
    python -m app.cli reindex
    python -m app.cli eval eval.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import answer, db, evaluate, ingest, store
from .config import cfg
from .extract import SUPPORTED


def _progress(frac: float, msg: str) -> None:
    print(f"\r[{frac:5.0%}] {msg[:100]:<100}", end="", flush=True)


def cmd_stats(_a) -> int:
    s = db.stats()
    ok, info = store.ping()
    print(f"Documents : {s['documents']}\nPassages  : {s['chunks']}\nCaractères: {s['chars']:,}".replace(",", " "))
    print(f"Qdrant    : {'OK' if ok else 'INJOIGNABLE — ' + info} ({store.count() if ok else '?'} vecteurs)")
    print(f"Modèle d'embedding de l'index : {db.get_meta('embed_model') or '—'}")
    print("Espaces   : " + (", ".join(db.list_collections()) or "—"))
    return 0


def cmd_ingest(a) -> int:
    root = Path(a.path)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        print("Aucun fichier à indexer.")
        return 1
    errors = 0
    with ingest.ingest_lock:
        for i, f in enumerate(files):
            collection = a.collection
            if root.is_dir() and not a.flat and len(f.relative_to(root).parts) > 1:
                collection = f.relative_to(root).parts[0]
            try:
                r = ingest.ingest_file(f, collection, progress=lambda fr, m, i=i: _progress((i + fr) / len(files), m))
            except Exception as exc:  # noqa: BLE001
                r = ingest.IngestResult(f.name, "error", str(exc))
            print("\r" + " " * 112 + "\r" + r.line())
            errors += r.status == "error"
    return 1 if errors else 0


def cmd_inbox(a) -> int:
    with ingest.ingest_lock:
        results = ingest.scan_inbox(a.collection, _progress)
    print("\r" + " " * 112 + "\r", end="")
    for r in results:
        print(r.line())
    print(f"{len(results)} fichier(s) traité(s) depuis {cfg.inbox_dir}")
    return 1 if any(r.status == "error" for r in results) else 0


def cmd_reindex(_a) -> int:
    with ingest.ingest_lock:
        n = ingest.reindex_all(_progress)
    print(f"\n{n} passages réindexés.")
    return 0


def cmd_ask(a) -> int:
    prep = answer.prepare(a.question, collections=a.collection or None, top_k=a.top_k)
    print()
    for piece in answer.stream_answer(prep):
        print(piece, end="", flush=True)
    print("\n\nSources :")
    for i, p in enumerate(prep.passages, start=1):
        print(f"  [{i}] {p.label()}  ({p.filename})")
    return 0


def cmd_eval(a) -> int:
    rows = evaluate.read_rows(a.csv)
    print(evaluate.evaluate(rows).render())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)

    s = sub.add_parser("ingest", help="indexer un fichier ou un dossier")
    s.add_argument("path")
    s.add_argument("-c", "--collection", default="Général")
    s.add_argument("--flat", action="store_true", help="ne pas utiliser les sous-dossiers comme espaces")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("inbox", help="indexer le dossier inbox (sous-dossiers = espaces)")
    s.add_argument("-c", "--collection", default="Général", help="espace des fichiers à la racine")
    s.set_defaults(fn=cmd_inbox)

    sub.add_parser("reindex", help="recalculer tous les vecteurs").set_defaults(fn=cmd_reindex)

    s = sub.add_parser("ask", help="poser une question")
    s.add_argument("question")
    s.add_argument("-c", "--collection", action="append")
    s.add_argument("-k", "--top-k", type=int)
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("eval", help="recall@k sur un CSV de questions")
    s.add_argument("csv")
    s.set_defaults(fn=cmd_eval)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except Exception as exc:  # noqa: BLE001
        print(f"\nErreur : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
