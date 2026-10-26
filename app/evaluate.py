"""Évaluation de la récupération : recall@k et MRR sur un jeu de questions annotées.

Format CSV (UTF-8, séparateur virgule ou point-virgule) :
    question,expected_doc,expected_page,collection
- expected_doc  : fragment du nom de fichier ou du titre du livre attendu (insensible à la casse)
- expected_page : (facultatif) page attendue, tolérance ±1
- collection    : (facultatif) restreint la recherche à un espace
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .config import load_settings
from .retrieve import _fetch_chunks, search_chunks


@dataclass
class EvalReport:
    n: int = 0
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    misses: list[tuple[str, str]] = field(default_factory=list)  # (question, attendu)

    def render(self) -> str:
        lines = [f"Questions évaluées : {self.n}"]
        for k in sorted(self.recall):
            lines.append(f"  recall@{k:<2} : {self.recall[k]:.1%}")
        lines.append(f"  MRR       : {self.mrr:.3f}")
        if self.misses:
            lines.append(f"\nÉchecs à @{max(self.recall)} ({len(self.misses)}) :")
            lines += [f"  - {q}  [attendu : {exp}]" for q, exp in self.misses]
        return "\n".join(lines)


def read_rows(path: str | Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8-sig")
    dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",;\t") if text.strip() else csv.excel
    rows = list(csv.DictReader(text.splitlines(), dialect=dialect))
    return [r for r in rows if (r.get("question") or "").strip()]


def _matches(chunk: dict, expected_doc: str, expected_page: int | None) -> bool:
    doc_ok = expected_doc.lower() in (chunk["filename"] + " " + chunk["title"]).lower()
    if not doc_ok:
        return False
    if expected_page is None:
        return True
    return chunk["page"] is not None and abs(chunk["page"] - expected_page) <= 1


def evaluate(rows: list[dict], ks: tuple[int, ...] = (1, 3, 5, 10)) -> EvalReport:
    settings = load_settings()
    kmax = max(ks)
    ranks: list[int | None] = []
    report = EvalReport()
    for row in rows:
        question = row["question"].strip()
        expected = (row.get("expected_doc") or "").strip()
        page_raw = (row.get("expected_page") or "").strip()
        page = int(page_raw) if page_raw.isdigit() else None
        collections = [c.strip() for c in (row.get("collection") or "").split("|") if c.strip()] or None
        hits = search_chunks(question, collections, kmax, settings)
        chunks = _fetch_chunks([h.chunk_id for h in hits])
        rank = None
        for i, h in enumerate(hits, start=1):
            ch = chunks.get(h.chunk_id)
            if ch and _matches(ch, expected, page):
                rank = i
                break
        ranks.append(rank)
        if rank is None:
            report.misses.append((question, expected + (f" p.{page}" if page else "")))
    report.n = len(rows)
    for k in ks:
        report.recall[k] = sum(1 for r in ranks if r is not None and r <= k) / max(len(ranks), 1)
    report.mrr = sum(1.0 / r for r in ranks if r) / max(len(ranks), 1)
    return report
