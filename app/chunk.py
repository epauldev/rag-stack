"""Découpage structurel parent/enfant.

- Le texte est d'abord découpé en sections selon les titres Markdown.
- Chaque section devient un ou plusieurs *parents* (≤ PARENT_CHARS), fournis au LLM comme contexte.
- Chaque parent est découpé en *enfants* (~CHILD_CHARS, avec recouvrement) : ce sont eux qui sont
  embeddés et retrouvés, pour la précision de la recherche.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field

from .extract import Segment

HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
MIN_SECTION_CHARS = 600  # une section plus courte est fusionnée avec la précédente si possible


@dataclass
class Child:
    ord: int
    page: int | None
    text: str


@dataclass
class Parent:
    ord: int
    section: str
    page_start: int | None
    page_end: int | None
    text: str
    children: list[Child] = field(default_factory=list)


def _clean_heading(text: str) -> str:
    return re.sub(r"[*_`]+", "", text).strip()


# ------------------------------------------------------------------ fenêtres de texte


def window_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Fenêtres de ~`size` caractères, calées sur la dernière coupure naturelle de la fin de fenêtre."""
    n = len(text)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            lo = start + int(size * 0.6)
            best = -1
            for sep, add in (("\n\n", 2), (". ", 1), ("? ", 1), ("! ", 1), ("\n", 1), ("; ", 1), (" ", 1)):
                i = text.rfind(sep, lo, end)
                if i != -1:
                    best = max(best, i + add)
            if best > start:
                end = best
        spans.append((start, end))
        if end >= n:
            break
        nxt = end - overlap
        if nxt <= start:
            nxt = end
        sp = text.find(" ", nxt, nxt + 60)  # on repart au début d'un mot
        if sp != -1 and sp < end:
            nxt = sp + 1
        start = nxt
    return spans


# ------------------------------------------------------------------ sections


@dataclass
class _Section:
    path: str
    paras: list[tuple[int | None, str]] = field(default_factory=list)  # (page, texte)

    def size(self) -> int:
        return sum(len(t) for _, t in self.paras)


def _sections(segments: list[Segment]) -> list[_Section]:
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    current = _Section(path="")

    def close() -> None:
        nonlocal current
        if current.paras:
            sections.append(current)

    for seg in segments:
        para: list[str] = []

        def end_para() -> None:
            if para:
                current.paras.append((seg.page, "\n".join(para).strip()))
                para.clear()

        for line in seg.text.splitlines():
            m = HEADING_RE.match(line)
            if m:
                end_para()
                close()
                level, title = len(m.group(1)), _clean_heading(m.group(2))
                if not title:
                    current = _Section(path=" > ".join(t for _, t in stack[-3:]))
                    continue
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                current = _Section(path=" > ".join(t for _, t in stack[-3:]))
                current.paras.append((seg.page, f"{'#' * level} {title}"))
            elif not line.strip():
                end_para()
            else:
                para.append(line)
        end_para()
    close()
    return sections


def _merge_small(sections: list[_Section], parent_chars: int) -> list[_Section]:
    merged: list[_Section] = []
    for sec in sections:
        if merged and sec.size() < MIN_SECTION_CHARS and merged[-1].size() + sec.size() <= parent_chars:
            merged[-1].paras.extend(sec.paras)
        else:
            merged.append(_Section(path=sec.path, paras=list(sec.paras)))
    return merged


# ------------------------------------------------------------------ parents / enfants


def _split_to_parents(sec: _Section, parent_chars: int) -> list[list[tuple[int | None, str]]]:
    """Regroupe les paragraphes d'une section en blocs ≤ parent_chars (un paragraphe trop long est coupé)."""
    groups: list[list[tuple[int | None, str]]] = []
    cur: list[tuple[int | None, str]] = []
    size = 0
    for page, text in sec.paras:
        pieces = [text]
        if len(text) > parent_chars:
            pieces = [text[a:b] for a, b in window_spans(text, parent_chars, 0)]
        for piece in pieces:
            if cur and size + len(piece) + 2 > parent_chars:
                groups.append(cur)
                cur, size = [], 0
            cur.append((page, piece))
            size += len(piece) + 2
    if cur:
        groups.append(cur)
    return groups


def chunk_document(
    segments: list[Segment], parent_chars: int = 4000, child_chars: int = 1500, child_overlap: int = 200
) -> list[Parent]:
    parents: list[Parent] = []
    child_ord = 0
    for sec in _merge_small(_sections(segments), parent_chars):
        for paras in _split_to_parents(sec, parent_chars):
            text = "\n\n".join(t for _, t in paras)
            offsets: list[int] = []
            pos = 0
            for _, t in paras:
                offsets.append(pos)
                pos += len(t) + 2
            pages = [p for p, _ in paras if p is not None]
            parent = Parent(
                ord=len(parents),
                section=sec.path,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                text=text,
            )
            for a, b in window_spans(text, child_chars, child_overlap):
                piece = text[a:b].strip()
                if len(piece) < 30 and parent.children:
                    continue
                idx = max(bisect_right(offsets, a) - 1, 0)
                parent.children.append(Child(ord=child_ord, page=paras[idx][0], text=piece))
                child_ord += 1
            if parent.children:
                parents.append(parent)
    for i, p in enumerate(parents):
        p.ord = i
    return parents


def embedding_text(title: str, section: str, text: str) -> str:
    """Texte réellement embeddé : le titre et la section donnent du contexte au fragment."""
    head = title if not section else f"{title} — {section}"
    return f"{head}\n{text}"
