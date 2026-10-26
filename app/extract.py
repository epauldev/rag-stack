"""Extraction du texte : PDF, DOCX, EPUB, HTML, TXT/MD → segments Markdown avec numéro de page."""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from .config import cfg

SUPPORTED = {".pdf", ".docx", ".epub", ".html", ".htm", ".txt", ".md"}

# En dessous de ce nombre moyen de caractères par page, un PDF est considéré comme scanné.
MIN_CHARS_PER_PAGE = 80


@dataclass
class Segment:
    text: str
    page: int | None = None


@dataclass
class Extracted:
    segments: list[Segment] = field(default_factory=list)
    pages: int | None = None
    warning: str | None = None

    @property
    def n_chars(self) -> int:
        return sum(len(s.text) for s in self.segments)


class ExtractionError(RuntimeError):
    pass


def title_from_filename(filename: str) -> str:
    """« Auteur_Titre_2020.pdf » → « Auteur Titre 2020 »."""
    stem = Path(filename).stem
    return re.sub(r"[_]+", " ", stem).strip() or stem


def extract(path: str | Path) -> Extracted:
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in SUPPORTED:
        raise ExtractionError(
            f"Format non pris en charge : {ext or '(sans extension)'}. "
            f"Formats acceptés : {', '.join(sorted(SUPPORTED))}."
        )
    try:
        if ext == ".pdf":
            return _pdf(p)
        if ext == ".docx":
            return _docx(p)
        if ext == ".epub":
            return _epub(p)
        if ext in {".html", ".htm"}:
            return _html_file(p)
        return _text(p)
    except ExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"Lecture impossible de {p.name} : {exc}") from exc


# ------------------------------------------------------------------ PDF


def _pdf(path: Path) -> Extracted:
    import pymupdf

    doc = pymupdf.open(path)
    n_pages = doc.page_count
    segments: list[Segment] = []
    try:
        import pymupdf4llm

        # Si Tesseract est installé (c'est le cas dans l'image Docker), pymupdf4llm applique l'OCR
        # de lui-même aux pages sans couche texte. Les anciennes versions ignorent ou refusent
        # certaines options : on retente avec des jeux d'options de plus en plus simples.
        chunks = None
        for opts in (
            {"header": False, "footer": False, "ocr_language": cfg.ocr_language},
            {"header": False, "footer": False},
            {},
        ):
            try:
                chunks = pymupdf4llm.to_markdown(doc, page_chunks=True, **opts)
                break
            except (TypeError, ValueError):
                continue
        if chunks is None:
            raise ExtractionError("pymupdf4llm n'a produit aucun résultat")
        for i, ch in enumerate(chunks):
            text = ch["text"] if isinstance(ch, dict) else str(ch)
            if text.strip():
                segments.append(Segment(text, i + 1))
    except Exception:  # noqa: BLE001 - repli sur l'extraction brute
        segments = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                segments.append(Segment(text, i + 1))
    finally:
        doc.close()

    out = Extracted(segments=segments, pages=n_pages)
    if n_pages and out.n_chars / n_pages < MIN_CHARS_PER_PAGE:
        out.warning = (
            "Très peu de texte extrait : le PDF est probablement un scan et l'OCR automatique n'a rien "
            "donné (Tesseract absent, ou scan illisible). Dans l'image Docker, Tesseract est inclus ; "
            "sinon passez le fichier à l'OCR (`ocrmypdf --language fra+eng in.pdf out.pdf`) puis réimportez-le."
        )
    return out


# ------------------------------------------------------------------ DOCX


def _docx(path: Path) -> Extracted:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(path))
    lines: list[str] = []
    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, d)
            text = para.text.strip()
            if not text:
                lines.append("")
                continue
            style = (para.style.name or "") if para.style is not None else ""
            m = re.match(r"(?i)(heading|titre)\s*(\d)", style)
            if m:
                lines.append(f"{'#' * min(int(m.group(2)), 6)} {text}")
            elif style.lower() == "title":
                lines.append(f"# {text}")
            else:
                lines.append(text)
            lines.append("")
        elif tag == "tbl":
            table = Table(child, d)
            for row in table.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                lines.append(" | ".join(cells))
            lines.append("")
    text = "\n".join(lines)
    return Extracted(segments=[Segment(text, None)] if text.strip() else [])


# ------------------------------------------------------------------ HTML / EPUB

_BLOCKS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre", "td"]


def _html_to_markdown(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()
    out: list[str] = []
    for el in soup.find_all(_BLOCKS):
        if el.find_parent(_BLOCKS):  # on ne garde que le bloc le plus externe
            continue
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
        if not text:
            continue
        if el.name[0] == "h" and el.name[1:].isdigit():
            out.append(f"{'#' * int(el.name[1])} {text}")
        else:
            out.append(text)
        out.append("")
    return "\n".join(out)


def _html_file(path: Path) -> Extracted:
    text = _html_to_markdown(_read_text(path))
    return Extracted(segments=[Segment(text, None)] if text.strip() else [])


def _epub(path: Path) -> Extracted:
    import ebooklib
    from ebooklib import epub

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        book = epub.read_epub(str(path), options={"ignore_ncx": True})
    segments: list[Segment] = []
    seen: set[str] = set()
    items = []
    for idref, _linear in book.spine:
        item = book.get_item_with_id(idref)
        if item is not None and item.get_type() == ebooklib.ITEM_DOCUMENT:
            items.append(item)
            seen.add(item.get_id())
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):  # documents hors spine
        if item.get_id() not in seen:
            items.append(item)
    for item in items:
        html = item.get_content().decode("utf-8", errors="replace")
        text = _html_to_markdown(html)
        if text.strip():
            segments.append(Segment(text, None))
    return Extracted(segments=segments)


# ------------------------------------------------------------------ TXT / MD


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _text(path: Path) -> Extracted:
    text = _read_text(path)
    return Extracted(segments=[Segment(text, None)] if text.strip() else [])
