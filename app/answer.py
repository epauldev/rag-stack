"""Construction du prompt, reformulation de la question, génération en streaming, affichage des sources."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

from . import llm
from .config import Settings, load_settings
from .retrieve import Passage, retrieve


@dataclass
class Prepared:
    question: str
    standalone: str
    passages: list[Passage]
    messages: list[dict]
    settings: Settings
    retrieval_seconds: float


def _condense(question: str, history: list[dict], s: Settings) -> str:
    """Reformule une question de suivi en question autonome (pour une recherche pertinente)."""
    turns = [m for m in history if m.get("role") in {"user", "assistant"} and m.get("content")][-6:]
    if not turns:
        return question
    transcript = "\n".join(f"{m['role']}: {str(m['content'])[:600]}" for m in turns)
    out = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Reformule la dernière question de l'utilisateur en une question autonome, compréhensible "
                    "sans l'historique, en gardant sa langue. Ne réponds pas à la question. "
                    "Renvoie uniquement la question reformulée."
                ),
            },
            {"role": "user", "content": f"Historique :\n{transcript}\n\nDernière question : {question}"},
        ],
        s,
        temperature=0.0,
    )
    return out.strip().strip('"') or question


def build_context(passages: list[Passage], max_chars: int) -> str:
    blocks: list[str] = []
    used = 0
    for i, p in enumerate(passages, start=1):
        text = p.text
        room = max_chars - used
        if room <= 200 and blocks:
            break
        if len(text) > room:
            text = text[:room].rsplit(" ", 1)[0] + " […]"
        blocks.append(f"[{i}] {p.label()}\n{text}")
        used += len(text)
    return "\n\n".join(blocks)


def prepare(
    question: str,
    history: list[dict] | None = None,
    collections: list[str] | None = None,
    top_k: int | None = None,
) -> Prepared:
    s = load_settings()
    history = history or []
    t0 = time.time()
    standalone = question
    if s.condense_history and history:
        try:
            standalone = _condense(question, history, s)
        except llm.LLMError:
            standalone = question
    passages = retrieve(standalone, collections, top_k, s)
    elapsed = time.time() - t0

    if passages:
        context = build_context(passages, s.max_context_chars)
        user = f"Extraits :\n\n{context}\n\nQuestion : {question}"
    else:
        user = (
            "Aucun extrait n'a été trouvé dans la bibliothèque pour cette question.\n\n"
            f"Question : {question}"
        )
    past = [
        {"role": m["role"], "content": str(m["content"])[:1500]}
        for m in history
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ][-6:]
    messages = [{"role": "system", "content": s.system_prompt}, *past, {"role": "user", "content": user}]
    return Prepared(question, standalone, passages, messages, s, elapsed)


def stream_answer(prep: Prepared) -> Iterator[str]:
    if not prep.passages:
        yield (
            "Je n'ai trouvé aucun passage dans la bibliothèque pour répondre. "
            "Vérifiez que des documents sont indexés (onglet Bibliothèque) et que les espaces cochés "
            "contiennent le sujet."
        )
        return
    yield from llm.chat_stream(prep.messages, prep.settings)


def format_sources(passages: list[Passage], standalone: str | None = None, question: str | None = None,
                   seconds: float | None = None) -> str:
    if not passages:
        return "_Aucune source._"
    lines = []
    if standalone and question and standalone != question:
        lines.append(f"_Recherche effectuée avec : « {standalone} »_\n")
    for i, p in enumerate(passages, start=1):
        excerpt = " ".join(p.text.split())
        if len(excerpt) > 700:
            excerpt = excerpt[:700].rsplit(" ", 1)[0] + " …"
        lines.append(
            f"**[{i}] {p.label()}**  \n"
            f"<sub>{p.collection} · `{p.filename}`</sub>\n\n"
            f"<details><summary>Voir l'extrait</summary>\n\n{excerpt}\n\n</details>"
        )
    if seconds is not None:
        lines.append(f"<sub>Recherche : {seconds:.1f} s</sub>")
    return "\n\n".join(lines)
