"""Client LM Studio (API compatible OpenAI) : embeddings, chat, diagnostic."""
from __future__ import annotations

import re
from typing import Iterable, Iterator

from openai import OpenAI

from .config import Settings, load_settings


class LLMError(RuntimeError):
    """Erreur de communication avec LM Studio, avec un message exploitable par l'utilisateur."""


def _client(s: Settings) -> OpenAI:
    # LM Studio peut charger un modèle à la volée : on laisse un délai généreux.
    return OpenAI(
        base_url=s.lmstudio_base_url,
        api_key=s.api_key or "lm-studio",
        timeout=600,
        max_retries=1,
    )


def _explain(exc: Exception, s: Settings) -> LLMError:
    msg = str(exc)
    if "onnect" in msg or "refused" in msg or "resolve" in msg:
        return LLMError(
            f"Impossible de joindre LM Studio sur {s.lmstudio_base_url}. Vérifiez que le serveur "
            "local est démarré (onglet Developer) et que l'URL est correcte. Depuis Docker, "
            "http://host.docker.internal:1234/v1 est attendu ; sous Linux, activez aussi "
            "« Serve on Local Network » dans LM Studio."
        )
    return LLMError(f"Erreur LM Studio : {msg}")


# ------------------------------------------------------------------ modèles


def list_models(s: Settings | None = None) -> list[str]:
    s = s or load_settings()
    try:
        return sorted(m.id for m in _client(s).models.list().data)
    except Exception as exc:  # noqa: BLE001
        raise _explain(exc, s) from exc


def looks_like_embedding_model(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ("embed", "bge", "e5", "gte", "minilm", "nomic", "arctic", "rerank"))


# ------------------------------------------------------------------ embeddings


def embed(texts: list[str], kind: str = "doc", s: Settings | None = None, batch: int = 16) -> list[list[float]]:
    """Vecteurs pour `texts`. kind = 'doc' ou 'query' (préfixes propres à certains modèles)."""
    s = s or load_settings()
    if not s.embed_model:
        raise LLMError(
            "Aucun modèle d'embedding choisi. Ouvrez l'onglet Réglages, testez la connexion "
            "puis sélectionnez le modèle d'embedding chargé dans LM Studio."
        )
    prefix = s.embed_query_prefix if kind == "query" else s.embed_doc_prefix
    client = _client(s)
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        part = [(prefix + t) if t.strip() else " " for t in texts[i : i + batch]]
        try:
            resp = client.embeddings.create(model=s.embed_model, input=part)
        except Exception as exc:  # noqa: BLE001
            raise _explain(exc, s) from exc
        data = sorted(resp.data, key=lambda d: d.index)
        if len(data) != len(part):
            raise LLMError("LM Studio a renvoyé un nombre de vecteurs inattendu.")
        out.extend(d.embedding for d in data)
    return out


# ------------------------------------------------------------------ chat

_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


def strip_think(chunks: Iterable[str]) -> Iterator[str]:
    """Retire les blocs <think>...</think> d'un flux (modèles à raisonnement type Qwen3)."""
    buf = ""
    in_think = False
    for piece in chunks:
        buf += piece
        while True:
            if in_think:
                end = buf.find(_THINK_CLOSE)
                if end == -1:
                    buf = buf[-len(_THINK_CLOSE) :]  # on garde une éventuelle balise partielle
                    break
                buf = buf[end + len(_THINK_CLOSE) :].lstrip("\n")
                in_think = False
            else:
                start = buf.find(_THINK_OPEN)
                if start == -1:
                    keep = len(_THINK_OPEN) - 1
                    if len(buf) > keep:
                        yield buf[:-keep]
                        buf = buf[-keep:]
                    break
                if start:
                    yield buf[:start]
                buf = buf[start + len(_THINK_OPEN) :]
                in_think = True
    if buf and not in_think:
        yield buf


def _require_llm(s: Settings) -> None:
    if not s.llm_model:
        raise LLMError(
            "Aucun modèle de langage choisi. Ouvrez l'onglet Réglages, testez la connexion "
            "puis sélectionnez le modèle de chat."
        )


def chat_stream(messages: list[dict], s: Settings | None = None) -> Iterator[str]:
    s = s or load_settings()
    _require_llm(s)

    def raw() -> Iterator[str]:
        try:
            stream = _client(s).chat.completions.create(
                model=s.llm_model, messages=messages, temperature=s.temperature, stream=True
            )
            for event in stream:
                if event.choices and event.choices[0].delta.content:
                    yield event.choices[0].delta.content
        except Exception as exc:  # noqa: BLE001
            raise _explain(exc, s) from exc

    yield from strip_think(raw())


def chat(messages: list[dict], s: Settings | None = None, temperature: float | None = None) -> str:
    s = s or load_settings()
    _require_llm(s)
    try:
        resp = _client(s).chat.completions.create(
            model=s.llm_model,
            messages=messages,
            temperature=s.temperature if temperature is None else temperature,
        )
    except Exception as exc:  # noqa: BLE001
        raise _explain(exc, s) from exc
    text = resp.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S).strip()
