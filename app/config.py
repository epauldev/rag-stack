"""Configuration.

Deux niveaux :
- `cfg` : infrastructure et découpage, lus depuis les variables d'environnement au démarrage.
- `Settings` : réglages modifiables depuis l'interface (modèles, prompt, top-k...),
  persistés dans `<DATA_DIR>/settings.json`. Les valeurs initiales viennent de l'environnement.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on", "oui"}


class Config:
    """Paramètres d'infrastructure (environnement uniquement)."""

    def __init__(self) -> None:
        self.data_dir = Path(_env("DATA_DIR", "./data")).resolve()
        self.inbox_dir = Path(_env("INBOX_DIR", "./inbox")).resolve()
        self.db_path = self.data_dir / "rag.sqlite"
        self.library_dir = self.data_dir / "library"
        self.extracted_dir = self.data_dir / "extracted"
        self.settings_path = self.data_dir / "settings.json"

        # Qdrant : serveur (QDRANT_URL), sinon fichier local (QDRANT_PATH), sinon mémoire (tests)
        self.qdrant_url = _env("QDRANT_URL", "")
        self.qdrant_path = _env("QDRANT_PATH", "")
        self.qdrant_api_key = _env("QDRANT_API_KEY", "") or None
        self.qdrant_collection = _env("QDRANT_COLLECTION", "documents")

        # Découpage (en caractères ; ~4 caractères par token en moyenne)
        self.child_chars = int(_env("CHILD_CHARS", "1500"))
        self.child_overlap = int(_env("CHILD_OVERLAP", "200"))
        self.parent_chars = int(_env("PARENT_CHARS", "4000"))

        # Recherche
        self.dense_k = int(_env("DENSE_K", "30"))
        self.lexical_k = int(_env("LEXICAL_K", "30"))
        self.rerank_candidates = int(_env("RERANK_CANDIDATES", "30"))
        self.reranker_model = _env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

        # OCR (Tesseract) des PDF scannés ; les langues doivent être installées (tesseract-ocr-fra...)
        self.ocr_language = _env("OCR_LANGUAGE", "fra+eng")

        # Interface
        self.host = _env("APP_HOST", "127.0.0.1")  # l'image Docker fixe APP_HOST=0.0.0.0 (port publié en local)
        self.port = int(_env("APP_PORT", "7860"))
        self.app_user = _env("APP_USER", "")
        self.app_password = _env("APP_PASSWORD", "")

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.library_dir, self.extracted_dir, self.inbox_dir):
            d.mkdir(parents=True, exist_ok=True)


cfg = Config()

DEFAULT_SYSTEM_PROMPT = (
    "Tu es un assistant de recherche documentaire. Tu réponds exclusivement à partir des "
    "extraits numérotés fournis dans le message de l'utilisateur.\n"
    "Règles :\n"
    "1. Appuie chaque affirmation sur un extrait et cite-le entre crochets, par exemple [1] ou [2][3].\n"
    "2. Si les extraits ne contiennent pas la réponse, ou seulement une partie, dis-le explicitement. "
    "N'invente rien et n'utilise pas de connaissances extérieures.\n"
    "3. Réponds dans la langue de la question ; les extraits peuvent être en français ou en anglais.\n"
    "4. Si les extraits ou les auteurs se contredisent, signale-le.\n"
    "5. Sois précis et structuré, sans remplissage."
)


@dataclass
class Settings:
    lmstudio_base_url: str = "http://host.docker.internal:1234/v1"
    api_key: str = "lm-studio"
    llm_model: str = ""
    embed_model: str = ""
    embed_query_prefix: str = ""
    embed_doc_prefix: str = ""
    temperature: float = 0.2
    top_k: int = 6
    max_context_chars: int = 16000
    use_reranker: bool = False
    condense_history: bool = True
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


def _env_defaults() -> Settings:
    return Settings(
        lmstudio_base_url=_env("LMSTUDIO_BASE_URL", Settings.lmstudio_base_url),
        api_key=_env("LMSTUDIO_API_KEY", Settings.api_key),
        llm_model=_env("LLM_MODEL", ""),
        embed_model=_env("EMBED_MODEL", ""),
        embed_query_prefix=_env("EMBED_QUERY_PREFIX", ""),
        embed_doc_prefix=_env("EMBED_DOC_PREFIX", ""),
        temperature=float(_env("TEMPERATURE", "0.2")),
        top_k=int(_env("TOP_K", "6")),
        max_context_chars=int(_env("MAX_CONTEXT_CHARS", "16000")),
        use_reranker=_env_bool("USE_RERANKER", False),
        condense_history=_env_bool("CONDENSE_HISTORY", True),
    )


def load_settings() -> Settings:
    """Valeurs d'environnement, surchargées par settings.json s'il existe."""
    s = _env_defaults()
    if cfg.settings_path.exists():
        try:
            saved = json.loads(cfg.settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        known = {f.name for f in fields(Settings)}
        for key, value in saved.items():
            if key in known:
                setattr(s, key, value)
    return s


def save_settings(s: Settings) -> None:
    cfg.ensure_dirs()
    cfg.settings_path.write_text(
        json.dumps(asdict(s), ensure_ascii=False, indent=2), encoding="utf-8"
    )
