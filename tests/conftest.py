"""Environnement de test : données dans un dossier temporaire, Qdrant en mémoire, LM Studio simulé."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

_TMP = Path(tempfile.mkdtemp(prefix="ragtest_"))
os.environ["DATA_DIR"] = str(_TMP / "data")
os.environ["INBOX_DIR"] = str(_TMP / "inbox")
os.environ["QDRANT_URL"] = ""
os.environ["QDRANT_PATH"] = ""
os.environ["CHILD_CHARS"] = "600"      # petits chunks : nos documents de test sont courts
os.environ["CHILD_OVERLAP"] = "80"
os.environ["PARENT_CHARS"] = "1800"

import pytest  # noqa: E402

import mock_lmstudio  # noqa: E402


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP


@pytest.fixture(scope="session", autouse=True)
def lmstudio():
    from app.config import Settings, save_settings

    server, url = mock_lmstudio.start()
    save_settings(Settings(lmstudio_base_url=url, llm_model="mock-chat-model", embed_model="mock-embed-bge-m3",
                           condense_history=True))
    yield url
    server.shutdown()
