"""Qdrant : stockage des vecteurs (le texte reste dans SQLite)."""
from __future__ import annotations

import atexit
import threading

from qdrant_client import QdrantClient, models

from .config import cfg


class EmbeddingMismatch(RuntimeError):
    """Les vecteurs de l'index n'ont pas la dimension du modèle d'embedding courant."""


_client: QdrantClient | None = None
_lock = threading.Lock()


def client() -> QdrantClient:
    global _client
    with _lock:
        if _client is None:
            if cfg.qdrant_url:
                _client = QdrantClient(
                    url=cfg.qdrant_url,
                    api_key=cfg.qdrant_api_key,
                    timeout=60,
                    check_compatibility=False,
                )
            elif cfg.qdrant_path:
                _client = QdrantClient(path=cfg.qdrant_path)
            else:
                _client = QdrantClient(":memory:")
        return _client


def close() -> None:
    """Ferme proprement le client (évite un message parasite à la sortie en mode fichier local)."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:  # noqa: BLE001
            pass
        _client = None


atexit.register(close)


def reset_client_for_tests() -> None:
    close()


def _name() -> str:
    return cfg.qdrant_collection


def ping() -> tuple[bool, str]:
    try:
        client().get_collections()
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def collection_dim() -> int | None:
    """Dimension des vecteurs de l'index, ou None si la collection n'existe pas encore."""
    c = client()
    if not c.collection_exists(_name()):
        return None
    params = c.get_collection(_name()).config.params.vectors
    return getattr(params, "size", None)


def ensure(dim: int) -> None:
    c = client()
    if not c.collection_exists(_name()):
        c.create_collection(
            _name(),
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        if cfg.qdrant_url:  # les index de payload n'existent que sur un vrai serveur Qdrant
            for field, schema in (
                ("collection", models.PayloadSchemaType.KEYWORD),
                ("doc_id", models.PayloadSchemaType.INTEGER),
            ):
                try:
                    c.create_payload_index(_name(), field_name=field, field_schema=schema)
                except Exception:  # noqa: BLE001 - l'index n'est qu'une optimisation
                    pass
        return
    existing = collection_dim()
    if existing is not None and existing != dim:
        raise EmbeddingMismatch(
            f"L'index contient des vecteurs de dimension {existing} mais le modèle d'embedding "
            f"actuel en produit {dim}. Utilisez « Réindexer tout » dans la Bibliothèque."
        )


def reset() -> None:
    c = client()
    if c.collection_exists(_name()):
        c.delete_collection(_name())


def upsert(ids: list[int], vectors: list[list[float]], payloads: list[dict], batch: int = 256) -> None:
    c = client()
    for i in range(0, len(ids), batch):
        c.upsert(
            _name(),
            points=models.Batch(
                ids=ids[i : i + batch],
                vectors=vectors[i : i + batch],
                payloads=payloads[i : i + batch],
            ),
            wait=True,
        )


def _doc_filter(doc_id: int) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])


def delete_doc(doc_id: int) -> None:
    c = client()
    if c.collection_exists(_name()):
        c.delete(_name(), points_selector=models.FilterSelector(filter=_doc_filter(doc_id)), wait=True)


def set_doc_collection(doc_id: int, collection: str) -> None:
    c = client()
    if c.collection_exists(_name()):
        c.set_payload(_name(), payload={"collection": collection}, points=_doc_filter(doc_id), wait=True)


def search(vector: list[float], limit: int, collections: list[str] | None = None) -> list[tuple[int, float]]:
    """Retourne [(chunk_id, score_cosinus)] triés par pertinence décroissante."""
    c = client()
    if not c.collection_exists(_name()):
        return []
    flt = None
    if collections:
        flt = models.Filter(
            must=[models.FieldCondition(key="collection", match=models.MatchAny(any=list(collections)))]
        )
    res = c.query_points(_name(), query=vector, limit=limit, query_filter=flt, with_payload=False)
    return [(int(p.id), float(p.score)) for p in res.points]


def count() -> int:
    c = client()
    if not c.collection_exists(_name()):
        return 0
    return c.count(_name(), exact=True).count
