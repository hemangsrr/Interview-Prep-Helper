import os
from typing import Optional
from pymongo import MongoClient, ASCENDING
from math import sqrt

_client: Optional[MongoClient] = None
_db = None


def get_db():
    global _client, _db
    if _db is not None:
        return _db
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB", "interview_prep")
    _client = MongoClient(uri)
    _db = _client[db_name]
    # Ensure collections exist and basic indexes
    _db["interviews"].create_index([("sid", ASCENDING)], unique=True)
    _db["panels"].create_index([("sid", ASCENDING)], unique=True)
    return _db


def save_panel(sid: str, panel: list, jd_text: str):
    db = get_db()
    db.panels.update_one({"sid": sid}, {"$set": {"sid": sid, "panel": panel, "jd_text": jd_text}}, upsert=True)


def save_interview_state(sid: str, state: dict):
    db = get_db()
    db.interviews.update_one({"sid": sid}, {"$set": {"sid": sid, "state": state}}, upsert=True)


def get_interview_state(sid: str) -> Optional[dict]:
    db = get_db()
    doc = db.interviews.find_one({"sid": sid})
    return doc.get("state") if doc else None


def get_panel(sid: str) -> Optional[dict]:
    """Return panel doc {sid, panel, jd_text, embedding?} if present."""
    db = get_db()
    return db.panels.find_one({"sid": sid})


def save_jd_panel_with_embedding(sid: str, jd_text: str, panel: list, embedding: list[float]):
    """Persist JD text, panel, and embedding for later semantic reuse."""
    db = get_db()
    db.panels.update_one(
        {"sid": sid},
        {"$set": {"sid": sid, "panel": panel, "jd_text": jd_text, "embedding": embedding}},
        upsert=True,
    )


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = sqrt(sum(x*x for x in a))
    nb = sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def find_similar_jd_panel(embedding: list[float], threshold: float = 0.9) -> Optional[dict]:
    """
    Brute-force cosine similarity search over stored JD embeddings.
    Returns best matching panel doc if similarity >= threshold.
    """
    db = get_db()
    best_doc = None
    best_score = 0.0
    for doc in db.panels.find({}, {"sid": 1, "panel": 1, "jd_text": 1, "embedding": 1}):
        emb = doc.get("embedding")
        if not emb:
            continue
        score = _cosine(embedding, emb)
        if score > best_score:
            best_score = score
            best_doc = doc
    if best_doc and best_score >= threshold:
        best_doc["similarity"] = best_score
        return best_doc
    return None
