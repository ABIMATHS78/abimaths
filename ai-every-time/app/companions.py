"""Companion lifecycle: seeding starters, designing, remixing, editing and following."""

from __future__ import annotations

import json
import re
import secrets

from fastapi import HTTPException

from app import db
from app.persona import Persona
from app.starter_companions import STARTER_COMPANIONS

MAX_VERSIONS = 50


def seed_starters() -> None:
    """Insert starter companions that don't exist yet. Existing ones are left untouched."""
    with db.connect() as conn:
        for slug, persona in STARTER_COMPANIONS.items():
            exists = conn.execute("SELECT 1 FROM companions WHERE slug = ?", (slug,)).fetchone()
            if not exists:
                ts = db.now()
                conn.execute(
                    "INSERT INTO companions (slug, owner_id, is_starter, persona, created_at, updated_at) "
                    "VALUES (?, NULL, 1, ?, ?, ?)",
                    (slug, persona.model_dump_json(), ts, ts),
                )


def _slugify(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "companion"
    return f"{base[:30]}-{secrets.token_hex(3)}"


def get(companion_id: int) -> dict:
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM companions WHERE id = ?", (companion_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Companion not found")
    return db.row_to_companion(row)


def visible_to(companion: dict, user_id: int) -> bool:
    """Starter companions are public; learner-designed ones are private to their owner."""
    return companion["is_starter"] or companion["owner_id"] == user_id


def get_visible(companion_id: int, user_id: int) -> dict:
    c = get(companion_id)
    if not visible_to(c, user_id):
        raise HTTPException(404, "Companion not found")
    return c


def get_owned(companion_id: int, user_id: int) -> dict:
    c = get(companion_id)
    if c["owner_id"] != user_id:
        # Starter companions belong to the platform; learners remix them instead.
        raise HTTPException(403, "You can only edit companions you designed. Remix this one to make it yours.")
    return c


def list_for(user_id: int) -> list[dict]:
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT c.*, (f.user_id IS NOT NULL) AS following FROM companions c "
            "LEFT JOIN follows f ON f.companion_id = c.id AND f.user_id = ? "
            "WHERE c.is_starter = 1 OR c.owner_id = ? ORDER BY c.is_starter ASC, c.id ASC",
            (user_id, user_id),
        ).fetchall()
    return [db.row_to_companion(r) | {"following": bool(r["following"])} for r in rows]


def create(user_id: int, persona: Persona, remixed_from: int | None = None) -> dict:
    ts = db.now()
    with db.connect() as conn:
        cur = conn.execute(
            "INSERT INTO companions (slug, owner_id, is_starter, persona, remixed_from, created_at, updated_at) "
            "VALUES (?, ?, 0, ?, ?, ?, ?)",
            (_slugify(persona.name), user_id, persona.model_dump_json(), remixed_from, ts, ts),
        )
        cid = cur.lastrowid
        conn.execute("INSERT INTO companion_versions (companion_id, persona, created_at) VALUES (?, ?, ?)",
                     (cid, persona.model_dump_json(), ts))
        conn.execute("INSERT OR IGNORE INTO follows (user_id, companion_id, created_at) VALUES (?, ?, ?)",
                     (user_id, cid, ts))
    return get(cid)


def remix(user_id: int, source_id: int, overrides: dict | None = None) -> dict:
    source = get_visible(source_id, user_id)
    data = source["persona"] | (overrides or {})
    if not overrides or "name" not in overrides:
        data["name"] = f"My {source['persona']['name']}"[:40]
    return create(user_id, Persona.model_validate(data), remixed_from=source_id)


def update(user_id: int, companion_id: int, persona: Persona) -> dict:
    get_owned(companion_id, user_id)
    ts = db.now()
    with db.connect() as conn:
        conn.execute("UPDATE companions SET persona = ?, updated_at = ? WHERE id = ?",
                     (persona.model_dump_json(), ts, companion_id))
        conn.execute("INSERT INTO companion_versions (companion_id, persona, created_at) VALUES (?, ?, ?)",
                     (companion_id, persona.model_dump_json(), ts))
        conn.execute(
            "DELETE FROM companion_versions WHERE companion_id = ? AND id NOT IN "
            "(SELECT id FROM companion_versions WHERE companion_id = ? ORDER BY id DESC LIMIT ?)",
            (companion_id, companion_id, MAX_VERSIONS),
        )
    return get(companion_id)


def versions(user_id: int, companion_id: int) -> list[dict]:
    get_owned(companion_id, user_id)
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT id, persona, created_at FROM companion_versions WHERE companion_id = ? ORDER BY id DESC",
            (companion_id,),
        ).fetchall()
    return [{"id": r["id"], "created_at": r["created_at"], "persona": json.loads(r["persona"])} for r in rows]


def restore(user_id: int, companion_id: int, version_id: int) -> dict:
    get_owned(companion_id, user_id)
    with db.connect() as conn:
        row = conn.execute("SELECT persona FROM companion_versions WHERE id = ? AND companion_id = ?",
                           (version_id, companion_id)).fetchone()
    if row is None:
        raise HTTPException(404, "Version not found")
    return update(user_id, companion_id, Persona.model_validate_json(row["persona"]))


def delete(user_id: int, companion_id: int) -> None:
    get_owned(companion_id, user_id)
    with db.connect() as conn:
        conn.execute("DELETE FROM companions WHERE id = ?", (companion_id,))


def set_follow(user_id: int, companion_id: int, follow: bool) -> None:
    get_visible(companion_id, user_id)
    with db.connect() as conn:
        if follow:
            conn.execute("INSERT OR IGNORE INTO follows (user_id, companion_id, created_at) VALUES (?, ?, ?)",
                         (user_id, companion_id, db.now()))
        else:
            conn.execute("DELETE FROM follows WHERE user_id = ? AND companion_id = ?", (user_id, companion_id))


def set_primary(user_id: int, companion_id: int) -> None:
    get_visible(companion_id, user_id)
    set_follow(user_id, companion_id, True)
    with db.connect() as conn:
        conn.execute("UPDATE users SET primary_companion_id = ?, onboarded = 1 WHERE id = ?", (companion_id, user_id))
