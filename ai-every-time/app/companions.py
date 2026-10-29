"""Companion lifecycle: seeding starters, designing, remixing, editing and following."""

from __future__ import annotations

import json
import logging
import re
import secrets

from fastapi import HTTPException

from app import db, paths
from app.persona import Persona
from app.starter_companions import STARTERS

log = logging.getLogger(__name__)
MAX_VERSIONS = 50
# A published companion is hidden for review once this many learners report it.
REPORTS_TO_HIDE = 3
STARTER_FIELDS = ("persona", "base_followers", "region", "photo_url", "real_person", "featured", "retired")
# What a companion's lessons don't depend on: its face, its read-aloud voice and its designer's private
# instructions (lessons are shared by everyone it teaches).
NOT_TAUGHT = {"avatar", "voice", "flag", "custom_instructions"}


def _teaching(persona: dict | Persona) -> dict | None:
    """The parts of a persona its lessons are written from (None for one saved in a shape that's no longer valid)."""
    try:
        if not isinstance(persona, Persona):
            persona = Persona.model_validate(persona)
    except ValueError:
        return None
    return persona.model_dump(exclude=NOT_TAUGHT)


def _forget_lessons(conn, companion_id: int) -> None:
    """Its lessons were written in its old voice: each is written again, in the new one, when next opened."""
    conn.execute("DELETE FROM lesson_content WHERE companion_id = ?", (companion_id,))


def seed_starters() -> None:
    """Make the starter companions match app/starter_companions.csv: add new ones, update the ones that changed,
    and retire the ones it no longer lists. A retired starter is kept, with its chats, memories and posts, for the
    learners who talked to it or have it as their companion; nobody else sees it, and it stops posting. Learners
    who followed it start following the featured starters. Listing it again in the CSV brings it back."""
    ts = db.now()
    with db.connect() as conn:
        existing = {r["slug"]: r for r in conn.execute(
            f"SELECT id, slug, {', '.join(STARTER_FIELDS)} FROM companions WHERE is_starter = 1")}
        for slug, s in STARTERS.items():
            values = (s.persona.model_dump_json(), s.followers, s.region, s.photo_url, int(s.real_person),
                      int(s.featured), 0)
            row = existing.pop(slug, None)
            if row is None:
                conn.execute(
                    f"INSERT INTO companions (slug, owner_id, is_starter, {', '.join(STARTER_FIELDS)}, created_at, updated_at) "
                    "VALUES (?, NULL, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (slug, *values, ts, ts))
            elif tuple(row[k] for k in STARTER_FIELDS) != values:
                if row["photo_url"] != s.photo_url:  # fetched again from the new address
                    conn.execute("DELETE FROM companion_photos WHERE companion_id = ?", (row["id"],))
                if _teaching(json.loads(row["persona"])) != _teaching(s.persona):
                    _forget_lessons(conn, row["id"])
                conn.execute(f"UPDATE companions SET {', '.join(f'{k} = ?' for k in STARTER_FIELDS)}, updated_at = ? "
                             "WHERE id = ?", (*values, ts, row["id"]))
        for slug, row in existing.items():
            if row["retired"]:
                continue
            log.info("Retiring the starter companion %s: it's no longer in starter_companions.csv. Learners who "
                     "talked to it keep it and their chats; nobody else sees it.", slug)
            conn.execute("INSERT OR IGNORE INTO follows (user_id, companion_id, created_at) "
                         "SELECT f.user_id, c.id, ? FROM follows f JOIN companions c ON c.is_starter = 1 AND c.featured = 1 "
                         "WHERE f.companion_id = ?", (ts, row["id"]))
            # Following it ends for learners who never talked to it (everyone followed the starters at sign-up).
            conn.execute("DELETE FROM follows WHERE companion_id = :id "
                         "AND user_id NOT IN (SELECT user_id FROM messages WHERE companion_id = :id) "
                         "AND user_id NOT IN (SELECT id FROM users WHERE primary_companion_id = :id)", {"id": row["id"]})
            conn.execute("UPDATE companions SET retired = 1, featured = 0, updated_at = ? WHERE id = ?", (ts, row["id"]))


def followers(row) -> int | None:
    """The follower count learners see: for a starter companion, the count it started with plus everyone who
    follows it here. Companions learners design don't show one. `row` has the companion's columns and `follows`."""
    return row["base_followers"] + row["follows"] if row["is_starter"] else None


def _slugify(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "companion"
    return f"{base[:30]}-{secrets.token_hex(3)}"


def get(companion_id: int) -> dict:
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM companions WHERE id = ?", (companion_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Companion not found")
    return db.row_to_companion(row)


def starter(slug: str) -> dict | None:
    """A starter companion by its slug, or None if there's none (or it has been retired)."""
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM companions WHERE slug = ? AND is_starter = 1 AND retired = 0", (slug,)).fetchone()
    return db.row_to_companion(row) if row else None


def get_by_ref(ref: str) -> dict:
    """A companion by its slug, as its page's address has it, or by its id, as addresses before slugs did."""
    with db.connect() as conn:
        found = paths.as_id(ref)
        row = (conn.execute("SELECT * FROM companions WHERE id = ?", (found,)) if found is not None
               else conn.execute("SELECT * FROM companions WHERE slug = ?", (ref,))).fetchone()
    if row is None:
        raise HTTPException(404, "Companion not found")
    return db.row_to_companion(row)


def public(companion: dict) -> bool:
    """Whether anyone may see this companion, signed in or not: a starter still listed, or a learner's companion
    published to the community and not hidden after reports."""
    return visible_to(companion, None) and not companion["retired"]


def public_view(companion: dict) -> dict:
    """What a visitor who isn't signed in sees of a companion: not its designer's account, nor their custom
    instructions."""
    return companion | {"owner_id": None, "persona": companion["persona"] | {"custom_instructions": ""}}


def visible_to(companion: dict, user_id: int | None) -> bool:
    """Starter companions are public. Learner-designed ones are private to their owner
    unless published to the community (and not hidden after reports). get_visible also leaves out
    companions by designers the learner blocked."""
    if companion["is_starter"] or (user_id is not None and companion["owner_id"] == user_id):
        return True
    return companion["is_public"] and not companion["hidden"]


# SQL equivalent of visible_to, for a query aliasing companions as `c` with the user id bound to `:uid`. Community
# companions by a designer the learner blocked are left out too.
VISIBLE_SQL = ("(c.is_starter = 1 OR c.owner_id = :uid OR (c.is_public = 1 AND c.hidden = 0 "
               "AND c.owner_id NOT IN (SELECT blocked_id FROM blocks WHERE user_id = :uid)))")


def designer_blocked(companion: dict, user_id: int) -> bool:
    """Whether the learner blocked the designer of this community companion."""
    if companion["is_starter"] or companion["owner_id"] in (None, user_id):
        return False
    with db.connect() as conn:
        return conn.execute("SELECT 1 FROM blocks WHERE user_id = ? AND blocked_id = ?",
                            (user_id, companion["owner_id"])).fetchone() is not None


def for_viewer(companion: dict, user_id: int) -> dict:
    """What a learner may see. Custom instructions are the designer's own and stay private."""
    if companion["owner_id"] == user_id:
        return companion
    persona = companion["persona"] | {"custom_instructions": ""}
    return companion | {"persona": persona}


def get_visible(companion_id: int, user_id: int) -> dict:
    c = get(companion_id)
    if not visible_to(c, user_id) or designer_blocked(c, user_id):
        raise HTTPException(404, "Companion not found")
    return c


def get_owned(companion_id: int, user_id: int) -> dict:
    c = get(companion_id)
    if c["owner_id"] != user_id:
        # Starter companions belong to the platform; learners remix them instead.
        raise HTTPException(403, "You can only edit companions you designed. Remix this one to make it yours.")
    return c


def persona_for(companion: dict, user_id: int | None) -> Persona:
    """The persona used when this companion talks to a learner. Custom instructions apply only to the
    designer's own sessions; pass user_id=None for content shared across learners (e.g. cached lessons)."""
    persona = Persona.model_validate(companion["persona"])
    if user_id is None or companion["owner_id"] != user_id:
        persona = persona.model_copy(update={"custom_instructions": ""})
    return persona


def list_for(user_id: int) -> list[dict]:
    """Every companion the learner can see: their own in the order they made them, then other learners' most
    followed first, then the starters most followed first. Retired starters only for learners who still have them."""
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT c.*, (f.user_id IS NOT NULL) AS following, u.display_name AS designer, "
            "(SELECT COUNT(*) FROM follows f2 WHERE f2.companion_id = c.id) AS follows "
            "FROM companions c "
            "LEFT JOIN follows f ON f.companion_id = c.id AND f.user_id = :uid "
            "LEFT JOIN users u ON u.id = c.owner_id "
            f"WHERE {VISIBLE_SQL} AND (c.retired = 0 OR f.user_id IS NOT NULL "
            "OR c.id = (SELECT primary_companion_id FROM users WHERE id = :uid)) "
            "ORDER BY c.is_starter ASC, (c.owner_id = :uid) DESC, "
            "CASE WHEN c.is_starter = 1 THEN c.base_followers + follows WHEN c.owner_id = :uid THEN 0 ELSE follows END DESC, "
            "c.id ASC",
            {"uid": user_id},
        ).fetchall()
    return [
        for_viewer(db.row_to_companion(r), user_id)
        | {"following": bool(r["following"]), "designer": r["designer"], "followers": followers(r)}
        for r in rows
    ]


def set_published(user_id: int, companion_id: int, public: bool) -> dict:
    c = get_owned(companion_id, user_id)
    if public and c["removed"]:
        raise HTTPException(409, "This companion was removed from the community by a moderator and can't be republished.")
    if public and c["hidden"]:
        raise HTTPException(409, "This companion was hidden after reports from other learners and can't be republished.")
    with db.connect() as conn:
        conn.execute("UPDATE companions SET is_public = ?, published_at = ? WHERE id = ?",
                     (int(public), db.now() if public else None, companion_id))
    return get(companion_id)


def report(user_id: int, companion_id: int, reason: str) -> None:
    c = get_visible(companion_id, user_id)
    if c["is_starter"] or c["owner_id"] == user_id:
        raise HTTPException(400, "Only other learners' community companions can be reported.")
    with db.connect() as conn:
        conn.execute("INSERT OR IGNORE INTO reports (user_id, companion_id, reason, created_at) VALUES (?, ?, ?, ?)",
                     (user_id, companion_id, reason, db.now()))
        n = conn.execute("SELECT COUNT(*) FROM reports WHERE companion_id = ?", (companion_id,)).fetchone()[0]
        if n >= REPORTS_TO_HIDE:
            conn.execute("UPDATE companions SET hidden = 1 WHERE id = ?", (companion_id,))


def block_designer(user_id: int, companion_id: int) -> None:
    """The learner stops seeing the companions this one's designer makes, and their posts: they unfollow them, and
    one that was the learner's own companion no longer is. Apple asks apps with things learners make to let them
    block each other (App Review guideline 1.2)."""
    c = get_visible(companion_id, user_id)
    if c["is_starter"] or c["owner_id"] in (None, user_id):
        raise HTTPException(400, "Only the designers of other learners' community companions can be blocked.")
    designer = c["owner_id"]
    with db.connect() as conn:
        conn.execute("INSERT OR IGNORE INTO blocks (user_id, blocked_id, created_at) VALUES (?, ?, ?)",
                     (user_id, designer, db.now()))
        conn.execute("DELETE FROM follows WHERE user_id = ? AND companion_id IN (SELECT id FROM companions WHERE owner_id = ?)",
                     (user_id, designer))
        conn.execute("UPDATE users SET primary_companion_id = NULL WHERE id = ? AND primary_companion_id IN "
                     "(SELECT id FROM companions WHERE owner_id = ?)", (user_id, designer))


def blocked(user_id: int) -> list[dict]:
    """The designers the learner blocked, most recent first."""
    with db.connect() as conn:
        rows = conn.execute("SELECT b.blocked_id AS id, u.display_name AS name, b.created_at AS since FROM blocks b "
                            "JOIN users u ON u.id = b.blocked_id WHERE b.user_id = ? ORDER BY b.created_at DESC, b.rowid DESC",
                            (user_id,)).fetchall()
    return [dict(r) for r in rows]


def unblock(user_id: int, designer_id: int) -> None:
    with db.connect() as conn:
        conn.execute("DELETE FROM blocks WHERE user_id = ? AND blocked_id = ?", (user_id, designer_id))


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


def remix(user_id: int, source_id: int, overrides: dict | None = None, keep_photo: bool = True) -> dict:
    """A new companion of the learner's own, starting from another. A starter companion's photo comes along
    unless they'd rather draw a face; other learners' photos stay theirs."""
    source = get_visible(source_id, user_id)
    if source["real_person"]:
        raise HTTPException(403, "Companions modelled on real people can't be remixed. Design your own instead.")
    overrides = overrides or {}
    data = source["persona"] | overrides
    if source["owner_id"] != user_id:
        # Another designer's private instructions never carry over into a remix.
        data["custom_instructions"] = overrides.get("custom_instructions", "")
    if "name" not in overrides:
        data["name"] = f"My {source['persona']['name']}"[:40]
    c = create(user_id, Persona.model_validate(data), remixed_from=source_id)
    if keep_photo and source["is_starter"]:
        with db.connect() as conn:
            conn.execute("INSERT INTO companion_photos (companion_id, mime, data, updated_at) "
                         "SELECT ?, mime, data, ? FROM companion_photos WHERE companion_id = ?", (c["id"], db.now(), source_id))
    return c


def update(user_id: int, companion_id: int, persona: Persona) -> dict:
    before = get_owned(companion_id, user_id)
    ts = db.now()
    with db.connect() as conn:
        conn.execute("UPDATE companions SET persona = ?, updated_at = ? WHERE id = ?",
                     (persona.model_dump_json(), ts, companion_id))
        if _teaching(before["persona"]) != _teaching(persona):
            _forget_lessons(conn, companion_id)
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
