"""Long-term memory for the learner-companion relationship (SQLite).

Stores learners, one conversation per (learner, companion) pair, the exact
message history, durable facts the companion has learned, and what content each
learner has consumed. Mirrors the production `users`, `chat_sessions`,
`chat_messages`, `reporter_follows` and `reading_history` tables closely enough to
swap in MySQL later.

Assistant turns are stored as the full content-block JSON returned by the API
(thinking and tool-use blocks included) and replayed unchanged, so history is
append-only across sessions.
"""

import json
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS learners (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id TEXT NOT NULL REFERENCES learners(id),
    companion_slug TEXT NOT NULL,
    origin_content_slug TEXT,
    created_at REAL NOT NULL,
    last_active_at REAL NOT NULL,
    UNIQUE (learner_id, companion_slug)
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS learner_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id TEXT NOT NULL REFERENCES learners(id),
    kind TEXT NOT NULL,
    fact TEXT NOT NULL,
    companion_slug TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS consumed_content (
    learner_id TEXT NOT NULL REFERENCES learners(id),
    content_slug TEXT NOT NULL,
    consumed_at REAL NOT NULL,
    PRIMARY KEY (learner_id, content_slug)
);
"""


class MemoryStore:
    def __init__(self, path: str = "companion_memory.db"):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # Learners -----------------------------------------------------------------

    def ensure_learner(self, learner_id: str, display_name: str | None = None) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO learners (id, display_name, created_at) VALUES (?, ?, ?)",
            (learner_id, display_name or learner_id, time.time()),
        )
        self.db.commit()

    def learner_name(self, learner_id: str) -> str:
        row = self.db.execute("SELECT display_name FROM learners WHERE id = ?", (learner_id,)).fetchone()
        return row["display_name"] if row else learner_id

    # Conversations ------------------------------------------------------------

    def get_or_create_conversation(
        self, learner_id: str, companion_slug: str, origin_content_slug: str | None = None
    ) -> tuple[int, bool]:
        """Return (conversation_id, created)."""
        row = self.db.execute(
            "SELECT id FROM conversations WHERE learner_id = ? AND companion_slug = ?",
            (learner_id, companion_slug),
        ).fetchone()
        if row:
            return row["id"], False
        now = time.time()
        cur = self.db.execute(
            "INSERT INTO conversations (learner_id, companion_slug, origin_content_slug, created_at, last_active_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (learner_id, companion_slug, origin_content_slug, now, now),
        )
        self.db.commit()
        return cur.lastrowid, True

    def append_message(self, conversation_id: int, role: str, content) -> None:
        now = time.time()
        self.db.execute(
            "INSERT INTO messages (conversation_id, role, content_json, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, json.dumps(content), now),
        )
        self.db.execute("UPDATE conversations SET last_active_at = ? WHERE id = ?", (now, conversation_id))
        self.db.commit()

    def history(self, conversation_id: int) -> list[dict]:
        rows = self.db.execute(
            "SELECT role, content_json FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return [{"role": r["role"], "content": json.loads(r["content_json"])} for r in rows]

    def idle_conversations(self, idle_seconds: float) -> list[sqlite3.Row]:
        """Conversations quiet for at least `idle_seconds` - candidates for a follow-up nudge."""
        return self.db.execute(
            "SELECT * FROM conversations WHERE last_active_at <= ? ORDER BY last_active_at",
            (time.time() - idle_seconds,),
        ).fetchall()

    # Learner facts and progress ----------------------------------------------

    def add_fact(self, learner_id: str, kind: str, fact: str, companion_slug: str) -> None:
        self.db.execute(
            "INSERT INTO learner_facts (learner_id, kind, fact, companion_slug, created_at) VALUES (?, ?, ?, ?, ?)",
            (learner_id, kind, fact, companion_slug, time.time()),
        )
        self.db.commit()

    def facts(self, learner_id: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT kind, fact, companion_slug FROM learner_facts WHERE learner_id = ? ORDER BY id",
            (learner_id,),
        ).fetchall()

    def mark_consumed(self, learner_id: str, content_slug: str) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO consumed_content (learner_id, content_slug, consumed_at) VALUES (?, ?, ?)",
            (learner_id, content_slug, time.time()),
        )
        self.db.commit()

    def consumed(self, learner_id: str) -> set[str]:
        rows = self.db.execute(
            "SELECT content_slug FROM consumed_content WHERE learner_id = ?", (learner_id,)
        ).fetchall()
        return {r["content_slug"] for r in rows}
