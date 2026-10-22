"""SQLite storage. One short-lived connection per operation keeps it thread-safe
under FastAPI's threadpool; swap this module for Postgres when you scale out."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT 'beginner',
    goals TEXT NOT NULL DEFAULT '',
    primary_companion_id INTEGER REFERENCES companions(id) ON DELETE SET NULL,
    onboarded INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS companions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL UNIQUE,
    owner_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    is_starter INTEGER NOT NULL DEFAULT 0,
    persona TEXT NOT NULL,
    remixed_from INTEGER REFERENCES companions(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS companion_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    companion_id INTEGER NOT NULL REFERENCES companions(id) ON DELETE CASCADE,
    persona TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS follows (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    companion_id INTEGER NOT NULL REFERENCES companions(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, companion_id)
);
-- Follower counts are worked out for every companion a page lists.
CREATE INDEX IF NOT EXISTS idx_follows_companion ON follows(companion_id);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    companion_id INTEGER NOT NULL REFERENCES companions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    post_id INTEGER REFERENCES posts(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(user_id, companion_id, id);
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    companion_id INTEGER NOT NULL REFERENCES companions(id) ON DELETE CASCADE,
    fact TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    companion_id INTEGER NOT NULL REFERENCES companions(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    topic TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_posts_companion ON posts(companion_id, id);
CREATE TABLE IF NOT EXISTS engagements (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    post_id INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('view', 'like', 'share', 'helpful', 'ask')),
    created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, post_id, kind)
);

-- Learning: lessons are taught in each companion's voice and cached per companion.
CREATE TABLE IF NOT EXISTS lesson_content (
    companion_id INTEGER NOT NULL REFERENCES companions(id) ON DELETE CASCADE,
    lesson_id TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (companion_id, lesson_id)
);
CREATE TABLE IF NOT EXISTS lesson_progress (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    lesson_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('started', 'completed')),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, lesson_id)
);
-- Quizzes and mock exams. Correct answers stay server-side until submission.
CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    companion_id INTEGER REFERENCES companions(id) ON DELETE SET NULL,
    kind TEXT NOT NULL CHECK (kind IN ('quiz', 'exam')),
    title TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    questions TEXT NOT NULL,
    time_limit_minutes INTEGER,
    answers TEXT,
    score INTEGER,
    total INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    submitted_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_assessments_user ON assessments(user_id, id);
CREATE TABLE IF NOT EXISTS decks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    companion_id INTEGER REFERENCES companions(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deck_id INTEGER NOT NULL REFERENCES decks(id) ON DELETE CASCADE,
    front TEXT NOT NULL,
    back TEXT NOT NULL,
    box INTEGER NOT NULL DEFAULT 1,
    due_at TEXT NOT NULL,
    reviews INTEGER NOT NULL DEFAULT 0,
    lapses INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_cards_deck ON cards(deck_id);
CREATE TABLE IF NOT EXISTS study_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    companion_id INTEGER REFERENCES companions(id) ON DELETE SET NULL,
    weekly_minutes INTEGER NOT NULL,
    plan TEXT NOT NULL,
    created_at TEXT NOT NULL
);
-- Community: reports against published companions.
CREATE TABLE IF NOT EXISTS reports (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    companion_id INTEGER NOT NULL REFERENCES companions(id) ON DELETE CASCADE,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, companion_id)
);
CREATE TABLE IF NOT EXISTS companion_photos (
    companion_id INTEGER PRIMARY KEY REFERENCES companions(id) ON DELETE CASCADE,
    mime TEXT NOT NULL,
    data BLOB NOT NULL,
    updated_at TEXT NOT NULL
);
-- Learning paths a companion designed for one learner on a topic they chose.
CREATE TABLE IF NOT EXISTS custom_paths (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    companion_id INTEGER REFERENCES companions(id) ON DELETE SET NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL
);
-- Every moderation or review decision, for accountability.
CREATE TABLE IF NOT EXISTS moderation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    staff_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    target_type TEXT NOT NULL CHECK (target_type IN ('companion', 'path', 'user')),
    target_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
-- A designer asking for a second look at the removal of their companion: one appeal per removal,
-- decided by someone other than the moderator who removed it.
CREATE TABLE IF NOT EXISTS appeals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    companion_id INTEGER NOT NULL REFERENCES companions(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    removal_id INTEGER NOT NULL UNIQUE REFERENCES moderation_log(id),
    message TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'upheld', 'overturned')),
    decided_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    decision_note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    decided_at TEXT
);
-- Messages to a learner about decisions on their companions or paths, shown with their reminders.
CREATE TABLE IF NOT EXISTS notices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    link TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    read_at TEXT
);
CREATE TABLE IF NOT EXISTS activity (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    day TEXT NOT NULL,
    PRIMARY KEY (user_id, day)
);
-- Phones and browsers a learner turned notifications on for. The endpoint is the device's
-- address at its browser's push service; the keys encrypt messages for that device only.
CREATE TABLE IF NOT EXISTS push_subscriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    endpoint TEXT NOT NULL UNIQUE,
    p256dh TEXT NOT NULL,
    auth TEXT NOT NULL,
    device TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    last_sent_at TEXT,
    failures INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_push_user ON push_subscriptions(user_id);
-- Links for choosing a new password. Each works once, for a short time; only a hash of the token is kept.
CREATE TABLE IF NOT EXISTS password_resets (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used_at TEXT
);
-- Links that confirm an email address belongs to the learner, sent before any reminder email.
CREATE TABLE IF NOT EXISTS email_confirmations (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used_at TEXT
);
-- Emails sent because someone asked (confirmation and password-reset links), by a hash of the
-- address and kept a day, so their limits hold even when an account is deleted and made again.
CREATE TABLE IF NOT EXISTS email_sends (
    address_hash TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_email_sends ON email_sends(address_hash, created_at);
-- Recent wrong passwords per email, to slow down guessing.
CREATE TABLE IF NOT EXISTS login_failures (
    email TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_login_failures ON login_failures(email, created_at);
-- One row per AI request: what it was for, what it cost against the learner's allowance, and the
-- tokens Claude reported. Rows outlive a deleted account, without the account, as usage totals.
-- Work no learner asked for (kind 'automatic') has no user. A failed request costs nothing.
CREATE TABLE IF NOT EXISTS ai_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,
    units INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    model TEXT NOT NULL DEFAULT '',
    fallback INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ai_usage_user ON ai_usage(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_ai_usage_time ON ai_usage(created_at);
-- AI requests turned away by a learner's allowance or the site ceiling, counted per learner per hour.
-- Like ai_usage, the counts outlive a deleted account, without the account.
CREATE TABLE IF NOT EXISTS ai_refusals (
    hour TEXT NOT NULL,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    reason TEXT NOT NULL CHECK (reason IN ('allowance', 'site')),
    count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (hour, user_id, reason)
);
-- Courses (app/catalog.py, app/courses.py): who's enrolled in which, their lab and project work and the instructor's
-- review of it (each attempt kept), certificates, and ratings. Courses are named by slug; lab items are "lab-{m}".
CREATE TABLE IF NOT EXISTS course_enrollments (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    course_slug TEXT NOT NULL,
    enrolled_at TEXT NOT NULL,
    PRIMARY KEY (user_id, course_slug)
);
CREATE TABLE IF NOT EXISTS course_submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    course_slug TEXT NOT NULL,
    item TEXT NOT NULL CHECK (item = 'project' OR item LIKE 'lab-%'),
    companion_id INTEGER REFERENCES companions(id) ON DELETE SET NULL,
    content TEXT NOT NULL,
    passed INTEGER NOT NULL,
    score INTEGER,
    review TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_course_submissions ON course_submissions(user_id, course_slug, item);
-- A certificate for a course or a track, with the code its public page is found by and the name it was made out to.
CREATE TABLE IF NOT EXISTS certificates (
    code TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('course', 'track')),
    ref TEXT NOT NULL,
    learner_name TEXT NOT NULL,
    issued_at TEXT NOT NULL,
    UNIQUE (user_id, kind, ref)
);
CREATE TABLE IF NOT EXISTS course_ratings (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    course_slug TEXT NOT NULL,
    stars INTEGER NOT NULL CHECK (stars BETWEEN 1 AND 5),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, course_slug)
);
-- Keys the server makes once and keeps, such as the VAPID key that push subscriptions are tied to.
CREATE TABLE IF NOT EXISTS server_keys (
    name TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at TEXT NOT NULL
);
-- Things the app remembers about itself, like which guides file it last loaded.
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Plans (app/billing.py): one row per learner. The plan they pay for through Stripe (or paid for on the old site), with
-- the state of its subscription; a plan given to them, by an admin or as a referral reward, until a date or for good;
-- and the plan they last chose on the pricing page without finishing paying for it. What they can use is the higher
-- of the paid plan (while it's active, on trial or being retried) and the given one. Card details are only ever at Stripe.
CREATE TABLE IF NOT EXISTS subscriptions (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    tier TEXT NOT NULL DEFAULT 'free' CHECK (tier IN ('free', 'digital', 'plus', 'pro', 'insider', 'teams')),
    billing_interval TEXT NOT NULL DEFAULT 'monthly' CHECK (billing_interval IN ('monthly', 'yearly')),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'trialing', 'past_due', 'canceled', 'incomplete', 'unpaid', 'paused')),
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT UNIQUE,
    stripe_price_id TEXT,
    seats INTEGER NOT NULL DEFAULT 1,
    amount_cents INTEGER,  -- what each billing period costs, when known
    current_period_start TEXT,
    current_period_end TEXT,
    trial_end TEXT,
    cancel_at_period_end INTEGER NOT NULL DEFAULT 0,
    canceled_at TEXT,
    trial_used INTEGER NOT NULL DEFAULT 0,
    grant_tier TEXT CHECK (grant_tier IN ('digital', 'plus', 'pro', 'insider', 'teams')),
    grant_until TEXT,
    grant_note TEXT NOT NULL DEFAULT '',
    intended_tier TEXT CHECK (intended_tier IN ('digital', 'plus', 'pro', 'insider', 'teams')),
    intended_interval TEXT CHECK (intended_interval IN ('monthly', 'yearly')),
    intended_at TEXT,
    legacy_id TEXT,  -- the subscription's id on the old site
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_subscriptions_customer ON subscriptions(stripe_customer_id);
-- Every change to the plan a learner can use: new (a first paid plan), upgrade, downgrade, cancel (back to free) and
-- reactivate (a paid plan again). Kept, without the learner, when an account is deleted, like the revenue.
CREATE TABLE IF NOT EXISTS subscription_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    from_tier TEXT NOT NULL,
    to_tier TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('new', 'upgrade', 'downgrade', 'cancel', 'reactivate')),
    source TEXT NOT NULL,  -- stripe, admin, referral or import
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_subscription_events_time ON subscription_events(created_at);
-- Money in (paid invoices) and out (refunds, as negative amounts), from Stripe, each once.
CREATE TABLE IF NOT EXISTS revenue_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    kind TEXT NOT NULL CHECK (kind IN ('payment', 'refund')),
    amount_cents INTEGER NOT NULL,
    currency TEXT NOT NULL,
    tier TEXT,
    billing_interval TEXT,
    stripe_ref TEXT NOT NULL UNIQUE,  -- the invoice or refund
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_revenue_events_time ON revenue_events(created_at);
-- Stripe events already handled: Stripe can send one more than once.
CREATE TABLE IF NOT EXISTS stripe_events (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    received_at TEXT NOT NULL
);
-- Subscriptions of deleted accounts still to be cancelled at Stripe, when it couldn't be reached at the time.
CREATE TABLE IF NOT EXISTS stripe_cancellations (
    stripe_subscription_id TEXT PRIMARY KEY,
    requested_at TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT ''
);
-- Addresses that have had a free trial, as hashes of the address as its mailbox sees it (app/billing.py), so a trial
-- isn't had again by deleting the account and making it again.
CREATE TABLE IF NOT EXISTS trials (
    address_hash TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
-- Referrals (app/referrals.py): who invited whom. An invite counts once the friend has confirmed their email address;
-- every three that count earn the inviter a free month of Plus.
CREATE TABLE IF NOT EXISTS referral_invites (
    invitee_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    referrer_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    qualified_at TEXT,
    subscribed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_referral_invites_referrer ON referral_invites(referrer_id);
CREATE TABLE IF NOT EXISTS referral_rewards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    claimed_at TEXT,
    applied_as TEXT CHECK (applied_as IN ('plan', 'credit'))  -- a month of Plus, or a month of Plus off their bill
);
CREATE INDEX IF NOT EXISTS idx_referral_rewards_user ON referral_rewards(user_id);
-- Emails about billing and from campaigns (app/campaigns.py), each sent at most once: `key` tells them apart.
CREATE TABLE IF NOT EXISTS email_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    sent_at TEXT NOT NULL,
    failed INTEGER NOT NULL DEFAULT 0,
    UNIQUE (user_id, kind, key)
);
CREATE INDEX IF NOT EXISTS idx_email_log_kind ON email_log(kind, sent_at);
-- Emails an admin wrote to send to everyone on a plan who agreed to news and offers.
CREATE TABLE IF NOT EXISTS campaigns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    audience TEXT NOT NULL,
    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    finished_at TEXT
);
-- The guides a learner on a plan without unlimited guides read in full, by month, for their free guides each month.
CREATE TABLE IF NOT EXISTS guide_reads (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    month TEXT NOT NULL,
    post_id INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, month, post_id)
);
-- Accounts from the old site that share an email address with another one, which became the account here (the one
-- with the best plan). Kept so nothing about them is lost and they can be looked into; see scripts/import_legacy.py.
CREATE TABLE IF NOT EXISTS legacy_accounts (
    legacy_id TEXT PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    login_method TEXT NOT NULL DEFAULT '',
    tier TEXT,
    status TEXT,
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    created_at TEXT,
    last_signed_in TEXT,
    imported_at TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(settings.database_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _confirm_existing(conn: sqlite3.Connection) -> None:
    """Addresses in use before confirmation existed that count as confirmed: learners who had reminder
    emails on keep getting them, and the admins named in AIET_ADMIN_EMAILS keep their role."""
    admins = sorted(settings.admin_emails)
    conn.execute("UPDATE users SET email_confirmed_at = created_at WHERE email_reminders = 1 "
                 f"OR lower(email) IN ({', '.join('?' * len(admins))})", admins)


# Columns added after the first release. CREATE TABLE IF NOT EXISTS won't add them to
# existing databases, so they are added here when missing. An optional fourth item fills
# the new column in (SQL, or a function given the connection), run once, right after the
# column is added.
MIGRATIONS = [
    ("companions", "is_public", "INTEGER NOT NULL DEFAULT 0"),
    ("companions", "hidden", "INTEGER NOT NULL DEFAULT 0"),
    ("companions", "published_at", "TEXT"),
    ("users", "feed_seen_post_id", "INTEGER NOT NULL DEFAULT 0"),
    ("users", "role", "TEXT NOT NULL DEFAULT 'learner'"),
    ("users", "email_reminders", "INTEGER NOT NULL DEFAULT 0"),
    ("users", "unsubscribe_token", "TEXT"),
    ("users", "last_digest_on", "TEXT"),
    ("companions", "removed", "INTEGER NOT NULL DEFAULT 0"),
    ("custom_paths", "status", "TEXT NOT NULL DEFAULT 'private'"),
    ("custom_paths", "review_note", "TEXT NOT NULL DEFAULT ''"),
    ("custom_paths", "reviewed_by", "INTEGER"),
    ("custom_paths", "submitted_at", "TEXT"),
    ("custom_paths", "reviewed_at", "TEXT"),
    ("users", "timezone", "TEXT NOT NULL DEFAULT ''"),
    ("users", "reminder_hour", "INTEGER"),
    ("users", "last_push_on", "TEXT"),
    ("notices", "pushed_at", "TEXT"),
    ("users", "email_confirmed_at", "TEXT", _confirm_existing),
    # A learner's own daily AI allowance, set by an admin: NULL for the site's, 0 for no limit.
    ("users", "ai_allowance", "INTEGER"),
    # Starter companions, from app/starter_companions.csv: the follower count each starts with (learners who
    # follow it add to it), its region, where its photo comes from, whether it's modelled on a real person,
    # and whether it's featured (the most followed in its region, which new learners start out following).
    ("companions", "base_followers", "INTEGER NOT NULL DEFAULT 0"),
    ("companions", "region", "TEXT NOT NULL DEFAULT ''"),
    ("companions", "photo_url", "TEXT NOT NULL DEFAULT ''"),
    ("companions", "real_person", "INTEGER NOT NULL DEFAULT 0"),
    ("companions", "featured", "INTEGER NOT NULL DEFAULT 0"),
    # A starter no longer in the CSV. It's kept, with its chats, for the learners who talked to it, but no one
    # else sees it, and it stops posting.
    ("companions", "retired", "INTEGER NOT NULL DEFAULT 0"),
    # An emoji reaction to a message from the other side of the chat: the companion's to the learner's
    # messages, the learner's to the companion's. NULL for none.
    ("messages", "reaction", "TEXT"),
    # Guides (app/guides.py) are posts with more to them: a summary, their kind (tutorial or cheat sheet), level,
    # reading time in minutes, the address of their graphic, and where they came from ("aieverytime:<id>"). Posts
    # companions write themselves have none of these.
    ("posts", "summary", "TEXT"),
    ("posts", "kind", "TEXT"),
    ("posts", "level", "TEXT"),
    ("posts", "minutes", "INTEGER"),
    ("posts", "graphic", "TEXT"),
    ("posts", "source", "TEXT"),
    # What the learner is into (hobbies, work, anything they love), in their words: companions start
    # conversations from it and draw examples from it.
    ("users", "interests", "TEXT NOT NULL DEFAULT ''"),
    # Accounts brought over from the old site (scripts/import_legacy.py): its id for them there, how they signed in
    # there (email, google, apple, microsoft), and when they last did, here or there.
    ("users", "legacy_id", "TEXT"),
    ("users", "login_method", "TEXT NOT NULL DEFAULT 'email'"),
    ("users", "last_signed_in", "TEXT"),
    # The code in the learner's invite link (app/referrals.py), made when they first ask for it.
    ("users", "referral_code", "TEXT"),
    # Whether the learner agreed to emails with news and offers (app/campaigns.py). Off until they say yes.
    ("users", "marketing_emails", "INTEGER NOT NULL DEFAULT 0"),
    # Where they came from when they signed up: the utm_source of the link, or "referral".
    ("users", "signup_source", "TEXT NOT NULL DEFAULT ''"),
    # Months of Plus earned with referrals run until here, apart from any plan an admin gave (app/referrals.py).
    ("subscriptions", "reward_until", "TEXT"),
]

# Indexes on columns added above, so made after them.
INDEXES = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_posts_source ON posts(source) WHERE source IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_posts_companion_created ON posts(companion_id, created_at);
CREATE INDEX IF NOT EXISTS idx_posts_kind_created ON posts(kind, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_legacy ON users(legacy_id) WHERE legacy_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_referral_code ON users(referral_code) WHERE referral_code IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_users_created ON users(created_at);
"""


def init_db() -> None:
    with connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
        for table, column, ddl, *backfill in MIGRATIONS:
            cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
                for fill in backfill:
                    fill(conn) if callable(fill) else conn.execute(fill)
        conn.executescript(INDEXES)


def _with_defaults(persona: dict) -> dict:
    """Fill in fields added after a companion was saved (e.g. voice) so every client sees the full shape."""
    from app.persona import Persona

    try:
        return Persona.model_validate(persona).model_dump()
    except ValueError:
        return persona


def row_to_companion(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "slug": row["slug"],
        "owner_id": row["owner_id"],
        "is_starter": bool(row["is_starter"]),
        "remixed_from": row["remixed_from"],
        "is_public": bool(row["is_public"]),
        "hidden": bool(row["hidden"]),
        "removed": bool(row["removed"]),
        "published_at": row["published_at"],
        "region": row["region"],
        "real_person": bool(row["real_person"]),
        "retired": bool(row["retired"]),
        "persona": _with_defaults(json.loads(row["persona"])),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
