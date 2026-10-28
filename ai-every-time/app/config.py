import json
import logging
import math
import os
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _price(p) -> tuple[float, float, float] | None:
    """Three finite, non-negative numbers, or None."""
    if not (isinstance(p, list) and len(p) == 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in p)):
        return None
    try:
        prices = tuple(float(x) for x in p)
    except OverflowError:  # an integer too big for a float
        return None
    return prices if all(math.isfinite(x) and x >= 0 for x in prices) else None


def _prices() -> dict:
    """AIET_PRICES: {"model": [input, output, cache read], ...} in US dollars per million tokens. An entry
    that isn't three prices is skipped, with a warning, and the others still count."""
    try:
        raw = json.loads(os.environ.get("AIET_PRICES") or "{}")
    except ValueError:
        raw = None
    if not isinstance(raw, dict):
        log.warning("AIET_PRICES isn't a JSON object of prices, so the built-in prices are used")
        return {}
    out = {}
    for model, p in raw.items():
        if (prices := _price(p)) is None:
            log.warning("AIET_PRICES: skipping %r, which isn't [input, output, cache read] in dollars", model)
        else:
            out[str(model)] = prices
    return out


def _json_object(name: str) -> dict:
    """An environment variable holding a JSON object, or {} (with a warning) if it holds anything else."""
    try:
        raw = json.loads(os.environ.get(name) or "{}")
    except ValueError:
        raw = None
    if not isinstance(raw, dict):
        log.warning("%s isn't a JSON object, so it's ignored", name)
        return {}
    return raw


@dataclass
class Settings:
    database_path: str = field(default_factory=lambda: os.environ.get("AIET_DATABASE", "ai_every_time.db"))
    model: str = field(default_factory=lambda: os.environ.get("AIET_MODEL", "claude-opus-5-5"))
    # The Claude workspace requests run in, for an API key that works across workspaces (one made without choosing
    # a workspace). An ID like wrkspc_01..., on the Workspaces page of the Claude Console. Leave it unset otherwise.
    workspace_id: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip())
    # Without an API key the app runs in offline demo mode with templated replies.
    offline: bool = field(
        default_factory=lambda: os.environ.get("AIET_OFFLINE") == "1"
        or not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    )
    # Minutes between automatic content runs for starter companions; 0 disables.
    autogen_interval_minutes: int = field(default_factory=lambda: _int("AIET_AUTOGEN_MINUTES", 0))
    session_days: int = field(default_factory=lambda: _int("AIET_SESSION_DAYS", 30))
    secure_cookies: bool = field(default_factory=lambda: os.environ.get("AIET_SECURE_COOKIES") == "1")
    history_turns: int = field(default_factory=lambda: _int("AIET_HISTORY_TURNS", 20))
    # Comma-separated emails that are always admins (how the first admin gets in).
    admin_emails: frozenset = field(default_factory=lambda: frozenset(
        e.strip().lower() for e in os.environ.get("AIET_ADMIN_EMAILS", "").split(",") if e.strip()))
    # Email reminders: any SMTP server. Leave AIET_SMTP_HOST unset to turn email off.
    smtp_host: str = field(default_factory=lambda: os.environ.get("AIET_SMTP_HOST", ""))
    smtp_port: int = field(default_factory=lambda: _int("AIET_SMTP_PORT", 587))
    smtp_user: str = field(default_factory=lambda: os.environ.get("AIET_SMTP_USER", ""))
    smtp_password: str = field(default_factory=lambda: os.environ.get("AIET_SMTP_PASSWORD", ""))
    smtp_starttls: bool = field(default_factory=lambda: os.environ.get("AIET_SMTP_STARTTLS", "1") == "1")
    mail_from: str = field(default_factory=lambda: os.environ.get("AIET_MAIL_FROM", ""))
    # Public address of the site, used for links in emails and Stripe's return pages. On Render it's the service's
    # onrender.com address until AIET_PUBLIC_URL is set (left blank, it counts as not set).
    public_url: str = field(default_factory=lambda: (
        os.environ.get("AIET_PUBLIC_URL", "").strip() or os.environ.get("RENDER_EXTERNAL_URL", "").strip()
        or "http://127.0.0.1:8000").rstrip("/"))
    # Default hour for the daily reminder (email and notifications) in each learner's own time zone,
    # or UTC until it's known. Learners can choose another hour. AIET_DIGEST_HOUR_UTC is the old name.
    reminder_hour: int = field(default_factory=lambda: _int("AIET_REMINDER_HOUR", _int("AIET_DIGEST_HOUR_UTC", 7)))
    # How push services can reach whoever runs this server: a mailto: address or an https:// address.
    # Defaults to the AIET_MAIL_FROM address, then an https AIET_PUBLIC_URL. Notifications are off without one.
    push_contact: str = field(default_factory=lambda: os.environ.get("AIET_PUSH_CONTACT", ""))
    # Notifications for the iPhone app, through Apple's push service: the text of the APNs key (the AuthKey .p8
    # file, line breaks or "\n"), its key ID, the Apple team's ID, and the app's bundle ID. The app can't turn
    # notifications on until the first three are set.
    apns_key: str = field(default_factory=lambda: os.environ.get("AIET_APNS_KEY", ""))
    apns_key_id: str = field(default_factory=lambda: os.environ.get("AIET_APNS_KEY_ID", "").strip())
    apns_team_id: str = field(default_factory=lambda: os.environ.get("AIET_APNS_TEAM_ID", "").strip())
    apns_bundle_id: str = field(default_factory=lambda: os.environ.get("AIET_APNS_BUNDLE_ID", "").strip() or "org.aieverytime.app")
    # AI requests each learner can make in any 24 hours (a chat message counts 1; see app/usage.py). 0: no limit.
    ai_allowance: int = field(default_factory=lambda: _int("AIET_AI_ALLOWANCE", 100))
    # The allowance until a learner confirms their email address (only while email is set up).
    ai_allowance_unconfirmed: int = field(default_factory=lambda: _int("AIET_AI_ALLOWANCE_UNCONFIRMED", 30))
    # A ceiling on everyone's AI requests together in any 24 hours, against floods of new accounts. 0: none.
    ai_site_limit: int = field(default_factory=lambda: _int("AIET_AI_SITE_LIMIT", 0))
    # Prices for the cost estimate on the Insights page, for models not in app/insights.py or when they change.
    prices: dict = field(default_factory=_prices)
    # Fetch starter companions' photos from their addresses in app/starter_companions.csv and keep small copies
    # here (see app/photos.py). With 0, browsers load them from those addresses every time.
    fetch_photos: bool = field(default_factory=lambda: os.environ.get("AIET_FETCH_PHOTOS", "1") != "0")
    # The education guides loaded at start-up (see app/guides.py): a file made by scripts/import_guides.py, by
    # default app/guides.json.gz. 0 loads none.
    guides: str = field(default_factory=lambda: os.environ.get("AIET_GUIDES", "").strip())
    # Paying for plans, through Stripe (see app/billing.py). The secret key (sk_live_... or sk_test_..., or a restricted
    # rk_ key) and the signing secret of the webhook endpoint Stripe sends events to, /api/billing/webhook (whsec_...).
    stripe_secret_key: str = field(default_factory=lambda: os.environ.get("STRIPE_SECRET_KEY", "").strip())
    stripe_webhook_secret: str = field(default_factory=lambda: os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip())
    # The Stripe price for each plan and billing period, where it isn't the one with the lookup key aiet_<plan>_<period>
    # (which `python -m app.billing setup` makes): {"plus_monthly": "price_...", ...}
    stripe_prices: dict = field(default_factory=lambda: _json_object("AIET_STRIPE_PRICES"))
    # Whether plans decide what learners can use: 1 on, 0 off. Unset, it's on once Stripe is set up.
    paywall_mode: str = field(default_factory=lambda: os.environ.get("AIET_PAYWALL", "").strip())
    # Daily AI requests on each plan while the paywall is on, where not the ones in app/plans.py: {"free": 15, ...}
    plan_allowances: dict = field(default_factory=lambda: _json_object("AIET_PLAN_ALLOWANCES"))
    # Guides a learner on the free plan can read in full each month.
    free_guides: int = field(default_factory=lambda: _int("AIET_FREE_GUIDES", 5))
    trial_days: int = field(default_factory=lambda: _int("AIET_TRIAL_DAYS", 7))
    # Where the Teams plan's "Contact us" goes.
    teams_email: str = field(default_factory=lambda: os.environ.get("AIET_TEAMS_EMAIL", "teams@aieverytime.org").strip())

    @property
    def paywall(self) -> bool:
        return self.paywall_mode == "1" or (self.paywall_mode != "0" and bool(self.stripe_secret_key))


settings = Settings()
