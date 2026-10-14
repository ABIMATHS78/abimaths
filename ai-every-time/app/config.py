import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    database_path: str = field(default_factory=lambda: os.environ.get("AIET_DATABASE", "ai_every_time.db"))
    model: str = field(default_factory=lambda: os.environ.get("AIET_MODEL", "claude-opus-5-5"))
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
    # Public address of the site, used for links in emails.
    public_url: str = field(default_factory=lambda: os.environ.get("AIET_PUBLIC_URL", "http://127.0.0.1:8000").rstrip("/"))
    # Default hour for the daily reminder (email and notifications) in each learner's own time zone,
    # or UTC until it's known. Learners can choose another hour. AIET_DIGEST_HOUR_UTC is the old name.
    reminder_hour: int = field(default_factory=lambda: _int("AIET_REMINDER_HOUR", _int("AIET_DIGEST_HOUR_UTC", 7)))
    # How push services can reach whoever runs this server: a mailto: address or an https:// address.
    # Defaults to the AIET_MAIL_FROM address, then an https AIET_PUBLIC_URL. Notifications are off without one.
    push_contact: str = field(default_factory=lambda: os.environ.get("AIET_PUSH_CONTACT", ""))


settings = Settings()
