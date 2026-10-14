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


settings = Settings()
