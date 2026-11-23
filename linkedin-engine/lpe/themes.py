"""Colour themes. The first six reproduce palettes seen in the training graphics; 'aiet' and 'midnight'
use the AI Every Time brand colours so posts can match the author's own site."""

BASE = {
    "bg": "#F4F0E8", "bg-2": "#ECE5D8", "panel": "#FBF8F2", "panel-head": "#EADFCF", "panel-head-ink": "#1A1816",
    "panel-border": "#D8CDBC", "ink": "#1A1816", "ink-2": "#3D3833", "muted": "#77706A", "line": "#D6CBBB",
    "line-strong": "#1A1816", "accent": "#D9774F", "accent-2": "#B8573A", "accent-soft": "#F3D9CB",
    "accent-ink": "#FFFFFF", "good": "#3F9B53", "good-soft": "#D9EED7", "good-ink": "#FFFFFF", "bad": "#E0362C",
    "bad-soft": "#F8D7D3", "bad-ink": "#FFFFFF", "warn": "#EE9F36", "warn-soft": "#FBE6C4", "warn-ink": "#FFFFFF",
    "link": "#B8573A", "highlight": "#FFE45C", "code-bg": "#1E1C1A", "code-ink": "#F4F0E8", "chip-bg": "#E6E0D6",
    "chip-ink": "#2B2724", "shadow": "0 10px 30px rgba(60, 40, 20, .12)", "dark": "0",
    "danger-panel": "#1E1C1A", "danger-panel-ink": "#F4F0E8",
}

THEMES = {
    # Cream paper with terracotta: the most common look in the training set.
    "cream": {},
    # Plain white page: link lists, chunky panel cheat sheets, prompt mockups.
    "paper": {
        "bg": "#FFFFFF", "bg-2": "#F4F4F2", "panel": "#FFFFFF", "panel-head": "#EEE7DA", "panel-border": "#1A1A1A",
        "ink": "#111111", "ink-2": "#2E2E2E", "muted": "#6B6B6B", "line": "#DADADA", "line-strong": "#111111",
        "bad": "#E8322D", "good": "#38B44A", "good-soft": "#DDF3DF", "warn": "#F2B53A", "link": "#1F5FD1",
        "highlight": "#FFE94D", "chip-bg": "#EFEFEF", "chip-ink": "#222222", "code-bg": "#F7F7F7", "code-ink": "#111111",
        "shadow": "0 8px 26px rgba(0, 0, 0, .10)",
    },
    # Near-black: delete/keep splits, terminal link lists, decision trees.
    "ink": {
        "bg": "#171615", "bg-2": "#22201E", "panel": "#2A2825", "panel-head": "#3A3632", "panel-head-ink": "#F5F1EA",
        "panel-border": "#47423C", "ink": "#F5F1EA", "ink-2": "#DCD5CA", "muted": "#A39C93", "line": "#4A4540",
        "line-strong": "#F5F1EA", "accent": "#E07B55", "accent-2": "#F0956F", "accent-soft": "#4A2E22",
        "good": "#8BD450", "good-soft": "#2C3B22", "good-ink": "#111111", "bad": "#E5321F", "bad-soft": "#4A1E19",
        "warn": "#F2B53A", "link": "#F07A4A", "highlight": "#E07B55", "code-bg": "#0F0E0D", "code-ink": "#F5F1EA",
        "chip-bg": "#3A3632", "chip-ink": "#F5F1EA", "shadow": "0 10px 30px rgba(0, 0, 0, .45)", "dark": "1",
        "danger-panel": "#0B0A09",
    },
    # A phone notes app in dark mode.
    "notes": {
        "bg": "#1C1C1E", "bg-2": "#2C2C2E", "panel": "#2C2C2E", "ink": "#F2F2F7", "ink-2": "#D1D1D6",
        "muted": "#8E8E93", "line": "#3A3A3C", "line-strong": "#F2F2F7", "accent": "#F5C344", "accent-2": "#F5C344",
        "accent-ink": "#1C1C1E", "link": "#F5C344", "good": "#F5C344", "dark": "1",
    },
    # Whiteboard and marker, for hand-drawn mind maps.
    "whiteboard": {
        "bg": "#F8F7F3", "bg-2": "#EFEDE6", "panel": "#FFFFFF", "ink": "#2A2A2A", "ink-2": "#3A3A3A",
        "muted": "#5A5A5A", "line": "#3A3A3A", "line-strong": "#2A2A2A", "accent": "#D7262E", "accent-2": "#B51D24",
        "link": "#D7262E",
    },
    # Solid terracotta, for badges and title cards.
    "terracotta": {
        "bg": "#D97757", "bg-2": "#C9673F", "panel": "#F8E4D9", "ink": "#2A1D17", "ink-2": "#3E2C24",
        "muted": "#6E4E40", "accent": "#D97757", "accent-2": "#B8573A", "line": "#B8573A",
    },
    # AI Every Time brand: lavender paper, violet accent.
    "aiet": {
        "bg": "#F6F4FF", "bg-2": "#EEEAFE", "panel": "#FFFFFF", "panel-head": "#E7E1FF", "panel-border": "#D8D2F2",
        "ink": "#1C1A29", "ink-2": "#34314A", "muted": "#6A6780", "line": "#DCD7F0", "line-strong": "#1C1A29",
        "accent": "#5B4CDB", "accent-2": "#4334B8", "accent-soft": "#E2DDFF", "good": "#1F9D6B", "good-soft": "#DCF6EC",
        "bad": "#D6455D", "bad-soft": "#FBE0E5", "warn": "#E0A849", "warn-soft": "#FAEBCF", "link": "#5B4CDB",
        "highlight": "#FFE07A", "code-bg": "#1C1A29", "code-ink": "#ECEBF5", "chip-bg": "#ECE8FB", "chip-ink": "#2B2840",
        "shadow": "0 10px 30px rgba(40, 30, 110, .12)",
    },
    # AI Every Time brand, dark.
    "midnight": {
        "bg": "#12111A", "bg-2": "#1B1A26", "panel": "#1F1D2C", "panel-head": "#2A2740", "panel-head-ink": "#ECEBF5",
        "panel-border": "#34314A", "ink": "#ECEBF5", "ink-2": "#CFCDE2", "muted": "#A19EB8", "line": "#34314A",
        "line-strong": "#ECEBF5", "accent": "#8A7DFF", "accent-2": "#B3A9FF", "accent-soft": "#2A2560",
        "accent-ink": "#12111A", "good": "#5BD6A0", "good-soft": "#123329", "good-ink": "#0D1F18", "bad": "#FF5C72",
        "bad-soft": "#3A1620", "warn": "#E0A849", "link": "#B3A9FF", "highlight": "#8A7DFF", "code-bg": "#0B0A12",
        "code-ink": "#ECEBF5", "chip-bg": "#2A2740", "chip-ink": "#ECEBF5", "shadow": "0 10px 30px rgba(0, 0, 0, .5)",
        "dark": "1", "danger-panel": "#0B0A12", "danger-panel-ink": "#ECEBF5",
    },
}


def theme_vars(name: str) -> dict:
    if name not in THEMES:
        raise ValueError(f"Unknown theme {name!r}. Choose from: {', '.join(THEMES)}")
    return {**BASE, **THEMES[name]}


def theme_css(name: str) -> str:
    return "".join(f"--{key}:{value};" for key, value in theme_vars(name).items())
