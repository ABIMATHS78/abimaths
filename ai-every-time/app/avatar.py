"""Render a companion's face as an SVG from its Avatar parameters.

All values reaching the SVG are validated by the Avatar model (hex colours and
fixed enums), so no user text is ever interpolated into markup, except the
escaped initials of render_initials.
"""

from html import escape

from app.persona import Avatar


def _hair(a: Avatar) -> tuple[str, str]:
    """Return (back_layer, front_layer) SVG for the hair style."""
    c = a.hair_color
    back, front = "", ""
    if a.hair_style == "short":
        front = f'<path d="M60 92 Q60 52 100 50 Q140 52 140 92 Q132 70 100 68 Q68 70 60 92Z" fill="{c}"/>'
    elif a.hair_style == "long":
        back = f'<path d="M54 95 Q52 48 100 46 Q148 48 146 95 L150 170 L50 170Z" fill="{c}"/>'
        front = f'<path d="M58 96 Q60 52 100 50 Q140 52 142 96 Q126 66 100 66 Q74 66 58 96Z" fill="{c}"/>'
    elif a.hair_style == "curly":
        circles = "".join(
            f'<circle cx="{x}" cy="{y}" r="15" fill="{c}"/>'
            for x, y in [(64, 78), (78, 60), (100, 52), (122, 60), (136, 78), (58, 100), (142, 100)]
        )
        front = circles
    elif a.hair_style == "bun":
        back = f'<circle cx="100" cy="40" r="18" fill="{c}"/>'
        front = f'<path d="M60 94 Q60 52 100 52 Q140 52 140 94 Q128 66 100 66 Q72 66 60 94Z" fill="{c}"/>'
    elif a.hair_style == "braids":
        back = (
            f'<rect x="54" y="80" width="14" height="80" rx="7" fill="{c}"/>'
            f'<rect x="132" y="80" width="14" height="80" rx="7" fill="{c}"/>'
        )
        front = f'<path d="M58 94 Q60 50 100 50 Q140 50 142 94 Q128 64 100 64 Q72 64 58 94Z" fill="{c}"/>'
    elif a.hair_style == "afro":
        back = f'<circle cx="100" cy="76" r="56" fill="{c}"/>'
    elif a.hair_style == "waves":
        front = (
            f'<path d="M60 92 Q58 54 100 50 Q142 54 140 92 Q134 72 120 70 Q110 62 100 70 '
            f'Q90 62 80 70 Q66 72 60 92Z" fill="{c}"/>'
        )
    # "bald": no hair
    return back, front


def _accessory(a: Avatar) -> str:
    c = a.accessory_color
    if a.accessory == "glasses":
        return (
            f'<rect x="70" y="92" width="24" height="16" rx="4" fill="none" stroke="{c}" stroke-width="3"/>'
            f'<rect x="106" y="92" width="24" height="16" rx="4" fill="none" stroke="{c}" stroke-width="3"/>'
            f'<line x1="94" y1="99" x2="106" y2="99" stroke="{c}" stroke-width="3"/>'
        )
    if a.accessory == "round_glasses":
        return (
            f'<circle cx="82" cy="100" r="11" fill="none" stroke="{c}" stroke-width="3"/>'
            f'<circle cx="118" cy="100" r="11" fill="none" stroke="{c}" stroke-width="3"/>'
            f'<line x1="93" y1="100" x2="107" y2="100" stroke="{c}" stroke-width="3"/>'
        )
    if a.accessory == "headphones":
        return (
            f'<path d="M56 100 Q56 40 100 40 Q144 40 144 100" fill="none" stroke="{c}" stroke-width="7"/>'
            f'<rect x="48" y="92" width="14" height="26" rx="6" fill="{c}"/>'
            f'<rect x="138" y="92" width="14" height="26" rx="6" fill="{c}"/>'
        )
    if a.accessory == "earrings":
        return f'<circle cx="62" cy="122" r="5" fill="{c}"/><circle cx="138" cy="122" r="5" fill="{c}"/>'
    if a.accessory == "headwrap":
        return (
            f'<path d="M56 88 Q56 36 100 34 Q144 36 144 88 Q124 70 100 70 Q76 70 56 88Z" fill="{c}"/>'
            f'<circle cx="128" cy="44" r="12" fill="{c}"/>'
        )
    if a.accessory == "cap":
        return (
            f'<path d="M58 82 Q60 46 100 46 Q140 46 142 82Z" fill="{c}"/>'
            f'<path d="M100 78 L160 80 Q158 88 140 88 L100 86Z" fill="{c}"/>'
        )
    return ""


def _mouth(a: Avatar) -> str:
    if a.expression == "grin":
        return '<path d="M84 126 Q100 144 116 126Z" fill="#7B241C"/><path d="M87 127 L113 127 L111 131 L89 131Z" fill="#FFFFFF"/>'
    if a.expression == "calm":
        return '<path d="M88 130 Q100 134 112 130" fill="none" stroke="#5D2E1F" stroke-width="3" stroke-linecap="round"/>'
    if a.expression == "curious":
        return '<ellipse cx="102" cy="130" rx="6" ry="5" fill="#7B241C"/>'
    return '<path d="M86 126 Q100 140 114 126" fill="none" stroke="#5D2E1F" stroke-width="3.5" stroke-linecap="round"/>'


def render_svg(a: Avatar, size: int = 200) -> str:
    back, front = _hair(a)
    brow_tilt = -4 if a.expression == "curious" else 0
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="{size}" height="{size}" '
        f'role="img" aria-label="Companion avatar">'
        f'<rect width="200" height="200" rx="100" fill="{a.background}"/>'
        f"{back}"
        f'<path d="M40 200 Q44 150 100 148 Q156 150 160 200Z" fill="{a.shirt}"/>'
        f'<rect x="88" y="132" width="24" height="22" fill="{a.skin}"/>'
        f'<ellipse cx="100" cy="100" rx="40" ry="46" fill="{a.skin}"/>'
        f'<ellipse cx="60" cy="104" rx="6" ry="9" fill="{a.skin}"/>'
        f'<ellipse cx="140" cy="104" rx="6" ry="9" fill="{a.skin}"/>'
        f"{front}"
        f'<path d="M74 {86 + brow_tilt} Q82 82 90 86" fill="none" stroke="#2B1A10" stroke-width="3" stroke-linecap="round"/>'
        f'<path d="M110 86 Q118 82 126 86" fill="none" stroke="#2B1A10" stroke-width="3" stroke-linecap="round"/>'
        f'<circle cx="82" cy="100" r="4.5" fill="#1B1B1B"/><circle cx="118" cy="100" r="4.5" fill="#1B1B1B"/>'
        f'<circle cx="83.5" cy="98.5" r="1.4" fill="#FFFFFF"/><circle cx="119.5" cy="98.5" r="1.4" fill="#FFFFFF"/>'
        f'<path d="M100 104 Q96 114 100 116" fill="none" stroke="#00000033" stroke-width="2.5" stroke-linecap="round"/>'
        f"{_mouth(a)}"
        f"{_accessory(a)}"
        f"</svg>"
    )


def render_initials(name: str, background: str, size: int = 200) -> str:
    """Initials on a circle: the stand-in for a companion modelled on a real person when their photo can't be
    shown, rather than a drawn face guessing at what they look like."""
    letters = escape("".join(w[0] for w in name.split()[:2]).upper())
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="{size}" height="{size}" '
        f'role="img" aria-label="Companion avatar">'
        f'<rect width="200" height="200" rx="100" fill="{background}"/>'
        f'<text x="100" y="100" dy=".35em" text-anchor="middle" fill="#FFFFFF" font-size="76" font-weight="700" '
        f'font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif">{letters}</text>'
        f"</svg>"
    )
