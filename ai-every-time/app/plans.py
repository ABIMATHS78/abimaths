"""The plans: what each costs and what it unlocks.

Prices are in US cents, each paid plan monthly or yearly. Teams is priced per seat, and set up by the AI Every Time
team (who give each member the plan) rather than bought on the pricing page. A plan unlocks everything the plans
below it do. The server checks what a plan unlocks (app/billing.py); the pricing page shows the same table (GET
/api/plans), so the two can't drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import settings

TIERS = ("free", "digital", "plus", "pro", "insider", "teams")
INTERVALS = ("monthly", "yearly")
SELF_SERVE = ("digital", "plus", "pro", "insider")  # bought on the pricing page

# What plans unlock beyond the free one, with the name used when a learner reaches for one without it.
FEATURES = {
    "guides": "Unlimited guides",
    "courses": "Full courses, labs, projects and certificates",
    "exams": "Mock exams",
    "custom_paths": "Learning paths designed for you",
    "event_pass": "A complimentary GDAI 2026 pass",
}


@dataclass(frozen=True)
class Plan:
    slug: str
    name: str
    tagline: str
    monthly: int   # cents a month (a seat a month for Teams)
    yearly: int    # cents a year
    allowance: int  # AI requests in any 24 hours
    event_discount: int  # percent off AI Every Time events
    unlocks: frozenset
    highlights: tuple
    badge: str = ""
    per_seat: bool = False

    @property
    def rank(self) -> int:
        return TIERS.index(self.slug)

    @property
    def paid(self) -> bool:
        return self.slug != "free"

    def price(self, interval: str) -> int:
        return self.yearly if interval == "yearly" else self.monthly

    def yearly_saving(self) -> int:
        """Percent saved paying yearly rather than twelve months."""
        full = self.monthly * 12
        return round(100 * (full - self.yearly) / full) if full else 0

    def daily_allowance(self) -> int:
        """AI requests in any 24 hours, AIET_PLAN_ALLOWANCES changing the ones here. 0: no limit."""
        value = settings.plan_allowances.get(self.slug, self.allowance)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else self.allowance


_ALL = frozenset(FEATURES)
PLANS = {p.slug: p for p in (
    Plan("free", "Free", "Start learning with AI companions", 0, 0, 15, 0, frozenset(), (
        "AI learning companions from around the world",
        "Learning paths, quizzes, flashcards and study plans",
        "The first module of every course",
        "{free_guides} education guides a month in full",
    )),
    Plan("digital", "Digital", "Read every guide", 199, 1999, 40, 10, frozenset({"guides"}), (
        "Everything in Free",
        "Every education guide, in full",
    )),
    Plan("plus", "Plus", "Learn with full courses", 299, 2999, 80, 15, frozenset({"guides", "courses", "exams"}), (
        "Everything in Digital",
        "All {courses} courses in full: every module, lab and project",
        "Certificates, Professional Certificates and Specializations",
        "Mock exams",
    ), badge="Most popular"),
    Plan("pro", "Pro", "Go further, faster", 699, 6999, 150, 25, _ALL - {"event_pass"}, (
        "Everything in Plus",
        "Learning paths your companion designs for you, on any topic",
    )),
    Plan("insider", "Insider", "The whole of AI Every Time", 999, 9999, 300, 50, _ALL, (
        "Everything in Pro",
        "A complimentary GDAI 2026 pass",
    ), badge="Premium"),
    Plan("teams", "Teams", "For teams and classrooms", 899, 8999, 300, 50, _ALL, (
        "Everything in Insider, for every member",
        "One invoice for the whole team",
        "Members added and removed for you",
    ), per_seat=True),
)}


def plan(tier: str | None) -> Plan:
    return PLANS.get(tier or "free", PLANS["free"])


def rank(tier: str | None) -> int:
    return plan(tier).rank


def best(*tiers: str | None) -> str:
    return max((t or "free" for t in tiers), key=rank)


def lowest_with(feature: str) -> Plan:
    """The cheapest plan that unlocks `feature`."""
    return next(p for p in PLANS.values() if feature in p.unlocks)


def money(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def public(course_count: int) -> list[dict]:
    """The plans as the pricing page shows them."""
    out = []
    for p in PLANS.values():
        allowance = p.daily_allowance()
        lines = [h.format(free_guides=settings.free_guides, courses=course_count) for h in p.highlights]
        # Then what every plan has, in figures, so the cards can be compared line by line.
        lines.append("No daily limit on AI requests" if allowance == 0 else f"{allowance} AI requests a day")
        if p.event_discount:
            lines.append(f"{p.event_discount}% off AI Every Time events")
        out.append({
            "slug": p.slug, "name": p.name, "tagline": p.tagline, "badge": p.badge, "per_seat": p.per_seat,
            "monthly": p.monthly, "yearly": p.yearly, "yearly_saving": p.yearly_saving(), "ai_allowance": allowance,
            "event_discount": p.event_discount, "unlocks": sorted(p.unlocks), "highlights": lines,
            "self_serve": p.slug in SELF_SERVE, "trial_days": settings.trial_days if p.slug in SELF_SERVE else 0,
        })
    return out
