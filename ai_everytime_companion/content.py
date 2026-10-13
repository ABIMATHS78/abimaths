"""Content library and the content-intelligence ranking.

The ranking models "what people genuinely love, engage with, share, and find
helpful": each signal is weighted, normalised by views, and decayed by age so
timely pieces surface. Companions use it to pick what to recommend next.

Sample items stand in for the production `articles` table (slug, title, summary,
content, category, difficulty, region, publishedAt, viewCount).
"""

import math
import re
import time
from dataclasses import dataclass, field

DAY = 86_400

# Weight per engagement signal. Shares and "helpful" votes say more about value
# than a raw view or a click, so they count for more.
SIGNAL_WEIGHTS = {
    "completions": 1.0,
    "saves": 2.0,
    "shares": 3.0,
    "helpful_votes": 3.0,
    "companion_chats": 2.5,  # a piece that starts conversations is doing its job
}
HALF_LIFE_DAYS = 30


@dataclass
class ContentItem:
    slug: str
    title: str
    summary: str
    body: str
    category: str
    difficulty: str  # beginner | intermediate | advanced
    published_at: float
    views: int = 0
    signals: dict[str, int] = field(default_factory=dict)
    prerequisites: tuple[str, ...] = ()


def engagement_score(item: ContentItem, now: float | None = None) -> float:
    """Value per view, smoothed for small samples, decayed by age."""
    now = time.time() if now is None else now
    weighted = sum(SIGNAL_WEIGHTS.get(k, 0.0) * v for k, v in item.signals.items())
    # Bayesian-style smoothing: a piece with 3 views and 3 shares shouldn't
    # outrank one with 3,000 views and 2,000 shares.
    value_rate = weighted / (item.views + 50)
    reach = math.log10(item.views + 10)
    age_days = max(0.0, (now - item.published_at) / DAY)
    freshness = 0.5 ** (age_days / HALF_LIFE_DAYS)
    return value_rate * reach * (0.4 + 0.6 * freshness)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


class ContentLibrary:
    def __init__(self, items: list[ContentItem]):
        self._items = {i.slug: i for i in items}

    def get(self, slug: str) -> ContentItem | None:
        return self._items.get(slug)

    def all(self) -> list[ContentItem]:
        return list(self._items.values())

    def search(self, query: str, limit: int = 5) -> list[ContentItem]:
        """Keyword relevance, tie-broken by engagement score.

        Swap for embeddings/vector search when the library grows past a few
        thousand items.
        """
        q = _tokens(query)
        scored = []
        for item in self._items.values():
            title_hits = len(q & _tokens(item.title))
            text_hits = len(q & _tokens(f"{item.summary} {item.body} {item.category}"))
            relevance = 3 * title_hits + text_hits
            if relevance:
                scored.append((relevance, engagement_score(item), item))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [item for _, _, item in scored[:limit]]

    def recommend(
        self,
        exclude: set[str],
        level: str | None = None,
        interests: list[str] | None = None,
        limit: int = 3,
    ) -> list[ContentItem]:
        """Best next pieces the learner hasn't consumed, whose prerequisites they have."""
        levels = ["beginner", "intermediate", "advanced"]
        interest_tokens = _tokens(" ".join(interests or []))
        candidates = []
        for item in self._items.values():
            if item.slug in exclude:
                continue
            if any(p not in exclude for p in item.prerequisites):
                continue
            score = engagement_score(item)
            if level in levels:
                gap = levels.index(item.difficulty) - levels.index(level)
                score *= {0: 1.0, 1: 0.8}.get(gap, 0.3)  # same level or one step up
            if interest_tokens & _tokens(f"{item.title} {item.category} {item.summary}"):
                score *= 1.5
            candidates.append((score, item))
        candidates.sort(key=lambda t: t[0], reverse=True)
        return [item for _, item in candidates[:limit]]


def sample_library(now: float | None = None) -> ContentLibrary:
    """A small illustrative library. Engagement numbers are synthetic test data."""
    now = time.time() if now is None else now
    return ContentLibrary([
        ContentItem(
            slug="what-is-a-large-language-model",
            title="What Is a Large Language Model?",
            summary="A plain-language tour of how LLMs predict text, what training does, and why they sometimes make things up.",
            body=(
                "A large language model is a neural network trained to predict the next token in a sequence. "
                "Pre-training on large text corpora teaches broad patterns of language and knowledge; "
                "post-training (instruction tuning and reinforcement learning from feedback) shapes it into a helpful assistant. "
                "Because the model generates plausible continuations rather than looking facts up, it can produce confident "
                "but wrong statements, often called hallucinations. Grounding answers in retrieved documents reduces this."
            ),
            category="Foundations",
            difficulty="beginner",
            published_at=now - 20 * DAY,
            views=4200,
            signals={"completions": 2900, "saves": 610, "shares": 380, "helpful_votes": 900, "companion_chats": 520},
        ),
        ContentItem(
            slug="prompting-fundamentals",
            title="Prompting Fundamentals: Getting Useful Answers From AI",
            summary="How to give context, examples, and clear success criteria so an AI model does what you meant.",
            body=(
                "Good prompts state the goal, the audience, the constraints, and what a good answer looks like. "
                "Providing examples shows the model the format you want. Asking for reasoning on hard problems, "
                "and splitting large tasks into steps, both improve reliability."
            ),
            category="Foundations",
            difficulty="beginner",
            published_at=now - 6 * DAY,
            views=1800,
            signals={"completions": 1300, "saves": 420, "shares": 160, "helpful_votes": 510, "companion_chats": 300},
            prerequisites=("what-is-a-large-language-model",),
        ),
        ContentItem(
            slug="retrieval-augmented-generation",
            title="Retrieval-Augmented Generation (RAG), Explained",
            summary="Why RAG pairs a search step with a language model, and the design choices that make it accurate.",
            body=(
                "RAG retrieves relevant documents for a question and gives them to the model as context. "
                "Key choices are how documents are chunked, how they are embedded and searched, how many are passed in, "
                "and whether the model must cite sources. Evaluating retrieval quality separately from answer quality "
                "makes problems much easier to find."
            ),
            category="Engineering",
            difficulty="intermediate",
            published_at=now - 12 * DAY,
            views=2600,
            signals={"completions": 1500, "saves": 700, "shares": 240, "helpful_votes": 620, "companion_chats": 410},
            prerequisites=("what-is-a-large-language-model",),
        ),
        ContentItem(
            slug="ai-agents-and-tool-use",
            title="AI Agents and Tool Use",
            summary="How models call tools in a loop to take actions, and how to keep agents reliable and safe.",
            body=(
                "An agent is a model that decides which tool to call, observes the result, and repeats until the task is done. "
                "Reliable agents have a small set of well-described tools, clear stopping conditions, limits on cost and actions, "
                "and human approval for anything irreversible."
            ),
            category="Engineering",
            difficulty="advanced",
            published_at=now - 3 * DAY,
            views=900,
            signals={"completions": 520, "saves": 300, "shares": 140, "helpful_votes": 260, "companion_chats": 190},
            prerequisites=("prompting-fundamentals", "retrieval-augmented-generation"),
        ),
        ContentItem(
            slug="ai-credit-scoring-thin-files",
            title="AI Credit Scoring for People With Thin Credit Files",
            summary="How alternative data like mobile-money history can widen access to credit, and the fairness risks to manage.",
            body=(
                "Lenders in mobile-first markets use signals such as airtime top-ups and mobile-money flows to assess people "
                "without formal credit histories. Models must be checked for bias against groups, explained to applicants, "
                "and governed so that data is collected with consent."
            ),
            category="AI in Finance",
            difficulty="intermediate",
            published_at=now - 45 * DAY,
            views=1500,
            signals={"completions": 800, "saves": 210, "shares": 190, "helpful_votes": 300, "companion_chats": 150},
            prerequisites=("what-is-a-large-language-model",),
        ),
        ContentItem(
            slug="eu-ai-act-for-builders",
            title="The EU AI Act for People Who Build Products",
            summary="The risk tiers, what counts as high-risk, and the practical obligations for teams shipping AI.",
            body=(
                "The EU AI Act sorts AI systems into risk tiers. Prohibited practices are banned; high-risk systems need risk "
                "management, data governance, logging, human oversight, and conformity assessment; general-purpose AI models "
                "carry transparency duties. Most teams start by inventorying their AI use cases and classifying each one."
            ),
            category="Policy & Ethics",
            difficulty="intermediate",
            published_at=now - 90 * DAY,
            views=3100,
            signals={"completions": 1400, "saves": 500, "shares": 260, "helpful_votes": 480, "companion_chats": 200},
        ),
    ])
