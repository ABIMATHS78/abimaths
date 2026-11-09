"""The About page (/about): what AI Every Time is, what it offers, and who leads it. The app draws it from GET
/api/about, and the server sends the same words in the page itself for search engines (app/seo.py), so the two can't
differ."""

from __future__ import annotations

from app import catalog, db

HEADLINE = "The world of AI, taught by local AI voices that never sleep"
MISSION = ("AI Every Time is building the world's first 24/7 learning hub for AI, where every piece of content on AI "
           "begins an ongoing learning relationship between learners and our AI content creators and learning "
           "companions.")
# What inspired AI Every Time.
STORY = (
    "The inspiration was the gap between content reach and human attention. A creator or educator on AI can reach "
    "millions of people, but cannot answer millions of follow-up questions, remember every learner, or be available "
    "at 2 a.m.",
    "That's where AI Every Time comes in: the world of AI, taught by your local AI voices that never sleep. On AI Every "
    "Time, you can chat and learn about AI with your local AI voices at any time. AI Every Time turns a piece of "
    "content into the beginning of a relationship, rather than the end of a view or a read.",
    "People follow, chat with and learn from always-available AI Learning Companions worldwide. Each represents a "
    "different country and brings a distinctive identity, expertise, perspective and cultural voice. Learning doesn't "
    "end when you finish reading AI content you like: it begins an ongoing relationship with our companions.",
)
# How AI Every Time is changing the way people learn about AI, keep up with it and engage with content on it.
DIFFERENT = (
    ("🌍", "Local AI voices, always on", "Our platform delivers always-on AI news and education through local AI voices "
     "that write content, explain it, answer questions and personalize learning for every learner in real time."),
    ("💬", "From reading to conversation", "We turn learning about AI from passive content consumption into "
     "interactive conversations, making the creator of every piece of AI content available to their audience at all "
     "times."),
    ("🤝", "Human, not robotic", "Much AI content sounds robotic or obviously AI-generated. We create learning "
     "experiences that feel human, clear and engaging."),
    ("📈", "Content intelligence", "Our content is powered by a proven content intelligence system that identifies what "
     "people genuinely love, engage with, share and find helpful, so every guide is timely, relevant and valuable."),
)
OFFERS = (
    ("🧑‍🏫", "AI Learning Companions", "Personalized companions who engage you in meaningful conversations about AI: "
     "they explain, quiz you, review your work and remember what you've learned, at any hour."),
    ("🎓", "Courses and certificates", "Comprehensive, structured learning that takes professionals from AI beginners "
     "to advanced practitioners, with hands-on labs in real tools, reviewed projects and certificates."),
    ("📚", "Guides and cheat sheets", "Timely tutorials, frameworks and cheat sheets on AI, each with an infographic, "
     "ready to put to work."),
    ("🎮", "Interactive and gamified", "Learning paths around your goals, with quizzes, flashcards, streaks and "
     "reminders that make learning interactive and help it stick."),
)
TRACTION = (
    ("Growing fast", "Our community of learners is growing rapidly, all around the world."),
    ("Paid subscribers", "Learners subscribe across our membership plans, including annual plans and our highest tiers."),
    ("Enterprise", "An enterprise customer already uses the AI Every Time API."),
)
TRACTION_NOTE = ("This early traction, and people's willingness to pay, strengthens our belief that people see real "
                 "value in AI Every Time: AI knowledge that's more accessible, personal, responsive and useful, at "
                 "global scale.")
FOUNDER = {
    "slug": "mojeed-abisiga",
    "name": "Mojeed Abisiga",
    "title": "Founder & CEO",
    "photo": "/static/about/mojeed-abisiga.jpg",
    "summary": "Award-winning Data and AI executive, entrepreneur, technology strategist, international speaker and "
               "thought leader.",
    "bio": (
        "Mojeed Abisiga is an award-winning Data and AI executive, entrepreneur, technology strategist, international "
        "speaker, and thought leader. He is the Founder and CEO of AI Every Time, a 24/7 AI learning hub, and the CEO "
        "and Co-founder of DataGlobal Hub, a global Data and AI media, technology, and community platform.",
        "Under his leadership, DataGlobal Hub has grown an international ecosystem of over 1,700 ambassadors, 200+ "
        "partners, 900+ media platforms, and 1,500+ scholarship opportunities, extending its reach across more than "
        "60 countries. He also leads the Global Data & AI Virtual Tech Conference, bringing together thousands of "
        "attendees and 140+ speakers to advance conversations, collaboration, and innovation across the global "
        "technology ecosystem. His leadership has earned DataGlobal Hub recognition as Technology News Platform of the "
        "Year 2025, while its Global Data & AI Conference received the Best Convention of the Year 2025 recognition at "
        "the Eventeny evAwards.",
        "With professional experience spanning American Express, KPMG, and Deloitte, Mojeed brings expertise in "
        "enterprise analytics, machine learning, artificial intelligence, and technology leadership. His achievements "
        "include recognition as a Top 50 Machine Learning and Data Science Expert, LinkedIn Top Voice in Machine "
        "Learning, Forbes Technology Council member, and Exceptional Data Scientist of the Year 2021. He is also a "
        "recipient of the EB-1A Extraordinary Ability Green Card, popularly known as the Einstein Visa, recognizing "
        "individuals with extraordinary ability in their fields.",
        "Academically, he earned a First-Class Bachelor's degree in Mathematics, graduating among the top students in "
        "his class, and a Master's degree in Informatics and Analytics, graduating Summa Cum Laude with a perfect 4.00 "
        "GPA. A former national mathematics gold medalist who represented Nigeria at an international mathematics "
        "competition in Bulgaria, Mojeed has also received numerous academic scholarships and distinctions.",
        "Today, he continues to build technology ventures, champion global talent, and create opportunities that make "
        "the future of Data and AI more accessible and inclusive.",
    ),
    "experience": ("American Express", "KPMG", "Deloitte"),
    "honours": (
        "Forbes Technology Council member",
        "LinkedIn Top Voice in Machine Learning",
        "Top 50 Machine Learning and Data Science Expert",
        "Exceptional Data Scientist of the Year 2021",
        "EB-1A Extraordinary Ability Green Card",
        "National mathematics gold medalist",
    ),
    "numbers": (
        ("1,700+", "DataGlobal Hub ambassadors"),
        ("60+", "countries reached"),
        ("140+", "conference speakers"),
        ("4.00", "GPA, Master's, Summa Cum Laude"),
    ),
    "also": "CEO & Co-founder, DataGlobal Hub",
}


def stats() -> dict:
    """What AI Every Time offers today, counted."""
    cat = catalog.data()
    with db.connect() as conn:
        companions = conn.execute("SELECT COUNT(*) FROM companions WHERE is_starter = 1 AND retired = 0").fetchone()[0]
        guides = conn.execute("SELECT COUNT(*) FROM posts p JOIN companions c ON c.id = p.companion_id "
                              "WHERE p.kind IS NOT NULL AND c.retired = 0 AND (c.is_starter = 1 OR "
                              "(c.is_public = 1 AND c.hidden = 0))").fetchone()[0]
    return {"courses": len(cat.courses), "tracks": len(cat.tracks), "companions": companions, "guides": guides}


def page() -> dict:
    return {"headline": HEADLINE, "mission": MISSION, "story": list(STORY),
            "different": [{"icon": i, "title": t, "text": x} for i, t, x in DIFFERENT],
            "offers": [{"icon": i, "title": t, "text": x} for i, t, x in OFFERS],
            "traction": [{"title": t, "text": x} for t, x in TRACTION], "traction_note": TRACTION_NOTE,
            "founder": FOUNDER | {"bio": list(FOUNDER["bio"]), "experience": list(FOUNDER["experience"]),
                                  "honours": list(FOUNDER["honours"]),
                                  "numbers": [{"value": v, "label": l} for v, l in FOUNDER["numbers"]]},
            "stats": stats()}
