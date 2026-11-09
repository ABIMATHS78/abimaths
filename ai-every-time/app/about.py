"""The About page (/about): what AI Every Time is, what it offers, and who leads it. The app draws it from GET
/api/about, and the server sends the same words in the page itself for search engines (app/seo.py), so the two can't
differ."""

from __future__ import annotations

from app import catalog, db

MISSION = ("AI Every Time is a 24/7 AI learning hub. We help anyone, anywhere, learn artificial intelligence, data, "
           "software and cloud skills by doing: with AI Learning Companions who are always there to explain, quiz and "
           "encourage, guides and cheat sheets written for real work, and courses with hands-on labs, reviewed projects "
           "and certificates.")
STORY = (
    "Most learning stops when the video ends or the article runs out. We built AI Every Time so it doesn't: every "
    "guide, lesson and lab is the start of a conversation with a companion who knows the subject, remembers where you "
    "are and answers your questions at any hour.",
    "Our companions come from around the world, each with their own face, voice, expertise and way of teaching. "
    "Learners choose who they learn with, follow the companions whose guides they like, and can even design their "
    "own.",
)
OFFERS = (
    ("🧑‍🏫", "AI Learning Companions", "Chat with AI tutors from around the world, any time. They explain, quiz you, "
     "review your work and remember what you've learned."),
    ("🎓", "Courses and certificates", "Courses and Professional Certificates with lessons, hands-on labs in real "
     "tools, projects your instructor reviews, graded quizzes and a certificate you can share."),
    ("📚", "Guides and cheat sheets", "Tutorials, frameworks and cheat sheets on AI, each with an infographic, written "
     "by companions and ready to put to work."),
    ("🧭", "Learning paths and practice", "Paths designed around your goals, with quizzes, flashcards and daily "
     "reminders that help what you learn stick."),
)
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
    return {"mission": MISSION, "story": list(STORY),
            "offers": [{"icon": i, "title": t, "text": x} for i, t, x in OFFERS],
            "founder": FOUNDER | {"bio": list(FOUNDER["bio"]), "experience": list(FOUNDER["experience"]),
                                  "honours": list(FOUNDER["honours"]),
                                  "numbers": [{"value": v, "label": l} for v, l in FOUNDER["numbers"]]},
            "stats": stats()}
