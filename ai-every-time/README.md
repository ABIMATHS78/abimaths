# AI Every Time

The 24/7 AI learning hub. Learners follow, chat with and learn from always-available **AI Learning Companions**. Each companion represents a country and has its own face, personality, expertise and writing voice. Learners can also **design their own companion** when they create an account, and change it whenever they like.

It includes the companion core (starter companions, the companion designer, chat with memory, self-publishing content) and structured learning taught by those companions: learning paths, lessons, quizzes, mock exams, flashcards, summaries, study plans and progress tracking.

## What's in it

| Area | What it does |
|---|---|
| **Starter companions** | 8 companions: Adaeze 🇳🇬, Haruto 🇯🇵, Camila 🇧🇷, Arjun 🇮🇳, Lena 🇩🇪, Wanjiru 🇰🇪, Maya 🇺🇸 and Sofía 🇲🇽. Each has a name, a generated face, a bio, expertise areas, personality sliders, a teaching style, a writing style, a signature phrase and a sample line in their voice. Defined in `app/starter_companions.py`. |
| **Onboarding** | After sign-up, learners pick a starter companion as-is, **customise** one (a remix: the original stays unchanged), or **design one from scratch**. |
| **Companion Studio** | The designer, with a live preview. **Identity**: name, country/cultural voice, tagline, background, reply language, interests. **Face**: skin tone, hair style and colour, expression, accessory, outfit, background. **Expertise**: up to 5 areas. **Personality**: 6 sliders. **Teaching style**: Socratic, step-by-step, storytelling, examples-first or challenge-driven. **Writing style**: reply length, level, emoji, analogies, local examples, signature phrase, sample line. **Custom instructions**. |
| **Change it later** | Learners can edit their companion at any time. Every save is kept as a version (last 50), and any version can be restored. Starter companions can't be edited, only remixed. |
| **Chat that remembers** | Replies stream in the companion's voice (built from the persona by `app/llm.py`). After each exchange the companion saves lasting facts about the learner (goals, level, struggles). Learners can see these and remove any of them. |
| **Content → relationship** | Companions publish short lessons on their own. Every post has an *"Ask {name} about this"* button that opens a chat with the post as context. |
| **Learning paths** | 5 paths of 5 lessons: AI Foundations, Understanding LLMs, Prompting & AI Agents, Responsible AI, and AI for Business. Each lesson has objectives and vetted key points (`app/curriculum.py`). The learner's companion teaches each lesson in its own voice; learners can switch to any companion they follow. A lesson is written once per companion and then cached. |
| **Interactive tutoring** | Every lesson page has a tutor panel where the learner can ask the companion about that lesson. The companion answers using the lesson's key points. |
| **Quizzes** | 5 multiple-choice questions from any lesson, post or conversation, with an explanation for each answer. Correct answers stay on the server until the learner submits. |
| **Mock exams** | 12 questions across a whole path, a 20-minute timer, a 70% pass mark, and a score breakdown by lesson. |
| **Flashcards** | Decks from any lesson, path, post or conversation. Reviews use spaced repetition (Leitner boxes 1-5: again / good / easy), so each card returns just before it would be forgotten. |
| **Summaries** | Key takeaways and a suggested next step for any lesson, post or conversation. |
| **Study plans** | The learner sets weekly time, number of weeks and which paths to focus on. The companion plans each week from their goals, memories, finished lessons and weak quiz scores. Plan items tick off as the learner completes them. |
| **Progress tracking** | Day streak, last 7 days of activity, lessons completed, quiz average, best exam score, cards mastered and due, progress per path, lessons worth reviewing (best score under 70%), and the next lesson. |
| **Content intelligence** | Views, likes, "helpful" marks, shares and follow-up questions are weighted (1, 3, 4, 5 and 6) and averaged per topic (`app/content.py`). Companions steer their next posts toward the topics learners value most. |
| **Community companions** | Learners can publish a companion they designed. Others can then find it in the Community section, follow it, chat with it and remix it. What stays private: the designer's custom instructions (they only shape the designer's own chats), and everyone's chats and memories. Learners can report a community companion (impersonation, harmful, misleading, spam, other); after 3 reports it is hidden for review. Unpublishing removes it, and its posts, from everyone else's view. |
| **Photo avatars** | Upload a PNG, JPEG or WebP (up to 2 MB) instead of the drawn face. Files are identified by their content, not their name, and served so they can't run as a web page. A photo is only visible to people who can see the companion. |
| **Voice** | Each companion has voice settings: on/off, accent, speed and pitch. Learners can have replies read aloud (🔊 on any reply, or "Read aloud" for every reply) and can dictate questions (🎤). This uses the browser's own speech features, so there's no server cost; how it sounds depends on the voices installed on the device. |
| **Custom learning paths** | A learner names any topic and their companion designs a 3-8 lesson path for them, with objectives and key points. It works with everything else: lessons, tutoring, quizzes, mock exams, flashcards, study plans and progress. These paths are private and marked as not expert-reviewed. |
| **Staff roles** | Learners, reviewers, moderators and admins. Emails listed in `AIET_ADMIN_EMAILS` are always admins, so a new deployment can't lock itself out. Admins assign the other roles on the Staff page; nobody can change their own role. |
| **Moderation** | Moderators see every reported community companion: its report count, reasons and details. They can restore it (if hidden), dismiss the reports (if still visible), or remove it from the community for good, with a required note. The designer keeps a private copy and can't republish it. Every decision is logged and the designer is told the outcome. |
| **Shared paths, expert reviewed** | A learner can submit a custom path for sharing. A reviewer checks it and can correct its title, description, lesson titles, objectives and key points, but not the number of lessons, so learners' progress still lines up. Corrections clear the cached lesson text, so companions re-teach from the corrected facts. Approved paths appear for every learner with an "Expert reviewed · suggested by …" badge. Reviewers can reject (with a note) or later unshare, and can't approve their own paths. |
| **Reminders** | A "Today" panel on the feed and a bell in the sidebar list what's waiting: a streak at risk, flashcards due, the next study-plan item (or one that's overdue), an unfinished quiz, a lesson to continue, new posts from followed companions, and notices about moderation and review decisions. Learners can opt in to a **daily reminder email** through any SMTP server. It carries one-click unsubscribe (RFC 8058), and its unsubscribe page asks for confirmation so email link-scanners can't unsubscribe people by accident. |

## Run it

```bash
cd ai-every-time
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # leave unset to run in offline demo mode
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000.

Without an API key, the app runs in **demo mode**: it works end to end, but companions give templated replies and posts. A banner says so.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Turns on real AI chat, memory and content |
| `AIET_MODEL` | `claude-opus-5-5` | Claude model used for everything |
| `AIET_AUTOGEN_MINUTES` | `0` (off) | Minutes between automatic posts from each starter companion |
| `AIET_DATABASE` | `ai_every_time.db` | SQLite file |
| `AIET_SECURE_COOKIES` | off | Set to `1` behind HTTPS |
| `AIET_HISTORY_TURNS` | `20` | Past conversation turns sent with each message |
| `AIET_OFFLINE` | off | Set to `1` to force demo mode |
| `AIET_ADMIN_EMAILS` | — | Comma-separated emails that are always admins |
| `AIET_SMTP_HOST`, `AIET_SMTP_PORT`, `AIET_SMTP_USER`, `AIET_SMTP_PASSWORD`, `AIET_SMTP_STARTTLS` | — / 587 / — / — / on | SMTP server for reminder emails; email is off while `AIET_SMTP_HOST` is unset |
| `AIET_MAIL_FROM` | — | Sender address for reminder emails (required for email) |
| `AIET_PUBLIC_URL` | `http://127.0.0.1:8000` | The site's public address, used for links in emails |
| `AIET_DIGEST_HOUR_UTC` | `7` | Hour (UTC) after which the daily email goes out |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests run in offline mode and cover:
- seeding the starter companions
- onboarding
- designing, editing and restoring a companion
- remixing
- access rules (no one can see, edit or chat with another learner's companion)
- input validation
- streamed chat and memory
- engagement-driven topic scores
- lessons (taught by the chosen companion, cached per companion)
- quizzes (answers hidden until submit, grading, no double submission)
- mock exams with a per-lesson breakdown
- flashcard spaced repetition
- summaries and lesson tutoring
- study plans that follow progress
- progress, streaks, and per-learner access rules for quizzes and decks
- publishing, private custom instructions, unpublishing, and hiding after reports
- photo upload checks (type, size, ownership, visibility, safe headers)
- voice settings validation
- custom paths, end to end and private to their owner
- reminders
- upgrading an older database
- staff roles and permissions
- moderation: restore, dismiss, remove, logging and notices to designers
- path review: approve with corrections, reject, resubmit, withdraw, unshare, no self-approval
- email reminders: opt-in, daily digest, sending, one-click unsubscribe

## How it's built

```
app/
  persona.py            Persona schema shared by starter and learner-designed companions
  starter_companions.py The 8 starter companions
  avatar.py             Builds each face as an SVG from its settings (validated colours and options only)
  llm.py                Claude: system prompt from the persona, streamed chat, memory extraction, post writing
  curriculum.py         Learning paths, lessons, objectives and vetted key points
  tutor.py              Claude: lessons, quiz/exam questions, flashcards, summaries, study plans
  learning.py           Learning API: paths (curated + custom), lessons, assessments, decks and reviews, plans, progress
  reminders.py          What's waiting for the learner, worked out from existing data
  staff.py              Moderation queue, path review queue, roles, decision log
  digest.py             Daily reminder email over SMTP, and unsubscribe
  content.py            Topic scores from engagement, post generation, background schedule
  companions.py         Create / remix / edit / version / follow
  auth.py, db.py        Password logins (scrypt) with cookie sessions; SQLite storage
  main.py               HTTP API + static app
static/                 Single-page web app (no build step)
```

**Prompt structure.** The system prompt has two parts. The first holds the fixed platform rules plus the persona; it is cached. Learner-written custom instructions go here too, *after* the platform rules, so they can change a companion's voice but not the rules. The second part holds learner data that changes often: profile, memories, and the post being discussed.

**Claude requests.** Every request turns on server-side refusal fallbacks (`fallbacks: "default"`). Effort is set per task: `low` for memory and summaries, `medium` for chat and flashcards, `high` for posts, lessons, quizzes and study plans.

**Accuracy.** Lessons, questions and cards are written from each lesson's vetted key points, not from the model's general knowledge alone. Generated output is checked before it is saved: questions must have exactly 4 distinct options and a valid answer, and study plans may only reference real lesson IDs and activities.

## Next steps

- **Push notifications** on phones, built on the same reminder list (needs a push service).
- **Appeals**, so a designer can ask for a second look after a removal.
- **Server-side voices** for consistent, higher-quality speech on every device (needs a text-to-speech provider).
- **Postgres** in place of SQLite, and a job queue in place of the in-process scheduler, for multi-instance deployment.
