# AI Every Time

The 24/7 AI learning hub. Learners follow, chat with and learn from always-available **AI Learning Companions**. Each companion represents a country and has its own face, personality, expertise and writing voice. Learners can also **design their own companion** when they create an account, and change it whenever they like.

This first build covers the companion core: starter companions, the companion designer, editing and history, chat with memory, and content that companions publish on their own and that follows what learners engage with.

## What's in it

| Area | What it does |
|---|---|
| **Starter companions** | 8 companions: Adaeze 🇳🇬, Haruto 🇯🇵, Camila 🇧🇷, Arjun 🇮🇳, Lena 🇩🇪, Wanjiru 🇰🇪, Maya 🇺🇸 and Sofía 🇲🇽. Each has a name, a generated face, a bio, expertise areas, personality sliders, a teaching style, a writing style, a signature phrase and a sample line in their voice. Defined in `app/starter_companions.py`. |
| **Onboarding** | After sign-up, learners pick a starter companion as-is, **customise** one (a remix: the original stays unchanged), or **design one from scratch**. |
| **Companion Studio** | The designer, with a live preview. **Identity**: name, country/cultural voice, tagline, background, reply language, interests. **Face**: skin tone, hair style and colour, expression, accessory, outfit, background. **Expertise**: up to 5 areas. **Personality**: 6 sliders. **Teaching style**: Socratic, step-by-step, storytelling, examples-first or challenge-driven. **Writing style**: reply length, level, emoji, analogies, local examples, signature phrase, sample line. **Custom instructions**. |
| **Change it later** | Learners can edit their companion at any time. Every save is kept as a version (last 50), and any version can be restored. Starter companions can't be edited, only remixed. |
| **Chat that remembers** | Replies stream in the companion's voice (built from the persona by `app/llm.py`). After each exchange the companion saves lasting facts about the learner (goals, level, struggles). Learners can see these and remove any of them. |
| **Content → relationship** | Companions publish short lessons on their own. Every post has an *"Ask {name} about this"* button that opens a chat with the post as context. |
| **Content intelligence** | Views, likes, "helpful" marks, shares and follow-up questions are weighted (1, 3, 4, 5 and 6) and averaged per topic (`app/content.py`). Companions steer their next posts toward the topics learners value most. |

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

## How it's built

```
app/
  persona.py            Persona schema shared by starter and learner-designed companions
  starter_companions.py The 8 starter companions
  avatar.py             Builds each face as an SVG from its settings (validated colours and options only)
  llm.py                Claude: system prompt from the persona, streamed chat, memory extraction, post writing
  content.py            Topic scores from engagement, post generation, background schedule
  companions.py         Create / remix / edit / version / follow
  auth.py, db.py        Password logins (scrypt) with cookie sessions; SQLite storage
  main.py               HTTP API + static app
static/                 Single-page web app (no build step)
```

**Prompt structure.** The system prompt has two parts. The first holds the fixed platform rules plus the persona; it is cached. Learner-written custom instructions go here too, *after* the platform rules, so they can change a companion's voice but not the rules. The second part holds learner data that changes often: profile, memories, and the post being discussed.

**Claude requests.** Every request turns on server-side refusal fallbacks (`fallbacks: "default"`). Effort is set per task: `medium` for chat, `low` for memory extraction, `high` for writing posts.

## Next steps

The schema and API already support these:
- **Learning features**: courses, learning paths, quizzes, flashcards, mock exams, study plans and progress tracking, all taught by the learner's companion. Quizzes and flashcards can use the same structured-output approach as posts.
- **Public sharing** of learner-designed companions (today they are private to their owner).
- **Voice**, and **uploaded photo avatars**.
- **Postgres** in place of SQLite, and a job queue in place of the in-process scheduler, for multi-instance deployment.
