# AI Every Time

The 24/7 AI learning hub. Learners follow, chat with and learn from always-available **AI Learning Companions**. Each companion represents a country and has its own face, personality, expertise and writing voice. Learners can also **design their own companion** when they create an account, and change it whenever they like.

It includes the companion core (starter companions, the companion designer, chat with memory, self-publishing content) and structured learning taught by those companions: learning paths, lessons, quizzes, mock exams, flashcards, summaries, study plans and progress tracking. It installs on phones and computers like an app, reminds learners at the time they choose in their own time zone, keeps AI costs predictable with a daily allowance per learner, and gives learners control of their account and data. Admins get an Insights page.

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
| **Appeals** | The designer of a removed companion sees why on its page and can ask for a second look, once per removal. Appeals go to the Staff page's Appeals tab. Neither the moderator who made the removal nor the designer (if they're staff themselves) can decide one. Another moderator either restores the companion to the community or keeps it removed, with a required note. The decision is logged and the designer is told. |
| **Shared paths, expert reviewed** | A learner can submit a custom path for sharing. A reviewer checks it and can correct its title, description, lesson titles, objectives and key points, but not the number of lessons, so learners' progress still lines up. Corrections clear the cached lesson text, so companions re-teach from the corrected facts. Approved paths appear for every learner with an "Expert reviewed · suggested by …" badge. Reviewers can reject (with a note) or later unshare, and can't approve their own paths. |
| **Reminders** | A "Today" panel on the feed and a bell in the sidebar list what's waiting: a streak at risk, flashcards due, the next study-plan item (or one that's overdue), an unfinished quiz, a lesson to continue, new posts from followed companions, and notices about moderation and review decisions. Staff also see the work waiting in their queues: reported companions, appeals they can decide, and paths to review. Learners pick the hour of their **daily reminder**, which follows their own time zone, as do streaks and study days. They can get it by email (any SMTP server, to a confirmed address) or as a notification. The email carries one-click unsubscribe (RFC 8058), and its unsubscribe page asks for confirmation so email link-scanners can't unsubscribe people by accident. |
| **Installable app and notifications** | The site installs on phones and computers like an app (manifest, icons and a service worker). Learners turn notifications on for each device from their profile and can send a test. The daily reminder arrives from their own companion; a reminder that would come more than 3 hours late is skipped rather than sent at night. Notices about decisions on their companions and paths arrive as they happen. On iPhone and iPad, notifications work once the app is on the Home Screen, and the profile explains how. Signing out turns notifications off for that device. |
| **Confirmed email addresses** | Reminder emails only go to an address its owner has confirmed (double opt-in), so nobody can sign someone else up for them. Sign-up emails a confirmation link, and learners can ask for another from their profile (up to 3 an hour). The link works once, for 7 days, only a hash of it is stored, and it opens a page with a button, so mail scanners that follow links can't confirm for anyone. Resetting the password by email also confirms the address. Learners who had reminder emails on before confirmation existed keep getting them. |
| **Daily AI allowance** | Each learner can make 100 AI requests in any 24 hours (configurable). A chat reply, a lesson the first time it's taught, a quiz, flashcards, a summary, a study plan or a post counts one; a mock exam or a custom path counts two. The profile shows what's left. When it runs out, the learner is told when they can ask again, and a chat message that was turned away goes back in the text box. A call to Claude that fails is refunded; an answer that came back but couldn't be used still counts, so failures can't be turned into free requests. Until learners confirm their email address the allowance is lower (30), which makes floods of throwaway accounts less useful. Admins have no limit, and an optional ceiling covers everyone together. |
| **Insights** | Admins see how the site is doing on the Staff page: learners (new, studying this week, confirmed, with notifications), learning this week, AI requests per day over the last 14 days, requests and tokens by kind, requests turned away and failed calls, the limits in force, the topics learners value most and the most followed community companions. |
| **Account and privacy** | Forgotten passwords are reset with an emailed link: it works once, for an hour, and only a hash of it is stored. Learners can change their password and sign out other devices; both also stop notifications on the other devices. After 10 wrong passwords for an email in 15 minutes, sign-in pauses for that email. **Download my data** gives one JSON file with everything the account holds, including its AI requests (answers to unfinished quizzes stay hidden). **Delete my account** (password required) removes it all, including the companions the learner designed and any paths they shared with the community. |

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

Notifications need two things. First, a contact address that browsers' push services can reach you at: `AIET_PUSH_CONTACT`, or else the `AIET_MAIL_FROM` address or an https `AIET_PUBLIC_URL`. Second, HTTPS in production (browsers allow notifications on `localhost` for testing). The server makes its own push (VAPID) key the first time it's needed and keeps it in the database.

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
| `AIET_SMTP_HOST`, `AIET_SMTP_PORT`, `AIET_SMTP_USER`, `AIET_SMTP_PASSWORD`, `AIET_SMTP_STARTTLS` | — / 587 / — / — / on | SMTP server for reminder, confirmation and password-reset emails; email is off while `AIET_SMTP_HOST` is unset |
| `AIET_MAIL_FROM` | — | Sender address for emails (required for email) |
| `AIET_PUBLIC_URL` | `http://127.0.0.1:8000` | The site's public address, used for links in emails |
| `AIET_REMINDER_HOUR` | `7` | Default hour of the daily reminder in each learner's own time zone (UTC until it's known); learners can change it. `AIET_DIGEST_HOUR_UTC` still works as the old name |
| `AIET_PUSH_CONTACT` | from `AIET_MAIL_FROM` or an https `AIET_PUBLIC_URL` | `mailto:` address or `https://` site that push services can contact you at; notifications are off without one |
| `AIET_AI_ALLOWANCE` | `100` | AI requests each learner can make in any 24 hours (mock exams and custom paths count two); `0` for no limit. It applies in demo mode too |
| `AIET_AI_ALLOWANCE_UNCONFIRMED` | `30` | The allowance until a learner confirms their email address; only while email is set up |
| `AIET_AI_SITE_LIMIT` | `0` (none) | A ceiling on everyone's AI requests together in any 24 hours, against floods of new accounts; admins aren't held to it |

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
- email reminders: opt-in, daily digest at the learner's hour, sending, one-click unsubscribe
- the installable app: manifest, icons and the service worker served from the root
- notifications: subscriptions only for browser push services (including addresses that parsers read differently), one device changing hands, ten devices per learner
- stored addresses checked again before sending, redirects refused, and one learner's failure not stopping the others
- messages encrypted for the device and signed with the server's VAPID key (decrypted in the tests with the device's own key)
- forgetting gone devices, and counting rejected messages but not push service outages
- the daily reminder at the learner's hour and time zone, at most once a day, and skipped when hours late or when there's nothing to say
- notices pushed once, and not ones from before notifications were on
- reminder time and time zone settings (including names like `Asia/Calcutta` that Chrome uses), and streaks on the learner's own calendar
- appeals: one per removal, not decided by the remover or the designer, restore or keep removed, notices and log, a later removal can be appealed
- staff queues in reminders
- password reset by email: no account enumeration, single use, expiry, at most three links an hour, signing out everywhere
- changing the password and signing out other devices (both cancel outstanding reset links), and limits on wrong passwords
- data download (no secrets, no one else's data, no answers to unfinished quizzes) and account deletion
- confirming email addresses: a link at sign-up and on request, only its hash stored, single use, expiry, at most three an hour; reminder emails wait for a confirmed address; a password reset confirms it; learners who had reminder emails on keep them after an upgrade
- the AI allowance: the limit and how long it says to wait, what each kind of request costs, lessons free once taught, refunds for failed calls but not for unusable answers, a lower allowance until the email address is confirmed, admins, the site ceiling, parallel requests that can't overspend, and a new companion's first post
- AI usage records: tokens from every attempt when a fallback model answers, chat replies and the memories taken from them, scheduled posts, the data download, and records outliving a deleted account
- Insights: for admins only, and the counts behind its numbers

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
  reminders.py          What's waiting for the learner (and for staff), worked out from existing data
  staff.py              Moderation queue, appeals, path review queue, roles, decision log
  digest.py             Daily reminder email over SMTP, confirming addresses, and unsubscribe
  usage.py              The daily AI allowance, and a record of every AI request with its tokens
  insights.py           The admin Insights page
  push.py               Notifications: subscriptions, the VAPID key, encrypted delivery, the daily schedule
  account.py            Password reset and change, signing out other devices, guessing limits, data download, deletion
  content.py            Topic scores from engagement, post generation, background schedule
  companions.py         Create / remix / edit / version / follow
  auth.py, db.py        Password logins (scrypt) with cookie sessions; SQLite storage
  main.py               HTTP API + static app
static/                 Single-page web app (no build step), service worker (sw.js), manifest and icons
scripts/make_icons.py   Draws the app icons (needs Pillow)
```

**Notifications.** Each message is encrypted for one device (RFC 8291) and signed with the server's VAPID key (RFC 8292), using `pywebpush`. The server only sends to the push services browsers use (Google, Mozilla, Apple, Microsoft), so a made-up subscription can't make it send requests anywhere else. To make that hold, an address must parse to the same host in both the standard library and urllib3 (which sends the request), it's checked again before every send, and redirects aren't followed. A device whose subscription has gone (404/410) is forgotten at once. A device that rejects five messages in a row is forgotten too; push service outages don't count against it. Each notice and each daily reminder is claimed before it's sent, so it goes out at most once, and one learner's failure doesn't hold up anyone else's. An in-process loop checks every minute; the email digest has its own loop.

**Prompt structure.** The system prompt has two parts. The first holds the fixed platform rules plus the persona; it is cached. Learner-written custom instructions go here too, *after* the platform rules, so they can change a companion's voice but not the rules. The second part holds learner data that changes often: profile, memories, and the post being discussed.

**Claude requests.** Every request turns on server-side refusal fallbacks (`fallbacks: "default"`). Effort is set per task: `low` for memory and summaries, `medium` for chat and flashcards, `high` for posts, lessons, quizzes and study plans. A structured reply that was cut short or refused doesn't fit its schema; it's handled as an answer that can't be used, not as a server error.

**AI allowance.** Before Claude is called, the request's cost is reserved against the learner's last 24 hours in one write transaction, so parallel requests can't both squeeze in. The request's record is put in a context variable, so the functions that ask Claude don't need to know whose request it is: each response adds its tokens to the record in hand, summing every attempt listed in `usage.iterations` when a fallback model answers, and the chat's memory extraction, which runs after the reply, adds to the chat's record. Only failed calls to Claude (API or connection errors) give the request back. Requests turned away are counted per learner per hour, so a stream of them takes almost no room.

**Accuracy.** Lessons, questions and cards are written from each lesson's vetted key points, not from the model's general knowledge alone. Generated output is checked before it is saved: questions must have exactly 4 distinct options and a valid answer, and study plans may only reference real lesson IDs and activities.

## Next steps

- **Estimated cost** on the Insights page, from the token counts and the model's prices.
- **Allowances per learner** that admins can raise, for a class or a paid plan.
- **Server-side voices** for consistent, higher-quality speech on every device (needs a text-to-speech provider).
- **Postgres** in place of SQLite, and a job queue in place of the in-process loops (posts, emails, notifications), for multi-instance deployment.
