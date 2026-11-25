# Handoff: AI Every Time (www.aieverytime.org)

This guide is for a teammate taking over work on the AI Every Time website. Start a new Claude Code session with the GitHub repository **ABIMATHS78/abimaths** attached, then paste in this whole file. It explains what the site is, how it runs, what's in progress, and the rules to keep.

Other projects in the same repository have their own guides: `mojeedabisiga.com/HANDOFF.md` covers Mojeed Abisiga's personal website, and `newsletter-engine/` and `linkedin-engine/` are separate content tools.

## 1. What it is

AI Every Time is a 24/7 AI learning hub, founded and run by **Mojeed Abisiga** (Founder & CEO). Learners chat with AI Learning Companions from around the world (AI personas powered by Claude), read guides and cheat sheets, take courses with labs, projects, quizzes and certificates, and subscribe to paid plans.

- **Live site:** https://www.aieverytime.org. The bare domain and `http://` forward to it.
- **Code:** the `ai-every-time/` folder of `ABIMATHS78/abimaths`. It's a Python FastAPI server (`app/`) with a single-page JavaScript front end (`static/app.js`, `static/styles.css`) and an SQLite database.
- **iPhone app:** `ai-every-time/ios/`. A native shell that shows the website, in TestFlight and heading for the App Store.
- **The full manual is `ai-every-time/README.md`.** It covers every feature, setting, test and deployment step. Read it before changing anything.

## 2. Accounts the new teammate needs

Only Mojeed (the owner) can grant these. Ask him to invite **your own email address** to each one. Never share one login between two people.

| Service | What it's for | How the owner invites you |
|---|---|---|
| **GitHub**, repo `ABIMATHS78/abimaths` | All code and pull requests | Repo **Settings → Collaborators → Add people** (Write access) |
| **Claude Code** (claude.ai/code) | Working on the code with Claude | Sign in with your own account and connect GitHub. The repo must be shared with you first. |
| **Render** (dashboard.render.com), service **ai-every-time** | Hosting, deploys, environment settings, logs | **Workspace settings → Members → Invite** |
| **Stripe** | Plans, payments, subscriptions | **Settings → Team and security → Add member** (Developer or Analyst role is enough to read; Administrator to change products) |
| **Claude Console** (platform.claude.com), org "AI Every Time, Inc." | The Claude API key the site uses | **Settings → Members → Invite** |
| **App Store Connect** | iPhone app, TestFlight | **Users and Access → +** (App Manager role) |
| **Google Search Console**, property `aieverytime.org` | SEO reports | **Settings → Users and permissions → Add user** |
| **ElevenLabs** | Course intro videos and voices | Workspace members |

**To get the Staff pages on the site** (Insights, Analytics, moderation, campaigns): the owner adds your email address to `AIET_ADMIN_EMAILS` on Render (**Environment**, comma-separated). Then sign up on www.aieverytime.org with that exact address and confirm it.

## 3. How it runs

- **Hosting:** Render, from `render.yaml` at the repo root. It's one Docker web service in Frankfurt with a 5 GB disk at `/data` holding the database.
- **Deploys:** every merge to `master` that touches `ai-every-time/` (except `ios/`) deploys automatically, in about 5 to 20 minutes.
  - Because of the disk, Render stops the old version before starting the new one, so the site answers **502 for a minute or two on every deploy**. Merge several pull requests together, or at quiet times.
  - The iPhone app waits this out with an "AI Every Time is updating" screen. That's in the code, but testers only get it in a TestFlight build made after 9 Nov 2026.
- **Settings:** environment variables on Render; `README.md` → Configuration lists them all. The ones that matter most:
  - `AIET_PUBLIC_URL` must be `https://www.aieverytime.org`. Canonical links, the sitemap and the redirect away from the onrender.com address all use it.
  - `ANTHROPIC_API_KEY` is the Claude key. It was rotated in Nov 2026, so check its expiry in the Claude Console and rotate it again before then.
  - `STRIPE_SECRET_KEY` and `STRIPE_WEBHOOK_SECRET` are for payments; `AIET_TRIAL_DAYS` is the free-trial length.
  - `AIET_ADMIN_EMAILS` lists staff, as above. `AIET_SUPPORT_EMAIL`, the SMTP settings, `AIET_APNS_*` and `AIET_IOS_*` cover email, iPhone notifications and the App Store links.
- **iPhone app releases:** on GitHub, go to **Actions → iOS release → Run workflow** (`.github/workflows/ios-release.yml`). No Mac is needed. `ai-every-time/ios/README.md` has the whole process.

## 4. Working on it

1. **Run it locally or in a cloud session:**
   ```
   cd ai-every-time
   python -m venv .venv && . .venv/bin/activate
   pip install -r requirements.txt -r requirements-dev.txt
   AIET_OFFLINE=1 uvicorn app.main:app --port 8000
   ```
   `AIET_OFFLINE=1` is demo mode, with templated AI replies and no key needed. Lesson and review text in demo mode is placeholder, so never use it for screenshots meant for real people.
   - **If `pip install` fails building `http-ece`,** use a virtual environment as above rather than the system Python.
2. **Test before every pull request:** `python -m pytest -q` from `ai-every-time/`. There are about 408 tests, and all must pass.
3. **Use a branch and a pull request.** Never push to `master`, and never force-push. The owner reviews and merges.
4. **Check the live site after a deploy.** For example: `curl -s -o /dev/null -w "%{http_code}\n" https://www.aieverytime.org/healthz`.

## 5. Rules to keep

- **Secrets never go in the repo, a pull request, a commit or a chat.** Not the Claude key, Stripe keys, the APNs/App Store `.p8` keys, SMTP passwords or staff tokens. They live only in Render's Environment and GitHub's repository secrets. If one leaks, rotate it.
- **Learners' data stays on the server.** Never put exports of users or subscriptions into prompts, the repo or a chat.
- **Course intro videos are made with ElevenLabs only** (not Higgsfield). Each video needs its own setting, camera move and co-instructor reaction, so they don't all look alike.
- **Never repeat an ElevenLabs generation to "retry".** Each call is charged.
- **SEO:** the sitemap must list only pages that load directly (200), aren't marked noindex, and name themselves as canonical. `tests/test_seo.py` checks this, so keep it passing. Old-site links (`/reporter/…`, `/article/…`, `/company/…`, `/login`) are redirected or marked gone in `app/pages.py`. Don't remove those rules.
- **The iPhone app must not link to payments elsewhere** (Apple's rules). That's why pricing and the Events tab are hidden inside the app (`inApp` in `static/app.js`).
- **Writing style:** plain, friendly English. Learners see AI companions, so never present them as real people. Course pages say the products they name (JetBrains, Cisco and so on) haven't endorsed them.

## 6. Where things stand (25 Nov 2026)

**Recently finished and live:**
- Course intro videos for 8 JetBrains courses: Python, Java, Kotlin, Web, Go, Rust, Teach Programming in the IDE, and SQL. They're in `static/course-intros/`, and `scripts/course-intros/README.md` explains how they're made.
- SEO:
  - sitemap, robots.txt, `llms.txt` and structured data;
  - server-rendered page content for crawlers;
  - redirects for the old site's addresses;
  - fixes for the Google Search Console reports deployed on 12 Nov, with "Validate fix" started in Search Console.
- **About page** (`/about`): the mission, the story, and Mojeed's bio and photo. The text is in `app/about.py`.
- **Events tab** (`/events`, `/events/gdai-2026`): a copy of DataGlobal Hub's pages for the Global Data & AI Virtual Tech Conference 2026, 20–22 November, 9:00 AM – 6:00 PM CST.
  - Files are in `static/events/`, and `scripts/events/README.md` explains how to refresh them.
  - The agenda PDF and Excel are the CST versions.
  - Hand-made changes to the copy are listed in `CORRECTIONS` in `scripts/events/build.py`.
- **Payments:** Stripe checkout with a free trial on the first paid plan, plan changes, the billing portal and webhooks (`app/billing.py`).

**Open items, roughly in priority order:**
1. **Events page after the conference:** the conference was 20–22 Nov. Decide with the owner what `/events` should say now (recordings, next year's dates), and update or archive the countdown page.
2. **"Check Your Eligibility" link** on `/events/gdai-2026` points to `/events/gdai-2026/eligibility`, which doesn't exist yet, so it shows "Page not found". Build it if it's still needed, or remove the link from the copy.
3. **GDAI 2025 recordings:** the video snippets on the conference page open DataGlobal Hub's recordings page. Copy it here only if the owner wants.
4. **Course intro videos for the other courses:**
   - Six JetBrains courses are ready to render but were blocked when the ElevenLabs credit ran out: pandas, Statistics, Machine Learning Foundations, and the three guided projects. That's about 520,000 credits.
   - The other ~89 courses have no video yet. That's about 85,000 to 90,000 credits per 15–16 s video at 720p, so agree the budget with the owner first.
5. **TestFlight build:** make a new one (Actions → iOS release) if it hasn't been done since 9 Nov, so testers get the "updating" screen.
6. **Search Console:** check the Page indexing report every week or two. The old `/article/…` news stories answer 410 and will drop out over time; anything else that grows needs a look.

## 7. Useful places in the code

| Path | What's there |
|---|---|
| `app/main.py` | All web addresses (API and pages), app setup |
| `app/pages.py` | Each page's title, description, status and old-address redirects |
| `app/seo.py` | Sitemap, robots.txt, llms.txt, structured data, crawler HTML |
| `app/billing.py`, `app/plans.py` | Stripe, plans, trials, the paywall |
| `app/courses.py`, `app/catalog.py`, `app/course_data/` | Courses (one JSON file each) |
| `app/about.py`, `app/events.py` | About page and Events data |
| `static/app.js`, `static/styles.css` | The whole front end |
| `tests/` | Tests; `test_seo.py`, `test_pages.py` and `test_courses.py` cover most of the above |
