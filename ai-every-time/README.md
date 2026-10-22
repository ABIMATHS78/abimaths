# AI Every Time

<img src="static/icons/logo-256.webp" width="112" height="112" alt="AI Every Time logo">

The 24/7 AI learning hub. Learners follow, chat with and learn from always-available **AI Learning Companions**. Each companion represents a country and has its own face, personality, expertise and writing voice. Learners can also **design their own companion** when they create an account, and change it whenever they like.

It includes the companion core (starter companions, the companion designer, chat with memory, self-publishing content) and structured learning taught by those companions: courses in the style of Coursera's, with tools shown step by step, hands-on labs, capstone projects and certificates; learning paths, lessons, quizzes, mock exams, flashcards, summaries, study plans and progress tracking. It installs on phones and computers like an app, reminds learners at the time they choose in their own time zone, keeps AI costs predictable with a daily allowance per learner, and gives learners control of their account and data. Admins get an Insights page.

## What's in it

| Area | What it does |
|---|---|
| **Starter companions** | 103 companions from every continent, listed in `app/starter_companions.csv`: name, country, region, photo and follower count. Each has a photo, a bio, expertise areas, personality sliders, a teaching style, a writing style, a signature phrase and a sample line in their voice (`app/starter_companions.py`), and a drawn face that stands in when the photo can't be shown. Three are modelled on real people: they say they're AI companions, don't speak for the person, can't be remixed, and show initials rather than a drawn face. See [Changing the starter companions](#changing-the-starter-companions). |
| **Follower counts** | Every starter companion shows its follower count: the number in the CSV, plus everyone who follows it on the site. Companions learners design don't show one. |
| **Onboarding** | Sign-up asks for the learner's AI level and, optionally, what they want to learn and what they're into (hobbies, work, anything they love); both can be changed on the profile. Then learners pick a starter companion as-is, **customise** one (a remix: the original stays unchanged, and its photo comes along once the server has stored it; changing its face shows the drawn face instead), or **design one from scratch**. Starters can be browsed by region or searched by name, country or topic, most followed first. New learners start out following the most followed companion of each region, so their feed isn't empty. |
| **Companion Studio** | The designer, with a live preview. **Identity**: name, country/cultural voice (with its flag, for any country), tagline, background, reply language, interests. **Face**: a photo, or a drawn face: skin tone, hair style and colour, expression, accessory, outfit, background. A companion with a photo shows whichever its designer chooses. Changing the drawn face shows it, since a photo can't change with it, and the photo is kept for switching back. **Expertise**: up to 5 areas. **Personality**: 6 sliders. **Teaching style**: Socratic, step-by-step, storytelling, examples-first or challenge-driven. **Writing style**: reply length, level, emoji, analogies, local examples, signature phrase, sample line. **Custom instructions**. |
| **Change it later** | Learners can edit their companion at any time. Every save is kept as a version (last 50), and any version can be restored. Starter companions can't be edited, only remixed. |
| **Companions start the conversation** | When a learner opens a chat with a companion for the first time, the companion speaks first: it greets them by name and opens a conversation from their interests, linking one to an idea from its expertise and asking a question to reply to. Without interests it starts from their learning goals, and without those it asks what they're into. Arriving from one of its posts, it starts from that. Claude writes it in the companion's voice; it's kept as the chat's first message, and it costs the learner nothing (each companion does it once per learner). In demo mode, or if Claude can't be reached, a template does the same. |
| **Chat that remembers** | Replies stream in the companion's voice (built from the persona by `app/llm.py`), with the learner's level, goals and interests to draw examples from. After each exchange the companion saves lasting facts about the learner (goals, level, struggles). Learners can see these and remove any of them. |
| **Reactions** | In chat, on the companion page and in the lesson tutor, the learner and the companion react to each other's messages: 👍 ❤️ 😂 😮 😢 🙏 🎉 💡. Learners react to any reply with the 🙂 button beside it (choosing the same one again takes it back). The companion reacts to the learner's message when a friend naturally would, such as ❤️ for thanks or 🎉 for good news: it decides as it starts its reply, at no extra cost, and the reaction is taken off the reply before anyone sees it (`app/reactions.py`). The companion also sees the learner's reaction to its last reply. Reactions are kept with the conversation and included in the data download. |
| **Guides** | 1,162 tutorials and cheat sheets that 102 of the starter companions wrote on aieverytime.org between 1 September and 15 October 2026. They're in the feed of anyone who follows their author, each on a page of its own, and in the **Guides** library: by region like the site's education page, by kind, level and topic, or searched by title or companion. Each shows its own graphic from the site, with the image generator's spelling mistakes corrected in the picture itself. Their text was repaired on the way in (see [The guides](#the-guides)). Learners can ask the author about any guide, quiz themselves on it or turn it into flashcards. |
| **Courses** | 103 courses in the style of Coursera's, across generative AI, machine learning, AI engineering and MLOps, data science and analytics, data engineering, computer science and system design, software development with JetBrains IDEs, networking that follows Cisco's CCNA exam topics, cybersecurity, cloud and DevOps, AI for business, responsible AI, AI in industry, and robotics. Thirty of them follow the outlines of the reporters' courses on aieverytime.org. Each is taught by one to three starter companions, its instructors, and has modules of lessons, hands-on labs and graded quizzes, usually a capstone project, a week-by-week study plan, a mock exam and flashcards for the whole course, and a certificate. Guided projects are short hands-on courses of one module. The catalogue is searched and filtered by subject, level, type, hands-on labs, JetBrains IDEs and Cisco alignment, and 17 Professional Certificates and Specializations group courses into series, from Data Scientist, Data Engineer, MLOps Engineer and NLP and LLM Engineer to Computer Science Foundations. Each course's page has what you'll learn, the skills and tools, the syllabus, the instructors, learners' ratings and, once enrolled, your progress. See [The courses](#the-courses). |
| **Course player** | The course's outline beside the item in hand, each item with its status. Lessons are written by their module's instructor in their own voice, from vetted key points, and cached like any lesson; any of the course's instructors can teach them instead. Beside every lesson, lab and project, the learner can ask the instructor about it; lessons can also be turned into a quiz, flashcards or a summary, or read aloud. |
| **Tool walkthroughs** | Lessons show their tool step by step, as a screencast would: PyCharm, IntelliJ IDEA, WebStorm, GoLand, Rider, CLion, DataGrip, Cisco Packet Tracer, Wireshark, a terminal or a chat assistant. Each step has its keyboard shortcut for macOS or for Windows and Linux (the learner chooses, and the choice is remembered), code to copy and what should happen. ▶ Play goes through the steps in the instructor's voice. The JetBrains shortcuts are checked against a table of the IDEs' default keymaps. A lesson can also have a recorded video (YouTube or Vimeo). |
| **Labs and projects** | A lab has what you need, starter files to download, steps to tick off and the checks it must pass. Learners do it on their own computer, paste in their work (code, command output, configuration) and the module's instructor reviews it against each check, saying what to fix. Capstone projects are scored out of 100 against their rubric. Every attempt is kept with its review. The learner's work is given to Claude as work to assess, not instructions to follow. |
| **Certificates** | Completing every lesson and passing every module quiz (70%), every lab and the project (70 points) earns the course's certificate, made out to the name the learner chooses. It has a page of its own that anyone with the link can open to check it, prints or saves as a PDF, and adds to LinkedIn. Finishing every course in a track earns the track's certificate too. They say what they are: not an academic credit, nor a certification from any company whose products a course teaches. |
| **Content → relationship** | Companions publish short lessons on their own. Every post has an *"Ask {name} about this"* button that opens a chat with the post as context. |
| **Learning paths** | 5 paths of 5 lessons: AI Foundations, Understanding LLMs, Prompting & AI Agents, Responsible AI, and AI for Business. Each lesson has objectives and vetted key points (`app/curriculum.py`). The learner's companion teaches each lesson in its own voice; learners can switch to any companion they follow. A lesson is written once per companion and then cached. When the companion's voice changes (anything but its face, read-aloud voice and private instructions), each lesson is written again in the new voice the next time it's opened. |
| **Interactive tutoring** | Every lesson page has a tutor panel where the learner can ask the companion about that lesson. The companion answers using the lesson's key points. |
| **Quizzes** | 5 multiple-choice questions from any lesson, post or conversation, with an explanation for each answer. Correct answers stay on the server until the learner submits. |
| **Mock exams** | 12 questions across a whole path, a 20-minute timer, a 70% pass mark, and a score breakdown by lesson. |
| **Flashcards** | Decks from any lesson, path, post or conversation. Reviews use spaced repetition (Leitner boxes 1-5: again / good / easy), so each card returns just before it would be forgotten. |
| **Summaries** | Key takeaways and a suggested next step for any lesson, post or conversation. |
| **Study plans** | The learner sets weekly time, number of weeks and which paths to focus on. The companion plans each week from their goals, memories, finished lessons and weak quiz scores. Plan items tick off as the learner completes them. |
| **Progress tracking** | Day streak, last 7 days of activity, lessons completed, quiz average, best exam score, cards mastered and due, progress per path, lessons worth reviewing (best score under 70%), and the next lesson. |
| **Content intelligence** | Views, likes, "helpful" marks, shares and follow-up questions are weighted (1, 3, 4, 5 and 6) and averaged per topic (`app/content.py`). Companions steer their next posts toward the topics learners value most. |
| **Community companions** | Learners can publish a companion they designed. Others can then find it in the Community section, follow it, chat with it and remix it. What stays private: the designer's custom instructions (they only shape the designer's own chats), and everyone's chats and memories. Learners can report a community companion (impersonation, harmful, misleading, spam, other); after 3 reports it is hidden for review. Unpublishing removes it, and its posts, from everyone else's view. |
| **Photo avatars** | Upload a PNG, JPEG or WebP (up to 2 MB) to show instead of the drawn face, and switch between the two at any time. Files are identified by their content, not their name, and served so they can't run as a web page. A photo is only visible to people who can see the companion, and only while it's the picture shown: a photo kept while the drawn face is shown is seen by its designer only. Starter companions' photos are fetched once from their addresses in the CSV and kept as small square images (320 pixels, a few kilobytes), so pages with many faces load fast on phones. Until a photo is kept, browsers load it from its own address, and the drawn face (initials, for a companion modelled on a real person) stands in if that fails too. |
| **Voice** | Each companion has voice settings: on/off, accent, speed and pitch. Learners can have replies read aloud (🔊 on any reply, or "Read aloud" for every reply) and can dictate questions (🎤). This uses the browser's own speech features, so there's no server cost; how it sounds depends on the voices installed on the device. |
| **Custom learning paths** | A learner names any topic and their companion designs a 3-8 lesson path for them, with objectives and key points. It works with everything else: lessons, tutoring, quizzes, mock exams, flashcards, study plans and progress. These paths are private and marked as not expert-reviewed. |
| **Staff roles** | Learners, reviewers, moderators and admins. Emails listed in `AIET_ADMIN_EMAILS` are always admins, so a new deployment can't lock itself out. While email is set up, such an address has to be confirmed first, so someone who signs up with it before its owner doesn't become an admin (the owner can take the account back with a password reset); admins from before confirmation existed keep their role. Admins assign the other roles on the Staff page; nobody can change their own role. |
| **Moderation** | Moderators see every reported community companion: its report count, reasons and details. They can restore it (if hidden), dismiss the reports (if still visible), or remove it from the community for good, with a required note. The designer keeps a private copy and can't republish it. Every decision is logged and the designer is told the outcome. |
| **Appeals** | The designer of a removed companion sees why on its page and can ask for a second look, once per removal. Appeals go to the Staff page's Appeals tab. Neither the moderator who made the removal nor the designer (if they're staff themselves) can decide one. Another moderator either restores the companion to the community or keeps it removed, with a required note. The decision is logged and the designer is told. |
| **Shared paths, expert reviewed** | A learner can submit a custom path for sharing. A reviewer checks it and can correct its title, description, lesson titles, objectives and key points, but not the number of lessons, so learners' progress still lines up. Corrections clear the cached lesson text, so companions re-teach from the corrected facts. Approved paths appear for every learner with an "Expert reviewed · suggested by …" badge. Reviewers can reject (with a note) or later unshare, and can't approve their own paths. |
| **Reminders** | A "Today" panel on the feed and a bell in the sidebar list what's waiting: a streak at risk, flashcards due, the next study-plan item (or one that's overdue), an unfinished quiz, a lesson to continue, new posts from followed companions, and notices about moderation and review decisions. Staff also see the work waiting in their queues: reported companions, appeals they can decide, and paths to review. Learners pick the hour of their **daily reminder**, which follows their own time zone, as do streaks and study days. They can get it by email (any SMTP server, to a confirmed address) or as a notification. The email carries one-click unsubscribe (RFC 8058), and its unsubscribe page asks for confirmation so email link-scanners can't unsubscribe people by accident. |
| **Installable app and notifications** | The site installs on phones and computers like an app (manifest, icons and a service worker). Learners turn notifications on for each device from their profile and can send a test. The daily reminder arrives from their own companion; a reminder that would come more than 3 hours late is skipped rather than sent at night. Notices about decisions on their companions and paths arrive as they happen. On iPhone and iPad, notifications work once the app is on the Home Screen, and the profile explains how. Signing out turns notifications off for that device. |
| **Confirmed email addresses** | Reminder emails only go to an address its owner has confirmed (double opt-in), so nobody can sign someone else up for them. Sign-up emails a confirmation link, and learners can ask for another from their profile. One address gets at most 3 links an hour and 10 emails of this kind (confirmation and password-reset links) a day, however many accounts ask, so nobody can use the site to flood someone's inbox; until it's confirmed, those emails leave out the name typed at sign-up. The link works once, for 7 days, only a hash of it is stored, and it opens a page with a button, so mail scanners that follow links can't confirm for anyone. Resetting the password by email also confirms the address. Learners who had reminder emails on before confirmation existed keep getting them. |
| **Daily AI allowance** | Each learner can make 100 AI requests in any 24 hours (configurable). A chat reply, a lesson the first time it's taught, a quiz, flashcards, a summary, a study plan, a post or a lab review counts one; a mock exam, a custom path or a project review counts two. The profile shows what's left. When it runs out, the learner is told when they can ask again, and a chat message that was turned away goes back in the text box. A call to Claude that fails is refunded; an answer that came back but couldn't be used still counts, so failures can't be turned into free requests. Until learners confirm their email address the allowance is lower (30), which makes floods of throwaway accounts less useful. Admins have no limit, and on the Staff page's People tab they can give a learner an allowance of their own, for a class or a paid plan. An optional ceiling covers everyone together; while email is set up it only pauses learners who haven't confirmed their address, so throwaway accounts can't pause everyone else. |
| **Insights** | Admins see how the site is doing on the Staff page: learners (new, studying this week, confirmed, with notifications), learning this week, AI requests per day over the last 14 days, requests, tokens and estimated cost by kind, requests turned away and failed calls, the limits in force, the topics learners value most and the most followed community companions. |
| **Account and privacy** | Forgotten passwords are reset with an emailed link: it works once, for an hour, and only a hash of it is stored. Learners can change their password and sign out other devices; both also stop notifications on the other devices. After 10 wrong passwords for an email in 15 minutes, sign-in pauses for that email. **Download my data** gives one JSON file with everything the account holds, including its AI requests (answers to unfinished quizzes stay hidden). **Delete my account** (password required) removes it all, including the companions the learner designed, any paths they shared with the community, and their certificates, whose pages stop working. |

## Try it in your browser

Open **[AI Every Time in GitHub Codespaces](https://codespaces.new/ABIMATHS78/abimaths)** and create the codespace. It installs everything, starts the app and opens it in a new tab, with nothing to install on your computer. When you create it you can give two optional values: `ANTHROPIC_API_KEY` for real AI replies (without it the app runs in demo mode), and `AIET_ADMIN_EMAILS` set to your email address, so that signing up with that address shows the Staff pages, including Insights. If the tab doesn't open, find port 8000 in the **Ports** panel and open its address. That address is private to your GitHub account. To try it on your phone, right-click the port and set its visibility to public, but sign up with your admin address first: anyone with a public address can use the app, and the AI on your key. The codespace caps learners' AI requests together at 200 a day (admins aren't held to it), and email isn't set up there, so there's no smaller allowance for new accounts. Starter companions' own posts come on top: each writes its first post when someone first follows it, at most 20 an hour. Set the port back to private when you're done.

## Run it

You need Python 3.11 or newer.

```bash
cd ai-every-time
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # leave unset to run in offline demo mode
export AIET_ADMIN_EMAILS=you@example.com # optional: sign up with this address to see the Staff pages
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000.

**If AI isn't working.** When Claude rejects the API key, the Claude account has a billing problem, it can't use the model, or the key works across workspaces and `ANTHROPIC_WORKSPACE_ID` doesn't name one, the server says so in one line of its log, admins see a banner with the fix, and learners are told AI isn't available. Check a key with `python -c "import anthropic, os; w = os.environ.get('ANTHROPIC_WORKSPACE_ID'); print(anthropic.Anthropic(default_headers={'anthropic-workspace-id': w} if w else None).models.retrieve('claude-opus-5-5').id)"` (it prints the model's name when the key works), set a working one and restart the app. In a codespace, update the `ANTHROPIC_API_KEY` secret (and `ANTHROPIC_WORKSPACE_ID`, if your key needs one) at github.com/settings/codespaces, then stop and reopen the codespace. Keep keys in those secrets, never in the code or a chat.

Without an API key, the app runs in **demo mode**: it works end to end, but companions give templated replies and posts. A banner says so.

Notifications need two things. First, a contact address that browsers' push services can reach you at: `AIET_PUSH_CONTACT`, or else the `AIET_MAIL_FROM` address or an https `AIET_PUBLIC_URL`. Second, HTTPS in production (browsers allow notifications on `localhost` for testing). The server makes its own push (VAPID) key the first time it's needed and keeps it in the database.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Turns on real AI chat, memory and content |
| `ANTHROPIC_WORKSPACE_ID` | — | Only for an API key that works across Claude workspaces (one made without choosing a workspace): the workspace to use, an ID starting `wrkspc_` from the [Workspaces](https://platform.claude.com/settings/workspaces) page of the Claude Console. A key made for one workspace doesn't need it |
| `AIET_MODEL` | `claude-opus-5-5` | Claude model used for everything |
| `AIET_AUTOGEN_MINUTES` | `0` (off) | Minutes between automatic posts from starter companions: each run, up to 8 that learners follow, those that posted longest ago first |
| `AIET_GUIDES` | `app/guides.json.gz` | The guides to load at start-up (see [The guides](#the-guides)); `0` loads none |
| `AIET_FETCH_PHOTOS` | `1` (on) | Fetch starter companions' photos and keep small copies. With `0`, browsers load them from their addresses in the CSV every time |
| `AIET_DATABASE` | `ai_every_time.db` | SQLite file |
| `AIET_SECURE_COOKIES` | off | Set to `1` behind HTTPS |
| `AIET_HISTORY_TURNS` | `20` | Past conversation turns sent with each message |
| `AIET_OFFLINE` | off | Set to `1` to force demo mode |
| `AIET_ADMIN_EMAILS` | — | Comma-separated emails that are always admins (once confirmed, while email is set up) |
| `AIET_SMTP_HOST`, `AIET_SMTP_PORT`, `AIET_SMTP_USER`, `AIET_SMTP_PASSWORD`, `AIET_SMTP_STARTTLS` | — / 587 / — / — / on | SMTP server for reminder, confirmation and password-reset emails; email is off while `AIET_SMTP_HOST` is unset |
| `AIET_MAIL_FROM` | — | Sender address for emails (required for email) |
| `AIET_PUBLIC_URL` | `http://127.0.0.1:8000` | The site's public address, used for links in emails |
| `AIET_REMINDER_HOUR` | `7` | Default hour of the daily reminder in each learner's own time zone (UTC until it's known); learners can change it. `AIET_DIGEST_HOUR_UTC` still works as the old name |
| `AIET_PUSH_CONTACT` | from `AIET_MAIL_FROM` or an https `AIET_PUBLIC_URL` | `mailto:` address or `https://` site that push services can contact you at; notifications are off without one |
| `AIET_AI_ALLOWANCE` | `100` | AI requests each learner can make in any 24 hours (mock exams and custom paths count two); `0` for no limit. It applies in demo mode too |
| `AIET_AI_ALLOWANCE_UNCONFIRMED` | `30` | The allowance until a learner confirms their email address; only while email is set up. `0` turns it off |
| `AIET_AI_SITE_LIMIT` | `0` (none) | A ceiling on everyone's AI requests together in any 24 hours, against floods of new accounts. Admins aren't held to it, nor, while email is set up, are learners who have confirmed their address |
| `AIET_PRICES` | built in | Prices for the cost estimate on Insights, as JSON: `{"model": [input, output, cache read]}` in US dollars per million tokens. Changes the built-in list prices or adds models |

### Changing the starter companions

`app/starter_companions.csv` lists them, one row each, with the columns `name`, `slug`, `country`, `region`, `background`, `ai learning companion image` and `followerCount`. Edit it (in a spreadsheet, saved as CSV in UTF-8) and restart the app:
- **Names, countries, regions, photos and follower counts** change as soon as the app restarts. Bios use the name from the CSV. A changed photo address is fetched again.
- **The slug is the companion's identity.** Changing it makes a new companion and retires the old one (see below), so leave slugs as they are.
- **A new row** needs a unique `slug`, and a country in `COUNTRY_CODES` in `app/starter_companions.py` for its flag. Its bio, expertise, personality and style come from its entry in `PERSONAS` in the same file, keyed by the slug; without one it has a plain persona. A companion modelled on a real person also goes in `REAL_PEOPLE` there, so it says so, can't be remixed and shows initials rather than a drawn face.
- **Removing a row** retires that companion. Learners who talked to it, or have it as their companion, keep it with their chats, under "Earlier companions"; nobody else sees it, and it stops posting. Everyone who followed it starts following the featured companions (the most followed in each region). Putting the row back brings it back.
- **Photos** can be any address that serves a PNG, JPEG or WebP picture. A Google Drive sharing link works if the file is shared with anyone who has the link; a direct address on the same host as the others is more reliable.
- A row the app can't read (a duplicate slug, a follower count that isn't a number) stops it from starting, with the line number, rather than removing that companion.

### The guides

The guides come from an export of aieverytime.org's articles table: those in education mode (tutorials and cheat sheets) published since a date you choose. `scripts/import_guides.py` turns the export into `app/guides.json.gz`, which the app loads when it starts:

```bash
python scripts/import_guides.py articles.csv --since 2026-09-01
```

It asks the site once which companion each author is (or give it a saved copy of the site's reporters list with `--reporters`). Each guide becomes a post by its companion, dated when the site published it. When the file changes, the app updates guides in place, so their likes, views and chats stay, adds new ones and removes any no longer in it. Guides aren't counted as new posts for learners: they're a library, not news.

On the way in, the script repairs what the site's own pipeline had damaged:
- **Capitals.** Words in capitals of four letters or more had been put in title case: "Gdpr", "Cuda", and SQL in code examples such as `Create Table … Varchar(100) NOT Null`. Acronyms, SQL and Dockerfile keywords and HTTP methods are back in capitals; ordinary words that look like them ("Select a model") are left alone.
- **Hyphens** had become spaces: "multi agent", "real time data", "end to end". They're back where words join to describe what follows ("real-time data"), not where they don't ("in real time").
- **Repeats and leftovers.** Sections repeated word for word (1.6 MB of them, in 385 guides), stray bold markers ("today.**."), headings that repeated the title, formulas written in LaTeX (now plain characters, like O(N²)), and a few spelling slips.
- **Titles, summaries and topics** were edited for punctuation and spelling ("Asyncio for AI: Building High-Performance Concurrent Applications"), and each guide filed under the topic that fits it. These edits are kept in `scripts/guide_edits.json`, by article id, and applied again whenever the script runs.

Each guide shows the graphic the site made for it, with the misspelled and garbled words the image generator put in it ("Warewarusing", "httx", "PREEQUISITES") corrected in place: everything else in the picture is as the site has it. The corrected graphics are in `static/guides/` (about 55 KB each), and what was corrected in each, and how, in `scripts/graphics/corrections/`; [scripts/graphics/README.md](scripts/graphics/README.md) tells how they're made and how to make them again. A guide whose graphic hasn't been corrected yet shows the site's own copy, and one the site gave no graphic (10 of them) has none.

### The courses

Each course is a JSON file in `app/course_data/`, named after its slug, and `tracks.json` there groups courses into Professional Certificates and Specializations. [scripts/courses/README.md](scripts/courses/README.md) explains the format and the standards a course meets, with a complete example (`python-programming-pycharm.json`), and `python scripts/courses/check.py` checks them all. The app reads them when it starts, and won't start with one that doesn't fit, saying which file and why.

- **Instructors** are starter companions, named by slug, the lead first; a module can name which of them teaches it. The three companions modelled on real people don't teach courses.
- **Lessons** are written by the instructor from the lesson's objectives and key points, so those must be accurate. Walkthroughs are shown exactly as written. When a lesson's objectives, key points or walkthrough change, its written text is forgotten when the app next starts, and written again in the new version.
- **Videos.** Give a lesson a `video`, a `https://www.youtube-nocookie.com/embed/…` or `https://player.vimeo.com/video/…` address, and it plays above the lesson. Recording a lesson's walkthrough in its IDE, PyCharm for example, makes it a screencast.
- **Progress follows positions**: a lesson's id is `course-{slug}-{module}-{lesson}`, so add lessons at the end of a module and modules at the end of a course, rather than reordering or removing those learners have started.
- **Products and exams.** A course that uses a JetBrains IDE or Cisco's tools says on its page that it's independent of them. Courses aligned with an exam follow its published topics; they don't give that certification.

## Deploy it

**Upgrading from the first eight starter companions** (Adaeze, Haruto, Camila, Arjun, Lena, Wanjiru, Maya and Sofía): the first start of this version retires them, as above. Learners who talked to one keep it, and their chats; nobody loses any data.

The app is one process with a SQLite database, so it runs as a single instance with storage that's kept between restarts. The Docker image works anywhere that runs containers with a persistent volume, such as a small virtual server, Fly.io, Railway, or Render with a disk. Run one copy per database: the reminder emails, notifications and scheduled posts run inside it.

```bash
cd ai-every-time
docker build -t ai-every-time .
docker run -d --name ai-every-time --restart unless-stopped -p 127.0.0.1:8000:8000 -v aiet-data:/data \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e AIET_PUBLIC_URL=https://learn.example.com -e AIET_SECURE_COOKIES=1 \
  -e AIET_ADMIN_EMAILS=you@example.com \
  -e AIET_SMTP_HOST=smtp.example.com -e AIET_SMTP_USER=... -e AIET_SMTP_PASSWORD=... \
  -e AIET_MAIL_FROM="AI Every Time <hello@example.com>" \
  ai-every-time
```

- Serve it over HTTPS (your host's, or a reverse proxy such as Caddy or nginx in front of port 8000), with `AIET_SECURE_COOKIES=1` and `AIET_PUBLIC_URL` set to the public address. Notifications need HTTPS. The example publishes port 8000 only on the machine itself, for a proxy running there; publishing it more widely would also offer the site without HTTPS.
- The app runs as an ordinary user (uid 10001). When the container starts as root, it first gives `/data` to that user, so a disk the host attached as root's works too. It always runs one process, whatever `WEB_CONCURRENCY` says.
- With email set up, sign up with the admin address and confirm it from the email; then the Staff pages appear.
- Learners' data is in the `/data` volume (`/data/ai_every_time.db`). Back it up while the app runs with `docker exec ai-every-time python -c "import sqlite3; sqlite3.connect('/data/ai_every_time.db').backup(sqlite3.connect('/data/backup.db'))"`, then copy `backup.db` somewhere safe.
- `/healthz` answers when the app and its database are up, and with 503 when the database file is missing or unreadable, for your host's health checks; the image checks itself with it too.
- Set `AIET_AI_ALLOWANCE` (and `AIET_AI_SITE_LIMIT`) to keep costs predictable, and watch the estimated cost on Insights. It's an estimate: when a fallback model answers after the first one declined, the whole request is priced at the fallback model's rates.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests run in offline mode and cover:
- seeding the starter companions
- the 103 starter companions: every CSV row has a persona and a flag, names and counts from the CSV, featured companions, rows that can't be read, follower counts on starters only, the learner's own companions first then the most followed, the welcome page's faces, retiring the earlier eight on an older site without losing anyone's chats (and bringing one back), real people (no remix, initials), photos fetched once and kept small (Google Drive links, failures, retries, turned off, never twice at once, at most four at a time), browsers keeping starters' faces but not learners' photos, remixes keeping a starter's photo, the photo or the drawn face as the designer chooses (the photo kept, and hidden from others while the drawn face is shown), a flag for every starter's country, a first post when first followed (capped, and not retried for an hour), and scheduled posts a few at a time
- the guides: loaded as posts by their companions (dated when published, authors who aren't starters skipped), updated in place when the file changes (likes kept, removed ones gone), not counted as new posts, the library's filters (region, kind, level, topic, words with or without accents, companion), lists newest first a page at a time, the whole guide on its own page, in chat and in study tools, and each guide's corrected graphic (or the site's, until it's corrected)
- importing the guides: acronyms, SQL, Dockerfile keywords and HTTP methods back in capitals without touching ordinary words or strings, hyphens back where they belong and not where they don't, formulas, repeated sections, headings' capitals, and reading an export (with each guide's graphic)
- reactions: the learner's (one per message, written any way, only to the companion's messages in their own chats), the companion's (taken off the start of its reply however it streams in, invalid ones dropped, a sentence in a tag left as written, demo-mode rules), a reply that's only a reaction, the companion seeing the learner's reaction, and the data download
- onboarding
- designing, editing and restoring a companion, and every setting reaching the system prompt
- remixing
- access rules (no one can see, edit or chat with another learner's companion)
- input validation
- streamed chat and memory
- companions starting chats: from the learner's interests, or their goals, or asking; from a post; once per chat and never after the learner has begun; kept in what Claude sees of the conversation, after the cached part of the prompt; free to learners but recorded; the template when Claude can't be reached; interests at sign-up, on the profile and in the data download
- engagement-driven topic scores
- lessons (taught by the chosen companion, cached per companion, written again when its voice changes but not its face)
- quizzes (answers hidden until submit, grading, no double submission)
- mock exams with a per-lesson breakdown
- flashcard spaced repetition
- summaries and lesson tutoring
- courses: every course file fits the format, its instructors are starter companions (none modelled on a real person), its JetBrains shortcuts are in the verified table, and it names no Claude model; ids that can't be confused; a broken file named when the app starts; the catalogue and a course's page; lessons taught by their module's instructor, with the walkthrough, and written again when the course changes; labs (enrolment first, reviewed against each check) and their starter files; chat about a lab; module quizzes and course mock exams; finishing a course, and its certificate (the name on it, its public page); ratings and leaving a course; tracks; an instructor's page listing their courses; the data download
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
- allowances that admins give a learner (including no limit, and back to the usual), logged and told to the learner
- the cost estimate: list prices, dated model names, models without a price, and `AIET_PRICES`
- the health check
- a rejected API key: checked at start, one line in the log, a banner for admins, a plain message for learners, start-up posts stopped after the first refusal, and nothing counted against anyone
- link limits that follow the address, so deleting and remaking an account doesn't reset them, and that hold for requests made at the same moment; no sign-up name in mail to an unconfirmed address
- tokens recorded for answers that were refused or cut short, with the real SDK's request and response shapes in the tests' stand-in for Claude
- lessons that other sites can't have written, the ceiling not pausing learners who confirmed, refusal counts outliving a deleted account, `AIET_AI_ALLOWANCE_UNCONFIRMED=0`, admin emails that need confirming, and no daily notification nagging to confirm

## How it's built

```
app/
  persona.py            Persona schema shared by starter and learner-designed companions
  starter_companions.csv The 103 starter companions: names, countries, regions, photos, follower counts
  starter_companions.py How each starter teaches and writes, and reading the CSV
  photos.py             Starter companions' photos: fetched once, cut to small squares, kept in the database
  reactions.py          Emoji reactions in chat, and taking the companion's off the start of its reply
  guides.json.gz        The education guides from aieverytime.org, made by scripts/import_guides.py
  guides.py             Loading the guides as posts by their companions, and keeping them up to date
  avatar.py             Builds each face as an SVG from its settings (validated colours and options only)
  llm.py                Claude: system prompt from the persona, streamed chat, memory extraction, post writing
  curriculum.py         Learning paths, lessons, objectives and vetted key points
  catalog.py            The course format, and the courses and tracks in course_data/ (one JSON file per course)
  courses.py            Courses API: catalogue, enrolling, progress, lab and project reviews, certificates, tracks, ratings
  tutor.py              Claude: lessons, quiz/exam questions, flashcards, summaries, study plans, lab and project reviews
  learning.py           Learning API: paths (curated + custom), lessons, assessments, decks and reviews, plans, progress
  reminders.py          What's waiting for the learner (and for staff), worked out from existing data
  staff.py              Moderation queue, appeals, path review queue, roles, decision log
  digest.py             Daily reminder email over SMTP, confirming addresses, and unsubscribe
  usage.py              The daily AI allowance, and a record of every AI request with its tokens
  insights.py           The admin Insights page
  push.py               Notifications: subscriptions, the VAPID key, encrypted delivery, the daily schedule
  account.py            Password reset and change, signing out other devices, guessing limits, data download, deletion
  content.py            Topic scores from engagement, post generation, the feed and the guides library
  companions.py         Create / remix / edit / version / follow
  auth.py, db.py        Password logins (scrypt) with cookie sessions; SQLite storage
  main.py               HTTP API + static app
static/                 Single-page web app (no build step), service worker (sw.js), manifest and icons;
                        guides/ holds the guides' corrected graphics
scripts/import_guides.py Turns an export of aieverytime.org's articles into app/guides.json.gz (see The guides);
                        scripts/guide_edits.json holds the editorial changes it applies
scripts/courses/        Writing courses: the guide to the format and its standards, and a checker (see its README)
scripts/graphics/       Correcting the spelling in the guides' graphics, and the corrections (see its README)
scripts/screenshots/    Screenshots of the app as a learner sees it, and slides for social media (see its README)
scripts/make_icons.py   Makes the logo image and the app icons from the logo (scripts/logo.webp); run it if the logo changes
Dockerfile              The image for deploying it (see Deploy it)
```

**Notifications.** Each message is encrypted for one device (RFC 8291) and signed with the server's VAPID key (RFC 8292), using `pywebpush`. The server only sends to the push services browsers use (Google, Mozilla, Apple, Microsoft), so a made-up subscription can't make it send requests anywhere else. To make that hold, an address must parse to the same host in both the standard library and urllib3 (which sends the request), it's checked again before every send, and redirects aren't followed. A device whose subscription has gone (404/410) is forgotten at once. A device that rejects five messages in a row is forgotten too; push service outages don't count against it. Each notice and each daily reminder is claimed before it's sent, so it goes out at most once, and one learner's failure doesn't hold up anyone else's. An in-process loop checks every minute; the email digest has its own loop.

**Prompt structure.** The system prompt has two parts. The first holds the fixed platform rules plus the persona and how to react to messages; it is cached. Learner-written custom instructions go here too, *after* the platform rules, so they can change a companion's voice but not the rules. The second part holds learner data that changes often: profile (with their interests), memories, the post being discussed, and the learner's reaction to the last reply. When a companion starts a chat, what it's asked to do goes there too, so the cached first part is the same as for chat; the conversation Claude is sent begins with a short turn for the learner opening the chat, since it has to begin with the learner.

**Claude requests.** Every request turns on server-side refusal fallbacks (`fallbacks: "default"`). Effort is set per task: `low` for memory and summaries, `medium` for chat and flashcards, `high` for posts, lessons, quizzes and study plans. Structured answers are read from the response after its tokens are recorded: one that was cut short or refused doesn't fit its schema, and counts as an answer that can't be used rather than causing a server error.

**AI allowance.** Before Claude is called, the request's cost is reserved against the learner's last 24 hours in one write transaction, so parallel requests can't both squeeze in. The request's record is put in a context variable, so the functions that ask Claude don't need to know whose request it is: each response adds its tokens to the record in hand, summing every attempt listed in `usage.iterations` when a fallback model answers, and the chat's memory extraction, which runs after the reply, adds to the chat's record. Only failed calls to Claude (API or connection errors) give the request back. Opening a lesson writes it on a GET, so a lesson is only written for requests from the app itself (by the browser's `Sec-Fetch-Site`), not for a page on another site that opens the address. Requests turned away are counted per learner per hour, so a stream of them takes almost no room.

**Accuracy.** Lessons, questions and cards are written from each lesson's vetted key points, not from the model's general knowledge alone. Generated output is checked before it is saved: questions must have exactly 4 distinct options and a valid answer, and study plans may only reference real lesson IDs and activities. A course's lessons are also told about the walkthrough beside them, and asked not to give menu paths or shortcuts of their own, so the only ones learners see are the checked ones.

## Next steps

- **Limits on sign-ups per network address**, once the app's hosting is known (they need the address the host's proxy reports).
- **Server-side voices** for consistent, higher-quality speech on every device (needs a text-to-speech provider).
- **Postgres** in place of SQLite, and a job queue in place of the in-process loops (posts, emails, notifications), for multi-instance deployment.
