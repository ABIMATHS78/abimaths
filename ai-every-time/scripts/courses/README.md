# Writing courses

The course catalogue is one JSON file per course in `app/course_data/`, named after the course's slug, plus
`tracks.json`, which groups courses into Professional Certificates and Specializations. The format is defined,
and checked, by the models in `app/catalog.py`; this guide explains it and the standards a course has to meet.
`python-programming-pycharm.json` is a complete example: read it before writing a course.

Check your files before anything else:

```bash
python scripts/courses/check.py                       # every course, plus the tracks
python scripts/courses/check.py app/course_data/my-course.json
```

It reports errors (the app won't start with them) and warnings (worth a second look). The app reads the files when
it starts, so restart it to see a change.

## How a course works

A course is taught by one to three of the starter AI Learning Companions, its instructors. It has **modules**,
taken in order (Coursera's "weeks"). Each module has:

- **Lessons.** The instructor writes each lesson in their own voice, from the lesson's *objectives* and *key points*,
  and learners can ask them about it in a chat beside it. The key points are the vetted core: every lesson, quiz,
  flashcard and exam question is built on them, so they must be accurate, specific and complete. A lesson can have
  a **walkthrough**: a tool shown step by step, with its keyboard shortcuts, shown beside the lesson exactly as you
  write it (the instructor introduces it but doesn't restate it), the way a screencast would show it. And a
  **video**, once one's recorded.
- An optional hands-on **lab**: setup, steps, the *checks* it must pass, and what to submit. Learners do it on their
  own computer, paste in their work (code, command output, configuration), and the instructor reviews it against
  the checks. Starter files download as a zip.
- A **quiz** on the module's lessons, written by the instructor (you don't write it; 70% passes).

A course can end with a capstone **project**, reviewed against its rubric. Finishing every lesson, passing every
quiz and lab, and passing the project earns the course certificate.

Every course also has, without you writing them:

- an **outline** (its modules and their lessons, labs and quizzes) and a **study plan**: the weeks to take it over, from
  `study_plan` if the course has one, otherwise one module a week;
- a **mock exam** on the whole course, timed, for practice (it doesn't count towards the certificate), and
  **flashcards**, for each lesson or the whole course. Both are written from the key points, like the quizzes.

A **guided project** is a short, hands-on course (one to two hours): one module of two or three short lessons that
ends with its lab, and usually no capstone.

## The format

```jsonc
{
  "slug": "python-programming-pycharm",      // lowercase words joined by hyphens, ending in a letter; the file name
  "title": "Python Programming with PyCharm",
  "subtitle": "One sentence: what you'll be able to do.",
  "category": "Software Development",        // one of the categories below
  "kind": "course",                           // or "guided-project"
  "level": "beginner",                        // beginner, intermediate or advanced
  "instructors": ["vladimir-novak", "donte-jacksonee"],   // starter companion slugs, the lead first (1-3)
  "skills": ["Python", "Debugging"],          // 3-10, "Skills you'll gain"
  "tools": ["PyCharm", "pytest"],             // 0-8 tools and products learners use
  "outcomes": ["Set up Python projects ..."],   // 3-6, "What you'll learn", each starting with a verb
  "prerequisites": "What learners need first, and what to install.",
  "description": "The course page's About text: one or two paragraphs.",
  "aligned_with": "",                         // e.g. "Cisco CCNA (200-301) exam topics: IP Connectivity"
  "modules": [
    {
      "title": "Get set up",
      "summary": "One or two sentences.",
      "instructor": "donte-jacksonee",       // optional: which instructor teaches this module (default: the first)
      "lessons": [
        {
          "title": "Projects and virtual environments",
          "minutes": 10,                      // 3-30
          "objectives": ["Explain why ...", "Create ..."],       // 1-4, each starting with a verb
          "key_points": ["A virtual environment is ...", "..."],  // 3-6 accurate sentences, at most 360 characters each
          "walkthrough": {                    // optional
            "tool": "PyCharm",
            "title": "Create a project with its own virtual environment",
            "steps": [                        // 2-14
              {"do": "Open the Terminal tool window and check which Python it uses.",
               "keys": {"mac": "Option+F12", "win": "Alt+F12"},   // optional: the shortcut; win means Windows and Linux
               "code": "python --version", "lang": "bash",         // optional: code, commands or config, and its language
               "see": "Python 3.x, and the prompt starts with (.venv)."}  // optional: what should happen
            ]
          },
          "video": ""                         // optional: https://www.youtube-nocookie.com/embed/ID or https://player.vimeo.com/video/ID
        }
      ],
      "lab": {                                // optional
        "title": "Set up your Python workspace",
        "tool": "PyCharm",
        "minutes": 30,                        // 10-240
        "goal": "What the lab achieves, in a sentence or two.",
        "setup": ["What to have installed or downloaded"],     // 0-6
        "steps": [ /* 3-16 steps, like a walkthrough's */ ],
        "checks": [                           // 1-8: what the submission must show to pass
          {"text": "The script runs on Python 3.", "look_for": ["Python 3."]}
        ],
        "submit": "Exactly what to paste in for review.",
        "files": [{"path": "check_setup.py", "content": "..."}],   // 0-10 starter files
        "hints": ["..."]                      // 0-6
      }
    }
  ],
  "study_plan": [                             // optional: the weeks to take the course over, in order
    {"week": 1, "focus": "Set up, and write and run your first scripts.", "modules": [1]},
    {"week": 2, "focus": "Control flow and functions; finish the module 2 lab.", "modules": [2, 3]},
    {"week": 3, "focus": "Build and submit the capstone project.", "modules": []}   // no modules: review or the project
  ],
  "project": {                                // optional capstone
    "title": "Build a study tracker",
    "minutes": 240,                           // 30-900
    "brief": "The scenario and what to build: a paragraph.",
    "deliverables": ["..."],                  // 2-8
    "rubric": [{"name": "It works", "points": 30, "description": "What earns the points."}],   // 3-8; points add up to 100
    "submit": "Exactly what to paste in.",
    "files": []                               // optional starter files
  }
}
```

**Categories:** Generative AI, Machine Learning, AI Engineering (MLOps, LLMOps, serving, AI system design), Data Science
(analysis, statistics, visualisation), Data Engineering (pipelines, warehouses, streaming), Software Development,
Computer Science (algorithms, systems, databases, distributed systems), Networking, Cybersecurity, Cloud & DevOps,
AI for Business, Responsible AI, AI in Industry, Robotics & Edge AI.

**Study plans.** A `study_plan` spreads the modules over weeks: number the weeks from 1, take the modules in order, and
cover every module (a long module can span two weeks; a week can also be for review or the capstone, with no modules).
Each week's `focus` says what to get done that week, in a sentence. The course page shows each week's time, worked
out from the minutes. Leave it out and the plan is one module a week.

The time a course takes is worked out from its minutes (each quiz adds 10), so set realistic minutes.

`look_for` holds short pieces of text a correct submission would contain (any case), such as an output line
(`"3 passed"`, `"Reply from 192.168.1.10"`, `"FULL/"`). In demo mode, without a Claude key, a check passes when all
its pieces are present; with a key, they guide the instructor, who judges the check itself. Choose distinctive
text, not words any answer would contain.

## Standards

**Accuracy comes first.** Key points, steps and code must be right: well-established facts, real commands, real
menu paths and shortcuts, code that runs. Where something differs between versions, say so rather than guess, and
prefer what's been stable for years. Don't include statistics, prices, dates of things still to come, or claims
you can't stand behind. No made-up studies or quotes.

**Make it hands-on.** Technical courses get a lab in most modules, with starter files where they help. Labs use
free tools on an ordinary laptop: PyCharm, IntelliJ IDEA and the other JetBrains IDEs (free versions or trials),
Cisco Packet Tracer (free with a Cisco Networking Academy account), Wireshark, VirtualBox, Docker, Python, a
browser. Run any code you include, if you can, and give the exact output a correct solution prints, so checks can
name it.

**Show the tools.** In a course whose tools include a JetBrains IDE, give most lessons a walkthrough in that IDE:
the IDE should be used and demonstrated throughout, as in a screencast course, not only mentioned. Other tools
(Packet Tracer, Wireshark, a terminal, a chat assistant) get walkthroughs where they help.

**Unique, and structured like Coursera's.** Every course has its own focus: don't repeat another course's lessons.
Courses have 3-5 modules of 2-4 lessons (8-15 minutes each); guided projects one module of 2-3 short lessons and a
substantial lab. Titles are specific; outcomes, skills and the description say what learners will be able to do.

**Writing.** Plain, clear, friendly sentences. British spelling in prose (organise, behaviour, analyse, programme
for a TV or training programme, program for software), but products' own names and labels as they are (Optimize
Imports, Settings | Editor | Color Scheme). The instructors bring their own voice when they teach, so keep course
text neutral: no first person, no instructor names in lessons.

**Instructors** are starter companions whose background suits the course (see the table below). Never one modelled
on a real person (mojeed-abisiga, gbenga-akingbulere, ogweda-haruna): the checker rejects them.

**Never name a Claude model ID** (the checker rejects them; they go out of date). Code that calls the Claude API reads
the model from an environment variable, and the course tells learners to set it to a current model from Anthropic's
documentation:

```python
import os
import anthropic

client = anthropic.Anthropic()               # reads ANTHROPIC_API_KEY
MODEL = os.environ["CLAUDE_MODEL"]           # a current model ID from docs.anthropic.com
reply = client.messages.create(model=MODEL, max_tokens=1024,
                               messages=[{"role": "user", "content": "Hello"}])
print(reply.content[0].text)
```

**Security courses are defensive.** They teach how to protect, monitor, detect and respond: hardening, secure configuration, secure coding, logging and monitoring, incident response and governance, practised on learners' own machines and on sample data that comes with the course. They don't teach how to attack systems.

**Cisco courses** use Cisco IOS commands that work in Cisco Packet Tracer (routers such as the 2911, switches such
as the 2960 and 3650, interfaces such as g0/0 and f0/1) and follow Cisco's published exam topics where they say
they're aligned with an exam. They're independent of Cisco, and say nothing that suggests otherwise.

## JetBrains IDE shortcuts

Use only these in `keys` (default keymaps; they're the same in PyCharm, IntelliJ IDEA, WebStorm, GoLand, CLion,
Rider, PhpStorm, RustRover and DataGrip). For anything else, give the menu path instead.

| Action | macOS | Windows and Linux |
|---|---|---|
| Search Everywhere | Shift Shift | Shift Shift |
| Find Action | Cmd+Shift+A | Ctrl+Shift+A |
| Settings | Cmd+, | Ctrl+Alt+S |
| Show Context Actions (quick-fixes, intentions) | Option+Enter | Alt+Enter |
| Basic completion | Ctrl+Space | Ctrl+Space |
| Parameter info | Cmd+P | Ctrl+P |
| Quick documentation | F1 | Ctrl+Q |
| Go to declaration | Cmd+B | Ctrl+B |
| Go to implementation | Cmd+Option+B | Ctrl+Alt+B |
| Find usages | Option+F7 | Alt+F7 |
| Go to class | Cmd+O | Ctrl+N |
| Go to file | Cmd+Shift+O | Ctrl+Shift+N |
| Go to symbol | Cmd+Option+O | Ctrl+Alt+Shift+N |
| Recent files | Cmd+E | Ctrl+E |
| File structure | Cmd+F12 | Ctrl+F12 |
| Go to test (and create one) | Cmd+Shift+T | Ctrl+Shift+T |
| Navigate back / forward | Cmd+[ / Cmd+] | Ctrl+Alt+Left / Ctrl+Alt+Right |
| Generate code / new file in the Project view | Cmd+N | Alt+Insert |
| Rename | Shift+F6 | Shift+F6 |
| Refactor This | Ctrl+T | Ctrl+Alt+Shift+T |
| Extract method | Cmd+Option+M | Ctrl+Alt+M |
| Extract variable | Cmd+Option+V | Ctrl+Alt+V |
| Extract constant | Cmd+Option+C | Ctrl+Alt+C |
| Extract parameter | Cmd+Option+P | Ctrl+Alt+P |
| Inline | Cmd+Option+N | Ctrl+Alt+N |
| Change signature | Cmd+F6 | Ctrl+F6 |
| Move | F6 | F6 |
| Safe delete | Cmd+Delete | Alt+Delete |
| Reformat code | Cmd+Option+L | Ctrl+Alt+L |
| Optimize imports | Ctrl+Option+O | Ctrl+Alt+O |
| Comment line | Cmd+/ | Ctrl+/ |
| Duplicate line | Cmd+D | Ctrl+D |
| Delete line | Cmd+Backspace | Ctrl+Y |
| Move line up / down | Option+Shift+Up / Down | Alt+Shift+Up / Down |
| Extend selection | Option+Up | Ctrl+W |
| Surround with | Cmd+Option+T | Ctrl+Alt+T |
| Insert live template | Cmd+J | Ctrl+J |
| Add or remove a caret | Option+Shift+Click | Alt+Shift+Click |
| Select next occurrence | Ctrl+G | Alt+J |
| Find / Replace in file | Cmd+F / Cmd+R | Ctrl+F / Ctrl+R |
| Find / Replace in files | Cmd+Shift+F / Cmd+Shift+R | Ctrl+Shift+F / Ctrl+Shift+R |
| Next highlighted error | F2 | F2 |
| Error description | Cmd+F1 | Ctrl+F1 |
| Run (last configuration) | Ctrl+R | Shift+F10 |
| Debug (last configuration) | Ctrl+D | Shift+F9 |
| Run what's at the caret (file, test) | Ctrl+Shift+R | Ctrl+Shift+F10 |
| Stop | Cmd+F2 | Ctrl+F2 |
| Toggle line breakpoint | Cmd+F8 | Ctrl+F8 |
| View breakpoints | Cmd+Shift+F8 | Ctrl+Shift+F8 |
| Step over / into / out | F8 / F7 / Shift+F8 | F8 / F7 / Shift+F8 |
| Run to cursor | Option+F9 | Alt+F9 |
| Resume program | Cmd+Option+R | F9 |
| Evaluate expression | Option+F8 | Alt+F8 |
| Commit | Cmd+K | Ctrl+K |
| Push | Cmd+Shift+K | Ctrl+Shift+K |
| Update project (pull) | Cmd+T | Ctrl+T |
| VCS operations popup | Ctrl+V | Alt+` |
| Rollback | Cmd+Option+Z | Ctrl+Alt+Z |
| Project / Run / Debug tool windows | Cmd+1 / Cmd+4 / Cmd+5 | Alt+1 / Alt+4 / Alt+5 |
| Problems / Structure / Services tool windows | Cmd+6 / Cmd+7 / Cmd+8 | Alt+6 / Alt+7 / Alt+8 |
| Git / Commit tool windows | Cmd+9 / Cmd+0 | Alt+9 / Alt+0 |
| Terminal | Option+F12 | Alt+F12 |
| Execute selection in Python Console (PyCharm) | Option+Shift+E | Alt+Shift+E |
| Run Anything | Ctrl Ctrl | Ctrl Ctrl |
| Recent locations | Cmd+Shift+E | Ctrl+Shift+E |

IntelliJ IDEA live templates: `psvm` or `main` (a main method), `sout` (System.out.println), `fori` (a for loop),
`iter` (a for-each loop), `ifn` (if null). Kotlin: `main`. Help | Learn IDE Features opens the IDE's own interactive
tutorials. JetBrains AI Assistant and Junie are features you install or enable; keep their steps to what's stable (open
the AI chat or Junie's tool window, ask, review what it proposes before accepting) rather than exact buttons.

## Tracks

`tracks.json` is a list of tracks:

```json
{"slug": "network-engineer", "title": "Network Engineer", "kind": "Professional Certificate",
 "subtitle": "...", "description": "...", "roles": ["Network engineer"], "courses": ["networking-fundamentals", "..."]}
```

A Professional Certificate prepares learners for a job and a Specialization builds one skill; courses are taken in
the order listed. Finishing every course in a track earns its certificate too.

## Instructors

The starter companions who can teach courses, with their expertise and the first line of their bio
(`app/starter_companions.py` has the rest).

| Slug | Name | Expertise | Background |
|---|---|---|---|
| `amahle-ndlovu` | Amahlé Ndlovù 🇿🇦 | AI Ethics & Safety, Data Science | Amahlé is a data scientist from Soweto who audits credit-scoring models for fairness at a Johannesburg bank. |
| `kwame-asante` | Kwamé Asantè 🇬🇭 | Machine Learning Foundations, AI for Business | Kwamé is a software engineer from Kumasi who builds models that forecast cocoa harvests for farming cooperatives. |
| `fatima-diallo` | Fatimà Diallò 🇸🇳 | Natural Language Processing, Large Language Models | Fatimà is a language-technology researcher in Dakar who records and labels Wolof speech so voice assistants can understand the people around her. |
| `chukwuemeka-obieche` | Chukwuemekà Obiechè 🇳🇬 | MLOps & Deployment, Machine Learning Foundations | Chukwuemekà is an ML engineer from Onitsha who keeps fraud-detection models running for a Lagos payments company. |
| `aissata-coulibaly` | Aïssatà Coulibàly 🇨🇮 | Computer Vision, Data Science | Aïssatà is a computer vision engineer in Abidjan whose phone app spots disease on cassava leaves from a single photo. |
| `tariqu-benai` | Tariqù Benaì 🇲🇦 | Robotics, AI Agents | Tariqù grew up in an Amazigh village in the High Atlas and now builds robots that clean solar panels in the desert near Ouarzazate. |
| `nkiruka-ezenwa` | Nkirukà Ezenwà 🇳🇬 | Prompt Engineering, Generative AI for Creators | Nkirukà is a prompt engineer from Ibadan who designs AI tutors for a Lagos edtech company. |
| `sekouu-camara` | Sekouù Camàra 🇬🇳 | Math for AI, Data Science | Sekouù taught mathematics at a lycée in Conakry before becoming a data scientist for a mobile network. |
| `amiraa-hassan` | Amiraà Hassàn 🇪🇬 | Large Language Models, Natural Language Processing | Amiraà is an NLP engineer from Alexandria who tests how well large language models handle Arabic, from classical poetry to Cairo street slang. |
| `thandiwe-dlamini` | Thandiwè Dlaminì 🇸🇿 | AI for Business, Data Science | Thandiwè is a data scientist in Manzini who forecasts medicine supplies for rural clinics, so nobody walks for hours to find empty shelves. |
| `mouhamadouu-ba` | Mouhamadouù Bâ 🇲🇱 | MLOps & Deployment, Machine Learning Foundations | Mouhamadouù is an engineer in Bamako who squeezes AI models onto cheap, solar-charged phones for places with patchy internet. |
| `zawadi-mutembo` | Zawadì Mutembò 🇹🇿 | Computer Vision, Machine Learning Foundations | Zawadì is a computer vision researcher in Arusha whose models sort millions of camera-trap photos from the Serengeti. |
| `youssoufau-traore` | Youssoufaù Traoré 🇧🇫 | Generative AI for Creators, Natural Language Processing | Youssoufaù is a filmmaker in Ouagadougou who uses generative AI to storyboard, subtitle and dub short films for festivals like FESPACO. |
| `naledi-mokoena` | Nalèdi Mokoèna 🇱🇸 | AI Policy & Regulation, AI Ethics & Safety | Nalèdi is a policy adviser in Maseru who helps her government write data-protection and AI rules that fit a small country. |
| `abderrahman-bensoussan` | Abderrahmàn Bensoussàn 🇩🇿 | AI Agents, MLOps & Deployment | Abderrahmàn is a software architect in Algiers who builds AI agents that schedule maintenance at power stations across the country. |
| `akosua-mensah` | Akosùa Mensàh 🇬🇭 | AI for Business, Prompt Engineering | Akosùa runs a consultancy in Accra that helps market traders, tailors and young founders put AI to work, from writing adverts to forecasting stock. |
| `idrissa-ouedraogo` | Idrissà Ouédraogò 🇧🇫 | Data Science, Math for AI | Idrissà is a climate data scientist in Ouagadougou who turns satellite and rain-gauge data into planting advice for farmers in the Sahel. |
| `zinhlee-khumalo` | Zinhlée Khumàlo 🇿🇼 | Large Language Models, Prompt Engineering | Zinhlée is a former maths teacher from Bulawayo who now trains teachers across Zimbabwe to use AI chat assistants in their lessons. |
| `mamadouu-dioufee` | Mamadouù Dioufée 🇸🇳 | Robotics, Computer Vision | Mamadouù is a robotics engineer in Dakar who builds drones that map the coastline and help fishing crews come home safely. |
| `lindiwe-sibanda` | Lindiwe Sibandà 🇿🇲 | AI Agents, AI for Business | Lindiwe founded a Lusaka start-up whose AI agents help small mining suppliers handle orders, invoices and customs paperwork. |
| `hiroshi-yamada` | Hiroshì Yamadà 🇯🇵 | Robotics, MLOps & Deployment | Hiroshì is a robotics engineer in Nagoya who teaches factory robots to work safely beside people on car assembly lines. |
| `mei-ling-zhang` | Mei-Líng Zhāng 🇨🇳 | Large Language Models, Machine Learning Foundations | Mei-Líng is a researcher in Hangzhou who studies how large language models learn from text, and where that learning runs out. |
| `rajesh-krishnan` | Rajèsh Krishnàn 🇮🇳 | Math for AI, Machine Learning Foundations | Rajèsh is a mathematics professor in Chennai who teaches the maths behind machine learning to engineers who swore they'd never need it again. |
| `soo-yeon-kimm` | Soo-Yéon Kimm 🇰🇷 | Generative AI for Creators, Computer Vision | Soo-Yéon is a creative technologist in Seoul who helps webtoon artists and music producers use generative AI as a tool, not a replacement. |
| `fatimah-rahimi` | Fatimàh Rahimì 🇦🇫 | Natural Language Processing, Prompt Engineering | Fatimàh is a computational linguist from Herat who builds translation and speech tools for Dari and Pashto speakers. |
| `arjun-sharma` | Arjùn Sharmà 🇮🇳 | AI Agents, AI for Business | Arjùn is a product manager in Gurugram who has shipped AI agents that answer millions of customer questions in Hindi, Punjabi and English. |
| `yuki-tanaka` | Yuki Tanakà 🇯🇵 | AI Ethics & Safety, Natural Language Processing | Yuki is a researcher in Kyoto who designs voice assistants for older people living alone, so the technology is gentle, honest and safe. |
| `wei-cheng-liu` | Wei-Chéng Liú 🇹🇼 | MLOps & Deployment, Math for AI | Wei-Chéng is a hardware engineer in Hsinchu who designs the chips that run AI models in phones and data centres. |
| `priya-naire` | Priyà Nairé 🇮🇳 | Computer Vision, AI Ethics & Safety | Priyà is a computer vision engineer in Kochi who builds models that help doctors read eye scans for early signs of diabetes. |
| `baraka-al-rashid` | Barakà Al-Rashíd 🇸🇦 | AI Policy & Regulation, AI for Business | Barakà is a policy adviser in Riyadh who works on how AI is used in public services, from traffic to health. |
| `ngo-thi-huunge` | Ngo Thi Huừngé 🇻🇳 | Data Science, AI for Business | Huừngé is a data scientist in Ho Chi Minh City who helps online sellers understand what their customers really want. |
| `khalida-sultan` | Khalidà Sultàn 🇵🇰 | Machine Learning Foundations, Data Science | Khalidà is a machine learning engineer in Lahore who taught herself to code from free online videos, and now makes the lessons she wished she'd had. |
| `chaiyapon-srisomboon` | Chaiyapòn Srisomboòn 🇹🇭 | AI Agents, MLOps & Deployment | Chaiyapòn is an engineer in Bangkok who builds the agents that route thousands of food deliveries through the city's traffic every evening. |
| `layla-al-mansouri` | Layla Al-Mansourì 🇦🇪 | Large Language Models, AI Policy & Regulation | Layla is an AI engineer in Abu Dhabi who works on Arabic-language models for government services, from permits to health questions. |
| `binta-yusupova` | Bintà Yusupovà 🇺🇿 | Math for AI, Machine Learning Foundations | Bintà coaches maths olympiad teams in Samarkand and teaches the probability and linear algebra that machine learning is built on. |
| `davaadorji-gantulag` | Davaadorjì Gantulàg 🇲🇳 | Computer Vision, Data Science | Davaadorjì is a remote-sensing engineer in Ulaanbaatar whose models read satellite images to track grazing land, snow and herds across the steppe. |
| `siti-nurhalizah-rahiman` | Siti Nurhalizah Rahimàn 🇲🇾 | Generative AI for Creators, Prompt Engineering | Siti is a digital marketer in Kuala Lumpur who teaches small brands to make posts, videos and campaigns with generative AI, honestly labelled and in their own voice. |
| `ravi-chandrasekharan` | Ravi Chandrasekharàn 🇱🇰 | Natural Language Processing, Machine Learning Foundations | Ravi is a software engineer from Jaffna who builds search and translation tools for Tamil and Sinhala. |
| `aynure-abdullayeva` | Aynurè Abdullayevà 🇦🇿 | Prompt Engineering, Large Language Models | Aynurè is an AI trainer in Baku who teaches teachers, nurses and shop owners to get useful, honest answers from AI chat assistants. |
| `jae-won-parkk` | Jae-Wòn Parkk 🇰🇷 | AI Agents, Machine Learning Foundations | Jae-Wòn is a game AI engineer in Pangyo who builds agents that learn to play by trial and error. |
| `divya-mehta` | Divyà Mehtà 🇮🇳 | AI for Business, AI Agents | Divyà is a business consultant in Ahmedabad who helps textile firms and family businesses adopt AI one sensible step at a time, from demand forecasts to customer chat. |
| `hamza-al-khalil` | Hamzà Al-Khalìl 🇯🇴 | Computer Vision, Robotics | Hamzà is a robotics engineer in Amman who builds drones that map ancient sites like Petra, spotting erosion before it's too late. |
| `nataliya-kovalenko` | Nataliyà Kovalenkò 🇰🇿 | Machine Learning Foundations, Math for AI | Nataliyà is a data scientist in Almaty who started her career analysing rocket telemetry at Baikonur. |
| `thida-kyawzin` | Thida Kyawzìn 🇲🇲 | Natural Language Processing, AI Ethics & Safety | Thida is an NLP researcher in Yangon who builds tools that sort and translate Burmese messages for aid organisations during floods and emergencies. |
| `ismaile-rahimovi` | Ismaìlè Rahimovì 🇹🇯 | MLOps & Deployment, Data Science | Ismaìlè is an engineer in Dushanbe who deploys and monitors the models that predict faults in the country's hydropower plants. |
| `elena-petrova` | Élèna Petrovà 🇷🇺 | Machine Learning Foundations, Math for AI | Élèna is a machine learning researcher in Saint Petersburg who teaches the foundations with a mathematician's care: definitions first, then examples, then the traps. |
| `luis-ferreiras` | Luís Ferreiràs 🇵🇹 | AI Agents, Prompt Engineering | Luís is a start-up engineer in Lisbon who builds AI agents for travel and customer support companies. |
| `annika-lindqvist` | Annikà Lindqvìst 🇸🇪 | AI Ethics & Safety, AI Policy & Regulation | Annikà is an AI ethics researcher in Stockholm who studies the energy use and fairness of large AI systems. |
| `konstantini-papadopoulous` | Konstantinì Papadopouloùs 🇬🇷 | Math for AI, AI Ethics & Safety | Konstantinì teaches mathematics and philosophy in Thessaloniki and loves showing that the newest AI rests on very old ideas: geometry, logic and careful argument. |
| `maia-duplessie` | Maïa Duplessiè 🇫🇷 | Generative AI for Creators, Computer Vision | Maïa is a designer and artist in Lyon who uses generative AI in exhibitions and posters, and teaches others to do the same without losing their style. |
| `bjorn-sigurdsson` | Björn Sigurdssòn 🇮🇸 | Natural Language Processing, Large Language Models | Björn is a language technologist in Reykjavík who helps large language models speak Icelandic, a language with fewer speakers than many cities. |
| `agnieszka-kowalska` | Agnieszka Kowalskà 🇵🇱 | Machine Learning Foundations, Prompt Engineering | Agnieszka is a software engineer in Kraków who runs free evening workshops where complete beginners train their first model in an hour. |
| `matteo-ferrari` | Mattèo Ferrarì 🇮🇹 | Robotics, Data Science | Mattèo is a robotics engineer from Modena who works on data and control systems for race cars and factory robots in Italy's Motor Valley. |
| `ingride-hansen` | Ingridè Hansèn 🇳🇴 | Computer Vision, Data Science | Ingridè is a computer vision engineer in Bergen whose cameras watch salmon pens, spotting sea lice and sick fish before they spread. |
| `ferencz-nagy` | Ferencz Nagŷ 🇭🇺 | Math for AI, Data Science | Ferencz is a mathematician in Budapest who grew up in maths camps and now teaches the probability behind machine learning. |
| `catalina-ionescu` | Cataliná Ionescù 🇷🇴 | AI Ethics & Safety, MLOps & Deployment | Cataliná is a security engineer in Cluj-Napoca who tests AI systems for weaknesses before attackers find them. |
| `nikolas-petrovici` | Nikolàs Petrovicì 🇷🇸 | AI Agents, Generative AI for Creators | Nikolàs is a game developer in Novi Sad who uses AI agents to playtest levels and generative tools to sketch worlds faster. |
| `siobhan-obrienn` | Siobhàn O'Briénn 🇮🇪 | AI Policy & Regulation, AI Ethics & Safety | Siobhàn is a lawyer in Dublin who advises tech companies on Europe's AI and data-protection rules. |
| `dariusz-wojciechowski` | Dariusz Wojciechowskì 🇵🇱 | MLOps & Deployment, Large Language Models | Dariusz is a platform engineer in Gdańsk who runs large language models in production for a logistics company. |
| `yildize-demirel` | Yildizè Demirèl 🇹🇷 | Natural Language Processing, Prompt Engineering | Yildizè is an NLP engineer in Istanbul who builds chatbots and search tools in Turkish, a language that stacks meaning onto the ends of words. |
| `aleksei-volkovi` | Alekseï Volkovì 🇷🇺 | Large Language Models, Math for AI | Alekseï is a physicist turned AI researcher in Akademgorodok, Novosibirsk's science town, who explains the maths inside large language models: vectors, probabilities and attention. |
| `marisolo-garcia` | Marisolò Garcíà 🇪🇸 | Data Science, Machine Learning Foundations | Marisolò is a sports data scientist in Seville who builds models that help a football club scout players and plan training. |
| `lasse-makinen` | Lasse Mäkìnen 🇫🇮 | Machine Learning Foundations, AI Ethics & Safety | Lasse is an educator in Helsinki who has helped design free online AI courses for thousands of people. |
| `zoe-papadakis` | Zoë Papadakìs 🇬🇷 | Computer Vision, Generative AI for Creators | Zoë is a heritage scientist in Heraklion, on Crete, who uses computer vision to piece together broken frescoes, and generative tools to imagine their missing parts, always clearly labelled as guesses. |
| `michel-lamberte` | Michèl Lambertè 🇧🇪 | AI Policy & Regulation, AI for Business | Michèl is a policy analyst in Brussels who follows Europe's AI rules from first draft to final vote. |
| `katarina-novakova` | Katarína Novákovà 🇸🇰 | Robotics, MLOps & Deployment | Katarína is a robotics engineer in Bratislava who programs and maintains the robots on a car assembly line. |
| `emre-yilmazi` | Emrè Yilmazì 🇹🇷 | AI for Business, AI Agents | Emrè is an entrepreneur in Istanbul whose online shop uses AI agents for customer chat, stock and delivery planning. |
| `sigridur-bjornsdottir` | Sigríður Björnsdóttìr 🇮🇸 | Math for AI, Data Science | Sigríður is a geophysicist in Reykjavík who uses machine learning on earthquake data to warn when an eruption may be on its way. |
| `vladimir-novak` | Vladimìr Novàk 🇨🇿 | Prompt Engineering, Large Language Models | Vladimìr is a software developer in Brno who builds dependable features on top of large language models. |
| `aoife-murphy` | Aoifè Murphŷ 🇮🇪 | Generative AI for Creators, Natural Language Processing | Aoifè is a writer and creative-technology teacher in Galway who shows authors how to use AI for research, editing and brainstorming, without letting it write the story. |
| `marisel-rodriguez` | Marisèl Rodriguèz 🇲🇽 | Prompt Engineering, AI for Business | Marisèl is a tech trainer in Guadalajara who teaches shop owners, accountants and students to save hours each week with AI assistants. |
| `deshawne-thompson` | Deshawné Thompsòn 🇺🇸 | AI Ethics & Safety, Machine Learning Foundations | Deshawné is a data scientist in Atlanta who audits hiring and lending algorithms for bias. |
| `chloe-tremblay` | Chloé Tremblàŷ 🇨🇦 | Large Language Models, Machine Learning Foundations | Chloé is a researcher in Montréal who studies how large language models learn and why they sometimes make things up. |
| `xiomara-hernandez` | Xiomàra Hernándèz 🇬🇹 | Natural Language Processing, AI Ethics & Safety | Xiomàra is a language-technology researcher from Quetzaltenango who builds speech and translation tools for K'iche' speakers. |
| `jamal-washingtonee` | Jamàl Washingtoneè 🇺🇸 | Robotics, Computer Vision | Jamàl is a robotics engineer in Detroit who works on how self-driving test vehicles see the road, in snow, rain and potholes. |
| `ingride-bjornsson` | Ingridè Bjornssòn 🇨🇦 | AI for Business, Data Science | Ingridè grew up in Gimli, Manitoba, in a family that came from Iceland generations ago, and now helps prairie farm co-ops use AI for crop planning, weather risk and selling smarter. |
| `alejandroo-riveras` | Alejandroó Riveràs 🇲🇽 | Generative AI for Creators, Prompt Engineering | Alejandroó is a creative director in Mexico City who uses generative AI to sketch campaigns, music videos and murals, then finishes them by hand. |
| `keisha-browne` | Keishà Brownè 🇯🇲 | AI for Business, Generative AI for Creators | Keishà is a music-business consultant in Kingston who helps artists and small labels use AI for marketing, rights and fan data, while protecting their sound and their ownership. |
| `tatianna-morrison` | Tatiànna Morrisòn 🇺🇸 | Data Science, AI Ethics & Safety | Tatiànna was a nurse in Brooklyn before becoming a health data scientist, and now builds models that flag patients who need extra care. |
| `carlos-ramirez` | Carlòs Ramirèz 🇨🇷 | MLOps & Deployment, AI Agents | Carlòs is an engineer in San José who deploys and maintains AI systems for medical-device and tourism companies. |
| `amelia-white` | Amèlia Whitè 🇺🇸 | AI Ethics & Safety, Large Language Models | Amèlia is an AI safety researcher in Boston who studies what goes on inside large language models and how to keep them honest. |
| `donte-jacksonee` | Dontè Jacksoneè 🇺🇸 | Machine Learning Foundations, Math for AI | Dontè is a high-school computer science teacher in Houston who teaches machine learning with the basketball stats his students actually care about. |
| `valentina-lopez` | Valentinà Lopèz 🇨🇴 | AI for Business, Data Science | Valentinà is a data scientist in Medellín who helps the city and local businesses use AI for transport, energy and customer service. |
| `rodrigoo-silva` | Rodrigoò Silvà 🇧🇷 | MLOps & Deployment, AI Agents | Rodrigoò is an engineer in São Paulo who keeps the AI that spots fraud in instant payments running day and night. |
| `isabela-martinez` | Isabelà Martinèz 🇦🇷 | Natural Language Processing, AI Ethics & Safety | Isabelà is a linguist and NLP researcher in Buenos Aires who studies how chatbots handle irony, slang and feelings. |
| `luciano-ferreiras` | Luciànò Ferreiràs 🇧🇷 | Computer Vision, Machine Learning Foundations | Luciànò is a computer vision engineer at Porto Digital, Recife's tech hub, who builds models for traffic safety and agriculture. |
| `camila-torrese` | Camilà Torresè 🇨🇱 | Machine Learning Foundations, Data Science | Camilà is a data scientist in Santiago who helps astronomers find rare objects in huge sky surveys from the Atacama desert. |
| `ximena-castillo` | Ximènà Castillò 🇵🇪 | Data Science, AI Ethics & Safety | Ximènà is a data scientist from Cusco who works with potato farmers and researchers to protect hundreds of native potato varieties. |
| `sebastian-vargas` | Sebastiàn Vargàs 🇻🇪 | Prompt Engineering, Large Language Models | Sebastiàn is a freelance developer in Caracas who builds AI chatbots for clinics, schools and shops across Latin America. |
| `marianna-sanchez` | Mariànnà Sanchèz 🇪🇨 | Robotics, Computer Vision | Mariànnà is a marine robotics engineer from Quito who builds underwater robots that survey reefs and sea life around the Galápagos Islands. |
| `diego-ramirez` | Diègo Ramirèz 🇧🇴 | AI Policy & Regulation, AI for Business | Diègo is a policy researcher in La Paz who studies how AI and data rules affect countries rich in resources like lithium. |
| `fernanda-oliveira` | Fernandà Oliveirà 🇧🇷 | Prompt Engineering, Generative AI for Creators | Fernandà is a copywriter in Rio de Janeiro who teaches marketers and creators to write prompts as sharp as their best headlines. |
| `aroha-ngata` | Arohà Ngàta 🇳🇿 | AI Ethics & Safety, Natural Language Processing | Arohà is a data scientist in Rotorua who works on speech technology for te reo Māori and on communities' rights over their own data. |
| `lachlanee-mitchell` | Lachlaneè Mitchèll 🇦🇺 | Robotics, MLOps & Deployment | Lachlaneè is an engineer in Perth who keeps self-driving mining trucks running safely across the Pilbara. |
| `tiare-teriifaatia` | Tiàre Teriifaàtia 🇵🇫 | Data Science, Computer Vision | Tiàre is a marine data scientist in Papeete who tracks coral reef health from photos taken by divers and drones. |
| `kaimana-kahananui` | Kaimànà Kahananùi 🇺🇸 | Math for AI, AI Agents | Kaimànà is a maths teacher in Hilo and a student of traditional Polynesian wayfinding, which crosses oceans by reading stars, swells and birds. |
| `merelaisa-tuivaga` | Merelaisà Tuivagà 🇫🇯 | AI Policy & Regulation, Data Science | Merelaisà is a climate policy adviser in Suva who uses data and AI to plan for cyclones and rising seas across Fiji's islands. |
| `braideun-osullivan` | Braideùn O'Sullivàn 🇦🇺 | Prompt Engineering, Large Language Models | Braideùn is an AI developer in Melbourne who builds writing and research assistants for universities and newsrooms. |
| `aleksandra-sorokina` | Aleksandrà Sorokinà 🇦🇶 | Data Science, Math for AI | Aleksandrà is a climate data scientist who spends her winters at a research station in Antarctica, where ice cores hold hundreds of thousands of years of climate history. |
| `erika-lindstrome` | Erikà Lindströmè 🇦🇶 | AI Agents, Robotics | Erikà is a robotics engineer who works summers at a Swedish research station in Antarctica, sending autonomous robots under the sea ice. |
