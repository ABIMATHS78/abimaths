# AI Every Time on the App Store

What to fill in on App Store Connect, ready to copy. [README.md](README.md) says where each part goes. The privacy
answers must match the app's privacy manifest (`AIEveryTime/PrivacyInfo.xcprivacy`) and the site's privacy page:
change all three together.

## The app's page

| Field | Value |
|---|---|
| Name (30 characters at most) | AI Every Time |
| Subtitle (30) | Learn AI with AI companions |
| Category | Education; second category Productivity |
| Price | Free |
| Support URL | https://www.aieverytime.org/support |
| Marketing URL | https://www.aieverytime.org |
| Privacy Policy URL | https://www.aieverytime.org/privacy |
| Copyright | 2026 and the name you publish under, such as `2026 AI Every Time Ltd` |

**Promotional text** (170 characters at most; you can change it any time without a new version):

> Chat with an AI companion who remembers you, then learn by doing: courses with certificates, quizzes, flashcards
> and over a thousand guides.

**Keywords** (100 characters at most, separated by commas):

```
learn AI,artificial intelligence,machine learning,AI course,AI tutor,prompts,flashcards,quiz,LLM
```

**Description** (4,000 characters at most):

```
AI Every Time is the friendly way to learn about AI, with AI companions from around the world who are there whenever you are.

Pick a companion, or design your own: their face, country, personality, teaching style and voice. Ask them anything about AI and they explain it your way, with examples from what you're into, and they remember what you talked about next time.

LEARN BY DOING
• Courses with lessons, labs, projects and quizzes, and a certificate when you finish
• Learning paths from the basics of AI to agents and building with it
• Over a thousand guides and cheat sheets written by the companions
• A tutor beside every lesson, for the questions you think of as you read

REMEMBER IT
• Quizzes and mock exams from any lesson, guide or conversation
• Flashcards that come back just before you'd forget them
• A study plan that fits your week, and a streak to keep

MADE FOR YOU
• Companions that know your level, your goals and what you're into
• A daily reminder at the time you choose
• Replies read aloud in your companion's voice

IN THE APP
• Notifications for your daily reminder and news about your companions
• Share your certificates, and invite friends, with the share sheet
• Touch and hold the app's icon to chat with your companion or carry on learning, or ask Siri

Companions are Claude, an AI made by Anthropic: what you write to them goes to Anthropic so it can reply. You can turn that off in your profile.

Design companions and share them with the community. Every community companion can be reported, and its designer blocked.

AI Every Time is free to use, with limits. Your account works on the web at aieverytime.org too.
```

**What's new in this version**, for later versions: a sentence or two on what changed, such as
`Notifications now open the page they're about, and the app starts faster.`

## Screenshots

In `screenshots/`, made from the site as the app shows it:

- `iphone-6.9/`: 1320 × 2868, for **iPhone 6.9" Display**. App Store Connect uses them for smaller iPhones too.
- `ipad-13/`: 2064 × 2752, for **iPad 13" Display**, needed because the app runs on iPads.

Drag each folder's images into its size on the version's page, in the order of their names. After the site's look
changes, `python ios/screenshots/make.py` (from `ai-every-time/`) makes them again, with the captions in it.

## App Privacy

In App Store Connect, open **App Privacy** and choose **Get Started**.

- **Do you or your third-party partners collect data from this app?** Yes.
- Choose these data types, then answer for each: **linked to the user's identity: Yes**, **used for tracking: No**,
  and these purposes:

| Data type | Purposes |
|---|---|
| Contact Info: Name | App Functionality |
| Contact Info: Email Address | App Functionality; Developer's Advertising or Marketing (news and offers, only to those who say yes) |
| User Content: Photos or Videos | App Functionality (a photo a learner gives a companion they design) |
| User Content: Other User Content | App Functionality; Product Personalization (chats, what companions remember, companions designed, projects, goals and interests) |
| Identifiers: User ID | App Functionality |
| Identifiers: Device ID | App Functionality (the device's address for notifications) |
| Usage Data: Product Interaction | App Functionality; Analytics (progress, streaks, guides read, AI requests) |
| Purchases: Purchase History | App Functionality (the plan a learner has, bought on the website) |

Nothing is used for tracking, so the app never asks to track and the page shows **Data Not Used to Track You**.

## Age rating

Answer **None** or **No** to every question about content (violence, sexual content, profanity, horror, alcohol and
drugs, gambling, contests, weapons, medical and wellness topics), and:

- **Parental controls / age assurance:** No.
- **Unrestricted web access:** No. The app shows AI Every Time; other sites open in Safari's viewer.
- **User-generated content:** Yes. Learners design companions and can publish them to the community.
- **Messaging and chat:** learners don't message each other, so No if it asks about people messaging each other;
  learners chat with AI companions, so Yes if it asks about AI chat.
- **Advertising:** No.

The privacy page says AI Every Time isn't for children under 13. If App Store Connect lets you choose a higher age
rating than your answers give, choose 13+.

## App Review Information

- **Sign-in required:** Yes. Make an account for Apple on the site (for example `appreview@aieverytime.org`, with a
  password only Apple will use), and as an admin give it a plan on the People tab of the Staff page, so the reviewer
  can try everything. Put its email and password in the **User name** and **Password** fields, never in the notes
  or this repository.
- **Contact information:** a name, phone number and email address Apple can reach while the app is reviewed.
- **Notes:**

```
AI Every Time is a place to learn about AI with AI companions: learners chat with a companion, follow its lessons, take quizzes and flashcards, and take courses that end in a certificate. The app shows our website, www.aieverytime.org, and adds what only an app can: notifications through Apple's push service, the share sheet, actions on the app's icon, Siri and Spotlight shortcuts, and an offline screen.

To try it with the account above:
1. Sign in. Companions are Claude, Anthropic's AI, so what learners write to them goes to Anthropic: sign-up says so, and Profile, under "Your data", is where learners turn it off.
2. Open "Chat" (or a companion from "Companions") and ask a question.
3. In Profile, turn notifications on, then "Send a test".
4. Touch and hold the app's icon for its shortcuts.

Purchases: the app sells nothing, and shows no prices or links to buy. Everyone can use AI Every Time free, with daily limits. Some learners have plans they bought on our website, and in the app they simply have what their plan includes. This review account has a plan so you can see everything.

User-generated content: learners can design AI companions and publish them to the community. Every community companion can be reported from its page (our moderators review reports, and a companion reported by several learners is hidden until they do) and its designer blocked ("Block designer"). Learners can't message each other. Contact: support@aieverytime.org, also on the Help page.

Account deletion: Profile, then "Delete my account".
```

## TestFlight: beta review for outside testers

| Field | Value |
|---|---|
| Beta App Description | Learn about AI with AI companions: chat, lessons, quizzes, flashcards and courses. This is a test version of the app. |
| Feedback Email | support@aieverytime.org |
| Privacy Policy URL | https://www.aieverytime.org/privacy |
| Sign-in information | The account made for App Review, above |
| What to Test | What's new in the build, or: `Sign in, chat with your companion, turn on notifications in Profile and send a test, and touch and hold the app's icon.` |
