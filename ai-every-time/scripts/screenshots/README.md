# Product screenshots

Screenshots of AI Every Time exactly as a learner sees it, and slides for social media made from them, from the
welcome page and signing up to chatting, customising and learning with a companion.

## Making them

From `ai-every-time/`:

```bash
pip install -r requirements.txt -r scripts/screenshots/requirements.txt
playwright install chromium        # not needed where Playwright's Chromium is already installed
export ANTHROPIC_API_KEY=sk-ant-...  # so the companions' words are Claude's (see below)
python scripts/screenshots/take.py screenshots
python scripts/screenshots/slides.py screenshots
```

`take.py` starts the app itself with a new database in the output folder (`screenshots/app.db`), fetches the
starter companions' real photos, loads the guides, and goes through the app twice as a new learner, Amara:

- **desktop**: 1440 x 900 browser window, saved at twice the pixels (2880 x 1800), in `screenshots/desktop/`
- **phone**: 390 x 844 screen, saved at three times (1170 x 2532), in `screenshots/phone/`

`slides.py` then makes the slides, 1080 x 1350 (the portrait size Instagram, LinkedIn and Facebook carousels
use), in `screenshots/slides/`: a headline and a line about each step, above the real screens shown in a browser
window with the phone beside them. The screenshots are left exactly as they were taken.

## The screens

| File | What it shows | Claude writes it |
|---|---|---|
| `01-welcome` | The welcome page, with the companions' faces | |
| `02-signup` | The sign-up form filled in: level, goals and interests (football, cooking, Afrobeats) | |
| `03-choose` | Choosing a companion: all 103, by region | |
| `04-chat-opener` | Amahlé Ndlovù starting the chat from Amara's interests | yes |
| `05-chat-reply` | Amara's reply and Amahlé's answer | yes |
| `06-customise` | Customising Agnieszka Kowalskà, with her photo | |
| `07-customise-face` | The face controls, and the choice between the photo and a drawn face | |
| `08-guide` | A guide (Mastering DuckDB), with its corrected graphic | |
| `09-learn` | The learning paths | |
| `10-lesson` | A lesson in Amahlé's voice | yes |
| `11-quiz` | A quiz on it | yes |
| `12-quiz-results` | The results, with explanations | yes |
| `13-feed` | The feed | yes |
| `14-progress` | Progress, after the lesson and quiz | |
| `15-companions` | The companions from around the world | |

The slides follow the same journey: meet, sign up, choose, chat, customise, guides, lesson (or the learning paths),
quiz, feed and progress. A slide whose screenshots aren't there is left out.

## Without a Claude key

The app runs in demo mode: companions' words are templates no learner sees once a key is set. So `take.py` leaves
out the screens Claude writes, and hides the yellow demo-mode banner, which learners don't see either. `--all`
takes those screens anyway, to try the steps; their words are demo text, not for posting.

With a key, a run makes about 10 AI requests (each chat's opener and reply, the lesson, each quiz), and waits up
to four minutes for any one of them.

## Things to know

- **The guides library isn't in the set.** Most guides still show the site's original graphics, with the image
  generator's spelling mistakes (`scripts/graphics/` corrects them; 18 are done). The guide shown is one whose
  graphic is corrected. Once the newest guides' graphics are corrected, the library itself makes a good screen.
- **Pictures from other sites** (the guides' graphics) are fetched by the script and handed to the browser, so
  they load wherever Python can reach the internet, through a proxy too.
- **Options:** `--only desktop` or `--only phone` takes just one of the two; `CHROMIUM=/path/to/chrome` picks the
  browser.
