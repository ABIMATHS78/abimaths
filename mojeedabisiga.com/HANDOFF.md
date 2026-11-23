# Handoff: mojeedabisiga.com and Mojeed Abisiga's Google knowledge panel

Paste this whole file into a new Claude Code session that has the GitHub repository **ABIMATHS78/abimaths** attached. It explains what was built, where things stand, and what to do next.

## 1. Goal

Mojeed Abisiga (full name **Abisiga Mojeed Damilola**, handle **@abimaths**) wants a Google knowledge panel that shows when people search his name. A panel already exists, and Google offers him "Claim this knowledge panel".

The strategy is: claim the panel, give Google one authoritative "entity home" (his website), make every profile link back to it, and strengthen the structured data and Wikidata.

## 2. Current state (23 Nov 2026)

| Item | Status |
|---|---|
| Website | **Live** at https://mojeedabisiga.com. Plain static HTML/CSS/JS with no build step. |
| HTTPS | Working. Let's Encrypt certificate for both the apex and www; www redirects to the apex; HTTP redirects to HTTPS. |
| Hosting | GitHub Pages from repo `ABIMATHS78/abimaths`, published by the workflow `.github/workflows/pages-mojeedabisiga.yml`. Source = GitHub Actions; custom domain = `mojeedabisiga.com`. |
| DNS (Squarespace) | A @ → 185.199.108.153, .109.153, .110.153, .111.153; CNAME www → abimaths78.github.io. The "Squarespace Defaults" preset was deleted and must **not** be re-added. The MX (Google Workspace) and TXT (google-site-verification) records exist; leave them alone. |
| First PR (#51) | Merged. Added the site, checker, workflow and Wikidata batches. |
| Second PR (#52) | Merged 23 Nov 2026 and live. Added the **/achievements/** page built from his CV, CV details on the Home, About and Press pages, his headshot as the favicon, header logo and social image, and two editorial portraits. Merging to `master` deploys automatically in about a minute. |
| Wikidata | Item **Q137917376** filled in by hand on 23 Nov 2026, because QuickStatements needs an autoconfirmed account (4 days old, 50 edits). It now has 21 statements: the official website (language: English), four aliases, nine profile IDs, citizenship, birthplace, birth date, occupations, education and employers, all with references. Batch 1 is therefore done. Batch 2 (optional) waits until the account is autoconfirmed. |
| Search Console | Done 23 Nov 2026. A Domain property for `mojeedabisiga.com` was already verified, the sitemap was submitted, and indexing was requested for `/`, `/about/`, `/achievements/` and `/press/`. |
| Knowledge panel claim | Not done yet. |

## 3. Repository layout (`mojeedabisiga.com/` in the repo)

- `site/`: everything that gets published.
  - Pages: `index.html`, `about/`, `achievements/`, `press/` and `404.html`.
  - Site files: `sitemap.xml`, `robots.txt`, `site.webmanifest` and `CNAME`.
  - Assets: `assets/css/styles.css` (colour tokens at the top; light and dark themes), `assets/js/main.js` and `assets/img/`.
- `tools/check_site.py`: the pre-publish check, which runs in CI on every pull request and on every push to master. Run it before every commit:
  `python3 mojeedabisiga.com/tools/check_site.py mojeedabisiga.com/site`
  It checks that every page's structured data (JSON-LD) is valid, that the **Person** node is **identical on every page**, that all internal links and #anchors resolve, and that each page has a title, a meta description and a canonical URL.
- `wikidata/`: QuickStatements batches for Q137917376. Run `1-update-…` first; `2-optional-…` creates items for DataGlobal Hub and his arXiv paper.
- `README.md`: the editing and publishing guide.

**Editing rules:**
- Keep facts identical across pages.
- When the Person JSON-LD changes, change it on **all four** pages. A small Python script that loads each page's JSON-LD block, edits the Person node and writes it back works well.
- After content changes, update `<lastmod>` in the sitemap and `dateModified` in the JSON-LD.
- Use a branch, then a pull request, then a merge. Do not push to master.

## 4. Facts the site states (keep these consistent everywhere)

| Fact | Value |
|---|---|
| Names | Mojeed Abisiga; Abisiga Mojeed Damilola; @abimaths |
| Birth | Born 17 May 1996 in Surulere, Lagos, Nigeria. The owner chose to keep this published. |
| Nationality and residence | Nigerian; U.S.-based; EB-1A "Einstein visa" green card |
| AI Every Time | Founder & CEO. https://www.aieverytime.org/ |
| DataGlobal Hub | Co-founder (with Justin Lewis), CEO and editor-in-chief. Runs the Global Data & AI Virtual Tech Conference (GDAI). |
| Other roles | Co-founder & COO of DataRango; chief data officer and advisory board member of The Strategemist |
| Forbes | Forbes Technology Council member since 2025 |
| Education | University of Lagos, B.Sc. Mathematics, First Class Honours (2018). UNC Greensboro, M.S. Informatics & Analytics, 4.00/4.00, summa cum laude (May 2023). |
| Mathematics | NAMCUS gold medal (2018). One of three students who represented Nigeria at the 25th IMC, Bulgaria (2018), on the University of Lagos team. |
| Career | KPMG Nigeria (2019–2021), Deloitte (2021), big data engineering (2023–present) |

**Official profiles** (also in `sameAs`):
- Social: LinkedIn `/in/mojeed-abisiga/`; Instagram and Threads `@abimaths`; https://profile.google.com/@abimaths
- Professional: Forbes Councils profile; Crunchbase; Sessionize; Product Hunt `@mojeed_abisiga`
- Code and research: GitHub `ABIMATHS78`; Google Scholar `ZUDHiSoAAAAJ`; ResearchGate; Academia.edu
- Reference: Wikiquote; Wikidata Q137917376

**Owner decisions, all final:**
- Six discrepancies between his CV and public sources were flagged: The Strategemist title, the scope of the Deloitte role, DataGlobal Hub founder vs co-founder, the end date as Graduate Student Association senator, "most early bed student", and the date and place of birth. The owner said to **leave them as they are now**. Don't raise them again.
- The climate-change item follows the CV: Lagos State quiz winner (2010), then a climate-change research tour of Germany (2011) sponsored by the Ministry of Education and Environment.

**Privacy:**
- Never publish his street address, phone number or personal email (they are on his CV).
- Contact on the site goes through LinkedIn and Instagram messages only.
- Don't put his email in requests to outside services.

The research notes (`RESEARCH_DOSSIER.md`) and the playbook (`KNOWLEDGE_PANEL_PLAYBOOK.md`) were kept out of the public repo. Ask the owner if you need them.

## 5. Next steps, in order

1. ~~Merge PR #52~~ Done. /achievements/, the favicon and the share image are live and confirmed.
2. **Google Search Console** (owner, with your guidance):
   - Add a Domain property for `mojeedabisiga.com` and verify it with a TXT record in Squarespace DNS.
   - Submit `https://mojeedabisiga.com/sitemap.xml`.
   - Use URL Inspection → Request indexing for `/`, `/about/`, `/achievements/` and `/press/`.
3. **Claim the knowledge panel**: search his name on Google, click **Claim this knowledge panel**, and complete "Get verified on Google". Once verified, use **Suggest edits** to set his official website, photo and description.
4. **Link profiles back to the site**: add https://mojeedabisiga.com as the website on LinkedIn, Instagram, Threads, the Google profile, GitHub, Google Scholar, Crunchbase, Sessionize, Product Hunt and ResearchGate.
5. **Wikidata**:
   - Run `wikidata/1-update-Q137917376.qs.txt` in QuickStatements while logged in to Wikidata. It adds the official website, aliases, education, employers and external IDs, each with references.
   - Optionally run batch 2 afterwards.
   - Don't create a Wikipedia article. The earlier one was deleted, and the conflict-of-interest rules apply.
6. **Validate**: test the home page at https://search.google.com/test/rich-results and https://validator.schema.org/.
7. **Later**: earned press coverage, and keeping the site's facts current. Add each new profile to `sameAs` on all pages, to the footer and to the About page's "Official profiles" list.

## 6. Gotchas from the cloud-session environment

- **GitHub API through the session proxy:**
  - Only repo-scoped REST calls work. GraphQL fails, so `gh pr create` fails. Create pull requests with REST instead: `gh api repos/ABIMATHS78/abimaths/pulls -f title=… -f head=… -f base=master -F body=@file`.
  - The Pages endpoints (`repos/…/pages`, `pages/health`) are blocked. Pages settings (custom domain, **Check again**, Enforce HTTPS) must be changed by the owner at https://github.com/ABIMATHS78/abimaths/settings/pages.
- **Stop-hook "unpushed commits" false alarm:** the clone is shallow and single-branch, so a new branch has no tracking ref. Fix it with:
  `git config --add remote.origin.fetch '+refs/heads/<branch>:refs/remotes/origin/<branch>' && git fetch origin <branch> && git branch --set-upstream-to=origin/<branch>`
- **DNS and HTTP checks:**
  - Port 53 is blocked, so use DNS over HTTPS, e.g. `https://dns.google/resolve?name=…&type=A`.
  - Plain HTTP requests ignore `curl --resolve`.
  - For HTTPS, `curl -v` shows the upstream certificate's subject through the proxy.
- **Past incident:** when the DNS moved from Squarespace, GitHub's check failed with NotServedByPagesError for about 4 hours. The old Squarespace records had a 4-hour lifetime, and GitHub kept using its cached copy until it expired. The fix was to wait, then click **Check again**. If DNS ever changes again, expect the same delay.
- **Images:** strip photo metadata (camera details, dates, any location) before publishing. Re-saving through Pillow without EXIF works. Playwright and Chromium are preinstalled for screenshots and accessibility checks with axe-core.
