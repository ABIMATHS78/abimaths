# mojeedabisiga.com

The official website of **Mojeed Abisiga (Abisiga Mojeed Damilola)**, and the structured data that supports his Google knowledge panel.

It is a plain static site: HTML, CSS and a little JavaScript, with no framework and no build step. The site is the "entity home": the page Google treats as the most authoritative source about him. Every page carries the same `Person` structured data (JSON-LD), which links his names, his Wikidata item (Q137917376), 14 official profiles, his schools and the companies he founded.

## What's here

| Path | What it is |
|---|---|
| `site/` | The website. Everything in this folder is published as is. |
| `site/index.html` | Home: the entity home, with `ProfilePage`, `Person`, `WebSite` and `Organization` structured data |
| `site/about/index.html` | Full biography with citations (`AboutPage`) |
| `site/press/index.html` | Press coverage, company announcements and a media kit with copyable bios (`CollectionPage`) |
| `site/assets/` | Stylesheet (colour tokens at the top, light and dark themes), script, headshot, share image, icons |
| `tools/check_site.py` | The checks that run before every publish (below). Standard library only |
| `wikidata/` | QuickStatements batches for the Wikidata item [Q137917376](https://www.wikidata.org/wiki/Q137917376). Run them after the site is live |
| `../.github/workflows/pages-mojeedabisiga.yml` | Checks the site and publishes it to GitHub Pages |

## How publishing works

- Change files in `site/` on a branch and open a pull request. The workflow runs the checks on the pull request.
- Merge to `master`. The workflow checks the site again and publishes `site/` to GitHub Pages at **https://mojeedabisiga.com**.
- The workflow only runs when the website, its checker or the workflow itself changes. AI Every Time app and iOS changes never trigger it. Website changes never trigger a Render deploy, which only watches `ai-every-time/`, or an iOS build, which only watches `ai-every-time/ios/`.

### What the checks cover

`python3 mojeedabisiga.com/tools/check_site.py mojeedabisiga.com/site` fails if any of these is wrong:

- The structured data on any page isn't valid JSON.
- The `Person` entity differs between pages. It must stay identical so Google sees one consistent set of facts.
- An internal link, image, stylesheet, script or `#anchor` points to something that doesn't exist.
- A page is missing its title, or an indexable page is missing its meta description or canonical URL.

## One-time setup

1. **GitHub Pages:** in **Settings → Pages**, set *Source* to **GitHub Actions** and *Custom domain* to `mojeedabisiga.com`. Once the certificate is issued, tick **Enforce HTTPS**.
2. **DNS at Squarespace** (**Domains → mojeedabisiga.com → DNS → DNS Settings**): delete the *Squarespace Defaults* records, leaving any email (MX) records alone, then add:

   | Type | Host | Data |
   |---|---|---|
   | A | @ | `185.199.108.153` |
   | A | @ | `185.199.109.153` |
   | A | @ | `185.199.110.153` |
   | A | @ | `185.199.111.153` |
   | CNAME | www | `abimaths78.github.io` |

   `www.mojeedabisiga.com` then forwards to `mojeedabisiga.com`, the main address used in the structured data.
3. **Google Search Console:** add a *Domain* property for `mojeedabisiga.com`. Verify it with the TXT record Google gives you, added in the same Squarespace DNS screen. Submit `https://mojeedabisiga.com/sitemap.xml`, then use *URL Inspection → Request indexing* for `/`, `/about/` and `/press/`.
4. **After launch:** test the home page at <https://search.google.com/test/rich-results> and <https://validator.schema.org/>. Then run the Wikidata batches in `wikidata/` with QuickStatements; delete the date-of-birth line first if it should stay private.

## Editing

- **Text:** edit the HTML in `site/` directly. Keep facts (names, titles, dates) the same on all three pages.
- **Structured data:** each page has one `<script type="application/ld+json">` block in its `<head>`. The `Person` part is identical on every page, and the check enforces that. When a fact changes, change it in all three files.
- **A new profile (YouTube, X and so on):** add its URL to the `sameAs` list in all three JSON-LD blocks, to the footer's "Elsewhere" list, and to "Official profiles" in `site/about/index.html`.
- **Photo:** replace `site/assets/img/mojeed-abisiga.jpg` and the two `.webp` versions with a professional headshot of at least 1200×1200 px, keeping the file names.
- **After any content change:** update `<lastmod>` in `site/sitemap.xml` and `dateModified` in the JSON-LD.
- **Contact email:** none is published yet; visitors are pointed to LinkedIn and Instagram. To add one, put a `mailto:` button in the Contact section of `site/index.html`. Then add an `"email"` field to the `Person` JSON-LD on all three pages.

## Preview locally

```bash
python3 -m http.server 8000 --directory mojeedabisiga.com/site
```

Then open <http://localhost:8000>.

The knowledge-panel playbook and the research notes behind this site are kept outside this public repository on purpose.
