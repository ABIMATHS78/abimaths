---
name: idea-to-website
description: Takes a website idea to a live site by agreeing the goal and pages, designing a simple look, building it, putting the code on GitHub and deploying it. Use when someone wants to launch a site.
---

# Idea to website

Get a real site live, small and good, then improve it. Work in five phases and finish each before the next.

## Phase 1: Plan (agree before building)

- What is the site for, and what should a visitor do? (Join, buy, contact, read.)
- Who are the visitors, and on what devices?
- Pages: start with the fewest that work, often one page with sections.
- Content: what text, images and logo exist? Mark the gaps.
- Domain name: do they have one? Where will it be hosted?

Write a one-page plan: goal, audience, pages and sections, content list, and launch checklist.

## Phase 2: Design

- Pick one main colour, one accent and a neutral, with text that passes contrast checks (4.5:1 for body text).
- Use one or two fonts. Body text at least 16px.
- Sketch each section in words: heading, supporting line, content, button.
- Design for phones first.

Show the design as a quick preview before building everything.

## Phase 3: Build

- For a simple site, use plain HTML and CSS: no build step, easy to host.
- For a site with many pages or app features, use a framework the person can maintain. Explain the choice in one sentence.
- Every page needs: a title and meta description, readable headings in order, alt text on images, a favicon, and links that work.
- Keep secrets (API keys, passwords) out of the code. Use the host's environment variables.

## Phase 4: Put it on GitHub

If you can run commands (for example in Claude Code), do this with the person's go-ahead. Otherwise, give them these exact steps:

```bash
git init
git add .
git commit -m "First version of the site"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

They create the empty repository on GitHub first, without a README, so the first push works.

## Phase 5: Deploy

Pick one host and walk them through it:
- **Vercel** or **Netlify**: sign in with GitHub, import the repository, accept the detected settings, deploy. Every push to `main` then redeploys.
- **GitHub Pages** for plain HTML: in the repository, open Settings, then Pages, and publish from the `main` branch.

To add their own domain, add it in the host's domain settings, then copy the DNS records it shows into their domain registrar. Changes can take up to a day to spread.

## Launch checklist

- [ ] Looks right on a phone and a laptop.
- [ ] Every link and button works; forms send.
- [ ] Spelling checked; contact details right.
- [ ] Page titles and descriptions set.
- [ ] Images compressed; the page loads fast.
- [ ] Privacy notice if it collects data or uses analytics.
- [ ] No secrets in the repository.

Finish by giving them the live address and three ideas for version two.
