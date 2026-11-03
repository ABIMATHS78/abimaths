// Renders cover.html to a pixel-exact 1584 x 396 PNG (LinkedIn's personal
// profile background size).
//
//   node linkedin-cover/render.cjs                      -> linkedin-cover/linkedin-cover-1584x396.png
//   node linkedin-cover/render.cjs out.png --debug      -> same, with safe-zone guides drawn on top
//
// Needs Playwright with Chromium (npm i -D playwright && npx playwright install chromium).
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require('playwright');

const W = 1584, H = 396;
const args = process.argv.slice(2);
const debug = args.includes('--debug');
const out = path.resolve(args.find(a => !a.startsWith('--')) ?? path.join(__dirname, 'linkedin-cover-1584x396.png'));

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  const url = pathToFileURL(path.join(__dirname, 'cover.html')).href + (debug ? '?debug' : '');
  await page.goto(url);
  await page.waitForSelector('body[data-ready="1"]');
  // The design's fonts come from Google Fonts; warn instead of silently rendering a fallback.
  const missing = await page.evaluate(() =>
    ["500 88px 'Bodoni Moda'", "400 30px 'Inter'"].filter(f => !document.fonts.check(f)));
  if (missing.length) console.warn(`warning: fonts not loaded (offline?): ${missing.join(', ')}`);
  await page.screenshot({ path: out, clip: { x: 0, y: 0, width: W, height: H } });
  await browser.close();
  console.log(`wrote ${out}`);
})();
