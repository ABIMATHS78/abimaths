// Exports the LinkedIn banner at exactly 1584 × 396 px, plus a layout preview.
//   npm install playwright && npx playwright install chromium   (first time only)
//   node render.js
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const W = 1584, H = 396;   // LinkedIn's recommended profile background size (4:1)
const here = (f) => path.join(__dirname, f);
const url = (f, q = '') => 'file://' + here(f) + q;

(async () => {
  const browser = await chromium.launch();

  // 1. The banner itself, 1 CSS px = 1 image px.
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  await page.goto(url('banner.html'));
  await page.waitForFunction(() => window.__ready === true);
  const out = here('linkedin-banner-1584x396.png');
  await page.screenshot({ path: out, clip: { x: 0, y: 0, width: W, height: H } });

  const png = fs.readFileSync(out);
  const w = png.readUInt32BE(16), h = png.readUInt32BE(20);
  if (w !== W || h !== H) throw new Error(`Expected ${W}×${H}, got ${w}×${h}`);
  console.log(`linkedin-banner-1584x396.png  ${w}×${h}  ${(png.length / 1024).toFixed(0)} KB`);

  // 2. Preview of the banner in LinkedIn-like desktop and mobile layouts (2× for sharpness).
  const prev = await browser.newPage({ viewport: { width: 1300, height: 600 }, deviceScaleFactor: 2 });
  await prev.goto(url('preview.html'));
  await prev.evaluate(() => Promise.all([document.fonts.ready,
    ...[...document.images].map(i => i.complete ? null : new Promise(r => { i.onload = i.onerror = r; }))]));
  await prev.locator('.wrap').screenshot({ path: here('linkedin-banner-preview.png') });
  console.log('linkedin-banner-preview.png');

  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
