'use strict';

/**
 * Browser test: drives the real page in Chromium against the app backed by a
 * fake SlideShare. Requires Playwright with Chromium (`npm i -D playwright`
 * and `npx playwright install chromium` if it is not already available).
 *
 * Set SCREENSHOT_DIR to save screenshots of the page.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const { PDFDocument } = require('pdf-lib');
const JSZip = require('jszip');
const { createApp } = require('../src/app');
const { createFakeSlideShare, startServer } = require('./helpers');

let chromium;
try {
  ({ chromium } = require('playwright'));
} catch {
  chromium = null;
}

const DECK = 'https://www.slideshare.net/janedoe/intro-to-machine-learning-262212345';
const shots = process.env.SCREENSHOT_DIR;

test('browser: look up a deck, choose slides and download every format', { skip: !chromium && 'playwright not installed' }, async (t) => {
  const fake = createFakeSlideShare();
  const server = await startServer(createApp({ fetch: fake.fetch, logger: { error() {} } }));
  const browser = await chromium.launch();
  t.after(async () => {
    await browser.close();
    await server.close();
  });

  const context = await browser.newContext({ acceptDownloads: true, viewport: { width: 1280, height: 900 } });
  // Google Fonts is not reachable in tests; fall back to system fonts quickly.
  await context.route(/fonts\.(googleapis|gstatic)\.com/, (route) => route.abort());
  const page = await context.newPage();
  const consoleErrors = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error' && !/fonts\.googleapis|ERR_FAILED/.test(msg.text())) consoleErrors.push(msg.text());
  });
  page.on('pageerror', (err) => consoleErrors.push(String(err)));

  await page.goto(server.base);
  await assert.doesNotReject(page.getByRole('heading', { level: 1 }).waitFor());
  if (shots) await page.screenshot({ path: path.join(shots, 'home.png'), fullPage: true });

  // A link that is not SlideShare is refused in the page.
  await page.getByLabel('SlideShare link').fill('https://example.com/deck');
  await page.getByRole('button', { name: 'Show slides' }).click();
  await assert.doesNotReject(page.getByRole('alert').filter({ hasText: 'Only slideshare.net links' }).waitFor());

  // A real presentation link shows every slide.
  await page.getByLabel('SlideShare link').fill(DECK);
  await page.getByRole('button', { name: 'Show slides' }).click();
  await page.locator('#deck-title', { hasText: 'Intro to Machine Learning' }).waitFor();
  const mounts = page.locator('button.mount');
  assert.equal(await mounts.count(), 6);
  assert.equal(await page.locator('#deck-meta').textContent(), '6 slides, all selected');
  assert.equal(await page.getByRole('alert').isHidden(), true, 'error from the previous attempt is cleared');

  // Previews actually load through the image proxy.
  await page.waitForFunction(() => [...document.querySelectorAll('.mount img')].every((i) => i.complete && i.naturalWidth > 0));
  assert.match(page.url(), /\?url=https%3A%2F%2Fwww\.slideshare\.net/);
  if (shots) await page.screenshot({ path: path.join(shots, 'results.png'), fullPage: true });

  // Leave slide 2 out.
  await page.getByRole('button', { name: 'Slide 2, included in download' }).click();
  assert.equal(await page.locator('#deck-meta').textContent(), '6 slides, 5 selected');
  assert.equal(await mounts.nth(1).getAttribute('aria-pressed'), 'false');

  // PDF with the 5 selected slides.
  let [dl] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Download PDF' }).click()]);
  assert.equal(dl.suggestedFilename(), 'Intro to Machine Learning.pdf');
  const pdf = await PDFDocument.load(fs.readFileSync(await dl.path()));
  assert.equal(pdf.getPageCount(), 5);
  await page.locator('#download-status', { hasText: 'Downloaded Intro to Machine Learning.pdf' }).waitFor();

  // PowerPoint.
  [dl] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Download PowerPoint' }).click()]);
  assert.equal(dl.suggestedFilename(), 'Intro to Machine Learning.pptx');
  const pptx = await JSZip.loadAsync(fs.readFileSync(await dl.path()));
  assert.equal(Object.keys(pptx.files).filter((n) => /^ppt\/slides\/slide\d+\.xml$/.test(n)).length, 5);

  // ZIP of images, after selecting everything again.
  await page.getByRole('button', { name: 'Select all' }).click();
  [dl] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Download images (ZIP)' }).click()]);
  assert.equal(dl.suggestedFilename(), 'Intro to Machine Learning.zip');
  const zip = await JSZip.loadAsync(fs.readFileSync(await dl.path()));
  assert.equal(Object.keys(zip.files).length, 6);

  // A single slide.
  [dl] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: 'Save slide 3 as an image' }).click()]);
  assert.equal(dl.suggestedFilename(), 'Intro to Machine Learning - slide 3.jpg');

  // Nothing selected gives guidance instead of an empty file.
  await page.getByRole('button', { name: 'Clear selection' }).click();
  await page.getByRole('button', { name: 'Download PDF' }).click();
  await page.locator('#download-status', { hasText: 'Select at least one slide' }).waitFor();

  // Opening a shared ?url= link loads the deck straight away.
  const shared = await context.newPage();
  await shared.goto(`${server.base}/?url=${encodeURIComponent(DECK)}`);
  await shared.locator('#deck-title', { hasText: 'Intro to Machine Learning' }).waitFor();
  assert.equal(await shared.locator('button.mount').count(), 6);

  // Phone layout has no horizontal scroll.
  const phone = await browser.newContext({ viewport: { width: 375, height: 800 }, deviceScaleFactor: 1 });
  await phone.route(/fonts\.(googleapis|gstatic)\.com/, (route) => route.abort());
  const mobile = await phone.newPage();
  await mobile.goto(`${server.base}/?url=${encodeURIComponent(DECK)}`);
  await mobile.locator('button.mount').first().waitFor();
  await mobile.waitForFunction(() => [...document.querySelectorAll('.mount img')].every((i) => i.complete && i.naturalWidth > 0));
  const overflow = await mobile.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  assert.equal(overflow, 0, 'no horizontal scroll at 375px');
  if (shots) await mobile.screenshot({ path: path.join(shots, 'mobile.png'), fullPage: true });
  await phone.close();

  assert.deepEqual(consoleErrors, []);
});
