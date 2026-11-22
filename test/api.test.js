'use strict';

/**
 * End-to-end tests of the HTTP API against a fake SlideShare (see helpers.js).
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const { PDFDocument } = require('pdf-lib');
const JSZip = require('jszip');
const { createApp } = require('../src/app');
const { createFakeSlideShare, startServer, image } = require('./helpers');

const DECK = 'https://www.slideshare.net/janedoe/intro-to-machine-learning-262212345';

let server;
let fake;

test.beforeEach(async () => {
  fake = createFakeSlideShare();
  server = await startServer(createApp({ fetch: fake.fetch, logger: { error() {} } }));
});

test.afterEach(async () => {
  await server.close();
});

function postSlides(url) {
  return fetch(`${server.base}/api/slides`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ url }),
  });
}

function download(params) {
  return fetch(`${server.base}/api/download?${new URLSearchParams(params)}`);
}

test('serves the page with security headers', async () => {
  const res = await fetch(`${server.base}/`);
  assert.equal(res.status, 200);
  const html = await res.text();
  assert.match(html, /<form id="lookup"/);
  assert.match(res.headers.get('content-security-policy'), /default-src 'self'/);
  assert.equal(res.headers.get('x-content-type-options'), 'nosniff');
  assert.equal(res.headers.get('x-powered-by'), null);
  for (const asset of ['/app.js', '/styles.css', '/favicon.svg']) {
    assert.equal((await fetch(server.base + asset)).status, 200, asset);
  }
});

test('POST /api/slides lists every slide of a presentation', async () => {
  const res = await postSlides(DECK);
  assert.equal(res.status, 200);
  const body = await res.json();
  assert.equal(body.title, 'Intro to Machine Learning');
  assert.equal(body.totalSlides, 6);
  assert.equal(body.slides.length, 6);
  assert.equal(body.url, DECK);
  const first = body.slides[0];
  assert.equal(first.number, 1);
  assert.match(first.thumb, /^\/api\/image\?src=https%3A%2F%2Fimage\.slidesharecdn\.com%2F.+-1-320\.jpg$/);
  assert.match(first.full, /-1-2048\.jpg$/);
  assert.match(first.download, /download=1/);
});

test('POST /api/slides accepts links without https:// and follows redirects', async () => {
  let res = await postSlides('www.slideshare.net/acme/quarterly-report-2024');
  assert.equal(res.status, 200);
  assert.equal((await res.json()).totalSlides, 5);

  res = await postSlides('https://www.slideshare.net/short-link');
  assert.equal(res.status, 200);
  const body = await res.json();
  assert.equal(body.url, DECK, 'reports the final presentation URL');
  assert.equal(body.totalSlides, 6);
});

test('POST /api/slides explains what went wrong', async () => {
  const cases = [
    ['', 400, /Paste a SlideShare link/],
    ['https://example.com/deck', 400, /Only slideshare\.net links/],
    ['https://www.slideshare.net/nobody/missing-deck', 404, /does not exist/],
    ['https://www.slideshare.net/janedoe', 422, /No slides were found/],
    ['https://www.slideshare.net/rate-limited', 502, /refused the request/],
    ['https://www.slideshare.net/evil-redirect', 400, /redirected somewhere/],
  ];
  for (const [url, status, message] of cases) {
    const res = await postSlides(url);
    assert.equal(res.status, status, url);
    const body = await res.json();
    assert.match(body.error, message, url);
  }
  assert.ok(!fake.requests.some((u) => u.includes('169.254.169.254')), 'never followed the redirect to a private address');
  assert.ok(!fake.requests.some((u) => u.includes('example.com')), 'never fetched a non-SlideShare page');
});

test('POST /api/slides rejects malformed JSON', async () => {
  const res = await fetch(`${server.base}/api/slides`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: '{"url":',
  });
  assert.equal(res.status, 400);
  assert.equal((await res.json()).code, 'bad_json');
});

test('the presentation page is fetched once and then cached', async () => {
  await postSlides(DECK);
  await postSlides(DECK);
  await download({ url: DECK, format: 'zip', slides: '1' });
  const pageFetches = fake.requests.filter((u) => u.includes('slideshare.net/janedoe/intro'));
  assert.equal(pageFetches.length, 1);
});

test('GET /api/image proxies slide images and can force a download', async () => {
  const src = 'https://image.slidesharecdn.com/introtomachinelearning-231004101112-5c1a2b3d/85/Intro-to-Machine-Learning-2-320.jpg';
  let res = await fetch(`${server.base}/api/image?src=${encodeURIComponent(src)}`);
  assert.equal(res.status, 200);
  assert.equal(res.headers.get('content-type'), 'image/jpeg');
  assert.ok(Buffer.from(await res.arrayBuffer()).equals(image('slide-2.jpg')));
  assert.equal(res.headers.get('content-disposition'), null);

  res = await fetch(`${server.base}/api/image?src=${encodeURIComponent(src)}&download=1&name=${encodeURIComponent('Intro to ML - slide 2')}`);
  assert.equal(res.status, 200);
  assert.equal(
    res.headers.get('content-disposition'),
    `attachment; filename="Intro to ML - slide 2.jpg"; filename*=UTF-8''Intro%20to%20ML%20-%20slide%202.jpg`
  );
});

test('GET /api/image will not fetch anything except SlideShare images', async () => {
  const targets = [
    'http://169.254.169.254/latest/meta-data/',
    'http://localhost:3000/healthz',
    'https://www.slideshare.net/janedoe/intro-to-machine-learning-262212345',
    'https://slidesharecdn.com.attacker.io/x-1-320.jpg',
    'file:///etc/passwd',
  ];
  for (const src of targets) {
    const res = await fetch(`${server.base}/api/image?src=${encodeURIComponent(src)}`);
    assert.equal(res.status, 400, src);
    assert.equal((await res.json()).code, 'blocked_host');
  }
  assert.equal(fake.requests.length, 0);
});

test('GET /api/download?format=pdf returns a PDF with every slide at full size', async () => {
  const res = await download({ url: DECK, format: 'pdf' });
  assert.equal(res.status, 200);
  assert.equal(res.headers.get('content-type'), 'application/pdf');
  assert.match(res.headers.get('content-disposition'), /^attachment; filename="Intro to Machine Learning\.pdf"/);
  assert.equal(res.headers.get('x-slide-count'), '6');
  const pdf = await PDFDocument.load(Buffer.from(await res.arrayBuffer()));
  assert.equal(pdf.getPageCount(), 6);
  assert.equal(pdf.getTitle(), 'Intro to Machine Learning');

  const imageFetches = fake.requests.filter((u) => u.includes('slidesharecdn.com'));
  assert.equal(imageFetches.length, 6);
  assert.ok(imageFetches.every((u) => u.endsWith('-2048.jpg')), 'downloads the largest JPEG of each slide');
});

test('GET /api/download?format=pptx returns a PowerPoint file', async () => {
  const res = await download({ url: DECK, format: 'pptx' });
  assert.equal(res.status, 200);
  assert.equal(res.headers.get('content-type'), 'application/vnd.openxmlformats-officedocument.presentationml.presentation');
  assert.match(res.headers.get('content-disposition'), /filename="Intro to Machine Learning\.pptx"/);
  const zip = await JSZip.loadAsync(Buffer.from(await res.arrayBuffer()));
  const slides = Object.keys(zip.files).filter((n) => /^ppt\/slides\/slide\d+\.xml$/.test(n));
  assert.equal(slides.length, 6);
  assert.ok(zip.file('[Content_Types].xml'));
});

test('GET /api/download?format=zip returns numbered images', async () => {
  const res = await download({ url: DECK, format: 'zip' });
  assert.equal(res.status, 200);
  assert.equal(res.headers.get('content-type'), 'application/zip');
  const zip = await JSZip.loadAsync(Buffer.from(await res.arrayBuffer()));
  assert.deepEqual(Object.keys(zip.files).sort(), [1, 2, 3, 4, 5, 6].map((n) => `slide-00${n}.jpg`));
  const slide4 = await zip.file('slide-004.jpg').async('nodebuffer');
  assert.ok(slide4.equals(image('slide-4.jpg')), 'slides keep their order and content');
});

test('GET /api/download honours a slide selection', async () => {
  const res = await download({ url: DECK, format: 'pdf', slides: '1,3-4' });
  assert.equal(res.status, 200);
  assert.equal(res.headers.get('x-slide-count'), '3');
  const pdf = await PDFDocument.load(Buffer.from(await res.arrayBuffer()));
  assert.equal(pdf.getPageCount(), 3);
  const fetched = fake.requests.filter((u) => u.includes('slidesharecdn.com')).map((u) => Number(u.match(/-(\d+)-2048/)[1]));
  assert.deepEqual(fetched.sort(), [1, 3, 4]);
});

test('GET /api/download reports bad input and upstream failures', async () => {
  let res = await download({ url: DECK, format: 'docx' });
  assert.equal(res.status, 400);
  assert.match((await res.json()).error, /Choose PDF, PowerPoint/);

  res = await download({ url: DECK, format: 'pdf', slides: '40-50' });
  assert.equal(res.status, 400);

  res = await download({ url: 'https://example.com/x', format: 'pdf' });
  assert.equal(res.status, 400);

  // A deck whose slide 7 is missing on the CDN.
  const html = (await import('node:fs')).readFileSync(require.resolve('./fixtures/next-data.html'), 'utf8').replace('"totalSlides":6', '"totalSlides":7');
  await server.close();
  fake = createFakeSlideShare({ pages: { '/janedoe/broken': { html: () => html } } });
  server = await startServer(createApp({ fetch: fake.fetch, logger: { error() {} } }));
  res = await download({ url: 'https://www.slideshare.net/janedoe/broken', format: 'pdf' });
  assert.equal(res.status, 404);
  assert.match((await res.json()).error, /does not exist|private or deleted/);
});

test('WebP-only slides still download as PDF via the JPEG variant', async () => {
  const html = `<!doctype html><title>WebP deck</title><script id="__NEXT_DATA__" type="application/json">${JSON.stringify({
    props: { pageProps: { slideshow: { title: 'WebP deck', totalSlides: 2, slides: {
      host: 'https://image.slidesharecdn.com', imageLocation: 'webpdeck-1', title: 'WebP-deck',
      imageSizes: [{ quality: 75, width: 2048, format: 'webp' }],
    } } } },
  })}</script>`;
  await server.close();
  fake = createFakeSlideShare({
    pages: { '/u/webp-deck': { html: () => html } },
    imageFor: (u) => (u.pathname.endsWith('.webp') ? new Response(image('slide-webp.webp')) : null),
  });
  server = await startServer(createApp({ fetch: fake.fetch, logger: { error() {} } }));
  const res = await download({ url: 'https://www.slideshare.net/u/webp-deck', format: 'pdf' });
  assert.equal(res.status, 200);
  const pdf = await PDFDocument.load(Buffer.from(await res.arrayBuffer()));
  assert.equal(pdf.getPageCount(), 2);
  assert.ok(fake.requests.some((u) => u.endsWith('-1-2048.jpg')), 'retried the slide as .jpg');
});

test('rate limiting kicks in after many requests', async () => {
  let last;
  for (let i = 0; i < 61; i++) last = await postSlides('https://example.com/x');
  assert.equal(last.status, 429);
  assert.ok(Number(last.headers.get('retry-after')) > 0);
});

test('unknown API routes return JSON 404', async () => {
  const res = await fetch(`${server.base}/api/nope`);
  assert.equal(res.status, 404);
  assert.equal((await res.json()).code, 'not_found');
});
