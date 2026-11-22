'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { PDFDocument } = require('pdf-lib');
const JSZip = require('jszip');

const { normalizePresentationUrl, isPageUrl, isImageUrl, safeFetch, PAGE_HOST_SUFFIXES } = require('../src/net');
const { extractSlides, cleanTitle } = require('../src/slideshare');
const { detectImageType, imageSize, buildPdf, buildPptx, buildZip } = require('../src/builders');
const { parseSlideSelection, safeFilename } = require('../src/app');
const { fixture, image } = require('./helpers');

test('normalizePresentationUrl accepts SlideShare presentation links', () => {
  const cases = [
    ['https://www.slideshare.net/janedoe/intro-262212345', 'https://www.slideshare.net/janedoe/intro-262212345'],
    ['www.slideshare.net/janedoe/intro-262212345', 'https://www.slideshare.net/janedoe/intro-262212345'],
    ['  https://de.slideshare.net/janedoe/intro?utm_source=x#slide=3 ', 'https://de.slideshare.net/janedoe/intro'],
    ['http://slideshare.net/slideshow/intro/262212345', 'https://slideshare.net/slideshow/intro/262212345'],
    ['https://WWW.SLIDESHARE.NET/janedoe/intro', 'https://www.slideshare.net/janedoe/intro'],
  ];
  for (const [input, expected] of cases) assert.equal(normalizePresentationUrl(input), expected, input);
});

test('normalizePresentationUrl rejects anything that is not a SlideShare page', () => {
  const bad = [
    '',
    '   ',
    'https://www.slideshare.net/',
    'https://example.com/janedoe/intro',
    'https://slideshare.net.evil.com/x',
    'https://evil.com/www.slideshare.net/x',
    'https://user:pass@www.slideshare.net/x',
    'https://www.slideshare.net:8443/x',
    'javascript:alert(1)',
    'file:///etc/passwd',
    'ftp://www.slideshare.net/x',
    'http://169.254.169.254/latest/meta-data/',
    'https://www.slideshare.net/' + 'a'.repeat(3000),
  ];
  for (const input of bad) {
    assert.throws(() => normalizePresentationUrl(input), (err) => err.status === 400 && err.message.length > 10, input);
  }
});

test('host allowlists only match the real domains', () => {
  assert.equal(isPageUrl('https://fr.slideshare.net/a/b'), true);
  assert.equal(isPageUrl('https://notslideshare.net/a'), false);
  assert.equal(isImageUrl('https://image.slidesharecdn.com/a/85/b-1-320.jpg'), true);
  assert.equal(isImageUrl('https://cdn.slidesharecdn.com/ss_thumbnails/x.jpg'), true);
  assert.equal(isImageUrl('https://slidesharecdn.com.evil.io/a.jpg'), false);
  assert.equal(isImageUrl('https://www.slideshare.net/a.jpg'), false);
  assert.equal(isImageUrl('http://127.0.0.1/a.jpg'), false);
});

test('extractSlides reads the Next.js page data (current SlideShare pages)', () => {
  const deck = extractSlides(fixture('next-data.html'));
  assert.equal(deck.source, 'next-data');
  assert.equal(deck.title, 'Intro to Machine Learning');
  assert.equal(deck.totalSlides, 6, 'uses totalSlides, not just the slides rendered in the HTML');
  const base = 'https://image.slidesharecdn.com/introtomachinelearning-231004101112-5c1a2b3d';
  assert.equal(deck.slides[0].full, `${base}/75/Intro-to-Machine-Learning-1-2048.jpg`, 'largest JPEG, not WebP');
  assert.equal(deck.slides[0].thumb, `${base}/85/Intro-to-Machine-Learning-1-320.jpg`);
  assert.equal(deck.slides[5].full, `${base}/75/Intro-to-Machine-Learning-6-2048.jpg`);
  assert.deepEqual(deck.slides.map((s) => s.number), [1, 2, 3, 4, 5, 6]);
  assert.ok(deck.slides.every((s) => !s.full.includes('neuralnets')), 'ignores the related presentation');
});

test('extractSlides falls back to scanning image URLs in the HTML', () => {
  const deck = extractSlides(fixture('html-only.html'));
  assert.equal(deck.source, 'html-scan');
  assert.equal(deck.title, 'Quarterly Report 2024');
  assert.equal(deck.totalSlides, 5, 'fills in lazy-loaded slides up to the declared total');
  const base = 'https://image.slidesharecdn.com/quarterlyreport2024-240115-9f8e7d';
  assert.equal(deck.slides[0].full, `${base}/75/Quarterly-Report-2024-1-2048.jpg`);
  // Slide 3 only appeared as an escaped 320px URL; it is upgraded to the largest size.
  assert.equal(deck.slides[2].full, `${base}/75/Quarterly-Report-2024-3-2048.jpg`);
  assert.equal(deck.slides[4].full, `${base}/75/Quarterly-Report-2024-5-2048.jpg`);
  assert.ok(deck.slides.every((s) => s.full.includes('quarterlyreport2024')), 'related deck is not mixed in');
});

test('extractSlides handles the legacy data-full layout', () => {
  const deck = extractSlides(fixture('legacy.html'));
  assert.equal(deck.title, 'Old deck title');
  assert.equal(deck.totalSlides, 4);
  assert.equal(
    deck.slides[3].full,
    'https://image.slidesharecdn.com/olddeck-140101123456-phpapp01/95/old-deck-title-4-1024.jpg?cb=1388534400'
  );
  assert.ok(deck.slides[0].thumb.endsWith('-1-638.jpg?cb=1388534400'));
});

test('extractSlides refuses pages that list several presentations', () => {
  assert.throws(() => extractSlides(fixture('profile-page.html')), (err) => err.code === 'no_slides');
  assert.throws(() => extractSlides('<html><body>Nothing here</body></html>'), (err) => err.code === 'no_slides');
  assert.throws(() => extractSlides('<p>This content is private</p>'), (err) => err.code === 'private');
  assert.throws(() => extractSlides(''), (err) => err.code === 'empty_page');
});

test('cleanTitle strips SlideShare suffixes', () => {
  assert.equal(cleanTitle('Intro to ML | PPT'), 'Intro to ML');
  assert.equal(cleanTitle('Intro to ML | PDF | Free Download'), 'Intro to ML');
  assert.equal(cleanTitle('Report - SlideShare'), 'Report');
  assert.equal(cleanTitle('A - B testing'), 'A - B testing');
});

test('parseSlideSelection understands lists and ranges', () => {
  assert.equal(parseSlideSelection(undefined, 10), null);
  assert.equal(parseSlideSelection('all', 10), null);
  assert.deepEqual(parseSlideSelection('1,3-5,9', 10), [1, 3, 4, 5, 9]);
  assert.deepEqual(parseSlideSelection('5-3, 3', 10), [3, 4, 5]);
  assert.deepEqual(parseSlideSelection('8-20', 10), [8, 9, 10]);
  assert.throws(() => parseSlideSelection('11,12', 10), /None of the selected slides/);
  assert.throws(() => parseSlideSelection('1;drop', 10), /not valid/);
});

test('safeFilename keeps names portable', () => {
  assert.equal(safeFilename('Intro to ML: part 1/2', 'pdf'), 'Intro to ML part 12.pdf');
  assert.equal(safeFilename('Café déjà vu', 'zip'), 'Cafe deja vu.zip');
  assert.equal(safeFilename('日本語', 'pdf'), 'slides.pdf');
});

test('detectImageType and imageSize read JPEG, PNG and WebP headers', () => {
  const jpg = image('slide-1.jpg');
  const png = image('slide-png.png');
  const webp = image('slide-webp.webp');
  assert.equal(detectImageType(jpg), 'jpeg');
  assert.equal(detectImageType(png), 'png');
  assert.equal(detectImageType(webp), 'webp');
  assert.equal(detectImageType(Buffer.from('<html>not an image</html>')), null);
  assert.deepEqual(imageSize(jpg), { width: 1024, height: 576 });
  assert.deepEqual(imageSize(png), { width: 800, height: 600 });
  assert.deepEqual(imageSize(webp), { width: 640, height: 360 });
});

function loadImages(names) {
  return names.map((name, i) => {
    const buffer = image(name);
    return { buffer, type: detectImageType(buffer), size: imageSize(buffer), number: i + 1 };
  });
}

test('buildPdf makes one page per slide, sized to the slide', async () => {
  const pdfBytes = await buildPdf(loadImages(['slide-1.jpg', 'slide-2.jpg', 'slide-png.png']), { title: 'Test deck' });
  assert.equal(pdfBytes.subarray(0, 5).toString(), '%PDF-');
  const pdf = await PDFDocument.load(pdfBytes);
  assert.equal(pdf.getPageCount(), 3);
  assert.deepEqual(pdf.getPage(0).getSize(), { width: 960, height: 540 });
  assert.deepEqual(pdf.getPage(2).getSize(), { width: 960, height: 720 }, 'a 4:3 slide gets a 4:3 page');
  assert.equal(pdf.getTitle(), 'Test deck');
});

test('buildPdf refuses WebP with a clear message', async () => {
  await assert.rejects(buildPdf(loadImages(['slide-webp.webp'])), (err) => err.code === 'unsupported_image');
});

test('buildPptx makes a valid PowerPoint file with one slide per image', async () => {
  const buf = await buildPptx(loadImages(['slide-1.jpg', 'slide-2.jpg', 'slide-3.jpg']), { title: 'Test deck' });
  const zip = await JSZip.loadAsync(buf);
  const slides = Object.keys(zip.files).filter((n) => /^ppt\/slides\/slide\d+\.xml$/.test(n));
  assert.equal(slides.length, 3);
  const media = Object.values(zip.files).filter((f) => f.name.startsWith('ppt/media/') && !f.dir);
  assert.equal(media.length, 3);
  assert.ok(media.every((f) => f.name.endsWith('.jpeg')));
  const presentation = await zip.file('ppt/presentation.xml').async('string');
  // 13.333in x 7.5in (16:9) in EMU
  assert.match(presentation, /<p:sldSz cx="12191695" cy="6858000"/);
  const slide1 = await zip.file('ppt/slides/slide1.xml').async('string');
  assert.match(slide1, /<p:pic>/);
});

test('buildZip stores numbered slide images', async () => {
  const buf = await buildZip(loadImages(['slide-1.jpg', 'slide-2.jpg', 'slide-png.png']));
  const zip = await JSZip.loadAsync(buf);
  assert.deepEqual(Object.keys(zip.files).sort(), ['slide-001.jpg', 'slide-002.jpg', 'slide-003.png']);
  const first = await zip.file('slide-001.jpg').async('nodebuffer');
  assert.ok(first.equals(image('slide-1.jpg')));
});

test('safeFetch follows redirects only inside the allowlist', async () => {
  const seen = [];
  const fetchImpl = async (url) => {
    seen.push(url);
    if (url.endsWith('/hop')) return new Response(null, { status: 302, headers: { location: '/final' } });
    if (url.endsWith('/final')) return new Response('ok');
    if (url.endsWith('/escape')) return new Response(null, { status: 302, headers: { location: 'http://10.0.0.5/admin' } });
    return new Response('nope', { status: 404 });
  };
  const res = await safeFetch(fetchImpl, 'https://www.slideshare.net/hop', { suffixes: PAGE_HOST_SUFFIXES, maxBytes: 1000 });
  assert.equal(res.body.toString(), 'ok');
  assert.equal(res.url, 'https://www.slideshare.net/final');

  await assert.rejects(
    safeFetch(fetchImpl, 'https://www.slideshare.net/escape', { suffixes: PAGE_HOST_SUFFIXES, maxBytes: 1000 }),
    (err) => err.code === 'blocked_host'
  );
  assert.ok(!seen.some((u) => u.includes('10.0.0.5')), 'never contacted the internal address');
});

test('safeFetch caps response size and times out', async () => {
  const big = async () => new Response(Buffer.alloc(5000));
  await assert.rejects(
    safeFetch(big, 'https://www.slideshare.net/x', { suffixes: PAGE_HOST_SUFFIXES, maxBytes: 1000 }),
    (err) => err.code === 'too_large'
  );
  const slow = (url, { signal }) =>
    new Promise((resolve, reject) => {
      signal.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' })));
    });
  await assert.rejects(
    safeFetch(slow, 'https://www.slideshare.net/x', { suffixes: PAGE_HOST_SUFFIXES, maxBytes: 1000, timeoutMs: 50 }),
    (err) => err.code === 'timeout'
  );
});
