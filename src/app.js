'use strict';

const path = require('path');
const express = require('express');
const { UserError, normalizePresentationUrl, isImageUrl } = require('./net');
const { fetchPresentation } = require('./slideshare');
const { fetchImage, fetchSlideImages, buildPdf, buildPptx, buildZip, EXT, MIME } = require('./builders');

const FORMATS = {
  pdf: { ext: 'pdf', mime: 'application/pdf', build: buildPdf },
  pptx: { ext: 'pptx', mime: 'application/vnd.openxmlformats-officedocument.presentationml.presentation', build: buildPptx },
  zip: { ext: 'zip', mime: 'application/zip', build: buildZip },
};

/** Small TTL cache so the preview and the download don't scrape the page twice. */
class TtlCache {
  constructor({ ttlMs = 15 * 60 * 1000, max = 200 } = {}) {
    this.ttlMs = ttlMs;
    this.max = max;
    this.map = new Map();
  }
  get(key) {
    const hit = this.map.get(key);
    if (!hit) return undefined;
    if (hit.expires < Date.now()) {
      this.map.delete(key);
      return undefined;
    }
    return hit.value;
  }
  set(key, value) {
    if (this.map.size >= this.max) this.map.delete(this.map.keys().next().value);
    this.map.set(key, { value, expires: Date.now() + this.ttlMs });
  }
}

/** Fixed-window rate limiter keyed by client IP. */
function rateLimit({ windowMs, max, message }) {
  const hits = new Map();
  return (req, res, next) => {
    const now = Date.now();
    const key = req.ip || 'unknown';
    let entry = hits.get(key);
    if (!entry || entry.reset < now) {
      entry = { count: 0, reset: now + windowMs };
      hits.set(key, entry);
      if (hits.size > 10000) {
        for (const [k, v] of hits) if (v.reset < now) hits.delete(k);
      }
    }
    entry.count++;
    if (entry.count > max) {
      res.set('Retry-After', String(Math.ceil((entry.reset - now) / 1000)));
      return res.status(429).json({ error: message, code: 'rate_limited' });
    }
    next();
  };
}

function safeFilename(title, ext) {
  const base =
    String(title || 'slides')
      .normalize('NFKD')
      .replace(/[\u0300-\u036f]/g, '')
      .replace(/[^A-Za-z0-9 ._-]+/g, '')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, 80) || 'slides';
  return `${base}.${ext}`;
}

/** attachment header with an ASCII fallback name plus the original (UTF-8) title. */
function contentDisposition(title, ext) {
  const ascii = safeFilename(title, ext);
  const unicode =
    String(title || 'slides')
      .replace(/[\u0000-\u001f\u007f/\\:*?"<>|]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, 120) || 'slides';
  const encoded = encodeURIComponent(`${unicode}.${ext}`).replace(
    /['()*]/g,
    (c) => '%' + c.charCodeAt(0).toString(16).toUpperCase()
  );
  return `attachment; filename="${ascii}"; filename*=UTF-8''${encoded}`;
}

/** Parse "1,3,5-8" into a sorted list of slide numbers within 1..total. */
function parseSlideSelection(spec, total) {
  if (spec == null || spec === '' || spec === 'all') return null;
  const picked = new Set();
  for (const part of String(spec).split(',')) {
    const p = part.trim();
    if (!p) continue;
    const range = p.match(/^(\d+)\s*-\s*(\d+)$/);
    if (range) {
      let a = Number(range[1]);
      let b = Number(range[2]);
      if (a > b) [a, b] = [b, a];
      for (let n = Math.max(1, a); n <= Math.min(total, b); n++) picked.add(n);
    } else if (/^\d+$/.test(p)) {
      const n = Number(p);
      if (n >= 1 && n <= total) picked.add(n);
    } else {
      throw new UserError('The slide selection is not valid. Use numbers like 1,3,5-8.');
    }
  }
  if (!picked.size) throw new UserError('None of the selected slides are in this presentation.');
  return [...picked].sort((a, b) => a - b);
}

function createApp({ fetch: fetchImpl = globalThis.fetch, maxConcurrentBuilds = 3, logger = console } = {}) {
  if (typeof fetchImpl !== 'function') throw new Error('A fetch implementation is required (Node 18+).');

  const app = express();
  const cache = new TtlCache();
  let activeBuilds = 0;

  app.disable('x-powered-by');
  app.set('trust proxy', 'loopback');

  app.use((req, res, next) => {
    res.set({
      'X-Content-Type-Options': 'nosniff',
      'Referrer-Policy': 'strict-origin-when-cross-origin',
      'X-Frame-Options': 'DENY',
      'Content-Security-Policy': [
        "default-src 'self'",
        "img-src 'self' data: blob:",
        "style-src 'self' https://fonts.googleapis.com",
        "font-src 'self' https://fonts.gstatic.com",
        "script-src 'self'",
        "connect-src 'self'",
        "frame-ancestors 'none'",
        "base-uri 'none'",
        "form-action 'self'",
      ].join('; '),
    });
    next();
  });

  app.use(express.static(path.join(__dirname, '..', 'public'), { maxAge: '1h', index: 'index.html' }));
  app.use('/api', express.json({ limit: '16kb' }));
  app.use('/api', rateLimit({ windowMs: 60_000, max: 60, message: 'Too many requests. Wait a minute and try again.' }));

  async function getDeck(rawUrl) {
    const url = normalizePresentationUrl(rawUrl);
    const cached = cache.get(url);
    if (cached) return cached;
    const deck = await fetchPresentation(fetchImpl, url);
    cache.set(url, deck);
    if (deck.url !== url) cache.set(deck.url, deck);
    return deck;
  }

  app.get('/healthz', (req, res) => res.json({ ok: true }));

  // Look up a presentation and list its slides.
  app.post('/api/slides', async (req, res, next) => {
    try {
      const deck = await getDeck(req.body && req.body.url);
      res.json({
        url: deck.url,
        title: deck.title,
        totalSlides: deck.totalSlides,
        slides: deck.slides.map((s) => ({
          number: s.number,
          thumb: `/api/image?src=${encodeURIComponent(s.thumb)}`,
          full: `/api/image?src=${encodeURIComponent(s.full)}`,
          download: `/api/image?src=${encodeURIComponent(s.full)}&download=1&name=${encodeURIComponent(
            `${deck.title} - slide ${s.number}`
          )}`,
        })),
      });
    } catch (err) {
      next(err);
    }
  });

  // Proxy a single slide image (previews and single-slide downloads).
  app.get('/api/image', async (req, res, next) => {
    try {
      const src = String(req.query.src || '');
      if (!isImageUrl(src)) throw new UserError('Only SlideShare slide images can be loaded.', 400, 'blocked_host');
      const img = await fetchImage(fetchImpl, src);
      res.set('Content-Type', MIME[img.type]);
      res.set('Cache-Control', 'public, max-age=86400');
      if (req.query.download) {
        res.set('Content-Disposition', contentDisposition(req.query.name || 'slide', EXT[img.type]));
      }
      res.send(img.buffer);
    } catch (err) {
      next(err);
    }
  });

  // Build the whole presentation (or selected slides) as PDF, PPTX or ZIP.
  app.get('/api/download', async (req, res, next) => {
    const format = FORMATS[String(req.query.format || 'pdf').toLowerCase()];
    if (!format) return next(new UserError('Choose PDF, PowerPoint (pptx) or images (zip).'));
    if (activeBuilds >= maxConcurrentBuilds) {
      res.set('Retry-After', '10');
      return res.status(503).json({ error: 'The server is busy building other files. Try again in a few seconds.', code: 'busy' });
    }
    activeBuilds++;
    try {
      const deck = await getDeck(req.query.url);
      const numbers = parseSlideSelection(req.query.slides, deck.totalSlides);
      const slides = numbers ? deck.slides.filter((s) => numbers.includes(s.number)) : deck.slides;
      const images = await fetchSlideImages(fetchImpl, slides);
      const file = await format.build(images, { title: deck.title });
      res.set('Content-Type', format.mime);
      res.set('Content-Length', String(file.length));
      res.set('Content-Disposition', contentDisposition(deck.title, format.ext));
      res.set('Cache-Control', 'no-store');
      res.set('X-Slide-Count', String(images.length));
      res.send(file);
    } catch (err) {
      next(err);
    } finally {
      activeBuilds--;
    }
  });

  app.use('/api', (req, res) => res.status(404).json({ error: 'Not found.', code: 'not_found' }));

  // eslint-disable-next-line no-unused-vars
  app.use((err, req, res, next) => {
    if (err instanceof UserError) {
      return res.status(err.status).json({ error: err.message, code: err.code });
    }
    if (err && err.type === 'entity.parse.failed') {
      return res.status(400).json({ error: 'The request was not valid JSON.', code: 'bad_json' });
    }
    if (err && err.type === 'entity.too.large') {
      return res.status(413).json({ error: 'The request was too large.', code: 'too_large' });
    }
    logger.error(err);
    res.status(500).json({ error: 'Something went wrong while preparing your file. Try again.', code: 'internal' });
  });

  return app;
}

module.exports = { createApp, parseSlideSelection, safeFilename, TtlCache };
