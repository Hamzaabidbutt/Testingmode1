'use strict';

/**
 * A fake slideshare.net + slidesharecdn.com for tests. It stands in for fetch()
 * so the app can be exercised end to end without network access, and records
 * every URL the app asked for.
 */

const fs = require('fs');
const path = require('path');

const FIXTURES = path.join(__dirname, 'fixtures');
const IMAGES = path.join(FIXTURES, 'images');

function fixture(name) {
  return fs.readFileSync(path.join(FIXTURES, name), 'utf8');
}

function image(name) {
  return fs.readFileSync(path.join(IMAGES, name));
}

const DEFAULT_PAGES = {
  '/janedoe/intro-to-machine-learning-262212345': { html: () => fixture('next-data.html') },
  '/acme/quarterly-report-2024': { html: () => fixture('html-only.html') },
  '/olduser/old-deck-title': { html: () => fixture('legacy.html') },
  '/janedoe': { html: () => fixture('profile-page.html') },
  '/short-link': { redirect: 'https://www.slideshare.net/janedoe/intro-to-machine-learning-262212345' },
  '/evil-redirect': { redirect: 'http://169.254.169.254/latest/meta-data/' },
  '/rate-limited': { status: 429 },
};

function createFakeSlideShare({ pages = {}, imageFor } = {}) {
  const routes = { ...DEFAULT_PAGES, ...pages };
  const requests = [];

  async function fakeFetch(url, opts = {}) {
    requests.push(String(url));
    const u = new URL(url);
    const host = u.hostname;

    if (host === 'slideshare.net' || host.endsWith('.slideshare.net')) {
      const route = routes[u.pathname.replace(/\/+$/, '')];
      if (!route) return new Response('<h1>Not found</h1>', { status: 404, headers: { 'content-type': 'text/html' } });
      if (route.redirect) return new Response(null, { status: 301, headers: { location: route.redirect } });
      if (route.status) return new Response('error', { status: route.status });
      if (route.delayMs) await sleep(route.delayMs, opts.signal);
      return new Response(route.html(), { status: 200, headers: { 'content-type': 'text/html; charset=utf-8' } });
    }

    if (host.endsWith('.slidesharecdn.com')) {
      if (imageFor) {
        const custom = await imageFor(u);
        if (custom) return custom;
      }
      const m = u.pathname.match(/-(\d+)-(\d+)\.(jpe?g|png|webp)$/i);
      if (!m) return new Response('not found', { status: 404 });
      const file = path.join(IMAGES, `slide-${m[1]}.jpg`);
      if (!fs.existsSync(file)) return new Response('not found', { status: 404 });
      const body = fs.readFileSync(file);
      return new Response(body, {
        status: 200,
        headers: { 'content-type': 'image/jpeg', 'content-length': String(body.length) },
      });
    }

    // Anything else would be a request the app must never make.
    throw new TypeError(`fetch failed: unexpected host ${host}`);
  }

  return { fetch: fakeFetch, requests };
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    if (signal) {
      signal.addEventListener('abort', () => {
        clearTimeout(t);
        const err = new Error('aborted');
        err.name = 'AbortError';
        reject(err);
      });
    }
  });
}

/** Start the app on a random port. Returns { base, close }. */
async function startServer(app) {
  return new Promise((resolve) => {
    const server = app.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      resolve({
        base: `http://127.0.0.1:${port}`,
        close: () => new Promise((r) => server.close(r)),
      });
    });
  });
}

module.exports = { createFakeSlideShare, startServer, fixture, image, FIXTURES, IMAGES };
