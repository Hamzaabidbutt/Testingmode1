'use strict';

/**
 * Outbound HTTP with a host allowlist.
 *
 * Every request (and every redirect hop) is checked against an allowlist so the
 * server can only ever talk to SlideShare and its image CDN. This prevents the
 * download endpoints from being abused to reach internal services (SSRF).
 */

const PAGE_HOST_SUFFIXES = ['slideshare.net'];
const IMAGE_HOST_SUFFIXES = ['slidesharecdn.com'];

const BROWSER_HEADERS = {
  'user-agent':
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
  'accept-language': 'en-US,en;q=0.9',
};

class UserError extends Error {
  constructor(message, status = 400, code = 'bad_request') {
    super(message);
    this.status = status;
    this.code = code;
  }
}

function hostMatches(hostname, suffixes) {
  const host = String(hostname || '').toLowerCase().replace(/\.$/, '');
  return suffixes.some((s) => host === s || host.endsWith('.' + s));
}

function isAllowedUrl(urlString, suffixes) {
  let u;
  try {
    u = new URL(urlString);
  } catch {
    return false;
  }
  if (u.protocol !== 'https:' && u.protocol !== 'http:') return false;
  if (u.username || u.password) return false;
  if (u.port && u.port !== '443' && u.port !== '80') return false;
  return hostMatches(u.hostname, suffixes);
}

const isPageUrl = (u) => isAllowedUrl(u, PAGE_HOST_SUFFIXES);
const isImageUrl = (u) => isAllowedUrl(u, IMAGE_HOST_SUFFIXES);

/**
 * Turn whatever the user pasted into a canonical SlideShare presentation URL,
 * or throw a UserError explaining what is wrong.
 */
function normalizePresentationUrl(input) {
  let raw = String(input || '').trim();
  if (!raw) throw new UserError('Paste a SlideShare link to get started.');
  if (raw.length > 2048) throw new UserError('That link is too long to be a SlideShare link.');
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(raw)) raw = 'https://' + raw.replace(/^\/+/, '');

  let u;
  try {
    u = new URL(raw);
  } catch {
    throw new UserError('That does not look like a web link. Copy the address of a SlideShare presentation.');
  }
  if (!isPageUrl(u.href)) {
    throw new UserError('Only slideshare.net links are supported. Copy the address of a SlideShare presentation.');
  }
  if (!u.pathname.replace(/\/+/g, '')) {
    throw new UserError('That is the SlideShare home page. Open a presentation and copy its address.');
  }
  u.protocol = 'https:';
  u.port = '';
  u.hash = '';
  // Keep the path, drop tracking parameters.
  u.search = '';
  return u.href;
}

/**
 * fetch() that validates the URL and every redirect hop against `suffixes`,
 * enforces a timeout and caps the body size.
 */
async function safeFetch(fetchImpl, url, { suffixes, maxBytes, timeoutMs = 20000, accept, maxRedirects = 5 }) {
  let current = url;
  for (let hop = 0; hop <= maxRedirects; hop++) {
    if (!isAllowedUrl(current, suffixes)) {
      throw new UserError('The link redirected somewhere this tool does not download from.', 400, 'blocked_host');
    }
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    let res;
    try {
      res = await fetchImpl(current, {
        method: 'GET',
        redirect: 'manual',
        signal: controller.signal,
        headers: { ...BROWSER_HEADERS, ...(accept ? { accept } : {}) },
      });
    } catch (err) {
      clearTimeout(timer);
      if (err && err.name === 'AbortError') {
        throw new UserError('SlideShare took too long to respond. Try again in a moment.', 504, 'timeout');
      }
      throw new UserError('Could not reach SlideShare. Check the link and try again.', 502, 'upstream_unreachable');
    }

    if (res.status >= 300 && res.status < 400 && res.headers.get('location')) {
      clearTimeout(timer);
      current = new URL(res.headers.get('location'), current).href;
      continue;
    }

    try {
      if (res.status === 404 || res.status === 410) {
        throw new UserError('SlideShare says this presentation does not exist. It may be private or deleted.', 404, 'not_found');
      }
      if (res.status === 403 || res.status === 429) {
        throw new UserError('SlideShare refused the request. Wait a minute and try again.', 502, 'upstream_refused');
      }
      if (!res.ok) {
        throw new UserError(`SlideShare responded with an error (${res.status}). Try again later.`, 502, 'upstream_error');
      }
      const declared = Number(res.headers.get('content-length') || 0);
      if (declared && declared > maxBytes) {
        throw new UserError('The file on SlideShare is larger than this tool allows.', 413, 'too_large');
      }
      const body = await readCapped(res, maxBytes);
      return { url: current, status: res.status, headers: res.headers, body };
    } catch (err) {
      if (err && err.name === 'AbortError') {
        throw new UserError('SlideShare took too long to respond. Try again in a moment.', 504, 'timeout');
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }
  throw new UserError('The link redirected too many times.', 502, 'too_many_redirects');
}

async function readCapped(res, maxBytes) {
  if (!res.body || typeof res.body.getReader !== 'function') {
    const buf = Buffer.from(await res.arrayBuffer());
    if (buf.length > maxBytes) throw new UserError('The file on SlideShare is larger than this tool allows.', 413, 'too_large');
    return buf;
  }
  const reader = res.body.getReader();
  const chunks = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > maxBytes) {
      try { await reader.cancel(); } catch { /* ignore */ }
      throw new UserError('The file on SlideShare is larger than this tool allows.', 413, 'too_large');
    }
    chunks.push(Buffer.from(value));
  }
  return Buffer.concat(chunks);
}

module.exports = {
  PAGE_HOST_SUFFIXES,
  IMAGE_HOST_SUFFIXES,
  UserError,
  isPageUrl,
  isImageUrl,
  normalizePresentationUrl,
  safeFetch,
};
