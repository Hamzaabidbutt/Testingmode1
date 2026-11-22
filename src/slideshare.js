'use strict';

/**
 * Reads a SlideShare presentation page and works out the image URL of every slide.
 *
 * SlideShare renders each slide as an image on image.slidesharecdn.com, e.g.
 *   https://image.slidesharecdn.com/<imageLocation>/<quality>/<title>-<slideNo>-<width>.jpg
 *
 * Two strategies are used, most reliable first:
 *  1. The Next.js page data (<script id="__NEXT_DATA__">) describes the deck:
 *     host, imageLocation, title, the available image sizes and totalSlides.
 *  2. Fallback: scan the HTML for slide image URLs (img src/srcset, data-full,
 *     JSON blobs, og:image), group them by deck and pick the largest size of
 *     each slide. Missing slides are filled in from the URL pattern when the
 *     page states the total slide count.
 */

const cheerio = require('cheerio');
const { UserError, safeFetch, PAGE_HOST_SUFFIXES, isImageUrl } = require('./net');

const MAX_SLIDES = 500;
const THUMB_MIN_WIDTH = 300;
const WEB_SAFE_FORMATS = ['jpg', 'jpeg', 'png'];

// <optional scheme>//<host>.slidesharecdn.com/<location>/<quality>/<title>-<n>-<width>.<ext>[?query]
const SLIDE_URL_RE =
  /(?:https?:)?\/\/([a-z0-9-]+(?:\.[a-z0-9-]+)*\.slidesharecdn\.com)\/([^\s"'<>\\/?#]+)\/(\d+)\/([^\s"'<>\\/?#]+?)-(\d+)-(\d+)\.(jpe?g|png|webp)(\?[^\s"'<>\\]*)?/gi;

function cleanTitle(raw) {
  if (!raw) return '';
  return String(raw)
    .replace(/\s+/g, ' ')
    .replace(/\s*[|\-–]\s*(PPT|PDF|PPTX|Free Download|SlideShare|Slideshare)(\s*[|\-–]\s*(PPT|PDF|PPTX|Free Download|SlideShare|Slideshare))*\s*$/i, '')
    .trim();
}

// Encode a URL path segment without double-encoding existing %XX escapes.
function pathSegment(s) {
  return String(s).replace(/[^A-Za-z0-9\-._~%!$&'()*+,;=:@]/g, (c) => encodeURIComponent(c));
}

function unescapeJsonish(html) {
  return html
    .replace(/\\u002[fF]/g, '/')
    .replace(/\\u0026/g, '&')
    .replace(/\\\//g, '/')
    .replace(/&amp;/g, '&');
}

function sizePreference(a, b) {
  // Larger width first; at equal width prefer jpg/png over webp.
  if (b.width !== a.width) return b.width - a.width;
  const aw = WEB_SAFE_FORMATS.includes(a.format) ? 0 : 1;
  const bw = WEB_SAFE_FORMATS.includes(b.format) ? 0 : 1;
  return aw - bw;
}

function pickSizes(imageSizes) {
  const sizes = (imageSizes || [])
    .map((s) => ({
      width: Number(s.width),
      quality: Number(s.quality),
      format: String(s.format || 'jpg').toLowerCase().replace('jpeg', 'jpg'),
    }))
    .filter((s) => Number.isFinite(s.width) && s.width > 0 && Number.isFinite(s.quality));
  if (!sizes.length) return null;
  // Prefer web-safe formats for the downloadable file (PDF/PPTX cannot embed webp).
  const safe = sizes.filter((s) => WEB_SAFE_FORMATS.includes(s.format));
  const pool = safe.length ? safe : sizes;
  const sorted = [...pool].sort(sizePreference);
  const full = sorted[0];
  const ascending = [...pool].sort((a, b) => a.width - b.width);
  const thumb = ascending.find((s) => s.width >= THUMB_MIN_WIDTH) || full;
  return { full, thumb };
}

/**
 * Breadth-first search through the parsed page data for the slide image
 * descriptor ({ host, imageLocation, title, imageSizes }). Breadth-first means
 * the presentation itself (props.pageProps.slideshow.slides) is found before
 * the "related presentations" nested deeper in the tree.
 */
function findSlidesDescriptor(root) {
  const queue = [{ node: root, ancestors: [] }];
  let fallback = null;
  for (let i = 0; i < queue.length && i < 200000; i++) {
    const { node, ancestors } = queue[i];
    if (!node || typeof node !== 'object') continue;
    if (
      !Array.isArray(node) &&
      typeof node.host === 'string' &&
      typeof node.imageLocation === 'string' &&
      Array.isArray(node.imageSizes)
    ) {
      let total = Number.isInteger(node.totalSlides) ? node.totalSlides : null;
      let deckTitle = null;
      for (let j = ancestors.length - 1; j >= 0; j--) {
        const a = ancestors[j];
        if (total == null && Number.isInteger(a.totalSlides)) total = a.totalSlides;
        if (deckTitle == null && typeof a.title === 'string' && a.title !== node.title) deckTitle = a.title;
      }
      const found = { descriptor: node, totalSlides: total, deckTitle };
      if (total != null) return found;
      if (!fallback) fallback = found;
      continue;
    }
    const next = Array.isArray(node) ? ancestors : ancestors.concat([node]);
    for (const key of Object.keys(node)) {
      const v = node[key];
      if (v && typeof v === 'object') queue.push({ node: v, ancestors: next });
    }
  }
  return fallback;
}

function fromNextData($) {
  const raw = $('script#__NEXT_DATA__').first().text();
  if (!raw) return null;
  let data;
  try {
    data = JSON.parse(raw);
  } catch {
    return null;
  }
  const found = findSlidesDescriptor(data);
  if (!found) return null;
  const { descriptor: d, totalSlides, deckTitle } = found;
  const sizes = pickSizes(d.imageSizes);
  const total = Number.isInteger(totalSlides) ? totalSlides : Number(d.totalSlides);
  if (!sizes || !Number.isInteger(total) || total < 1) return null;

  const host = d.host.replace(/\/+$/, '');
  const base = `${host}/${pathSegment(d.imageLocation)}`;
  const name = pathSegment(d.title || 'slide');
  const url = (s, n) => `${base}/${s.quality}/${name}-${n}-${s.width}.${s.format}`;

  const slides = [];
  for (let n = 1; n <= Math.min(total, MAX_SLIDES); n++) {
    slides.push({ number: n, full: url(sizes.full, n), thumb: url(sizes.thumb, n) });
  }
  if (!slides.every((s) => isImageUrl(s.full))) return null;
  return { slides, totalSlides: total, deckTitle, source: 'next-data' };
}

function findTotalSlides(text) {
  const patterns = [
    /"totalSlides"\s*:\s*(\d+)/,
    /"total_slides"\s*:\s*(\d+)/,
    /data-total-slides=["'](\d+)["']/,
    /"numberOfPages"\s*:\s*"?(\d+)/,
  ];
  for (const re of patterns) {
    const m = text.match(re);
    if (m) return Number(m[1]);
  }
  return null;
}

function fromHtmlScan(html, ogImage) {
  const text = unescapeJsonish(html);
  const groups = new Map();
  for (const m of text.matchAll(SLIDE_URL_RE)) {
    const [, host, location, quality, title, num, width, ext, query] = m;
    const key = `${host.toLowerCase()}/${location}/${title}`;
    const entry = {
      host: host.toLowerCase(),
      location,
      quality: Number(quality),
      title,
      number: Number(num),
      width: Number(width),
      format: ext.toLowerCase().replace('jpeg', 'jpg'),
      query: query || '',
    };
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(entry);
  }
  if (!groups.size) return null;

  // The presentation itself is the group og:image points at, otherwise the group
  // with the most distinct slides (related decks only show their cover).
  let chosenKey = null;
  if (ogImage) {
    const m = [...unescapeJsonish(ogImage).matchAll(SLIDE_URL_RE)][0];
    if (m) {
      const key = `${m[1].toLowerCase()}/${m[2]}/${m[4]}`;
      if (groups.has(key)) chosenKey = key;
    }
  }
  if (!chosenKey) {
    let best = -1;
    for (const [key, entries] of groups) {
      const distinct = new Set(entries.map((e) => e.number)).size;
      if (distinct > best) {
        best = distinct;
        chosenKey = key;
      }
    }
    // Several decks that each show only a cover (a profile or search page) is
    // a list of presentations, not one presentation.
    if (best < 2 && groups.size > 1) return null;
  }

  const entries = groups.get(chosenKey);
  const byNumber = new Map();
  for (const e of entries) {
    if (!byNumber.has(e.number)) byNumber.set(e.number, []);
    byNumber.get(e.number).push(e);
  }

  const toUrl = (e, n = e.number) =>
    `https://${e.host}/${e.location}/${e.quality}/${e.title}-${n}-${e.width}.${e.format}${e.query}`;

  const pick = (list) => {
    const safe = list.filter((e) => WEB_SAFE_FORMATS.includes(e.format));
    const pool = safe.length ? safe : list;
    const full = [...pool].sort(sizePreference)[0];
    const thumb = [...pool].sort((a, b) => a.width - b.width).find((e) => e.width >= THUMB_MIN_WIDTH) || full;
    return { full, thumb };
  };

  // Template for slides the page did not list (lazy-loaded): the largest
  // variant seen anywhere in the deck, with the slide number swapped.
  const allPicked = [...byNumber.values()].map(pick);
  const templateFull = allPicked.map((p) => p.full).sort(sizePreference)[0];
  const templateThumb =
    allPicked
      .map((p) => p.thumb)
      .sort((a, b) => a.width - b.width)
      .find((e) => e.width >= THUMB_MIN_WIDTH) || templateFull;

  const declared = findTotalSlides(text);
  const maxSeen = Math.max(...byNumber.keys());
  const total = Math.min(Math.max(declared || 0, maxSeen), MAX_SLIDES);

  const slides = [];
  for (let n = 1; n <= total; n++) {
    const list = byNumber.get(n);
    if (list) {
      const { full, thumb } = pick(list);
      // Upgrade to the deck's largest size when this slide was only seen small.
      const fullUrl = full.width < templateFull.width ? toUrl({ ...templateFull, query: '' }, n) : toUrl(full);
      slides.push({ number: n, full: fullUrl, thumb: toUrl(thumb) });
    } else {
      slides.push({
        number: n,
        full: toUrl({ ...templateFull, query: '' }, n),
        thumb: toUrl({ ...templateThumb, query: '' }, n),
      });
    }
  }
  return { slides, totalSlides: total, deckTitle: null, source: 'html-scan' };
}

/**
 * Parse a SlideShare page. Returns { title, totalSlides, slides[], source }.
 * Throws UserError when no slides can be found.
 */
function extractSlides(html, pageUrl = '') {
  if (typeof html !== 'string' || !html) {
    throw new UserError('SlideShare returned an empty page.', 502, 'empty_page');
  }
  const $ = cheerio.load(html);
  const ogTitle = $('meta[property="og:title"]').attr('content');
  const ogImage = $('meta[property="og:image"]').attr('content') || '';

  const result = fromNextData($) || fromHtmlScan(html, ogImage);
  if (!result || !result.slides.length) {
    const lower = html.toLowerCase();
    if (lower.includes('this content is private') || lower.includes('private content')) {
      throw new UserError('This presentation is private, so its slides cannot be downloaded.', 403, 'private');
    }
    throw new UserError(
      'No slides were found on that page. Make sure the link opens a single SlideShare presentation.',
      422,
      'no_slides'
    );
  }

  const title =
    cleanTitle(ogTitle) || cleanTitle(result.deckTitle) || cleanTitle($('title').first().text()) || titleFromUrl(pageUrl);

  return { title: title || 'SlideShare presentation', totalSlides: result.slides.length, slides: result.slides, source: result.source };
}

function titleFromUrl(pageUrl) {
  try {
    const parts = new URL(pageUrl).pathname.split('/').filter(Boolean);
    const slug = parts[parts.length - 1] || '';
    return decodeURIComponent(slug.replace(/-\d+$/, '')).replace(/[-_]+/g, ' ').trim();
  } catch {
    return '';
  }
}

/** Fetch a SlideShare page and extract its slides. */
async function fetchPresentation(fetchImpl, pageUrl) {
  const res = await safeFetch(fetchImpl, pageUrl, {
    suffixes: PAGE_HOST_SUFFIXES,
    maxBytes: 8 * 1024 * 1024,
    accept: 'text/html,application/xhtml+xml',
  });
  const html = res.body.toString('utf8');
  const deck = extractSlides(html, res.url);
  return { url: res.url, ...deck };
}

module.exports = { extractSlides, fetchPresentation, cleanTitle, MAX_SLIDES };
