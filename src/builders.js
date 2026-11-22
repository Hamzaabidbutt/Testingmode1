'use strict';

/**
 * Downloads slide images and packages them as PDF, PowerPoint or ZIP.
 */

const { PDFDocument } = require('pdf-lib');
const PptxGenJS = require('pptxgenjs');
const JSZip = require('jszip');
const { UserError, safeFetch, IMAGE_HOST_SUFFIXES } = require('./net');

const MAX_IMAGE_BYTES = 15 * 1024 * 1024;
const MAX_TOTAL_BYTES = 300 * 1024 * 1024;

function detectImageType(buf) {
  if (!buf || buf.length < 12) return null;
  if (buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff) return 'jpeg';
  if (buf.slice(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))) return 'png';
  if (buf.toString('ascii', 0, 4) === 'RIFF' && buf.toString('ascii', 8, 12) === 'WEBP') return 'webp';
  if (buf.toString('ascii', 0, 3) === 'GIF') return 'gif';
  return null;
}

const EXT = { jpeg: 'jpg', png: 'png', webp: 'webp', gif: 'gif' };
const MIME = { jpeg: 'image/jpeg', png: 'image/png', webp: 'image/webp', gif: 'image/gif' };

/** Pixel size of a JPEG, PNG or WebP image, or null. */
function imageSize(buf) {
  const type = detectImageType(buf);
  if (type === 'png') return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
  if (type === 'jpeg') {
    let i = 2;
    while (i + 9 < buf.length) {
      if (buf[i] !== 0xff) { i++; continue; }
      const marker = buf[i + 1];
      if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) { i += 2; continue; }
      const len = buf.readUInt16BE(i + 2);
      // SOF0..SOF15 except DHT(C4), JPG(C8), DAC(CC)
      if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker)) {
        return { height: buf.readUInt16BE(i + 5), width: buf.readUInt16BE(i + 7) };
      }
      i += 2 + len;
    }
    return null;
  }
  if (type === 'webp') {
    const chunk = buf.toString('ascii', 12, 16);
    if (chunk === 'VP8X') return { width: 1 + buf.readUIntLE(24, 3), height: 1 + buf.readUIntLE(27, 3) };
    if (chunk === 'VP8 ') return { width: buf.readUInt16LE(26) & 0x3fff, height: buf.readUInt16LE(28) & 0x3fff };
    if (chunk === 'VP8L') {
      const b = buf.readUInt32LE(21);
      return { width: 1 + (b & 0x3fff), height: 1 + ((b >> 14) & 0x3fff) };
    }
  }
  return null;
}

async function fetchImage(fetchImpl, url) {
  const res = await safeFetch(fetchImpl, url, {
    suffixes: IMAGE_HOST_SUFFIXES,
    maxBytes: MAX_IMAGE_BYTES,
    accept: 'image/jpeg,image/png,image/*;q=0.8',
  });
  let buf = res.body;
  let type = detectImageType(buf);
  // PDF and PowerPoint need JPEG/PNG; the CDN serves the same slide as .jpg too.
  if (type === 'webp' && /\.webp(\?|$)/i.test(url)) {
    try {
      const alt = await safeFetch(fetchImpl, url.replace(/\.webp(\?|$)/i, '.jpg$1'), {
        suffixes: IMAGE_HOST_SUFFIXES,
        maxBytes: MAX_IMAGE_BYTES,
        accept: 'image/jpeg',
      });
      const altType = detectImageType(alt.body);
      if (altType === 'jpeg' || altType === 'png') {
        buf = alt.body;
        type = altType;
      }
    } catch {
      /* keep webp */
    }
  }
  if (!type) throw new UserError('SlideShare sent something that is not an image.', 502, 'bad_image');
  return { buffer: buf, type, size: imageSize(buf) };
}

/** Fetch slide images with limited concurrency, preserving order. */
async function fetchSlideImages(fetchImpl, slides, { concurrency = 6, onProgress } = {}) {
  const out = new Array(slides.length);
  let next = 0;
  let total = 0;
  let done = 0;
  let failed = false;
  async function worker() {
    while (!failed && next < slides.length) {
      const i = next++;
      try {
        const img = await fetchImage(fetchImpl, slides[i].full);
        total += img.buffer.length;
        if (total > MAX_TOTAL_BYTES) {
          throw new UserError('This presentation is too large to download in one file. Select fewer slides.', 413, 'too_large');
        }
        out[i] = { ...img, number: slides[i].number };
      } catch (err) {
        failed = true; // stop the other workers from starting new downloads
        throw err;
      }
      done++;
      if (onProgress) onProgress(done, slides.length);
    }
  }
  const workers = Array.from({ length: Math.min(concurrency, slides.length) }, worker);
  await Promise.all(workers);
  return out;
}

function requireEmbeddable(images, what) {
  const bad = images.find((i) => i.type !== 'jpeg' && i.type !== 'png');
  if (bad) {
    throw new UserError(
      `Slide ${bad.number} is in a format that cannot go into a ${what}. Download the images instead.`,
      422,
      'unsupported_image'
    );
  }
}

async function buildPdf(images, { title } = {}) {
  requireEmbeddable(images, 'PDF');
  const pdf = await PDFDocument.create();
  pdf.setTitle(title || 'Slides');
  pdf.setCreator('Slide Saver');
  pdf.setProducer('Slide Saver');
  const PAGE_WIDTH = 960; // points; height follows each slide's aspect ratio
  for (const img of images) {
    const embedded = img.type === 'png' ? await pdf.embedPng(img.buffer) : await pdf.embedJpg(img.buffer);
    const height = Math.round((PAGE_WIDTH * embedded.height) / embedded.width);
    const page = pdf.addPage([PAGE_WIDTH, height]);
    page.drawImage(embedded, { x: 0, y: 0, width: PAGE_WIDTH, height });
  }
  return Buffer.from(await pdf.save());
}

async function buildPptx(images, { title } = {}) {
  requireEmbeddable(images, 'PowerPoint file');
  const pptx = new PptxGenJS();
  const first = images.find((i) => i.size) || { size: { width: 16, height: 9 } };
  const W = 13.333;
  const H = Math.round(((W * first.size.height) / first.size.width) * 1000) / 1000;
  pptx.defineLayout({ name: 'DECK', width: W, height: H });
  pptx.layout = 'DECK';
  pptx.title = title || 'Slides';
  pptx.company = 'Slide Saver';

  for (const img of images) {
    const slide = pptx.addSlide();
    slide.background = { color: 'FFFFFF' };
    const size = img.size || first.size;
    // Fit inside the slide, centred, keeping the aspect ratio.
    const scale = Math.min(W / size.width, H / size.height);
    const w = size.width * scale;
    const h = size.height * scale;
    slide.addImage({
      data: `${MIME[img.type]};base64,${img.buffer.toString('base64')}`,
      x: (W - w) / 2,
      y: (H - h) / 2,
      w,
      h,
    });
  }
  return pptx.write({ outputType: 'nodebuffer' });
}

async function buildZip(images) {
  const zip = new JSZip();
  const pad = String(images.length).length < 3 ? 3 : String(images.length).length;
  for (const img of images) {
    zip.file(`slide-${String(img.number).padStart(pad, '0')}.${EXT[img.type]}`, img.buffer, { binary: true });
  }
  // Slide images are already compressed, so store them as-is.
  return zip.generateAsync({ type: 'nodebuffer', compression: 'STORE' });
}

module.exports = {
  detectImageType,
  imageSize,
  fetchImage,
  fetchSlideImages,
  buildPdf,
  buildPptx,
  buildZip,
  EXT,
  MIME,
};
