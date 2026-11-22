'use strict';

(function () {
  const form = document.getElementById('lookup');
  const input = document.getElementById('url');
  const lookupBtn = document.getElementById('lookup-btn');
  const lookupError = document.getElementById('lookup-error');
  const results = document.getElementById('results');
  const titleEl = document.getElementById('deck-title');
  const metaEl = document.getElementById('deck-meta');
  const list = document.getElementById('slides');
  const template = document.getElementById('slide-template');
  const statusEl = document.getElementById('download-status');
  const downloadButtons = Array.from(document.querySelectorAll('[data-format]'));

  const FORMAT_NAMES = { pdf: 'PDF', pptx: 'PowerPoint file', zip: 'ZIP of images' };

  let deck = null; // { url, title, totalSlides, slides[] }
  let busy = false;

  function showLookupError(message) {
    lookupError.textContent = message;
    lookupError.hidden = !message;
    input.setAttribute('aria-invalid', message ? 'true' : 'false');
  }

  function setStatus(message, isError) {
    statusEl.textContent = message || '';
    statusEl.classList.toggle('is-error', Boolean(isError));
  }

  function selectedNumbers() {
    return Array.from(list.querySelectorAll('.mount'))
      .filter((b) => b.getAttribute('aria-pressed') === 'true')
      .map((b) => Number(b.dataset.number));
  }

  function updateMeta() {
    if (!deck) return;
    const total = deck.totalSlides;
    const selected = selectedNumbers().length;
    const slidesWord = total === 1 ? 'slide' : 'slides';
    metaEl.textContent =
      selected === total ? `${total} ${slidesWord}, all selected` : `${total} ${slidesWord}, ${selected} selected`;
  }

  /** [1,2,3,5,7,8] -> "1-3,5,7-8" */
  function compactRanges(numbers) {
    const parts = [];
    for (let i = 0; i < numbers.length; i++) {
      const start = numbers[i];
      let end = start;
      while (numbers[i + 1] === end + 1) end = numbers[++i];
      parts.push(start === end ? String(start) : `${start}-${end}`);
    }
    return parts.join(',');
  }

  function showSkeleton(count) {
    list.innerHTML = '';
    list.classList.add('is-loading');
    for (let i = 0; i < count; i++) {
      const li = document.createElement('li');
      li.className = 'slide';
      li.innerHTML = '<span class="mount" aria-hidden="true"><span class="window"></span><span class="mount-label">&nbsp;</span></span>';
      list.appendChild(li);
    }
  }

  function renderDeck() {
    titleEl.textContent = deck.title;
    list.innerHTML = '';
    list.classList.remove('is-loading');
    const frag = document.createDocumentFragment();
    for (const s of deck.slides) {
      const node = template.content.firstElementChild.cloneNode(true);
      const mount = node.querySelector('.mount');
      const img = node.querySelector('img');
      const win = node.querySelector('.window');
      mount.dataset.number = String(s.number);
      mount.setAttribute('aria-label', `Slide ${s.number}, included in download`);
      node.querySelector('.num').textContent = String(s.number);
      img.src = s.thumb;
      img.alt = `Slide ${s.number}`;
      img.addEventListener('load', () => {
        if (img.naturalWidth && img.naturalHeight) win.style.aspectRatio = `${img.naturalWidth} / ${img.naturalHeight}`;
      });
      img.addEventListener('error', () => win.classList.add('is-broken'));
      const save = node.querySelector('.save-one');
      save.href = s.download;
      save.setAttribute('aria-label', `Save slide ${s.number} as an image`);
      frag.appendChild(node);
    }
    list.appendChild(frag);
    updateMeta();
    setStatus('');
  }

  async function readError(res) {
    try {
      const data = await res.json();
      if (data && data.error) return data.error;
    } catch (_) {
      /* not JSON */
    }
    if (res.status === 429) return 'Too many requests. Wait a minute and try again.';
    return `The server could not finish the request (error ${res.status}). Try again.`;
  }

  async function lookup(url) {
    showLookupError('');
    const value = url.trim();
    if (!value) {
      showLookupError('Paste a SlideShare link to get started.');
      input.focus();
      return;
    }
    if (!/slideshare\.net/i.test(value)) {
      showLookupError('Only slideshare.net links are supported. Copy the address of a SlideShare presentation.');
      input.focus();
      return;
    }

    lookupBtn.disabled = true;
    lookupBtn.textContent = 'Finding slides…';
    results.hidden = false;
    titleEl.textContent = 'Looking up the presentation…';
    metaEl.textContent = '';
    showSkeleton(8);

    try {
      const res = await fetch('/api/slides', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: value }),
      });
      if (!res.ok) throw new Error(await readError(res));
      deck = await res.json();
      renderDeck();
      const params = new URLSearchParams(window.location.search);
      params.set('url', deck.url);
      history.replaceState(null, '', `${window.location.pathname}?${params}`);
      titleEl.setAttribute('tabindex', '-1');
      titleEl.focus({ preventScroll: true });
      results.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
    } catch (err) {
      deck = null;
      results.hidden = true;
      list.innerHTML = '';
      showLookupError(err && err.message ? err.message : 'Could not load that presentation. Try again.');
      input.focus();
    } finally {
      lookupBtn.disabled = false;
      lookupBtn.textContent = 'Show slides';
    }
  }

  function filenameFrom(res, fallback) {
    const cd = res.headers.get('Content-Disposition') || '';
    const star = cd.match(/filename\*=UTF-8''([^;]+)/i);
    if (star) {
      try {
        return decodeURIComponent(star[1]);
      } catch (_) {
        /* fall through */
      }
    }
    const plain = cd.match(/filename="([^"]+)"/i);
    return plain ? plain[1] : fallback;
  }

  async function download(format, button) {
    if (!deck || busy) return;
    const numbers = selectedNumbers();
    if (!numbers.length) {
      setStatus('Select at least one slide to download.', true);
      return;
    }
    const params = new URLSearchParams({ url: deck.url, format });
    if (numbers.length !== deck.totalSlides) params.set('slides', compactRanges(numbers));

    busy = true;
    const label = button.innerHTML;
    downloadButtons.forEach((b) => (b.disabled = true));
    button.textContent = 'Preparing…';
    const count = numbers.length;
    setStatus(`Building your ${FORMAT_NAMES[format]} from ${count} ${count === 1 ? 'slide' : 'slides'}. Large decks take a little longer.`);

    try {
      const res = await fetch(`/api/download?${params}`);
      if (!res.ok) throw new Error(await readError(res));
      const blob = await res.blob();
      const name = filenameFrom(res, `slides.${format}`);
      const href = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = href;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(href), 60_000);
      setStatus(`Downloaded ${name}.`);
    } catch (err) {
      setStatus(err && err.message ? err.message : 'The download failed. Try again.', true);
    } finally {
      busy = false;
      button.innerHTML = label;
      downloadButtons.forEach((b) => (b.disabled = false));
    }
  }

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    lookup(input.value);
  });

  input.addEventListener('input', () => {
    if (!lookupError.hidden) showLookupError('');
  });

  list.addEventListener('click', (e) => {
    const mount = e.target.closest('button.mount');
    if (!mount) return;
    const on = mount.getAttribute('aria-pressed') !== 'true';
    mount.setAttribute('aria-pressed', String(on));
    mount.setAttribute('aria-label', `Slide ${mount.dataset.number}, ${on ? 'included in' : 'left out of'} download`);
    updateMeta();
    setStatus('');
  });

  function setAll(on) {
    list.querySelectorAll('button.mount').forEach((m) => {
      m.setAttribute('aria-pressed', String(on));
      m.setAttribute('aria-label', `Slide ${m.dataset.number}, ${on ? 'included in' : 'left out of'} download`);
    });
    updateMeta();
    setStatus('');
  }

  document.getElementById('select-all').addEventListener('click', () => setAll(true));
  document.getElementById('select-none').addEventListener('click', () => setAll(false));
  downloadButtons.forEach((b) => b.addEventListener('click', () => download(b.dataset.format, b)));

  // Support links like /?url=https://www.slideshare.net/...
  const initial = new URLSearchParams(window.location.search).get('url');
  if (initial) {
    input.value = initial;
    lookup(initial);
  }
})();
