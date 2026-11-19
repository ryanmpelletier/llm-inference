/* Shared helpers: theme, citations, canvas sizing. */
(function () {
  const root = document.documentElement;
  const body = document.body;
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)');

  // ---- theme -------------------------------------------------------------
  function currentTheme() {
    const t = root.getAttribute('data-theme');
    if (t) return t;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  function setTheme(t) {
    root.setAttribute('data-theme', t);
    try { localStorage.setItem('vllm-theme', t); } catch (e) {}
    document.dispatchEvent(new CustomEvent('themechange'));
    updateThemeBtn();
  }
  function updateThemeBtn() {
    const b = document.getElementById('theme-btn');
    if (!b) return;
    const dark = currentTheme() === 'dark';
    b.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
    b.querySelector('.sun').toggleAttribute('hidden', !dark);
    b.querySelector('.moon').toggleAttribute('hidden', dark);
  }
  try {
    const saved = localStorage.getItem('vllm-theme');
    if (saved === 'dark' || saved === 'light') root.setAttribute('data-theme', saved);
  } catch (e) {}
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    document.dispatchEvent(new CustomEvent('themechange'));
    updateThemeBtn();
  });

  // ---- css var reader for canvases ---------------------------------------
  const cssCache = {};
  let cssStamp = 0;
  function cssVar(name) {
    const key = cssStamp + name;
    if (cssCache[key]) return cssCache[key];
    const v = getComputedStyle(root).getPropertyValue(name).trim();
    cssCache[key] = v;
    return v;
  }
  document.addEventListener('themechange', () => { cssStamp++; });

  // ---- canvas helper: HiDPI sizing + resize observer ---------------------
  function setupCanvas(canvas, logicalHeight, onResize) {
    const ctx = canvas.getContext('2d');
    function resize() {
      const w = canvas.clientWidth || canvas.parentElement.clientWidth || 600;
      const h = typeof logicalHeight === 'function' ? logicalHeight(w) : logicalHeight;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      canvas.style.height = h + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      canvas._w = w; canvas._h = h;
      if (onResize) onResize(w, h);
    }
    resize();
    if (window.ResizeObserver) new ResizeObserver(resize).observe(canvas.parentElement);
    else window.addEventListener('resize', resize);
    return ctx;
  }

  // ---- animation loop that pauses offscreen ------------------------------
  function animate(canvas, draw) {
    let visible = true, last = performance.now(), raf = 0;
    function frame(now) {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      draw(dt, now / 1000);
      if (visible) raf = requestAnimationFrame(frame);
    }
    const io = new IntersectionObserver(entries => {
      entries.forEach(e => {
        const was = visible;
        visible = e.isIntersecting;
        if (visible && !was) { last = performance.now(); raf = requestAnimationFrame(frame); }
      });
    }, { rootMargin: '80px' });
    io.observe(canvas);
    raf = requestAnimationFrame(frame);
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) { cancelAnimationFrame(raf); visible = false; }
      else if (!visible) { visible = true; last = performance.now(); raf = requestAnimationFrame(frame); }
    });
  }

  // ---- citations ---------------------------------------------------------
  // <span data-cite="key"></span> becomes a numbered superscript linking to #src-key
  function buildCitations() {
    const list = document.getElementById('source-list');
    if (!list) return;
    const items = Array.from(list.querySelectorAll('li[id]'));
    const index = {};
    items.forEach((li, i) => { index[li.id.replace(/^src-/, '')] = i + 1; });
    document.querySelectorAll('[data-cite]').forEach(el => {
      const keys = el.dataset.cite.split(/\s*,\s*/).filter(Boolean);
      const sup = document.createElement('sup');
      sup.className = 'cite';
      sup.innerHTML = keys.map(k => index[k] ? `<a href="#src-${k}" title="Source ${index[k]}">[${index[k]}]</a>` : `<a title="missing source ${k}">[?]</a>`).join('');
      el.replaceWith(sup);
    });
  }

  // ---- canvas text fitting ----------------------------------------------
  // Pick the first candidate string that fits in maxW; truncate the last one with an ellipsis if none do.
  function fitText(ctx, candidates, maxW) {
    const list = Array.isArray(candidates) ? candidates : [candidates];
    for (const t of list) if (ctx.measureText(t).width <= maxW) return t;
    let t = list[list.length - 1];
    while (t.length > 1 && ctx.measureText(t + '…').width > maxW) t = t.slice(0, -1);
    return t + '…';
  }
  window.VLLM = { cssVar, setupCanvas, animate, currentTheme, setTheme, prefersReduced, fitText,
    fmt: {
      gb: v => v >= 1000 ? (v / 1000).toFixed(2) + ' TB' : v.toFixed(v < 10 ? 2 : 1) + ' GB',
      int: v => Math.round(v).toLocaleString('en-US'),
      pct: v => Math.round(v * 100) + '%'
    }
  };

  document.addEventListener('DOMContentLoaded', () => {
    updateThemeBtn();
    const tb = document.getElementById('theme-btn');
    if (tb) tb.addEventListener('click', () => setTheme(currentTheme() === 'dark' ? 'light' : 'dark'));
    buildCitations();
  });
})();
