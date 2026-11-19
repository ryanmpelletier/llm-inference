/* Glossary: search and topic filter for the glossary section, plus inline term links.
   The first mention of each glossary term in every section becomes a link to its entry;
   hovering or focusing it shows the short definition, and after jumping to an entry a
   "Back to where you were" button returns the reader to the same spot. */
(function () {
  const V = window.VLLM;
  const root = document.getElementById('glossary');
  if (!root) return;
  const entries = Array.from(root.querySelectorAll('dl.gloss > div[id]'));
  const groups = Array.from(root.querySelectorAll('.gloss-group'));
  const q = document.getElementById('gloss-q');
  const count = document.getElementById('gloss-count');
  const empty = document.getElementById('gloss-empty');
  const chips = Array.from(root.querySelectorAll('[data-gcat]'));
  let cat = 'all';

  // ---------------------------------------------------------------- filter
  const norm = s => s.toLowerCase().replace(/[‐-―]/g, '-');
  entries.forEach(e => { e._text = norm(e.textContent + ' ' + (e.dataset.match || '').replace(/\|/g, ' ')); });
  function applyFilter() {
    const words = norm(q.value.trim()).split(/\s+/).filter(Boolean);
    let shown = 0;
    groups.forEach(g => {
      const catOk = cat === 'all' || g.dataset.cat === cat;
      let any = false;
      g.querySelectorAll('dl.gloss > div[id]').forEach(e => {
        const ok = catOk && words.every(w => e._text.includes(w));
        e.hidden = !ok;
        if (ok) { any = true; shown++; }
      });
      g.hidden = !any;
    });
    count.textContent = shown === entries.length ? shown + ' terms' : shown + ' of ' + entries.length + ' terms';
    empty.hidden = shown > 0;
  }
  function setCat(c) {
    cat = c;
    chips.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.gcat === c)));
    applyFilter();
  }
  q.addEventListener('input', applyFilter);
  chips.forEach(b => b.addEventListener('click', () => setCat(b.dataset.gcat)));
  // make sure a linked entry is visible even when a filter would hide it
  function reveal(id) {
    const el = document.getElementById(id);
    if (el && el.hidden) { q.value = ''; setCat('all'); }
  }
  window.addEventListener('hashchange', () => { if (location.hash.startsWith('#g-')) reveal(location.hash.slice(1)); });
  if (location.hash.startsWith('#g-')) reveal(location.hash.slice(1));

  // ---------------------------------------------------------- inline links
  const byLower = new Map();
  entries.forEach(e => {
    if (!e.dataset.match) return;
    e.dataset.match.split('|').forEach(alias => {
      const a = alias.trim();
      if (a) byLower.set(a.toLowerCase(), { id: e.id, alias: a, cs: a !== a.toLowerCase() });
    });
  });
  const escRe = s => s.replace(/[.*+?^${}()|[\]\\\/]/g, '\\$&');
  const alts = Array.from(byLower.values()).map(v => v.alias).sort((a, b) => b.length - a.length).map(escRe);
  // whole words only; allow a plural "s"/"es"; hyphenated compounds are not split
  const RE = new RegExp('(?<![\\w-])(' + alts.join('|') + ')(?:es|s)?(?![\\w-])', 'gi');
  // everyday terms are linked once on the whole page; core concepts once per section
  const ONCE = new Set(['g-token', 'g-request', 'g-prompt', 'g-layer', 'g-weights', 'g-batching', 'g-latency', 'g-throughput',
    'g-serving', 'g-inference', 'g-llm', 'g-concurrency', 'g-queue', 'g-scheduler', 'g-interconnect', 'g-node', 'g-rank',
    'g-tensor', 'g-kernel', 'g-util', 'g-bandwidth', 'g-activations', 'g-attention',
    'g-transformer', 'g-lockstep', 'g-collective', 'g-step', 'g-context', 'g-replica', 'g-expert', 'g-stage']);
  const pageUsed = new Set();
  const SKIP = '.hero .lede b, a, code, pre, kbd, h1, h2, h3, h4, button, label, select, option, input, textarea, svg, canvas, sup, .lab, .eyebrow, .tag, .k, .scn, th, .cmd, .verdict, .hero-legend, .gloss, #glossary, .sources, .rail';

  function scopeKey(node, sec) { return sec.id || 'hero'; }
  function linkSection(sec) {
    const seen = {};
    const walker = document.createTreeWalker(sec, NodeFilter.SHOW_TEXT, {
      acceptNode: n => (!n.nodeValue.trim() || n.parentElement.closest(SKIP)) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(node => {
      const key = scopeKey(node, sec);
      const used = seen[key] || (seen[key] = new Set());
      const text = node.nodeValue;
      let m, last = 0, frag = null;
      RE.lastIndex = 0;
      while ((m = RE.exec(text))) {
        const hit = byLower.get(m[1].toLowerCase());
        if (!hit || used.has(hit.id) || (ONCE.has(hit.id) && pageUsed.has(hit.id))) continue;
        if (hit.cs && m[1] !== hit.alias) continue;
        used.add(hit.id); pageUsed.add(hit.id);
        frag = frag || document.createDocumentFragment();
        frag.appendChild(document.createTextNode(text.slice(last, m.index)));
        const a = document.createElement('a');
        a.className = 'term-link'; a.href = '#' + hit.id; a.textContent = m[0];
        frag.appendChild(a);
        last = m.index + m[0].length;
      }
      if (frag) { frag.appendChild(document.createTextNode(text.slice(last))); node.replaceWith(frag); }
    });
  }
  const main = document.querySelector('main');
  if (main) main.querySelectorAll(':scope > section').forEach(sec => { if (sec.id !== 'glossary' && sec.id !== 'sources') linkSection(sec); });

  // --------------------------------------------------------------- popover
  const pop = document.createElement('div');
  pop.className = 'term-pop'; pop.id = 'term-pop'; pop.setAttribute('role', 'tooltip'); pop.hidden = true;
  document.body.appendChild(pop);
  let hideT = 0, showT = 0, current = null;
  const TARGET = 'a.term-link, .tip[data-tip]';
  function show(a) {
    pop.innerHTML = '';
    if (a.matches('.tip')) {
      // plain tooltip (e.g. a graphic's legend): title + one paragraph from data attributes
      const t = document.createElement('b'); t.textContent = a.dataset.tipTitle || ''; pop.appendChild(t);
      const p = document.createElement('p'); p.textContent = a.dataset.tip; pop.appendChild(p);
    } else {
      const e = document.getElementById(a.hash.slice(1));
      if (!e) return;
      const def = e.querySelector('.g-def'), bus = e.querySelector('.g-bus');
      const t = document.createElement('b'); t.textContent = e.querySelector('dt').textContent; pop.appendChild(t);
      const p = document.createElement('p'); p.textContent = def ? def.textContent : ''; pop.appendChild(p);
      if (bus) { const pb = document.createElement('p'); pb.className = 'bus'; pb.textContent = bus.textContent; pop.appendChild(pb); }
      const more = document.createElement('p'); more.className = 'more'; more.textContent = 'Click for the full glossary entry'; pop.appendChild(more);
    }
    pop.hidden = false;
    const r = a.getBoundingClientRect(), pw = pop.offsetWidth, ph = pop.offsetHeight, m = 12;
    let x = Math.min(Math.max(m, r.left + r.width / 2 - pw / 2), window.innerWidth - pw - m);
    let y = r.bottom + 8;
    if (y + ph > window.innerHeight - m) y = r.top - ph - 8;
    pop.style.left = x + 'px'; pop.style.top = Math.max(m, y) + 'px';
    a.setAttribute('aria-describedby', 'term-pop');
    current = a;
  }
  function hide() {
    clearTimeout(showT);
    pop.hidden = true;
    if (current) current.removeAttribute('aria-describedby');
    current = null;
  }
  document.addEventListener('pointerover', ev => {
    if (ev.pointerType !== 'mouse') return;
    const a = ev.target.closest && ev.target.closest(TARGET);
    if (!a) return;
    clearTimeout(hideT); clearTimeout(showT);
    showT = setTimeout(() => show(a), 140);
  });
  document.addEventListener('pointerout', ev => {
    if (ev.pointerType !== 'mouse') return;   // a tap also fires pointerout; tapped tooltips close on the next tap
    const a = ev.target.closest && ev.target.closest(TARGET);
    if (!a) return;
    clearTimeout(showT); hideT = setTimeout(hide, 80);
  });
  document.addEventListener('focusin', ev => {
    const a = ev.target.closest && ev.target.closest(TARGET);
    if (a && a.matches(':focus-visible')) show(a); else if (current) hide();
  });
  document.addEventListener('keydown', ev => { if (ev.key === 'Escape' && current) hide(); });
  document.addEventListener('click', ev => {
    const a = ev.target.closest && ev.target.closest('.tip[data-tip]');
    if (a) { if (current === a) hide(); else show(a); }
    else if (current && current.matches('.tip')) hide();
  });
  window.addEventListener('scroll', () => { if (current) hide(); }, { passive: true });

  // ---------------------------------------------------- back to where you were
  const back = document.createElement('button');
  back.type = 'button'; back.className = 'term-back'; back.hidden = true;
  back.textContent = '← Back to where you were';
  document.body.appendChild(back);
  let returnY = null;
  document.addEventListener('click', ev => {
    const a = ev.target.closest && ev.target.closest('a');
    if (!a) return;
    if (a.classList.contains('term-link')) {
      hide();
      returnY = window.scrollY;
      reveal(a.hash.slice(1));
      back.hidden = false;
    } else if (!a.closest('#glossary')) {
      // any other navigation (rail, buttons) means the reader has moved on
      back.hidden = true; returnY = null;
    }
  });
  back.addEventListener('click', () => {
    if (returnY !== null) window.scrollTo({ top: returnY, behavior: V && V.prefersReduced.matches ? 'auto' : 'smooth' });
    back.hidden = true; returnY = null;
  });
})();
