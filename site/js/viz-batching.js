/* Continuous batching on one GPU, with a paged KV cache grid. */
(function () {
  const V = window.VLLM;
  const canvas = document.getElementById('batch-canvas');
  if (!canvas) return;

  const SLOTS = 8;            // max-num-seqs
  const BLOCKS = 96;          // KV cache blocks per panel
  const BLOCK_TOKENS = 4;     // tokens per block
  const MAXLEN = 64;          // longest prompt + answer; sets the scale of the slot bars
  const state = { rate: 0.2, speed: 8, paused: false, t: 0, acc: 0, seed: 1, nextId: 1 };

  function rnd() { state.seed = (state.seed * 1664525 + 1013904223) % 4294967296; return state.seed / 4294967296; }
  function newRequest(step) {
    const prompt = 4 + Math.floor(rnd() * 12);
    const out = 8 + Math.floor(rnd() * 40);
    const r = { id: state.nextId++, prompt, out, arrived: step, hue: (state.nextId % 8) + 1 };
    return r;
  }
  function mkPanel(kind) {
    return { kind, queue: [], slots: new Array(SLOTS).fill(null), done: 0, tokens: 0, busySteps: 0, slotSteps: 0, waitSum: 0, steps: 0, blocksUsed: 0, hist: [] };
  }
  let P = { cont: mkPanel('continuous') };
  let step = 0;

  function reset() { P = { cont: mkPanel('continuous') }; step = 0; state.nextId = 1; state.seed = 7; }

  function blocksFor(seq) { return Math.ceil((seq.prompt + seq.gen) / BLOCK_TOKENS); }  // paged: grows as tokens are produced
  function usedBlocks(p) { return p.slots.reduce((s, q) => s + (q ? blocksFor(q) : 0), 0); }

  function tick() {
    step++;
    state.acc += state.rate;
    const arrivals = [];
    while (state.acc >= 1) { state.acc -= 1; arrivals.push(newRequest(step)); }
    {
      const p = P.cont;
      arrivals.forEach(r => p.queue.push({ ...r, gen: 0 }));
      // admission
      for (let i = 0; i < SLOTS && p.queue.length; i++) {
        if (p.slots[i]) continue;
        const cand = p.queue[0];
        if (usedBlocks(p) + Math.ceil(cand.prompt / BLOCK_TOKENS) > BLOCKS) break; // no KV space: stay queued
        p.slots[i] = p.queue.shift(); p.slots[i].started = step; p.waitSum += step - p.slots[i].arrived;
      }
      // one decode step
      let active = 0;
      p.slots.forEach((q, i) => {
        if (!q) return;
        if (q.gen < q.out) { q.gen++; p.tokens++; active++; if (q.gen === q.out) { p.done++; p.slots[i] = null; } }
      });
      p.steps++; p.slotSteps += active; if (active > 0) p.busySteps++;
      p.hist.push(active / SLOTS); if (p.hist.length > 120) p.hist.shift();
      p.blocksUsed = usedBlocks(p);
    }
  }

  // ---- drawing ------------------------------------------------------------
  function roundRect(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }

  function drawPanel(ctx, p, x, y, w, h, title, sub) {
    const fg = V.cssVar('--lab-fg'), muted = V.cssVar('--lab-muted'), grid = V.cssVar('--lab-grid');
    const mono = '11px ' + V.cssVar('--font-mono');
    ctx.fillStyle = '#12171f'; roundRect(ctx, x, y, w, h, 8); ctx.fill(); ctx.strokeStyle = grid; ctx.stroke();
    ctx.font = '600 12px ' + V.cssVar('--font-body'); ctx.fillStyle = fg; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(title, x + 12, y + 16);
    ctx.font = mono; ctx.fillStyle = muted; ctx.fillText(V.fitText(ctx, sub, w - 24), x + 12, y + 32);
    // queue
    const qy = y + 50;
    ctx.fillStyle = muted; ctx.fillText('queued: ' + p.queue.length, x + 12, qy);
    p.queue.slice(0, 12).forEach((q, i) => { ctx.fillStyle = V.cssVar('--g' + q.hue); roundRect(ctx, x + 100 + i * 14, qy - 5, 10, 10, 2); ctx.fill(); });
    if (p.queue.length > 12) { ctx.fillStyle = muted; ctx.fillText('+' + (p.queue.length - 12), x + 100 + 12 * 14, qy); }
    // slots
    const sy = y + 78, rowH = 14, barX = x + 60, barW = w - 60 - 12;
    ctx.fillStyle = muted; ctx.fillText('batch slots', x + 12, sy - 6);
    for (let i = 0; i < SLOTS; i++) {
      const ry = sy + i * rowH;
      ctx.fillStyle = grid; roundRect(ctx, barX, ry + 2, barW, rowH - 4, 3); ctx.fill();
      const q = p.slots[i];
      ctx.fillStyle = muted; ctx.fillText('seq ' + i, x + 12, ry + rowH / 2);
      if (q) {
        const col = V.cssVar('--g' + q.hue);
        const pw = barW * (q.prompt / MAXLEN), gw = barW * (q.gen / MAXLEN);
        ctx.fillStyle = col; ctx.globalAlpha = .45; roundRect(ctx, barX, ry + 2, pw, rowH - 4, 3); ctx.fill(); ctx.globalAlpha = 1;
        // dashed outline up to the finish line: the whole answer, drawn for illustration (an engine cannot know it)
        const ow = barW * (q.out / MAXLEN), finX = barX + pw + ow;
        if (q.gen < q.out) { ctx.fillStyle = col; ctx.globalAlpha = .12; ctx.fillRect(barX + pw + gw, ry + 2, ow - gw, rowH - 4); ctx.strokeStyle = col; ctx.globalAlpha = .6; ctx.lineWidth = 1; ctx.setLineDash([3, 2]); ctx.strokeRect(barX + pw + gw + .5, ry + 2.5, ow - gw - 1, rowH - 5); ctx.setLineDash([]); ctx.globalAlpha = 1; }
        ctx.fillStyle = muted; ctx.fillRect(Math.round(finX) - 1, ry, 2, rowH);
        ctx.fillStyle = col; roundRect(ctx, barX + pw, ry + 2, gw, rowH - 4, 2); ctx.fill();
      }
    }
    // utilization sparkline
    const uy = sy + SLOTS * rowH + 14;
    ctx.fillStyle = muted; ctx.fillText('slot utilization', x + 12, uy);
    const ux = x + 120, uw = w - 120 - 12, uh = 26;
    ctx.fillStyle = grid; ctx.fillRect(ux, uy - uh / 2, uw, uh);
    if (p.hist.length > 1) {
      ctx.beginPath();
      p.hist.forEach((v, i) => { const px = ux + (i / 119) * uw, py = uy + uh / 2 - v * uh; i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.strokeStyle = V.cssVar('--accent'); ctx.lineWidth = 1.5; ctx.stroke();
    }
    // KV blocks
    const ky = uy + 30;
    const kvLabel = ['KV cache: paged blocks, grown on demand', 'KV: paged, grown on demand', 'KV: paged on demand'];
    const cols = 32, cell = Math.min(10, (w - 24) / cols), rows = BLOCKS / cols;
    let b = 0;
    const owners = [];
    p.slots.forEach(q => { if (q) { const n = blocksFor(q); const real = Math.ceil((q.prompt + q.gen) / BLOCK_TOKENS); for (let i = 0; i < n; i++) owners.push({ hue: q.hue, waste: i >= real }); } });
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++, b++) {
      const bx = x + 12 + c * cell, by = ky + 10 + r * cell;
      const o = owners[b];
      if (o) { ctx.fillStyle = V.cssVar('--g' + o.hue); ctx.globalAlpha = o.waste ? .25 : .9; ctx.fillRect(bx, by, cell - 1.5, cell - 1.5); ctx.globalAlpha = 1; }
      else { ctx.fillStyle = grid; ctx.fillRect(bx, by, cell - 1.5, cell - 1.5); }
    }
    const wasted = owners.filter(o => o.waste).length;
    ctx.fillStyle = muted; ctx.textAlign = 'right';
    const count = V.fitText(ctx, [
      owners.length + '/' + BLOCKS + ' blocks' + (wasted ? ' · ' + wasted + ' reserved but empty' : ''),
      owners.length + '/' + BLOCKS + ' blocks' + (wasted ? ' · ' + wasted + ' empty' : '')
    ], (w - 24) * 0.55);
    ctx.fillText(count, x + w - 12, ky);
    ctx.textAlign = 'left';
    ctx.fillText(V.fitText(ctx, kvLabel, w - 24 - ctx.measureText(count).width - 10), x + 12, ky);
  }

  function draw(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    drawPanel(ctx, P.cont, 0, 0, w, 274, 'Continuous batching', ['finished requests leave and queued ones join at every step', 'finished leave, queued join, every step']);
  }

  function updateReadout() {
    const el = id => document.getElementById(id);
    const f = p => ({ thr: p.steps ? (p.tokens / p.steps).toFixed(2) : '0', util: p.steps ? Math.round(100 * p.slotSteps / (p.steps * SLOTS)) : 0, done: p.done, wait: p.done ? (p.waitSum / Math.max(1, p.done)).toFixed(1) : '0' });
    const b = f(P.cont);
    if (el('b-cont-thr')) { el('b-cont-thr').textContent = b.thr; el('b-cont-util').textContent = b.util + '%'; el('b-cont-done').textContent = b.done; el('b-cont-wait').textContent = b.wait; }
  }

  const ctx = V.setupCanvas(canvas, () => 274);
  let acc = 0;
  reset();
  V.animate(canvas, (dt) => {
    if (!state.paused) {
      acc += dt * state.speed;
      let n = 0;
      while (acc >= 1 && n < 50) { acc -= 1; tick(); n++; }
      if (n) updateReadout();
    }
    draw(ctx, canvas._w, canvas._h);
  });
  const rate = document.getElementById('batch-rate');
  if (rate) rate.addEventListener('input', () => { state.rate = +rate.value / 100; document.getElementById('batch-rate-val').textContent = (state.rate).toFixed(2); });
  const pause = document.getElementById('batch-pause');
  if (pause) pause.addEventListener('click', () => { state.paused = !state.paused; pause.textContent = state.paused ? 'Play' : 'Pause'; });
  const rst = document.getElementById('batch-reset');
  if (rst) rst.addEventListener('click', () => { reset(); updateReadout(); });
})();
