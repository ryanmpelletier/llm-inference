/* The bus loop: the beginner analogy for decode, made literal.
   The route is the model, each stop a layer. One lap = one decode step: every rider who is
   ready gets one more word. A lap costs the same whether the bus is full or empty. Riders board
   and leave at the depot between laps (continuous batching), bring their prompt as luggage that
   must be loaded first (prefill, optionally chunked), and their luggage grows by one bag per word
   (KV cache). When the rack overflows, the newest rider is bumped and must reload (preemption).
   Toy numbers, chosen to make the shape visible, not to model any GPU. */
(function () {
  const V = window.VLLM;
  const canvas = document.getElementById('bus-canvas');
  if (!canvas) return;
  const $ = id => document.getElementById(id);

  const SEATS = 8;            // like --max-num-seqs
  const RACK = 72;            // luggage capacity per bus, like KV cache blocks
  const LAP = 2.0;            // s of driving per lap, independent of load
  const DWELL = 0.3;          // s at the stop even with nothing to load
  const PER_BAG = 0.03;       // s of extra stop time per bag loaded
  const MAX_OUT = 14;         // longest answer a rider can want; sets the scale of the seat bars
  const CHUNK = 16;           // bags loaded per stop when chunked (like --max-num-batched-tokens)
  const STOPS = 10;
  const WATERMARK = 6;        // free bags kept in reserve before admitting anyone, like vLLM's free-block watermark

  const knobs = { rate: 0.4, chunked: true, speed: 0.5, paused: false, n: 1 };
  let buses = [], queue = [], t = 0, seed = 5, nextId = 1, nextArrival = 0;
  let wordEvents = [], ttfts = [], done = 0, bumped = 0, particles = [], floaters = [], lastWords = 0;

  function rnd() { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; }
  function expo(mean) { return -Math.log(Math.max(1e-9, 1 - rnd())) * mean; }
  function mkRider(prompt, out) {
    const id = nextId++;
    return { id, hue: ((id - 1) % 8) + 1, prompt, out, gen: 0, lug: 0, toLoad: prompt, ready: false, arrived: t, boarded: 0, first: 0 };
  }
  function randomRider() { return mkRider(4 + Math.floor(rnd() * 9), 5 + Math.floor(rnd() * 10)); }
  function mkBus(i, pos) { return { i, pos, phase: 'drive', left: LAP * (1 - pos), seats: new Array(SEATS).fill(null), laps: 0, flash: 0, lastLoad: 0 }; }
  const used = b => b.seats.reduce((s, r) => s + (r ? r.lug : 0), 0);
  // rack space spoken for: bags on the rack plus bags still waiting to be loaded
  const claimed = b => b.seats.reduce((s, r) => s + (r ? r.lug + r.toLoad : 0), 0);
  const onBoard = b => b.seats.filter(Boolean).length;

  function reset() {
    t = 0; seed = 5; nextId = 1; queue = []; wordEvents = []; ttfts = []; done = 0; bumped = 0;
    particles = []; floaters = []; lastWords = 0;
    buses = [];
    for (let i = 0; i < knobs.n; i++) buses.push(mkBus(i, i / knobs.n));
    nextArrival = expo(LAP / knobs.rate);
    resize();
  }
  function setBuses(n) {
    if (n === buses.length) return;
    knobs.n = n;
    if (n > buses.length) {
      const lead = buses[0];
      buses.push(mkBus(1, (lead.pos + 0.5) % 1));
    } else {
      // the removed bus's riders go back to the front of the queue and must reload
      const gone = buses.pop();
      const back = gone.seats.filter(Boolean).sort((a, b) => a.boarded - b.boarded);
      back.forEach(r => { r.toLoad = r.lug; r.lug = 0; r.ready = false; });
      queue = back.concat(queue);
    }
    resize();
  }

  // ---- what happens at the stop ----
  function arriveAtStop(b) {
    b.laps++;
    // 1. every rider who was ready during the lap gets one word; luggage grows by one bag
    let words = 0;
    b.seats.forEach((r, i) => {
      if (!r || !r.ready) return;
      r.gen++; r.lug++; words++;
      if (r.gen === 1) { r.first = t; ttfts.push(t - r.arrived); if (ttfts.length > 8) ttfts.shift(); }
      if (r.gen >= r.out) { b.seats[i] = null; done++; particles.push({ kind: 'exit', hue: r.hue, bus: b.i, t0: performance.now() }); }
    });
    if (words) wordEvents.push({ t, n: words });
    lastWords = words;
    floaters.push({ life: Math.min(1300, 900 * (LAP + DWELL) / knobs.speed / buses.length), text: words ? '+' + words + (words === 1 ? ' word' : ' words') : 'empty lap: 0 words', bus: b.i, t0: performance.now(), empty: !words });
    // 2. rack overflow: bump the newest rider back to the queue (vLLM preempts and recomputes)
    while (claimed(b) > RACK) {
      let idx = -1;
      b.seats.forEach((r, i) => { if (r && (idx < 0 || r.boarded > b.seats[idx].boarded)) idx = i; });
      if (idx < 0) break;
      const r = b.seats[idx]; b.seats[idx] = null;
      r.toLoad = r.lug; r.lug = 0; r.ready = false;
      queue.unshift(r); bumped++; b.flash = performance.now();
    }
    // 3. board waiting riders into free seats if their luggage fits on the rack
    for (let i = 0; i < SEATS && queue.length; i++) {
      if (b.seats[i]) continue;
      const r = queue[0];
      if (claimed(b) + r.toLoad + WATERMARK > RACK) break;
      queue.shift(); r.boarded = t; b.seats[i] = r;
      particles.push({ kind: 'board', hue: r.hue, bus: b.i, t0: performance.now() });
    }
    // 4. load luggage, oldest boarders first; chunked loading caps bags per stop
    let budget = knobs.chunked ? CHUNK : Infinity, loaded = 0;
    b.seats.filter(Boolean).sort((a, c) => a.boarded - c.boarded).forEach(r => {
      if (r.toLoad <= 0 || budget <= 0) return;
      const take = Math.min(r.toLoad, budget);
      r.toLoad -= take; r.lug += take; budget -= take; loaded += take;
    });
    b.seats.forEach(r => { if (r && r.toLoad === 0) r.ready = true; });
    b.lastLoad = loaded;
    b.phase = 'dwell'; b.left = DWELL + loaded * PER_BAG; b.pos = 0;
  }

  function advance(dt) {
    t += dt;
    while (nextArrival <= t) { queue.push(randomRider()); nextArrival += expo(LAP / knobs.rate); }
    for (const b of buses) {
      let rem = dt;
      while (rem > 0) {
        if (rem < b.left) { b.left -= rem; rem = 0; }
        else {
          rem -= b.left;
          if (b.phase === 'drive') arriveAtStop(b);
          else { b.phase = 'drive'; b.left = LAP; }
        }
      }
      b.pos = b.phase === 'drive' ? 1 - b.left / LAP : 0;
    }
    const cut = t - 12;
    while (wordEvents.length && wordEvents[0].t < cut) wordEvents.shift();
  }

  // ---- geometry ----
  let L = null;
  function layout(w) {
    const B = buses.length || knobs.n;
    const wide = w >= 760;
    const PANEL_H = 234;
    let trackW, trackH, panels = [];
    if (wide) {
      trackW = Math.round(w * 0.54); trackH = 262;
      const px = trackW + 16, pw = w - px;
      const side = B === 2 && pw >= 440;
      for (let i = 0; i < B; i++) {
        if (side) panels.push({ x: px + i * ((pw + 10) / 2), y: 0, w: (pw - 10) / 2, h: PANEL_H });
        else panels.push({ x: px, y: i * (PANEL_H + 10), w: pw, h: PANEL_H });
      }
    } else {
      trackW = w; trackH = w < 480 ? 232 : 262;
      const side = B === 2 && w >= 520;
      for (let i = 0; i < B; i++) {
        if (side) panels.push({ x: i * ((w + 10) / 2), y: trackH + 12, w: (w - 10) / 2, h: PANEL_H });
        else panels.push({ x: 0, y: trackH + 12 + i * (PANEL_H + 10), w, h: PANEL_H });
      }
    }
    const pad = 30;
    const top = 30;
    const r = Math.max(40, Math.min((trackH - top - 62) / 2, (trackW - 2 * pad) / 4));
    const yc = top + r;
    const xl = pad + r, xr = trackW - pad - r;
    const h = Math.max(trackH, ...panels.map(p => p.y + p.h));
    return { w, h, trackW, trackH, r, yc, xl, xr, st: xr - xl, mid: (xl + xr) / 2, yTop: yc - r, yBot: yc + r, panels, small: w < 480 };
  }
  function pointAt(u) {
    const { r, yc, xl, xr, st, mid, yTop, yBot } = L;
    const len = 2 * st + 2 * Math.PI * r;
    let s = ((u % 1) + 1) % 1 * len;
    if (s < st / 2) return { x: mid + s, y: yBot, a: 0 };
    s -= st / 2;
    if (s < Math.PI * r) { const th = s / r; return { x: xr + r * Math.sin(th), y: yc + r * Math.cos(th), a: -th }; }
    s -= Math.PI * r;
    if (s < st) return { x: xr - s, y: yTop, a: Math.PI };
    s -= st;
    if (s < Math.PI * r) { const th = s / r; return { x: xl - r * Math.sin(th), y: yc - r * Math.cos(th), a: Math.PI - th }; }
    s -= Math.PI * r;
    return { x: xl + s, y: yBot, a: 0 };
  }
  const queueSpot = i => ({ x: L.mid - 30 - i * 17, y: L.yBot + 34 });
  const exitSpot = () => ({ x: L.mid + 34, y: L.yBot + 34 });

  // ---- drawing ----
  const ctx = canvas.getContext('2d');
  function resize() {
    const w = canvas.clientWidth || canvas.parentElement.clientWidth || 600;
    L = layout(w);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(L.h * dpr);
    canvas.style.height = L.h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function rr(x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }
  function stadium() {
    const { r, yc, xl, xr, yTop, yBot } = L;
    ctx.beginPath(); ctx.moveTo(xl, yBot); ctx.lineTo(xr, yBot);
    ctx.arc(xr, yc, r, Math.PI / 2, -Math.PI / 2, true);
    ctx.lineTo(xl, yTop);
    ctx.arc(xl, yc, r, -Math.PI / 2, Math.PI / 2, true);
    ctx.closePath();
  }
  function rider(x, y, hue, bags, rad) {
    ctx.fillStyle = V.cssVar('--g' + hue);
    ctx.beginPath(); ctx.arc(x, y, rad, 0, Math.PI * 2); ctx.fill();
    if (bags > 0) {
      const bw = Math.min(16, 3 + bags * 0.5), bh = bags > 20 ? 7 : 5;
      ctx.globalAlpha = .55; ctx.fillRect(x - bw / 2, y + rad + 3, bw, bh); ctx.globalAlpha = 1;
    }
  }

  function drawTrack(now) {
    const fg = V.cssVar('--lab-fg'), muted = V.cssVar('--lab-muted'), grid = V.cssVar('--lab-grid');
    const mono = V.cssVar('--font-mono'), body = V.cssVar('--font-body');
    // road
    stadium(); ctx.strokeStyle = grid; ctx.lineWidth = 16; ctx.stroke();
    stadium(); ctx.strokeStyle = muted; ctx.globalAlpha = .35; ctx.lineWidth = 1; ctx.setLineDash([6, 8]); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    // stops = layers
    ctx.font = '10px ' + mono; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let i = 0; i < STOPS; i++) {
      const p = pointAt((i + 1) / (STOPS + 1));
      const nx = -Math.sin(p.a), ny = Math.cos(p.a);
      ctx.fillStyle = muted; ctx.beginPath(); ctx.arc(p.x + nx * 12, p.y + ny * 12, 2.5, 0, Math.PI * 2); ctx.fill();
      // labels outside the road, except along the bottom where the queue lives
      // outside the road, except along the bottom where the queue lives: there, inside
      const off = ny > 0.5 ? -22 : 22;
      ctx.fillText(String(i + 1), p.x + nx * off, p.y + ny * off);
    }
    // the stop (depot)
    const d = pointAt(0);
    ctx.fillStyle = V.cssVar('--accent'); rr(d.x - 14, d.y + 10, 28, 6, 2); ctx.fill();
    ctx.fillStyle = fg; ctx.font = '600 10px ' + mono; ctx.fillText('STOP', d.x, d.y + 26);
    // queue
    const maxQ = Math.max(1, Math.floor((L.mid - 40) / 17));
    queue.slice(0, maxQ).forEach((r, i) => { const q = queueSpot(i); rider(q.x, q.y, r.hue, r.toLoad, 5); });
    ctx.font = '10px ' + mono; ctx.textAlign = 'left'; ctx.fillStyle = muted;
    const qLabel = queue.length > maxQ ? 'queued: ' + queue.length + ' (+' + (queue.length - maxQ) + ' off screen)' : 'queued: ' + queue.length;
    ctx.fillText(V.fitText(ctx, [qLabel, 'queued: ' + queue.length], L.mid - 20), 8, L.yBot + 56);
    ctx.textAlign = 'right';
    ctx.fillText(V.fitText(ctx, ['done: ' + done, '' + done], L.trackW - L.mid - 48), L.trackW - 8, L.yBot + 56);
    // centre caption
    ctx.textAlign = 'center';
    const cx = L.mid, maxW = L.st + L.r * 1.2;
    const total = buses.reduce((s, b) => s + onBoard(b), 0);
    ctx.font = '600 13px ' + body; ctx.fillStyle = fg;
    ctx.fillText(V.fitText(ctx, [buses.length === 1 ? 'Lap ' + buses[0].laps + ' · ' + total + ' of ' + SEATS + ' seats taken' : total + ' riders on ' + buses.length + ' buses', total + ' on board'], maxW), cx, L.yc - 22);
    ctx.font = '11px ' + mono; ctx.fillStyle = muted;
    ctx.fillText(V.fitText(ctx, ['every lap visits every stop (layer)', 'every lap: every layer', 'all layers'], maxW), cx, L.yc - 2);
    const stalled = buses.find(b => b.phase === 'dwell' && b.lastLoad >= 12);
    if (stalled) {
      ctx.fillStyle = V.cssVar('--idle'); ctx.font = '600 11px ' + mono;
      ctx.fillText(V.fitText(ctx, ['stopped to load ' + stalled.lastLoad + ' bags: no words meanwhile', 'loading ' + stalled.lastLoad + ' bags: no words', 'loading…'], maxW), cx, L.yc + 14);
    } else ctx.fillText(V.fitText(ctx, ['a lap takes the same time, full or empty', 'same lap time, full or empty', 'same lap time'], maxW), cx, L.yc + 14);
    // particles: boarding and leaving
    particles = particles.filter(p => now - p.t0 < 600);
    particles.forEach(p => {
      const k = Math.min(1, (now - p.t0) / 600), e = k * k * (3 - 2 * k);
      const from = p.kind === 'board' ? queueSpot(0) : pointAt(0), to = p.kind === 'board' ? pointAt(0) : exitSpot();
      ctx.globalAlpha = p.kind === 'exit' ? 1 - k * .7 : 1;
      rider(from.x + (to.x - from.x) * e, from.y + (to.y - from.y) * e - Math.sin(Math.PI * e) * 14, p.hue, 0, 5);
      ctx.globalAlpha = 1;
    });
    // buses
    buses.forEach(b => drawBus(b, now));
    // word pops at the stop
    floaters = floaters.filter(f => now - f.t0 < f.life);
    floaters.forEach(f => {
      const k = (now - f.t0) / f.life;
      ctx.globalAlpha = 1 - k; ctx.font = '600 12px ' + mono; ctx.textAlign = 'center';
      ctx.fillStyle = f.empty ? V.cssVar('--idle') : fg;
      // rise from just above the road, staying clear of the centre caption
      const y0 = Math.max(L.yc + 34, d.y - 22);
      ctx.fillText(f.text, d.x, y0 - k * Math.min(18, y0 - L.yc - 30));
      ctx.globalAlpha = 1;
    });
  }

  function drawBus(b, now) {
    const p = pointAt(b.pos);
    const scale = L.small ? 0.85 : 1;
    const bw = 58 * scale, bh = 20 * scale;
    ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.a);
    const flashing = now - b.flash < 700;
    ctx.fillStyle = V.cssVar('--lab-fg'); rr(-bw / 2, -bh / 2, bw, bh, 5 * scale); ctx.fill();
    if (flashing) { ctx.strokeStyle = V.cssVar('--crit'); ctx.lineWidth = 2; ctx.stroke(); }
    // windows: one per seat; dim while that rider's luggage is still loading
    const ww = (bw - 12 * scale) / SEATS;
    for (let i = 0; i < SEATS; i++) {
      const r = b.seats[i];
      ctx.fillStyle = r ? V.cssVar('--g' + r.hue) : '#2a3340';
      ctx.globalAlpha = r && !r.ready ? .4 : 1;
      ctx.fillRect(-bw / 2 + 4 * scale + i * ww, -bh / 2 + 4 * scale, ww - 1.5, bh - 8 * scale);
      ctx.globalAlpha = 1;
    }
    // nose
    ctx.fillStyle = '#2a3340'; ctx.fillRect(bw / 2 - 5 * scale, -bh / 2 + 4 * scale, 2.5 * scale, bh - 8 * scale);
    ctx.restore();
    if (buses.length > 1) {
      // bus number in a badge riding on the bus, so it never collides with other labels
      ctx.font = '600 10px ' + V.cssVar('--font-mono'); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillStyle = V.cssVar('--accent'); ctx.beginPath(); ctx.arc(p.x, p.y, 7, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.fillText(String(b.i + 1), p.x, p.y + 0.5);
    }
  }

  function drawPanel(b, P, now) {
    const fg = V.cssVar('--lab-fg'), muted = V.cssVar('--lab-muted'), grid = V.cssVar('--lab-grid');
    const mono = V.cssVar('--font-mono'), body = V.cssVar('--font-body');
    ctx.fillStyle = '#12171f'; rr(P.x, P.y, P.w, P.h, 8); ctx.fill(); ctx.strokeStyle = grid; ctx.lineWidth = 1; ctx.stroke();
    ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
    ctx.font = '600 12px ' + body; ctx.fillStyle = fg;
    const title = buses.length > 1 ? 'Inside bus ' + (b.i + 1) : 'Inside the bus';
    ctx.fillText(title, P.x + 12, P.y + 16);
    ctx.font = '10px ' + mono; ctx.fillStyle = muted; ctx.textAlign = 'right';
    const state = b.phase === 'dwell' ? (b.lastLoad ? 'at the stop, loading ' + b.lastLoad + ' bags' : 'at the stop') : 'driving';
    ctx.fillText(V.fitText(ctx, [state, b.phase === 'dwell' ? 'at stop' : 'driving'], P.w - 24 - ctx.measureText(title).width - 30), P.x + P.w - 12, P.y + 16);
    // legend for the seat bars
    {
      const ly = P.y + 34, sw = 16, col = fg;
      ctx.font = '10px ' + mono; ctx.textAlign = 'left';
      const full = ['words so far', 'whole answer (shown for illustration)'], short = ['so far', 'whole answer'];
      const need = l => sw * 2 + 6 * 2 + 14 + ctx.measureText(l[0]).width + ctx.measureText(l[1]).width;
      const lab = need(full) <= P.w - 24 ? full : short;
      let x = P.x + 12;
      ctx.fillStyle = col; rr(x, ly - 4, sw, 8, 2); ctx.fill(); x += sw + 6;
      ctx.fillStyle = muted; ctx.fillText(lab[0], x, ly); x += ctx.measureText(lab[0]).width + 14;
      ctx.fillStyle = col; ctx.globalAlpha = .14; rr(x, ly - 4, sw, 8, 2); ctx.fill();
      ctx.strokeStyle = col; ctx.globalAlpha = .7; ctx.lineWidth = 1; ctx.setLineDash([3, 2]); rr(x + .5, ly - 3.5, sw - 1, 7, 2); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      x += sw + 6; ctx.fillStyle = muted; ctx.fillText(lab[1], x, ly);
    }
    // seats
    const y0 = P.y + 50, rowH = 15, barX = P.x + 32, textW = Math.min(92, P.w * 0.32), barW = P.w - 32 - 12 - textW - 8;
    for (let i = 0; i < SEATS; i++) {
      const y = y0 + i * rowH, r = b.seats[i];
      ctx.fillStyle = grid; rr(barX, y + 3, barW, rowH - 6, 3); ctx.fill();
      ctx.textAlign = 'right'; ctx.font = '10px ' + mono;
      if (!r) { ctx.fillStyle = '#2a3340'; ctx.beginPath(); ctx.arc(P.x + 18, y + rowH / 2, 4.5, 0, Math.PI * 2); ctx.fill(); ctx.fillStyle = muted; ctx.globalAlpha = .6; ctx.fillText('empty seat', P.x + P.w - 12, y + rowH / 2); ctx.globalAlpha = 1; continue; }
      const col = V.cssVar('--g' + r.hue);
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(P.x + 18, y + rowH / 2, 4.5, 0, Math.PI * 2); ctx.fill();
      // every bar shares one scale (MAX_OUT words = full width). The dashed outline is the whole answer and its
      // right edge is the finish line; the solid bar is the words produced so far. A real engine does not
      // know the answer length in advance; this is drawn for illustration.
      const unit = barW / MAX_OUT, goalW = r.out * unit, finX = barX + goalW;
      ctx.fillStyle = col; ctx.globalAlpha = .14; rr(barX, y + 3, goalW, rowH - 6, 3); ctx.fill(); ctx.strokeStyle = col; ctx.globalAlpha = .7; ctx.lineWidth = 1; ctx.setLineDash([3, 2]); rr(barX + .5, y + 3.5, goalW - 1, rowH - 7, 3); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      ctx.fillStyle = muted; ctx.fillRect(Math.round(finX) - 1, y + 1, 2, rowH - 2);
      ctx.fillStyle = col;
      if (!r.ready) {
        // prompt still loading: a thin line along the bottom shows how many bags are on the rack
        const k = r.lug / Math.max(1, r.lug + r.toLoad);
        ctx.globalAlpha = .7; ctx.fillRect(barX, y + rowH - 5, Math.max(2, goalW * k), 2); ctx.globalAlpha = 1;
        ctx.fillStyle = muted; ctx.fillText(V.fitText(ctx, ['loading ' + r.lug + '/' + (r.lug + r.toLoad) + ' bags', 'loading'], textW), P.x + P.w - 12, y + rowH / 2);
      } else {
        if (r.gen > 0) { rr(barX, y + 3, Math.max(4, r.gen * unit), rowH - 6, 3); ctx.fill(); }
        ctx.fillStyle = fg; ctx.fillText(V.fitText(ctx, [r.gen + ' of ' + r.out + ' words', r.gen + '/' + r.out], textW), P.x + P.w - 12, y + rowH / 2);
      }
    }
    // luggage rack = KV cache
    const ry = y0 + SEATS * rowH + 14, u = used(b);
    ctx.font = '10px ' + mono; ctx.textAlign = 'left'; ctx.fillStyle = muted;
    const rackR = u + ' / ' + RACK + ' bags';
    ctx.textAlign = 'right'; ctx.fillStyle = u > RACK * 0.85 ? V.cssVar('--warn') : fg; ctx.fillText(rackR, P.x + P.w - 12, ry);
    ctx.textAlign = 'left'; ctx.fillStyle = muted;
    ctx.fillText(V.fitText(ctx, ['luggage rack = KV cache', 'luggage = KV cache', 'KV cache'], P.w - 36 - ctx.measureText(rackR).width), P.x + 12, ry);
    const bx = P.x + 12, bwid = P.w - 24, by = ry + 9, bh = 12;
    ctx.fillStyle = grid; rr(bx, by, bwid, bh, 3); ctx.fill();
    let x = bx;
    b.seats.filter(Boolean).sort((a, c) => a.boarded - c.boarded).forEach(r => {
      const segW = bwid * (r.lug / RACK);
      if (segW <= 0) return;
      ctx.fillStyle = V.cssVar('--g' + r.hue); ctx.globalAlpha = r.ready ? .9 : .45;
      ctx.fillRect(x, by, Math.max(0, segW - 2), bh); ctx.globalAlpha = 1;
      x += segW;
    });
    if (now - b.flash < 1200) {
      ctx.strokeStyle = V.cssVar('--crit'); ctx.lineWidth = 2; rr(bx, by, bwid, bh, 3); ctx.stroke();
      ctx.fillStyle = V.cssVar('--crit'); ctx.textAlign = 'left'; ctx.font = '600 10px ' + mono;
      ctx.fillText(V.fitText(ctx, ['rack full: newest rider bumped back to the stop', 'rack full: rider bumped', 'bumped'], P.w - 24), P.x + 12, by + bh + 13);
    }
  }

  function draw(now) {
    if (!L) return;
    ctx.clearRect(0, 0, L.w, L.h);
    drawTrack(now);
    buses.forEach((b, i) => { if (L.panels[i]) drawPanel(b, L.panels[i], now); });
  }

  // ---- readout ----
  let lastReadout = 0;
  function updateReadout(now) {
    if (now - lastReadout < 250) return;
    lastReadout = now;
    const span = Math.min(12, Math.max(1e-6, t));
    const wps = wordEvents.reduce((s, e) => s + e.n, 0) / span;
    const set = (id, v) => { const el = $(id); if (el) el.textContent = v; };
    set('bus-wps', t > 0.5 ? wps.toFixed(1) : '0');
    set('bus-onboard', buses.reduce((s, b) => s + onBoard(b), 0) + ' / ' + SEATS * buses.length);
    set('bus-last', String(lastWords));
    set('bus-wait', String(queue.length));
  }

  // ---- controls ----
  const rate = $('bus-rate');
  if (rate) rate.addEventListener('input', () => { knobs.rate = +rate.value; $('bus-rate-val').textContent = knobs.rate.toFixed(1); nextArrival = t + expo(LAP / knobs.rate); });
  document.querySelectorAll('[data-bus-n]').forEach(btn => btn.addEventListener('click', () => {
    document.querySelectorAll('[data-bus-n]').forEach(x => x.setAttribute('aria-pressed', String(x === btn)));
    setBuses(+btn.dataset.busN);
  }));
  const rush = $('bus-rush');
  if (rush) rush.addEventListener('click', () => { for (let i = 0; i < 6; i++) queue.push(randomRider()); });
  const speedBtns = Array.from(document.querySelectorAll('[data-bus-speed]'));
  speedBtns.forEach(b => b.addEventListener('click', () => {
    knobs.speed = +b.dataset.busSpeed;
    speedBtns.forEach(x => x.setAttribute('aria-pressed', String(x === b)));
  }));
  const pause = $('bus-pause');
  if (pause) pause.addEventListener('click', () => { knobs.paused = !knobs.paused; pause.textContent = knobs.paused ? 'Resume' : 'Pause'; });
  const rst = $('bus-reset');
  // Reset restores every control to its starting value, then restarts the simulation
  if (rst) rst.addEventListener('click', () => {
    knobs.rate = 0.4; knobs.speed = 0.5; knobs.n = 1; knobs.paused = false;
    if (rate) { rate.value = '0.4'; $('bus-rate-val').textContent = '0.4'; }
    document.querySelectorAll('[data-bus-n]').forEach(x => x.setAttribute('aria-pressed', String(x.dataset.busN === '1')));
    speedBtns.forEach(x => x.setAttribute('aria-pressed', String(x.dataset.busSpeed === '0.5')));
    if (pause) pause.textContent = 'Pause';
    reset();
  });

  if (window.ResizeObserver) new ResizeObserver(resize).observe(canvas.parentElement);
  else window.addEventListener('resize', resize);
  reset();
  V.animate(canvas, dt => {
    const now = performance.now();
    if (!knobs.paused) advance(dt * knobs.speed);
    draw(now);
    updateReadout(now);
  });
})();
