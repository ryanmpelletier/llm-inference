/* Split visualizer: shows how a model is laid out across GPUs for each
   parallelism strategy, and animates requests flowing through it. */
(function () {
  const V = window.VLLM;
  const canvas = document.getElementById('split-canvas');
  if (!canvas) return;

  const L = 8;        // layer bands drawn per model
  const EXPERTS = 8;  // expert cells per MoE layer
  const state = { strategy: 'tp', gpus: 4, load: 3, speed: 1, t: 0, tokens: [], pulses: [], busy: [], memo: {} };

  const DESCR = {
    tp: 'Every GPU holds a vertical slice of every layer. One batch runs on all GPUs at once; after each layer the partial results are summed with an all-reduce (the flashes).',
    pp: 'Each GPU holds a contiguous block of layers. A batch walks GPU to GPU. With fewer micro-batches in flight than stages, downstream GPUs sit idle (the hatched bubble).',
    dp: 'Every GPU holds a full copy of the model. A load balancer sends each request to one replica; replicas never talk to each other.',
    ep: 'Attention weights are replicated (data-parallel), but each MoE layer\'s experts are spread across GPUs. Tokens hop to whichever GPU owns their chosen expert (all-to-all).'
  };

  // colour follows the layer, not the GPU, so you can see which part of which layer each GPU holds
  function layerColor(l) { return V.cssVar('--g' + ((l % 8) + 1)); }

  // ---- layout ----------------------------------------------------------
  function layout(w, h) {
    const N = state.gpus;
    const top = 58, bottom = h - 54;
    const gap = w < 520 ? 8 : 14;
    const left = w < 520 ? 16 : 58, right = 16;   // room for the layer labels on wider screens
    const gw = (w - left - right - gap * (N - 1)) / N;
    const boxes = [];
    for (let i = 0; i < N; i++) {
      const x = left + i * (gw + gap);
      boxes.push({ x, y: top, w: gw, h: bottom - top, pad: Math.max(6, Math.min(12, gw * 0.08)) });
    }
    return { boxes, top, bottom, gw, gap };
  }

  // ---- simulation of moving tokens --------------------------------------
  // token: { gpu, layer (0..L float), id, color, kind }
  let spawnAcc = 0;
  function step(dt) {
    const N = state.gpus;
    const s = state.strategy;
    const rate = state.load;        // batches in flight target
    const speed = 1.6 * state.speed; // layers per second
    state.busy = new Array(N).fill(0);

    if (s === 'tp') {
      // a single batch moves through all layers on all GPUs together; up to `rate` batches queued
      if (state.tokens.length === 0) state.tokens.push({ layer: 0, id: Math.random() });
      const tkn = state.tokens[0];
      const before = Math.floor(tkn.layer);
      tkn.layer += speed * dt;
      const after = Math.floor(tkn.layer);
      if (after !== before && after < L) state.pulses.push({ layer: after, t: 0, kind: 'allreduce' });
      if (tkn.layer >= L) { state.tokens.shift(); state.pulses.push({ layer: L, t: 0, kind: 'out' }); }
      for (let i = 0; i < N; i++) state.busy[i] = 1;
    } else if (s === 'pp') {
      // microbatches enter at GPU 0, traverse L layers, each GPU owns L/N layers
      const inflight = Math.min(rate, N);
      spawnAcc += dt;
      const spacing = (L / N) / speed; // time for one stage
      if (state.tokens.length < inflight && spawnAcc >= spacing) { spawnAcc = 0; state.tokens.push({ layer: 0, id: Math.random(), hue: state.tokens.length }); }
      state.tokens.forEach(tkn => { tkn.layer += speed * dt; });
      state.tokens = state.tokens.filter(tkn => tkn.layer < L);
      state.tokens.forEach(tkn => { const g = Math.min(N - 1, Math.floor(tkn.layer / (L / N))); state.busy[g] = 1; });
    } else if (s === 'dp') {
      // independent requests; each goes to one replica; replicas busy only if they have a request
      const want = Math.min(rate, N * 3);
      spawnAcc += dt;
      if (state.tokens.length < want && spawnAcc > 0.35) {
        spawnAcc = 0;
        // round-robin assignment
        const counts = new Array(N).fill(0); state.tokens.forEach(t => counts[t.gpu]++);
        let g = 0; for (let i = 1; i < N; i++) if (counts[i] < counts[g]) g = i;
        state.tokens.push({ layer: -1.2, gpu: g, id: Math.random() });
      }
      state.tokens.forEach(tkn => { tkn.layer += speed * dt * (tkn.layer < 0 ? 2 : 1); });
      state.tokens = state.tokens.filter(tkn => tkn.layer < L + 0.3);
      state.tokens.forEach(tkn => { if (tkn.layer >= 0) state.busy[tkn.gpu] = 1; });
    } else if (s === 'ep') {
      // tokens traverse layers on a home GPU for attention, hop to an expert owner for the MoE half of each layer
      const want = Math.min(rate, N * 2);
      spawnAcc += dt;
      if (state.tokens.length < want && spawnAcc > 0.3) {
        spawnAcc = 0;
        const g = state.tokens.length % N;
        state.tokens.push({ layer: 0, gpu: g, home: g, expert: null, id: Math.random() });
      }
      state.tokens.forEach(tkn => {
        const before = tkn.layer;
        tkn.layer += speed * dt;
        const frac = tkn.layer - Math.floor(tkn.layer);
        const fracBefore = before - Math.floor(before);
        if (fracBefore < 0.5 && frac >= 0.5 && Math.floor(tkn.layer) === Math.floor(before)) {
          // dispatch: choose an expert, owned by GPU floor(expert / (EXPERTS/N))
          tkn.expert = Math.floor(Math.random() * EXPERTS);
          const owner = Math.floor(tkn.expert / (EXPERTS / N));
          if (owner !== tkn.gpu) state.pulses.push({ layer: Math.floor(tkn.layer) + 0.5, from: tkn.gpu, to: owner, t: 0, kind: 'a2a' });
          tkn.gpu = owner;
        }
        if (Math.floor(tkn.layer) !== Math.floor(before) && tkn.layer < L) {
          // combine: return home
          if (tkn.gpu !== tkn.home) state.pulses.push({ layer: Math.floor(tkn.layer), from: tkn.gpu, to: tkn.home, t: 0, kind: 'a2a' });
          tkn.gpu = tkn.home; tkn.expert = null;
        }
      });
      state.tokens = state.tokens.filter(tkn => tkn.layer < L);
      state.tokens.forEach(tkn => { state.busy[tkn.gpu] = 1; });
    }
    state.pulses.forEach(p => p.t += dt);
    state.pulses = state.pulses.filter(p => p.t < 0.6);
    // smoothed busy for meters
    if (!state.smooth || state.smooth.length !== N) state.smooth = new Array(N).fill(0);
    for (let i = 0; i < N; i++) state.smooth[i] += (state.busy[i] - state.smooth[i]) * Math.min(1, dt * 3);
  }

  // ---- drawing ---------------------------------------------------------
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  function hatch(ctx, x, y, w, h, color) {
    ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
    ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.globalAlpha = 0.5;
    for (let d = -h; d < w; d += 8) { ctx.beginPath(); ctx.moveTo(x + d, y + h); ctx.lineTo(x + d + h, y); ctx.stroke(); }
    ctx.restore();
  }

  function draw(ctx, w, h, dt) {
    const N = state.gpus, s = state.strategy;
    const lab = V.cssVar('--lab'), fg = V.cssVar('--lab-fg'), muted = V.cssVar('--lab-muted'), grid = V.cssVar('--lab-grid');
    const idle = V.cssVar('--idle');
    const mono = '11px ' + V.cssVar('--font-mono');
    ctx.clearRect(0, 0, w, h);
    const { boxes } = layout(w, h);

    // header: incoming requests / load balancer
    ctx.font = mono; ctx.fillStyle = muted; ctx.textBaseline = 'middle';
    if (s === 'dp') {
      const cx = w / 2;
      ctx.fillStyle = grid; roundRect(ctx, cx - 70, 14, 140, 26, 6); ctx.fill();
      ctx.fillStyle = fg; ctx.textAlign = 'center'; ctx.fillText('load balancer', cx, 27);
      ctx.strokeStyle = grid; ctx.lineWidth = 1;
      boxes.forEach(b => { ctx.beginPath(); ctx.moveTo(cx, 40); ctx.lineTo(b.x + b.w / 2, b.y); ctx.stroke(); });
    } else if (s === 'pp') {
      ctx.textAlign = 'left'; ctx.fillText(V.fitText(ctx, ['micro-batches enter at GPU 0 and flow left to right', 'micro-batches enter at GPU 0, flow right'], w - 32), 16, 27);
    } else if (s === 'tp') {
      ctx.textAlign = 'left'; ctx.fillText('one batch, all GPUs in lockstep', 16, 27);
    } else {
      ctx.textAlign = 'left'; ctx.fillText(V.fitText(ctx, ['attention replicated · experts sharded · tokens hop to their expert', 'attention replicated · experts sharded'], w - 32), 16, 27);
    }

    const bandH = (b) => (b.h - b.pad * 2 - 40) / L;

    boxes.forEach((b, i) => {
      // box
      ctx.fillStyle = '#12171f'; roundRect(ctx, b.x, b.y, b.w, b.h, 8); ctx.fill();
      ctx.strokeStyle = grid; ctx.lineWidth = 1; ctx.stroke();
      // title
      ctx.font = mono; ctx.textAlign = 'left'; ctx.fillStyle = fg; ctx.fillText('GPU ' + i, b.x + b.pad, b.y + 16);
      // layer bands
      const bh = bandH(b);
      const innerX = b.x + b.pad, innerW = b.w - b.pad * 2, y0 = b.y + 28;
      for (let l = 0; l < L; l++) {
        const y = y0 + l * bh, hh = bh - 3, col = layerColor(l);
        // ghost outline of the full layer
        ctx.strokeStyle = grid; ctx.lineWidth = 1; roundRect(ctx, innerX, y, innerW, hh, 3); ctx.stroke();
        if (s === 'tp') {
          const sw = innerW / N;
          ctx.fillStyle = col; ctx.globalAlpha = 0.85; roundRect(ctx, innerX + i * sw, y, sw - 1, hh, 2); ctx.fill(); ctx.globalAlpha = 1;
        } else if (s === 'pp') {
          const owner = Math.min(N - 1, Math.floor(l / (L / N)));
          if (owner === i) { ctx.fillStyle = col; ctx.globalAlpha = 0.85; roundRect(ctx, innerX, y, innerW, hh, 3); ctx.fill(); ctx.globalAlpha = 1; }
        } else if (s === 'dp') {
          ctx.fillStyle = col; ctx.globalAlpha = 0.85; roundRect(ctx, innerX, y, innerW, hh, 3); ctx.fill(); ctx.globalAlpha = 1;
        } else if (s === 'ep') {
          // top half attention (replicated), bottom half experts (sharded)
          const ah = hh * 0.42;
          ctx.fillStyle = col; ctx.globalAlpha = 0.85; roundRect(ctx, innerX, y, innerW, ah, 2); ctx.fill(); ctx.globalAlpha = 1;
          const ew = innerW / EXPERTS;
          for (let e = 0; e < EXPERTS; e++) {
            const owner = Math.floor(e / (EXPERTS / N));
            const ex = innerX + e * ew, ey = y + ah + 2, eh = hh - ah - 2;
            if (owner === i) { ctx.fillStyle = col; ctx.globalAlpha = 0.85; ctx.fillRect(ex + 0.5, ey, ew - 1.5, eh); ctx.globalAlpha = 1; }
            else { ctx.strokeStyle = grid; ctx.strokeRect(ex + 0.5, ey, ew - 1.5, eh); }
          }
        }
      }
      // layer labels (first box only)
      if (i === 0 && b.x >= 50) {
        ctx.fillStyle = muted; ctx.font = '9px ' + V.cssVar('--font-mono'); ctx.textAlign = 'right';
        ctx.fillText('layer 1', b.x - 6, y0 + bh / 2);
        ctx.fillText('layer ' + L, b.x - 6, y0 + (L - 0.5) * bh);
        ctx.font = mono;
      }
      // idle hatch
      const busy = state.smooth ? state.smooth[i] : 0;
      if (busy < 0.5) hatch(ctx, innerX, y0, innerW, bh * L - 3, idle);
      // utilization meter
      const my = b.y + b.h - b.pad - 8;
      ctx.fillStyle = grid; roundRect(ctx, innerX, my, innerW, 6, 3); ctx.fill();
      ctx.fillStyle = busy > 0.5 ? fg : idle; roundRect(ctx, innerX, my, Math.max(4, innerW * busy), 6, 3); ctx.fill();
      ctx.fillStyle = muted; ctx.textAlign = 'left'; ctx.font = '9px ' + V.cssVar('--font-mono');
      ctx.fillText(busy > 0.5 ? 'busy' : 'idle', innerX, my - 7);
      ctx.font = mono;
    });

    // tokens
    const tokenY = (b, layer) => b.y + 28 + (layer + 0.5) * bandH(b);
    const drawToken = (x, y, color, r = 5) => {
      ctx.beginPath(); ctx.arc(x, y, r + 3, 0, Math.PI * 2); ctx.fillStyle = color; ctx.globalAlpha = 0.25; ctx.fill(); ctx.globalAlpha = 1;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fillStyle = '#fff'; ctx.fill();
    };
    if (s === 'tp') {
      state.tokens.forEach(tkn => boxes.forEach((b, i) => {
        const sw = (b.w - b.pad * 2) / N;
        drawToken(b.x + b.pad + i * sw + sw / 2, tokenY(b, Math.min(tkn.layer, L - 0.5)), fg, 4);
      }));
    } else if (s === 'pp') {
      state.tokens.forEach(tkn => {
        const g = Math.min(N - 1, Math.floor(tkn.layer / (L / N)));
        const b = boxes[g];
        drawToken(b.x + b.w / 2, tokenY(b, tkn.layer), fg);
      });
      // arrows between stages
      for (let i = 0; i < N - 1; i++) {
        const a = boxes[i], c = boxes[i + 1];
        const ly = a.y + 28 + (L / N) * (i + 1) * bandH(a) - bandH(a) / 2;
        ctx.strokeStyle = muted; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(a.x + a.w, ly); ctx.lineTo(c.x, ly); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(c.x, ly); ctx.lineTo(c.x - 5, ly - 3); ctx.lineTo(c.x - 5, ly + 3); ctx.closePath(); ctx.fillStyle = muted; ctx.fill();
      }
    } else if (s === 'dp') {
      state.tokens.forEach(tkn => {
        const b = boxes[tkn.gpu];
        if (tkn.layer < 0) {
          // travelling from LB to box
          const f = (tkn.layer + 1.2) / 1.2; const sx = w / 2, sy = 40, ex = b.x + b.w / 2, ey = b.y;
          drawToken(sx + (ex - sx) * f, sy + (ey - sy) * f, fg, 4);
        } else drawToken(b.x + b.w / 2 + ((tkn.id * 7) % 1 - 0.5) * (b.w * 0.5), tokenY(b, Math.min(tkn.layer, L - 0.5)), fg);
      });
    } else if (s === 'ep') {
      state.tokens.forEach(tkn => {
        const b = boxes[tkn.gpu];
        const frac = tkn.layer - Math.floor(tkn.layer);
        let x = b.x + b.w / 2;
        if (tkn.expert !== null && frac >= 0.5) { const ew = (b.w - b.pad * 2) / EXPERTS; x = b.x + b.pad + (tkn.expert + 0.5) * ew; }
        drawToken(x, tokenY(b, tkn.layer), fg, 4);
      });
    }

    // pulses: all-reduce flashes / all-to-all hops
    state.pulses.forEach(p => {
      const a = 1 - p.t / 0.6;
      if (p.kind === 'allreduce') {
        const b0 = boxes[0], bN = boxes[N - 1];
        const y = b0.y + 28 + p.layer * bandH(b0) - 1.5;
        ctx.strokeStyle = V.cssVar('--accent'); ctx.globalAlpha = a; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(b0.x, y); ctx.lineTo(bN.x + bN.w, y); ctx.stroke();
        ctx.fillStyle = fg; ctx.font = mono; ctx.textAlign = 'right';
        if (p.layer < L) ctx.fillText('all-reduce', bN.x + bN.w - 4, y - 8);
        ctx.globalAlpha = 1;
      } else if (p.kind === 'a2a') {
        const b1 = boxes[p.from], b2 = boxes[p.to];
        const y = b1.y + 28 + p.layer * bandH(b1);
        ctx.strokeStyle = V.cssVar('--ep'); ctx.globalAlpha = a; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(b1.x + b1.w / 2, y); ctx.quadraticCurveTo((b1.x + b2.x + b2.w / 2 + b1.w / 2) / 2, y - 18, b2.x + b2.w / 2, y); ctx.stroke();
        ctx.globalAlpha = 1;
      }
    });

    // footer labels
    ctx.fillStyle = muted; ctx.font = mono; ctx.textAlign = 'left';
    const foot = {
      tp: ['memory per GPU: 1/' + N + ' of weights · 1/' + N + ' of KV cache', 'per GPU: 1/' + N + ' weights · 1/' + N + ' KV cache'],
      pp: ['memory per GPU: 1/' + N + ' of layers · KV cache only for its own layers', 'per GPU: 1/' + N + ' layers · KV for own layers'],
      dp: ['memory per GPU: full weights · its own KV cache', 'per GPU: full weights · own KV cache'],
      ep: ['memory per GPU: full attention weights · 1/' + N + ' of experts · its own KV cache', 'per GPU: all attention · 1/' + N + ' experts · own KV']
    }[s];
    ctx.fillText(V.fitText(ctx, foot, w - 32), 16, h - 14);
  }

  // ---- wiring ------------------------------------------------------------
  const ctx = V.setupCanvas(canvas, w => (w < 520 ? 380 : 420));
  V.animate(canvas, (dt) => {
    if (!V.prefersReduced.matches && !state.paused) step(dt);
    else if (state.paused) step(0);
    draw(ctx, canvas._w, canvas._h, dt);
  });

  const descr = document.getElementById('split-descr');
  function setStrategy(s) {
    state.strategy = s; state.tokens = []; state.pulses = []; spawnAcc = 0; state.smooth = null;
    document.querySelectorAll('[data-split]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.split === s)));
    if (descr) descr.textContent = DESCR[s];
    const loadLbl = document.getElementById('split-load-label');
    if (loadLbl) loadLbl.textContent = s === 'pp' ? 'micro-batches in flight' : s === 'tp' ? 'batches queued' : 'requests in flight';
  }
  document.querySelectorAll('[data-split]').forEach(b => b.addEventListener('click', () => setStrategy(b.dataset.split)));
  document.querySelectorAll('[data-split-gpus]').forEach(b => b.addEventListener('click', () => {
    state.gpus = +b.dataset.splitGpus; state.tokens = []; state.pulses = []; state.smooth = null;
    document.querySelectorAll('[data-split-gpus]').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    document.querySelectorAll('.split-n').forEach(el => { el.textContent = state.gpus; });
    document.querySelectorAll('.split-all').forEach(el => { el.textContent = state.gpus === 2 ? 'Both' : 'All ' + state.gpus; });
  }));
  const load = document.getElementById('split-load');
  if (load) load.addEventListener('input', () => { state.load = +load.value; document.getElementById('split-load-val').textContent = load.value; });
  const pause = document.getElementById('split-pause');
  if (pause) pause.addEventListener('click', () => { state.paused = !state.paused; pause.textContent = state.paused ? 'Play' : 'Pause'; pause.setAttribute('aria-pressed', String(state.paused)); });
  window.VLLM.setSplitStrategy = setStrategy;
  setStrategy('tp');
})();
