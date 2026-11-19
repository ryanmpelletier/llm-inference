/* One layer at a time: inside one GPU, each layer's weights are copied from VRAM
   onto the chip before the math on them can run. For a decode step the copy takes
   far longer than the math, so the compute units mostly wait.
   Illustrative ratio, no specific model or GPU: the math for one token takes
   1/RIDGE of the copy time, so the two are equal at RIDGE tokens per batch. */
(function () {
  const V = window.VLLM;
  const canvas = document.getElementById('layer-canvas');
  if (!canvas) return;

  const LAYERS = 32;          // drawn layers (illustrative)
  const RIDGE = 150;          // tokens at which math time equals copy time (illustrative)
  const LAYER_ANIM = 1.6;     // seconds of animation per layer
  const CELLS_C = 12, CELLS_R = 5;

  const state = { batch: 1, layer: 0, t: 0, flow: 0, paused: false };

  // times in units of one layer's copy time
  function times() {
    const copy = 1, math = state.batch / RIDGE, layer = Math.max(copy, math);
    return { copy, math, layer, cBusy: math / layer, mBusy: copy / layer };
  }
  const pct = v => (v >= 0.995 ? '100' : v < 0.01 ? (v * 100).toFixed(1) : Math.round(v * 100)) + '%';

  function rr(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h); }

  function draw(ctx, w, h) {
    const fg = V.cssVar('--lab-fg'), muted = V.cssVar('--lab-muted'), grid = V.cssVar('--lab-grid');
    const mono = V.cssVar('--font-mono'), labBg = V.cssVar('--lab');
    const memC = V.cssVar('--tp'), mathC = V.cssVar('--pp');
    const T = times();
    const p = state.t / LAYER_ANIM;                 // progress through this layer, 0..1
    const narrow = w < 560;
    ctx.clearRect(0, 0, w, h);
    ctx.textBaseline = 'middle';

    // ---- geometry
    const top = 26, boxH = narrow ? 190 : 200;
    const vx = 12, vw = narrow ? Math.round(w * 0.34) : Math.min(240, w * 0.28);
    const cw = narrow ? Math.round(w * 0.36) : Math.min(270, w * 0.32), cx = w - 12 - cw;
    const busX0 = vx + vw, busX1 = cx, busY = top + boxH / 2;

    // ---- VRAM with every layer
    ctx.font = '600 11px ' + mono; ctx.fillStyle = fg; ctx.textAlign = 'left';
    ctx.fillText(narrow ? 'VRAM' : 'VRAM: holds every layer', vx, 12);
    ctx.strokeStyle = '#2a3340'; ctx.lineWidth = 1; rr(ctx, vx + .5, top + .5, vw - 1, boxH - 1, 8); ctx.stroke();
    const sx = vx + 10, sw = vw - 20, sy = top + 10, sh = boxH - 20, slab = sh / LAYERS;
    for (let l = 0; l < LAYERS; l++) {
      const on = l === state.layer;
      ctx.fillStyle = on ? memC : (l < state.layer ? '#26303c' : '#3b4656');
      ctx.fillRect(sx, sy + l * slab, on ? sw : sw * 0.8, Math.max(1, slab - 1.5));
    }
    const ly = sy + state.layer * slab + slab / 2;
    ctx.font = '11px ' + mono; ctx.textAlign = 'right';
    const lbl = 'layer ' + (state.layer + 1);
    const lw = ctx.measureText(lbl).width + 10;
    const lx = sx + sw * 0.8 - 4, lyy = Math.min(sy + sh - 8, Math.max(sy + 8, ly));
    ctx.fillStyle = labBg; ctx.fillRect(lx - lw, lyy - 8, lw, 16);
    ctx.fillStyle = fg; ctx.fillText(lbl, lx - 4, lyy);
    ctx.textAlign = 'left';

    // ---- memory bus: tiles stream at a steady rate; density shows how busy the bus is
    ctx.strokeStyle = grid; ctx.lineWidth = 14; ctx.lineCap = 'butt';
    ctx.beginPath(); ctx.moveTo(busX0, busY); ctx.lineTo(busX1, busY); ctx.stroke();
    const span = busX1 - busX0 - 10, slots = Math.max(4, Math.floor(span / 18));
    const show = Math.max(1, Math.round(slots * T.mBusy));
    ctx.fillStyle = memC;
    for (let i = 0; i < slots; i++) {
      // spread the shown tiles evenly among the slots
      if (Math.floor((i + 1) * show / slots) === Math.floor(i * show / slots)) continue;
      const f = ((i / slots) + state.flow) % 1;
      ctx.fillRect(busX0 + 5 + f * span - 4, busY - 4, 8, 8);
    }
    ctx.font = '11px ' + mono; ctx.textAlign = 'center';
    ctx.fillStyle = muted; ctx.fillText('memory bus', (busX0 + busX1) / 2, busY - 22);
    ctx.fillStyle = T.mBusy < 0.995 ? fg : muted;
    ctx.fillText('busy ' + pct(T.mBusy), (busX0 + busX1) / 2, busY + 22);

    // ---- GPU chip: on-chip memory + compute units
    ctx.font = '600 11px ' + mono; ctx.fillStyle = fg; ctx.textAlign = 'left';
    ctx.fillText(narrow ? 'GPU chip' : 'GPU chip: on-chip memory + compute', cx, 12);
    ctx.strokeStyle = '#2a3340'; ctx.lineWidth = 1; rr(ctx, cx + .5, top + .5, cw - 1, boxH - 1, 8); ctx.stroke();
    const om = { x: cx + 10, y: top + 10, w: cw - 20, h: 44 };
    ctx.fillStyle = '#151b24'; rr(ctx, om.x, om.y, om.w, om.h, 5); ctx.fill();
    ctx.font = '10px ' + mono; ctx.fillStyle = muted; ctx.fillText(narrow ? 'on-chip memory' : 'on-chip memory (small)', om.x + 6, om.y + 10);
    const tiles = 6, tw = (om.w - 12 - (tiles - 1) * 4) / tiles;
    for (let i = 0; i < tiles; i++) {
      ctx.fillStyle = i < 3 ? memC : grid;
      ctx.fillRect(om.x + 6 + i * (tw + 4), om.y + 22, tw, 14);
    }
    // compute units: a steady share of the cells lit = the share of time they are busy
    const cuTop = om.y + om.h + 18;
    ctx.font = '10px ' + mono; ctx.fillStyle = muted; ctx.fillText('compute units', om.x, cuTop - 8);
    const gap = 3, N = CELLS_C * CELLS_R;
    const cu = Math.min((cw - 20 - (CELLS_C - 1) * gap) / CELLS_C, (top + boxH - 30 - cuTop - (CELLS_R - 1) * gap) / CELLS_R);
    const lit = Math.max(1, Math.round(T.cBusy * N));
    for (let i = 0; i < N; i++) {
      const r = Math.floor(i / CELLS_C), c = i % CELLS_C;
      ctx.fillStyle = i < lit ? mathC : grid;
      ctx.fillRect(om.x + c * (cu + gap), cuTop + r * (cu + gap), cu, cu);
    }
    ctx.font = '11px ' + mono; ctx.textAlign = 'center';
    ctx.fillStyle = T.cBusy < 0.995 ? fg : muted;
    ctx.fillText('busy ' + pct(T.cBusy), cx + cw / 2, top + boxH - 14);
    ctx.textAlign = 'left';

    // ---- per-layer timing bars, in units of the layer's time
    const by = top + boxH + 26, bx = 12, bw = w - 24;
    ctx.font = '600 11px ' + mono; ctx.fillStyle = fg;
    ctx.fillText('time for one layer, ' + state.batch + (state.batch === 1 ? ' token' : ' tokens') + ' in the batch', bx, by);
    const bar = (y, label, t, color) => {
      ctx.font = '11px ' + mono; ctx.fillStyle = muted; ctx.textAlign = 'left';
      ctx.fillText(V.fitText(ctx, label, bw - 60), bx, y);
      ctx.fillStyle = fg; ctx.textAlign = 'right'; ctx.fillText(pct(t / T.layer), bx + bw, y); ctx.textAlign = 'left';
      const full = Math.max(2, bw * t / T.layer);
      ctx.fillStyle = grid; ctx.fillRect(bx, y + 10, bw, 12);
      ctx.globalAlpha = 0.35; ctx.fillStyle = color; ctx.fillRect(bx, y + 10, full, 12);
      ctx.globalAlpha = 1; ctx.fillRect(bx, y + 10, Math.max(2, full * p), 12);
    };
    bar(by + 20, narrow ? 'copy the weights (fixed)' : 'copy the layer\'s weights from VRAM (the same for any batch size)', T.copy, memC);
    bar(by + 56, narrow ? 'do the math (per token)' : 'do the math on them (grows with every token)', T.math, mathC);
    ctx.font = '11px ' + mono; ctx.fillStyle = fg;
    const line = T.math < T.copy ? 'memory sets the pace: the compute units wait for data'
      : T.math > T.copy ? 'compute sets the pace: the memory bus waits for the math' : 'balanced: copy and math take the same time';
    const short = T.math < T.copy ? 'compute waits for memory' : T.math > T.copy ? 'memory waits for compute' : 'balanced';
    ctx.fillText(V.fitText(ctx, [line, short], bw), bx, by + 100);
  }

  const ctx = V.setupCanvas(canvas, () => 360);
  function render() { draw(ctx, canvas._w, canvas._h); }
  document.addEventListener('themechange', render);
  if (window.ResizeObserver) new ResizeObserver(render).observe(canvas.parentElement);

  if (V.prefersReduced.matches) { state.t = LAYER_ANIM * 0.4; state.layer = 12; render(); }
  else V.animate(canvas, dt => {
    if (!state.paused) {
      state.t += dt; state.flow = (state.flow + dt * 0.1) % 1;
      if (state.t >= LAYER_ANIM) { state.t = 0; state.layer = (state.layer + 1) % LAYERS; }
    }
    render();
  });

  const slider = document.getElementById('layer-batch');
  if (slider) slider.addEventListener('input', () => {
    state.batch = Math.round(Math.pow(2, +slider.value));
    document.getElementById('layer-batch-val').textContent = state.batch;
    render();
  });
  const pause = document.getElementById('layer-pause');
  if (pause) pause.addEventListener('click', () => { state.paused = !state.paused; pause.textContent = state.paused ? 'Play' : 'Pause'; });
})();
