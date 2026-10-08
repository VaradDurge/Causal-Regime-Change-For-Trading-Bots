(() => {
  const w = window;
  if (w.__crtaNav) {
    w.__crtaNav.scan();
    return;
  }
  const api = { hover: null };
  w.__crtaNav = api;
  const busy = new WeakSet();
  const cache = new WeakMap();
  const axisKey = (ref) => ref.replace(/^x/, "xaxis").replace(/^y/, "yaxis");
  const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  // Embedded previews can stall requestAnimationFrame; a timer keeps the glide moving.
  const tick = (fn) => {
    let done = false;
    const run = () => {
      if (done) return;
      done = true;
      fn(performance.now());
    };
    requestAnimationFrame(run);
    setTimeout(run, 34);
  };

  function linX(gd, tr) {
    const hit = cache.get(tr);
    if (hit) return hit;
    const ax = gd._fullLayout[axisKey(tr.xaxis || "x")];
    const src = tr.x || (tr._input && tr._input.x) || [];
    const out = new Float64Array(src.length);
    for (let k = 0; k < src.length; k++) out[k] = ax.d2l(src[k]);
    cache.set(tr, out);
    return out;
  }

  function yNames(gd) {
    return Object.keys(gd._fullLayout).filter(
      (k) => /^yaxis\d*$/.test(k) && gd._fullLayout[k].type !== "category",
    );
  }

  function extent(gd) {
    let lo = Infinity;
    let hi = -Infinity;
    gd._fullData.forEach((tr) => {
      const xs = linX(gd, tr);
      for (let k = 0; k < xs.length; k++) {
        if (xs[k] < lo) lo = xs[k];
        if (xs[k] > hi) hi = xs[k];
      }
    });
    return [lo, hi];
  }

  function yRange(gd, name, lo, hi) {
    let min = Infinity;
    let max = -Infinity;
    gd._fullData.forEach((tr) => {
      if (tr.visible !== true || tr.type === "heatmap") return;
      if (axisKey(tr.yaxis || "y") !== name) return;
      const candle = tr.type === "candlestick" || tr.type === "ohlc";
      const tops = candle ? tr.high : tr.y;
      const bots = candle ? tr.low : tr.y;
      if (!tops) return;
      const xs = linX(gd, tr);
      for (let k = 0; k < xs.length; k++) {
        if (xs[k] < lo || xs[k] > hi) continue;
        const a = +bots[k];
        const b = +tops[k];
        if (Number.isFinite(a) && a < min) min = a;
        if (Number.isFinite(b) && b > max) max = b;
      }
    });
    if (!Number.isFinite(min)) return null;
    const pad = (max - min) * 0.06 || Math.abs(max) * 0.02 || 0.05;
    return [min - pad, max + pad];
  }

  function current(gd) {
    const ax = gd._fullLayout.xaxis;
    return ax.range.map((v) => ax.r2l(v));
  }

  function clamp(gd, lo, hi) {
    const [mn, mx] = extent(gd);
    const full = mx - mn;
    let span = Math.max(Math.min(hi - lo, full * 1.04), full / 400);
    const mid = (lo + hi) / 2;
    lo = mid - span / 2;
    hi = mid + span / 2;
    const slack = span * 0.08;
    if (lo < mn - slack) {
      hi += mn - slack - lo;
      lo = mn - slack;
    }
    if (hi > mx + slack) {
      lo -= hi - (mx + slack);
      hi = mx + slack;
    }
    return [lo, hi];
  }

  function apply(gd, update) {
    busy.add(gd);
    return w.Plotly.relayout(gd, update).finally(() => busy.delete(gd));
  }

  function glide(gd, lo, hi, ms = 280) {
    const xa = gd._fullLayout.xaxis;
    const [a0, b0] = current(gd);
    const ys = yNames(gd)
      .map((n) => [n, gd._fullLayout[n].range.slice(), yRange(gd, n, lo, hi)])
      .filter((t) => t[2]);
    const t0 = performance.now();
    gd.__glide = t0;
    const frame = (now) => {
      if (gd.__glide !== t0) return;
      const t = Math.min(1, (now - t0) / ms);
      const e = ease(t);
      const update = { "xaxis.range": [xa.l2r(a0 + (lo - a0) * e), xa.l2r(b0 + (hi - b0) * e)] };
      ys.forEach(([n, from, to]) => {
        update[`${n}.range`] = [from[0] + (to[0] - from[0]) * e, from[1] + (to[1] - from[1]) * e];
      });
      apply(gd, update).then(() => {
        if (t < 1) tick(frame);
      });
    };
    tick(frame);
  }

  function zoom(gd, factor) {
    const [a, b] = current(gd);
    const mid = (a + b) / 2;
    const half = ((b - a) * factor) / 2;
    glide(gd, ...clamp(gd, mid - half, mid + half));
  }

  function shift(gd, fraction) {
    const [a, b] = current(gd);
    const d = (b - a) * fraction;
    glide(gd, ...clamp(gd, a + d, b + d));
  }

  function fit(gd, ms) {
    const [a, b] = current(gd);
    glide(gd, ...clamp(gd, a, b), ms);
  }

  const ACTIONS = {
    left: (gd) => shift(gd, -0.25),
    out: (gd) => zoom(gd, 1 / 0.6),
    in: (gd) => zoom(gd, 0.6),
    right: (gd) => shift(gd, 0.25),
  };

  function hook(gd) {
    if (!gd._fullLayout || !gd._fullLayout.xaxis) return;
    const host = gd.closest('[data-testid="stPlotlyChart"]') || gd.parentElement;
    if (!host.querySelector(".crta-nav")) {
      host.style.position = "relative";
      const bar = document.createElement("div");
      bar.className = "crta-nav";
      // No tag-like strings in this file: st.html's sanitizer drops a script that contains them.
      [
        ["left", "Shift left", "‹"],
        ["out", "Zoom out", "−"],
        ["in", "Zoom in", "+"],
        ["right", "Shift right", "›"],
      ].forEach(([action, title, glyph]) => {
        const btn = document.createElement("button");
        btn.dataset.a = action;
        btn.title = title;
        btn.textContent = glyph;
        bar.appendChild(btn);
      });
      bar.addEventListener("click", (ev) => {
        const btn = ev.target.closest("button");
        const plot = host.querySelector(".js-plotly-plot");
        if (btn && plot) ACTIONS[btn.dataset.a](plot);
      });
      host.appendChild(bar);
    }
    if (gd.__crta) return;
    gd.__crta = true;
    gd.on("plotly_relayout", (ev) => {
      if (busy.has(gd)) return;
      if (!Object.keys(ev || {}).some((k) => k.startsWith("xaxis"))) return;
      fit(gd, 170);
    });
    gd.on("plotly_react", () => tick(() => fit(gd, 1)));
    gd.addEventListener("mouseenter", () => (api.hover = gd));
    gd.addEventListener("mouseleave", () => {
      if (api.hover === gd) api.hover = null;
    });
    fit(gd, 1);
  }

  api.scan = () => document.querySelectorAll(".js-plotly-plot").forEach(hook);

  document.addEventListener("keydown", (ev) => {
    const gd = api.hover;
    if (!gd || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (ev.target.closest && ev.target.closest("input, textarea, [role=slider], [contenteditable]")) return;
    const keys = { ArrowLeft: "left", ArrowRight: "right", "+": "in", "=": "in", "-": "out" };
    const name = keys[ev.key];
    if (!name) return;
    ev.preventDefault();
    ACTIONS[name](gd);
  });

  let pending = 0;
  new MutationObserver(() => {
    clearTimeout(pending);
    pending = setTimeout(api.scan, 60);
  }).observe(document.body, { childList: true, subtree: true });
  api.scan();
})();
