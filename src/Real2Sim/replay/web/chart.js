(function () {
  'use strict';
  function lowerBound(a, value) {
    let lo = 0, hi = a.length;
    while (lo < hi) { const mid = (lo + hi) >>> 1; if (a[mid] < value) lo = mid + 1; else hi = mid; }
    return lo;
  }
  function nearest(a, value) {
    const i = lowerBound(a, value);
    if (!i) return 0;
    if (i === a.length) return i - 1;
    return value - a[i - 1] < a[i] - value ? i - 1 : i;
  }
  function bounds(values) {
    const good = values.filter(Number.isFinite);
    if (!good.length) return [-1, 1];
    let low = Infinity, high = -Infinity;
    for (const value of good) { low = Math.min(low, value); high = Math.max(high, value); }
    const padding = Math.max((high - low) * .12, Math.max(Math.abs(low), Math.abs(high)) * .02, 1e-3);
    return [low - padding, high + padding];
  }
  function viewRange(t, cursor, seconds) {
    const first = t[0], last = t[t.length - 1];
    const width = seconds === 'all' ? last - first : Math.min(Number(seconds), last - first);
    const left = Math.max(first, Math.min(cursor - width / 2, last - width));
    return [left, left + width];
  }
  function tick(value) {
    if (Math.abs(value) >= 100) return value.toFixed(0);
    if (Math.abs(value) >= 10) return value.toFixed(1);
    return value.toFixed(2);
  }
  class Chart {
    constructor(canvas, output, onSeek) {
      this.canvas = canvas; this.output = output; this.onSeek = onSeek;
      canvas.addEventListener('pointerdown', event => {
        if (!this.range) return;
        this.dragRange = this.range.slice(); canvas.setPointerCapture(event.pointerId); this.seek(event);
      });
      canvas.addEventListener('pointermove', event => { if (this.dragRange) this.seek(event); });
      canvas.addEventListener('pointerup', () => { this.dragRange = null; });
      canvas.addEventListener('pointercancel', () => { this.dragRange = null; });
    }
    seek(event) {
      if (!this.range || !this.box) return;
      const rect = this.canvas.getBoundingClientRect();
      const proportion = Math.max(0, Math.min(1, (event.clientX - rect.left - this.box.left) / this.box.width));
      const range = this.dragRange || this.range;
      this.onSeek(range[0] + proportion * (range[1] - range[0]));
    }
    draw(t, real, sim, range, cursor, dual = false, labels = ['R', 'S'], markers = []) {
      if (!t.length) return;
      this.range = range;
      const canvas = this.canvas, rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(100, rect.width), height = Math.max(70, rect.height);
      if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
        canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
      }
      const ctx = canvas.getContext('2d');
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
      const left = 47, right = dual ? 43 : 13, top = 10, bottom = 24;
      const w = width - left - right, h = height - top - bottom;
      this.box = {left, width: w};
      const start = Math.max(0, lowerBound(t, range[0]) - 1), end = Math.min(t.length, lowerBound(t, range[1]) + 1);
      const yr = bounds(real.slice(start, end).concat(dual ? [] : sim.slice(start, end)));
      const ys = dual ? bounds(sim.slice(start, end)) : yr;
      const x = value => left + (value - range[0]) / Math.max(1e-9, range[1] - range[0]) * w;
      const y = (value, scale) => top + h - (value - scale[0]) / (scale[1] - scale[0]) * h;
      ctx.lineWidth = 1; ctx.font = '9px system-ui, sans-serif';
      for (let i = 0; i <= 3; i++) {
        const yy = top + i * h / 3;
        ctx.strokeStyle = '#edf1f4'; ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(left + w, yy); ctx.stroke();
        ctx.textAlign = 'right'; ctx.fillStyle = dual ? '#6e99c2' : '#96a5b0';
        ctx.fillText(tick(yr[1] - i * (yr[1] - yr[0]) / 3), left - 8, yy + 3);
        if (dual) {
          ctx.textAlign = 'left'; ctx.fillStyle = '#c5986c';
          ctx.fillText(tick(ys[1] - i * (ys[1] - ys[0]) / 3), left + w + 7, yy + 3);
        }
      }
      for (let i = 0; i <= 4; i++) {
        const xx = left + i * w / 4;
        ctx.strokeStyle = '#f0f3f6'; ctx.beginPath(); ctx.moveTo(xx, top); ctx.lineTo(xx, top + h); ctx.stroke();
        ctx.textAlign = 'center'; ctx.fillStyle = '#93a3af';
        ctx.fillText((range[0] + i * (range[1] - range[0]) / 4).toFixed(2), xx, height - 6);
      }
      ctx.save(); ctx.beginPath(); ctx.rect(left, top, w, h); ctx.clip();
      for (const marker of markers) {
        if (marker.time < range[0] || marker.time > range[1]) continue;
        ctx.strokeStyle = '#b76c55'; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(x(marker.time), top); ctx.lineTo(x(marker.time), top + h); ctx.stroke();
        ctx.setLineDash([]); ctx.fillStyle = '#b76c55'; ctx.textAlign = 'left';
        ctx.fillText(marker.label, Math.min(x(marker.time) + 3, left + w - 50), top + 10);
      }
      [[real, yr, '#267bd5'], [sim, ys, '#e48a37']].forEach(([values, scale, color]) => {
        ctx.beginPath(); ctx.lineWidth = 1.5; ctx.strokeStyle = color;
        let active = false;
        for (let i = start; i < end; i++) {
          if (!Number.isFinite(values[i])) { active = false; continue; }
          if (active) ctx.lineTo(x(t[i]), y(values[i], scale)); else ctx.moveTo(x(t[i]), y(values[i], scale));
          active = true;
        }
        ctx.stroke();
      });
      ctx.setLineDash([3, 4]); ctx.strokeStyle = '#9caeb8'; ctx.lineWidth = .8;
      ctx.beginPath(); ctx.moveTo(x(cursor), top); ctx.lineTo(x(cursor), top + h); ctx.stroke(); ctx.setLineDash([]);
      const point = nearest(t, cursor);
      [[real, yr, '#267bd5'], [sim, ys, '#e48a37']].forEach(([values, scale, color]) => {
        if (Number.isFinite(values[point])) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x(t[point]), y(values[point], scale), 2.6, 0, Math.PI * 2); ctx.fill(); }
      });
      ctx.restore();
      const value = v => Number.isFinite(v) ? v.toFixed(3) : '—';
      if (this.output) this.output.textContent = `${t[point].toFixed(2)}s  ·  ${labels[0]} ${value(real[point])}  /  ${labels[1]} ${value(sim[point])}`;
    }
  }
  globalThis.ReplayCharts = {Chart, lowerBound, nearest, bounds, viewRange};
})();
