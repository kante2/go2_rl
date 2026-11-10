const $ = id => document.getElementById(id);
const KINDS = ['measured', 'predicted'];
const COLORS = { measured: '#65dfbd', predicted: '#ffbc79', over: '#ff6b8b', under: '#6ea8ff', grid: '#25354a', text: '#8fa4bd', zero: '#7b8ea6' };
const FILL = { over: 'rgba(255,107,139,.30)', under: 'rgba(110,168,255,.30)' };
// Measured windows are 20 ms and predictions arrive near 50 Hz; 50 ms bins pair them up.
const BIN = .05;
const LEFT = 56, RIGHT_PAD = 16, TOP = 22, BOTTOM_PAD = 30;
let points = [], frozen = null, cursor = 0, snapshot = null, connected = false;
let lastResponse = 0, clockOffset = 0, hoverTime = null, rows = [];
const format = (value, digits=1) => Number.isFinite(value) ? value.toFixed(digits) : '—';
// Below display precision a gap has no direction.
const sign = value => value >= .05 ? 'over' : value <= -.05 ? 'under' : '';
const signed = (value, digits=1) => !Number.isFinite(value) ? '—' : `${{ over: '+', under: '−' }[sign(value)] || ''}${Math.abs(value).toFixed(digits)}`;

function serverNow() { return performance.now()/1000 + clockOffset; }
function settings() {
  return { span: Number($('window').value), average: Number($('average').value), tolerance: Number($('tolerance').value) };
}
function currentView() { return frozen || { now: serverNow(), points }; }

// Bins both traces on one grid, then adds a trailing moving average per trace.
function analyze({ now, points }, { span, average }) {
  const start = now-span, origin = start-average, n = Math.ceil((now+.05-origin)/BIN);
  const sum = {}, count = {}, run = {};
  for (const kind of KINDS) { sum[kind] = new Float64Array(n); count[kind] = new Uint32Array(n); run[kind] = { sum: 0, count: 0 }; }
  for (const p of points) {
    const i = Math.floor((p.time-origin)/BIN);
    if (i >= 0 && i < n) { sum[p.kind][i] += p.value; count[p.kind][i]++; }
  }
  const width = Math.max(1, Math.round(average/BIN)), result = [];
  for (let i=0; i<n; i++) {
    const row = { time: origin+(i+.5)*BIN };
    for (const kind of KINDS) {
      // Average bin means, not raw samples, so 2- vs 3-sample bins do not add ripple.
      const r = run[kind], mean = i => count[kind][i] ? sum[kind][i]/count[kind][i] : NaN;
      row[kind] = mean(i);
      if (Number.isFinite(row[kind])) { r.sum += row[kind]; r.count++; }
      if (i >= width && Number.isFinite(mean(i-width))) { r.sum -= mean(i-width); r.count--; }
      row[kind+'Avg'] = r.count ? r.sum/r.count : NaN;
    }
    row.diff = row.predicted-row.measured;
    row.avgDiff = row.predictedAvg-row.measuredAvg;
    if (row.time >= start) result.push(row);
  }
  return result;
}

function summarize(rows, tolerance) {
  const paired = rows.filter(r => Number.isFinite(r.diff));
  if (!paired.length) return null;
  const mean = key => paired.reduce((total, r) => total+r[key], 0)/paired.length;
  return {
    measured: mean('measured'), predicted: mean('predicted'), bias: mean('diff'),
    mae: paired.reduce((total, r) => total+Math.abs(r.diff), 0)/paired.length,
    within: paired.filter(r => Math.abs(r.diff) <= tolerance).length/paired.length*100,
  };
}
const lastFinite = (rows, key) => { for (let i=rows.length-1; i>=0; i--) if (Number.isFinite(rows[i][key])) return rows[i]; };

function updateCards() {
  const fresh = connected && performance.now()-lastResponse < 1500;
  const status = snapshot?.status || {};
  const now = serverNow();
  const robotFresh = fresh && status.low_age_ms !== null && status.low_age_ms < 100 && now-(snapshot?.now || 0) < .5;
  const audioFresh = fresh && status.audio_alive && status.audio_age_ms !== null && status.audio_age_ms < 500;
  const latest = kind => {
    for (let i=points.length-1; i>=0; i--) if (points[i].kind === kind) return points[i];
    return undefined;
  };
  const measured = latest('measured'), predicted = latest('predicted');
  $('measured').textContent = audioFresh && measured && now-measured.time < 1 ? format(measured.value) : '—';
  $('predicted').textContent = robotFresh && predicted && now-predicted.time < .5 ? format(predicted.value) : '—';

  const { average, tolerance } = settings();
  const gapRow = audioFresh && robotFresh ? lastFinite(rows, 'avgDiff') : undefined;
  const gap = gapRow?.avgDiff, instant = lastFinite(rows, 'diff')?.diff;
  $('gap').textContent = signed(gap);
  $('gapCard').className = `card gap ${Number.isFinite(gap) ? sign(gap) : ''} ${Math.abs(gap) > tolerance ? 'outside' : ''}`;
  $('gapState').textContent = !Number.isFinite(gap) ? 'Needs both measured and predicted samples'
    : `${average} s average · latest 50 ms bin ${signed(instant)} dB · ${Math.abs(gap) <= tolerance ? 'within' : 'outside'} ± ${tolerance} dB`;

  $('horizon').textContent = robotFresh ? `${format(status.remaining_horizon_ms)} ms` : '—';
  $('audioState').textContent = !status.audio_enabled ? 'Disabled · Start with --audio-host' : audioFresh ? (measured?.clipping ? 'PCM stream active · Clipping detected' : 'PCM stream active · Estimated audio clock') : 'Audio unavailable or stale';
  $('robotState').textContent = robotFresh ? `LowState active · Source age ${format(status.low_age_ms)} ms` : 'LowState unavailable or stale';
  $('connection').textContent = !fresh ? 'Disconnected' : status.simulated ? 'Simulated preview' : robotFresh && audioFresh ? 'Live · Both inputs' : 'Waiting for inputs';
  $('connection').className = `badge ${fresh && robotFresh && audioFresh ? 'live' : 'stale'}`;
  $('demo').hidden = !status.simulated;
  $('model').textContent = status.model || '—';
  $('count').textContent = Number(status.predictions || 0).toLocaleString();
  $('message').textContent = !fresh ? 'No response from the prediction process. The page will reconnect automatically.' : status.audio_error || status.last_error || (!status.audio_enabled ? 'Measured noise requires --audio-host. Predictions continue using LowState only.' : 'Live comparison uses approximate audio timestamps. Match microphone gain and placement to the training recordings.');
}

function updateStats(stats, tolerance) {
  $('meanMeasured').textContent = stats ? `${format(stats.measured)} dBFS` : '—';
  $('meanPredicted').textContent = stats ? `${format(stats.predicted)} dBFS` : '—';
  $('bias').textContent = stats ? `${signed(stats.bias)} dB` : '—';
  $('bias').className = stats ? sign(stats.bias) : '';
  $('mae').textContent = stats ? `${format(stats.mae)} dB` : '—';
  $('within').textContent = stats ? `${format(stats.within, 0)} % (± ${tolerance} dB)` : '—';
}

// Canvas helpers shared by all three charts so their time axes line up.
function prepare(canvas) {
  const rect = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
  const width = Math.max(260, rect.width), height = rect.height;
  if (canvas.width !== Math.round(width*ratio) || canvas.height !== Math.round(height*ratio)) {
    canvas.width = Math.round(width*ratio); canvas.height = Math.round(height*ratio);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
  return { ctx, width, height, left: LEFT, right: width-RIGHT_PAD, top: TOP, bottom: height-BOTTOM_PAD };
}
function niceStep(range) {
  for (const step of [.5, 1, 2, 5, 10, 20]) if (range/step <= 6) return step;
  return 50;
}
function dbRange(values) {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return [-45, -15];
  let low = Math.floor((Math.min(...finite)-2)/5)*5, high = Math.ceil((Math.max(...finite)+2)/5)*5;
  if (high-low < 10) { const mid = (low+high)/2; low = Math.floor((mid-5)/5)*5; high = low+10; }
  return [low, high];
}
function axes(chart, now, span, low, high, unit) {
  const { ctx, left, right, top, bottom, height } = chart;
  chart.x = t => left+(t-now+span)/span*(right-left);
  chart.y = v => bottom-(v-low)/(high-low)*(bottom-top);
  chart.time = px => now-span+(px-left)/(right-left)*span;
  ctx.font = '11px system-ui'; ctx.fillStyle = COLORS.text; ctx.strokeStyle = COLORS.grid; ctx.lineWidth = 1;
  ctx.fillText(unit, 8, 12);
  const step = niceStep(high-low);
  for (let v=Math.ceil(low/step)*step; v<=high+1e-9; v+=step) {
    ctx.beginPath(); ctx.moveTo(left, chart.y(v)); ctx.lineTo(right, chart.y(v)); ctx.stroke();
    ctx.fillText(Number.isInteger(v) ? String(v) : v.toFixed(1), 10, chart.y(v)+4);
  }
  for (let i=0; i<=4; i++) {
    const t = now-span+i*span/4, px = chart.x(t);
    ctx.beginPath(); ctx.moveTo(px, top); ctx.lineTo(px, bottom); ctx.stroke();
    ctx.fillText(i === 4 ? 'Now' : `−${format(span-i*span/4)}s`, Math.min(px, right-25), height-9);
  }
}
function clip(chart) {
  chart.ctx.save(); chart.ctx.beginPath();
  chart.ctx.rect(chart.left, chart.top, chart.right-chart.left, chart.bottom-chart.top); chart.ctx.clip();
}
function line(chart, series, key, color, width, maxGap) {
  const { ctx, x, y } = chart;
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
  let previous, last;
  for (const p of series) {
    if (!Number.isFinite(p[key])) { previous = undefined; continue; }
    if (!previous || p.time-previous.time > maxGap) ctx.moveTo(x(p.time), y(p[key])); else ctx.lineTo(x(p.time), y(p[key]));
    previous = last = p;
  }
  ctx.stroke();
  if (last) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x(last.time), y(last[key]), 3, 0, Math.PI*2); ctx.fill(); }
}
// Vertical band per bin from `fromKey` to `toKey`, coloured by which is larger.
function gapBands(chart, series, fromKey, toKey) {
  const { ctx, x, y } = chart, w = Math.max(1, x(BIN)-x(0));
  for (const r of series) {
    const a = r[fromKey], b = r[toKey];
    if (!Number.isFinite(a) || !Number.isFinite(b) || a === b) continue;
    ctx.fillStyle = b > a ? FILL.over : FILL.under;
    const y1 = y(a), y2 = y(b);
    ctx.fillRect(x(r.time)-w/2, Math.min(y1, y2), w+.5, Math.abs(y2-y1));
  }
}
function placeholder(chart, text) {
  const { ctx, left, right, top, bottom } = chart;
  ctx.fillStyle = COLORS.text; ctx.textAlign = 'center';
  ctx.fillText(text, (left+right)/2, (top+bottom)/2); ctx.textAlign = 'left';
}
function readout(chart, entries) {
  if (hoverTime === null || !rows.length) return;
  const { ctx, x, top, bottom, right } = chart, px = x(hoverTime);
  if (px < chart.left || px > right) return;
  ctx.strokeStyle = '#c7d6e8'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(px, top); ctx.lineTo(px, bottom); ctx.stroke(); ctx.setLineDash([]);
  const i = Math.round((hoverTime-rows[0].time)/BIN), row = rows[Math.max(0, Math.min(rows.length-1, i))];
  ctx.font = '12px system-ui';
  const texts = entries.map(([label, key, fmt, color]) => [`${label} ${fmt(row[key])}`, color]);
  const boxWidth = Math.max(...texts.map(([t]) => ctx.measureText(t).width))+20, boxHeight = texts.length*17+10;
  const bx = px+boxWidth+12 > right ? px-boxWidth-8 : px+8;
  ctx.fillStyle = 'rgba(11,18,32,.88)'; ctx.fillRect(bx, top+4, boxWidth, boxHeight);
  texts.forEach(([t, color], k) => { ctx.fillStyle = color; ctx.fillText(t, bx+10, top+20+k*17); });
}

function drawMain(view, s) {
  const chart = prepare($('chart'));
  const visible = view.points.filter(p => p.time >= view.now-s.span && p.time <= view.now+.05);
  const [low, high] = dbRange(visible.map(p => p.value));
  axes(chart, view.now, s.span, low, high, 'dBFS');
  clip(chart);
  gapBands(chart, rows, 'measured', 'predicted');
  for (const kind of KINDS) {
    const series = visible.filter(p => p.kind === kind).sort((a, b) => a.time-b.time);
    line(chart, series, 'value', COLORS[kind], 1.6, .12);
  }
  chart.ctx.restore();
  if (!visible.length) placeholder(chart, 'Waiting for measured or predicted noise…');
  readout(chart, [['Measured', 'measured', v => `${format(v)} dBFS`, COLORS.measured],
                  ['Predicted', 'predicted', v => `${format(v)} dBFS`, COLORS.predicted],
                  ['P − M', 'diff', v => `${signed(v)} dB`, '#e7eff9']]);
  const counts = kind => visible.filter(p => p.kind === kind).length;
  $('viewState').textContent = `${frozen ? 'View paused · Data collection continues' : 'Live view'} · ${s.span} s window · ${counts('measured')} measured / ${counts('predicted')} predicted samples · ${rows.filter(r => Number.isFinite(r.diff)).length} paired bins`;
}

function drawAverages(view, s) {
  const chart = prepare($('avgChart'));
  const [low, high] = dbRange(rows.flatMap(r => [r.measuredAvg, r.predictedAvg]));
  axes(chart, view.now, s.span, low, high, 'dBFS');
  clip(chart);
  gapBands(chart, rows, 'measuredAvg', 'predictedAvg');
  line(chart, rows, 'measuredAvg', COLORS.measured, 2.2, BIN*1.5);
  line(chart, rows, 'predictedAvg', COLORS.predicted, 2.2, BIN*1.5);
  chart.ctx.restore();
  if (!rows.some(r => Number.isFinite(r.measuredAvg) || Number.isFinite(r.predictedAvg))) placeholder(chart, 'Waiting for samples…');
  readout(chart, [['Measured avg', 'measuredAvg', v => `${format(v)} dBFS`, COLORS.measured],
                  ['Predicted avg', 'predictedAvg', v => `${format(v)} dBFS`, COLORS.predicted]]);
}

function drawDifference(view, s) {
  const chart = prepare($('diffChart'));
  const peak = Math.max(s.tolerance+1, 3, ...rows.map(r => Math.abs(r.avgDiff)).filter(Number.isFinite));
  const limit = Math.ceil(peak/niceStep(peak*2))*niceStep(peak*2);
  axes(chart, view.now, s.span, -limit, limit, 'dB');
  const { ctx, x, y, left, right } = chart;
  clip(chart);
  gapBands(chart, rows.map(r => ({ time: r.time, zero: 0, avgDiff: r.avgDiff })), 'zero', 'avgDiff');
  line(chart, rows, 'avgDiff', '#e7eff9', 1.8, BIN*1.5);
  ctx.restore();
  ctx.strokeStyle = COLORS.zero; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(left, y(0)); ctx.lineTo(right, y(0)); ctx.stroke();
  ctx.setLineDash([6, 4]); ctx.strokeStyle = '#9daec1';
  for (const v of [s.tolerance, -s.tolerance]) { ctx.beginPath(); ctx.moveTo(left, y(v)); ctx.lineTo(right, y(v)); ctx.stroke(); }
  ctx.setLineDash([]);
  ctx.fillStyle = COLORS.over; ctx.fillText('Predicted higher ↑', right-110, y(limit)+14);
  ctx.fillStyle = COLORS.under; ctx.fillText('Predicted lower ↓', right-110, y(-limit)-6);
  if (!rows.some(r => Number.isFinite(r.avgDiff))) placeholder(chart, 'Needs both measured and predicted samples…');
  readout(chart, [['Avg difference', 'avgDiff', v => `${signed(v)} dB`, '#e7eff9']]);
}

function draw() {
  const view = currentView(), s = settings();
  rows = analyze(view, s);
  for (const el of document.querySelectorAll('.avgLabel')) el.textContent = s.average;
  drawMain(view, s); drawAverages(view, s); drawDifference(view, s);
  updateStats(summarize(rows, s.tolerance), s.tolerance);
}

async function poll() {
  const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 2000);
  try {
    const response = await fetch(`/api/state?since=${cursor}`, { cache: 'no-store', signal: controller.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (data.reset) points = [];
    snapshot = data; cursor = data.cursor; clockOffset = data.now-performance.now()/1000;
    points.push(...data.points.filter(p => Number.isFinite(p.time) && Number.isFinite(p.value) && KINDS.includes(p.kind)));
    points = points.filter(p => p.time >= data.now-60).slice(-8000);
    connected = true; lastResponse = performance.now();
  } catch { connected = false; }
  finally { clearTimeout(timeout); if (!frozen) draw(); updateCards(); setTimeout(poll, 200); }
}

$('pause').onclick = () => {
  frozen = frozen ? null : { now: serverNow(), points: points.slice() };
  $('pause').textContent = frozen ? 'Resume live' : 'Pause view';
  draw();
};
for (const id of ['window', 'average', 'tolerance']) $(id).onchange = () => { draw(); updateCards(); };
for (const id of ['chart', 'avgChart', 'diffChart']) {
  const canvas = $(id);
  canvas.addEventListener('mousemove', event => {
    const rect = canvas.getBoundingClientRect(), { span } = settings(), now = currentView().now;
    const px = event.clientX-rect.left, right = rect.width-RIGHT_PAD;
    hoverTime = px < LEFT || px > right ? null : now-span+(px-LEFT)/(right-LEFT)*span;
    if (frozen) draw();
  });
  canvas.addEventListener('mouseleave', () => { hoverTime = null; if (frozen) draw(); });
}
$('export').onclick = () => {
  const header = 'time_monotonic_s,measured_dbfs,predicted_dbfs,diff_db,measured_avg_dbfs,predicted_avg_dbfs,avg_diff_db';
  const cell = v => Number.isFinite(v) ? v.toFixed(3) : '';
  const csv = header+'\n'+rows.map(r => [r.time.toFixed(3), cell(r.measured), cell(r.predicted), cell(r.diff),
    cell(r.measuredAvg), cell(r.predictedAvg), cell(r.avgDiff)].join(',')).join('\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' })), anchor = document.createElement('a');
  anchor.href = url; anchor.download = `robot_noise_comparison_avg${settings().average}s.csv`; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
window.addEventListener('resize', draw);
setInterval(updateCards, 500);
draw(); poll();
