(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const state = {mode: 'before', joint: 0, cursor: 0, window: 'all', forceZ: false, playing: false, speed: 1, rmseQuantity: 'q'};
  const names = {hip: 'Hip', thigh: 'Thigh', calf: 'Calf'};
  const units = {q: 'rad', dq: 'rad/s', ddq: 'rad/s²', tau: 'N·m'};
  let manifest, trial, cache = new Map(), token = 0, charts = {}, footCharts = [], animationTime = 0;
  let channels = {}, activeData, rmseChart;
  const num = (value, digits = 4) => value == null || !Number.isFinite(value) ? '—' : Math.abs(value) > 0 && Math.abs(value) < 1e-4 ? value.toExponential(2) : value.toFixed(digits);
  const node = (tag, text, cls) => { const item = document.createElement(tag); if (text != null) item.textContent = text; if (cls) item.className = cls; return item; };
  function clear(element) { element.replaceChildren(); }
  function stop() { state.playing = false; $('play').textContent = '▶'; $('play').setAttribute('aria-label', '재생'); }
  function data() { return trial[state.mode]; }
  async function json(path) { const response = await fetch(path, {cache: 'no-cache'}); if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`); return response.json(); }
  function fail(error) { stop(); $('error').hidden = false; $('error').textContent = `불러오지 못했습니다: ${error.message}. run_replay.sh로 실행한 HTTP 주소에서 열어 주세요.`; }
  function addStep(title, description, index, highlight) {
    const step = node('div', null, `step${highlight ? ' highlight' : ''}`);
    step.append(node('i', String(index)), node('b', title), node('small', description)); $('diagram').append(step);
  }
  function notes(text, emphasis = false) { $('notes').append(node('p', text, emphasis ? 'emphasis' : '')); }
  function bindChannels() {
    activeData = data(); const d = activeData;
    Object.assign(units, d.units);
    channels = {};
    ['q', 'dq', 'ddq', 'tau'].forEach(key => { channels[key] = [d.real[key].map(row => row[state.joint]), d.sim[key].map(row => row[state.joint])]; });
    const error = d.errors.quantities[state.rmseQuantity];
    channels.rmse = [error.joint.map(row => row[state.joint]), error.combined];
    channels.force = manifest.feet.map((_, foot) => {
      let a = d.force.real_counts.map(row => row[foot]), b = d.force.sim_N.map(row => row[foot]);
      if (state.forceZ) {
        a = a.map(value => d.force.real_std[foot] > 1e-9 ? (value - d.force.real_mean[foot]) / d.force.real_std[foot] : NaN);
        b = b.map(value => d.force.sim_std[foot] > 1e-9 ? (value - d.force.sim_mean[foot]) / d.force.sim_std[foot] : NaN);
      }
      return [a, b];
    });
  }
  function sidebar() {
    const d = activeData, sync = trial.synchronization;
    clear($('metric-cards')); clear($('diagram')); clear($('notes')); clear($('contract'));
    $('scope').textContent = '전후 동일한 정상 보행 샘플 · sim − real 오차 · 화면 확대와 별개';
    ['q', 'dq', 'ddq', 'tau'].forEach(quantity => {
      const metric = d.metrics[quantity][state.joint];
      const card = node('div', null, 'metric');
      const value = node('strong', `${num(metric.rmse)} `); value.append(node('span', units[quantity], 'unit'));
      card.append(node('span', `${quantity} RMSE${state.mode === 'after' && ['q','dq'].includes(quantity) ? ' · 정책 입력' : ''}`), value,
        node('small', `MAE ${num(metric.mae)} · bias ${num(metric.bias)}`), node('small', `n=${metric.n}`));
      if (state.mode === 'after') {
        const reference = node('small', `보정 전 RMSE ${num(trial.before.metrics[quantity][state.joint].rmse)} ${trial.before.units[quantity]} · 01 기준 물리 오차`);
        reference.style.gridColumn = '1 / -1'; card.append(reference);
      }
      $('metric-cards').append(card);
    });
    addStep('기존 real · sim 로그 유지', '동일 baseline 정책·명령의 기록 · 재시뮬레이션 없음', 1, false);
    addStep('관절 · 발 순서와 단위 확인', 'SDK FR·FL·RR·RL / hip·thigh·calf · q는 rad 그대로', 2, false);
    addStep('동일 전처리 · 기록 시간축', `tick 중복 ${trial.timing.duplicate_ticks_removed}개 제거 · 양쪽 15 Hz 필터 · 공통 200 Hz`, 3, false);
    addStep(`공통 위상 정렬 ${num(sync.applied_lag_s * 1000, 1)} ms`, '01·02 모두 동일한 sim(t + Δt)와 평가 샘플 사용', 4, false);
    if (state.mode === 'after') {
      addStep('배포 관측 좌표 변환', 'policy.json 관절 순서·기본각 → 실제 ONNX 고정 정규화 · 숫자 역변환 없음', 5, true);
      addStep('목표각 scale 검증', '기록 목표각 ↔ action 좌표 왕복 · 관측 q에 action_scale을 곱하지 않음', 6, true);
    }
    notes('전후 모두 같은 실제 시각·정상 보행 샘플로 오차를 계산합니다. q/dq는 15 Hz 필터 값이며 ddq는 필터 dq의 미분입니다.');
    notes(`위상 추정은 시작 구간, 검증은 끝 구간입니다. 끝 잔차 p90 ${num(sync.held_out_abs_residual_p90_s == null ? null : sync.held_out_abs_residual_p90_s * 1000, 1)} ms. 독립 보행의 위상 차이이며 하드웨어 지연을 뜻하지 않습니다.`);
    if (!sync.passed) notes('위상 검증 기준 미통과: 두 탭 모두 잠정 정렬을 사용합니다.', true);
    notes('발 힘은 counts와 N을 유지합니다. z-score는 파형 표시·오차 계산용이며 센서 교정이 아닙니다. 보정 전 기준 평균·표준편차를 전후 동일하게 사용합니다.');
    const dl = node('dl');
    const deployment = trial.mapping.deployment;
    const policyJoint = deployment.policy_to_sdk.indexOf(state.joint);
    [['데이터', '기존 real / sim NPZ'], ['시간 정렬', '01·02 공통'], ['배포 관절 순서', 'FL·FR·RL·RR'],
      ['선택 관절 기본각', `${num(deployment.default_joint_pos[policyJoint], 3)} rad`],
      ['목표각 action scale', num(deployment.action_scale[policyJoint], 3)],
      ['02 q 배율', num(1 / deployment.normalization_denominator[11 + policyJoint], 4)],
      ['02 q 오프셋', num((-deployment.default_joint_pos[policyJoint] - deployment.normalization_mean[11 + policyJoint]) / deployment.normalization_denominator[11 + policyJoint], 4)],
      ['02 dq 배율', num(1 / deployment.normalization_denominator[23 + policyJoint], 4)], ['관절 관측 parity', `${deployment.joint_parity_frames} frames`],
      ['q 왕복 최대오차', `${num(Math.max(deployment.real_q_roundtrip_max_error_rad, deployment.sim_q_roundtrip_max_error_rad))} rad`],
      ['tau 매핑', '배포 규약 없음 · N·m 유지'], ['발 힘 매핑', '배포 규약 없음 · 유지'], ['재시뮬레이션', '없음']].forEach(([key, value]) => dl.append(node('dt', key), node('dd', value)));
    notes('02 q/dq는 배포 정책이 사용하는 정규화 관측(무차원)입니다. 01의 물리 단위 RMSE와 숫자를 직접 비교해 개선률로 해석하지 않습니다. ddq·tau는 정책 입력에 없으므로 물리값과 단위를 유지합니다. tau는 real 추정 모터 토크와 sim 액추에이터 토크를 비교하며, 배포 토크 스케일은 없습니다.');
    $('contract').append(dl);
    $('provenance').textContent = `real ${manifest.trials.find(t => t.id === trial.trial).realSha256.slice(0, 12)} · ${trial.trial}`;
  }
  function refresh(resetCursor = false) {
    if (!trial) return;
    if (resetCursor) state.cursor = data().suggestedTime;
    const d = data(); state.cursor = Math.max(d.t[0], Math.min(d.t[d.t.length - 1], state.cursor));
    const name = manifest.joints[state.joint].replace('_joint', '').split('_');
    $('joint-title').textContent = `${name[0]} · ${names[name[1]]}`;
    $('page').textContent = String(state.joint + 1).padStart(2, '0');
    Array.from($('joints').children).forEach((button, index) => { button.classList.toggle('selected', index === state.joint); button.setAttribute('aria-current', index === state.joint ? 'page' : 'false'); });
    ['before', 'after'].forEach(mode => { $(mode).classList.toggle('selected', mode === state.mode); $(mode).setAttribute('aria-pressed', String(mode === state.mode)); });
    $('sim-label').textContent = '기존 baseline 시뮬 로그';
    $('mode-note').textContent = state.mode === 'before'
      ? '01 보정 전: 기존 real·sim 로그의 공통 전처리와 시간·보행 위상 정렬까지 완료한 기준 데이터입니다.'
      : '02 보정 후: 같은 시간 정렬 데이터의 q/dq를 실제 배포 정책 입력으로 매핑해 그대로 표시합니다(무차원). ddq·tau·발 힘은 배포 매핑 규약이 없어 물리값을 유지합니다.';
    const review = !trial.synchronization.passed;
    $('status').classList.toggle('review', review);
    $('status').textContent = review ? '공통 시간 정렬 · 검토 필요' : state.mode === 'before' ? '시간 정렬 기준 데이터' : '배포 정책 입력 매핑 적용';
    ['force-raw', 'force-z'].forEach((id, i) => { const selected = Boolean(i) === state.forceZ; $(id).classList.toggle('selected', selected); $(id).setAttribute('aria-pressed', String(selected)); });
    bindChannels(); sidebar();
    clear($('force-rmse-cards'));
    $('force-rmse-scope').textContent = '배포 파일에 발 힘 교정 규약이 없어 01·02는 동일합니다. 같은 시간 정렬과 표준화 기준의 파형 RMSE이며 N 단위 오차가 아닙니다.';
    manifest.feet.forEach((foot, i) => {
      const card = node('div', null, 'force-rmse-card');
      card.append(node('span', `${foot} RMSE`), node('strong', num(d.force.zscoreRmse[i], 3)),
        node('small', `전 ${num(trial.before.force.zscoreRmse[i], 3)} → 후 ${num(trial.after.force.zscoreRmse[i], 3)}`), node('small', `상관계수 r ${num(d.force.correlation[i], 3)}`));
      $('force-rmse-cards').append(card);
      $(`force-corr-${i}`).textContent = `r ${num(d.force.correlation[i], 3)} · z-RMSE ${num(d.force.zscoreRmse[i], 3)}`;
      $(`force-unit-${i}`).textContent = state.forceZ ? '각 신호의 z-score · 보정된 힘이 아님' : '실제 counts';
      $(`force-unit-right-${i}`).textContent = state.forceZ ? '' : '시뮬레이션 N';
    });
    draw();
  }
  function draw() {
    if (!activeData) return;
    const d = activeData, range = ReplayCharts.viewRange(d.t, state.cursor, state.window);
    const markers = [];
    ['q', 'dq', 'ddq', 'tau'].forEach(key => {
      $(`joint-unit-${key}`).textContent = units[key] + (state.mode === 'after' ? (['ddq','tau'].includes(key) ? ' · 물리값 유지' : ' · 정규화 정책 입력') : '');
    });
    ['q', 'dq', 'ddq', 'tau'].forEach(key => charts[key].draw(d.t, ...channels[key], range, state.cursor, false, ['R', 'S'], markers));
    $('rmse-unit').textContent = `${state.rmseQuantity} RMSE [${units[state.rmseQuantity]}]`;
    rmseChart.draw(d.t, ...channels.rmse, range, state.cursor, false, ['선택 관절', '12관절'], markers);
    footCharts.forEach((chart, foot) => chart.draw(d.t, ...channels.force[foot], range, state.cursor, !state.forceZ, ['R', 'S'], markers));
    $('clock').textContent = `${state.cursor.toFixed(2)} s`;
    $('seek').value = (state.cursor - d.t[0]) / Math.max(1e-9, d.t[d.t.length - 1] - d.t[0]) * 1000;
  }
  function seek(value) { stop(); state.cursor = value; draw(); }
  function selectMode(mode) { if (!trial) return; stop(); state.mode = mode; refresh(); }
  function joint(delta) { state.joint = (state.joint + delta + 12) % 12; refresh(); }
  async function selectTrial(id) {
    const request = ++token; stop(); $('status').textContent = '데이터를 불러오는 중';
    try {
      const entry = manifest.trials.find(t => t.id === id);
      let value = cache.get(id);
      if (!value) { value = await json(`data/${entry.file}`); cache.set(id, value); }
      if (request !== token) return;
      trial = value; $('velocity').textContent = value.velocity.toFixed(1);
      refresh(true); $('error').hidden = true;
    } catch (error) { fail(error); }
  }
  function createCharts() {
    ['q', 'dq', 'ddq', 'tau'].forEach(key => {
      const row = node('section', null, 'chart-row'), heading = node('div', null, 'chart-head');
      const output = node('output'), unit = node('small', units[key]); unit.id = `joint-unit-${key}`; heading.append(node('strong', key), unit, output);
      const canvas = node('canvas', null, 'chart'); canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', `${key} 실제·시뮬레이션 비교`);
      row.append(heading, canvas); $('joint-charts').append(row); charts[key] = new ReplayCharts.Chart(canvas, output, seek);
    });
    manifest.feet.forEach((name, foot) => {
      const box = node('section', null, 'foot'), heading = node('div', null, 'foot-heading');
      const correlation = node('small'); correlation.id = `force-corr-${foot}`; heading.append(node('b', name), correlation);
      const axisNote = node('div', null, 'axis-note'), left = node('span'), right = node('span');
      left.id = `force-unit-${foot}`; right.id = `force-unit-right-${foot}`; axisNote.append(left, right);
      const canvas = node('canvas', null, 'chart'); canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', `${name} 실제 센서와 시뮬레이션 힘`);
      box.append(heading, axisNote, canvas); $('force-charts').append(box); footCharts.push(new ReplayCharts.Chart(canvas, null, seek));
    });
    rmseChart = new ReplayCharts.Chart($('rmse-canvas'), $('rmse-output'), seek);
  }
  async function initialize() {
    try {
      manifest = await json('data/index.json');
      manifest.trials.forEach(entry => { const option = node('option', `${entry.id} · ${entry.velocity.toFixed(1)} m/s`); option.value = entry.id; $('trial').append(option); });
      manifest.joints.forEach((name, index) => { const parts = name.split('_'); const button = node('button', parts[0]); button.append(node('span', names[parts[1]])); button.addEventListener('click', () => { state.joint = index; refresh(); }); $('joints').append(button); });
      createCharts();
      $('trial').addEventListener('change', event => selectTrial(event.target.value));
      $('before').addEventListener('click', () => selectMode('before')); $('after').addEventListener('click', () => selectMode('after'));
      $('prev').addEventListener('click', () => joint(-1)); $('next').addEventListener('click', () => joint(1));
      $('window').addEventListener('change', event => { state.window = event.target.value; draw(); });
      $('rmse-quantity').addEventListener('change', event => { state.rmseQuantity = event.target.value; bindChannels(); draw(); });
      $('speed').addEventListener('change', event => { state.speed = Number(event.target.value); });
      $('force-raw').addEventListener('click', () => { state.forceZ = false; refresh(); }); $('force-z').addEventListener('click', () => { state.forceZ = true; refresh(); });
      $('seek').addEventListener('input', event => { const d = data(); seek(d.t[0] + Number(event.target.value) / 1000 * (d.t[d.t.length - 1] - d.t[0])); });
      function togglePlay() {
        if (!state.playing && activeData && state.cursor >= activeData.t[activeData.t.length - 1]) state.cursor = activeData.t[0];
        state.playing = !state.playing; $('play').textContent = state.playing ? 'Ⅱ' : '▶';
        $('play').setAttribute('aria-label', state.playing ? '일시정지' : '재생'); animationTime = 0;
      }
      $('play').addEventListener('click', togglePlay);
      window.addEventListener('keydown', event => {
        if (['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(event.target.tagName)) return;
        if (event.key === 'ArrowLeft') { event.preventDefault(); joint(-1); }
        if (event.key === 'ArrowRight') { event.preventDefault(); joint(1); }
        if (event.code === 'Space') { event.preventDefault(); togglePlay(); }
      });
      window.addEventListener('resize', draw);
      await selectTrial(manifest.trials[0].id);
      function animate(now) {
        if (state.playing && activeData) {
          if (animationTime) state.cursor += Math.min((now - animationTime) / 1000, .1) * state.speed;
          const end = activeData.t[activeData.t.length - 1];
          if (state.cursor >= end) { state.cursor = end; stop(); }
          animationTime = now; draw();
        } else animationTime = 0;
        requestAnimationFrame(animate);
      }
      requestAnimationFrame(animate);
    } catch (error) { fail(error); }
  }
  globalThis.ReplayApp = {state, initialize, selectMode, selectTrial, refresh, seek, joint};
  initialize();
})();
