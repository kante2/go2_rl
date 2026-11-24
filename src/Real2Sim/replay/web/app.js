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
    if (manifest.excludedTrials?.length) notes(`비교 제외: ${manifest.excludedTrials.map(item => item.id).join(', ')} (상태 샘플 없음)`);
    $('scope').textContent = '전후 동일한 평가 샘플 · sim − real 오차 · 화면 확대와 별개';
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
      const timing = sync.physicalComparison?.quantities[quantity];
      if (timing) {
        const a = timing.clockOnly[state.joint].rmse, b = timing.aligned[state.joint].rmse;
        const comparison = node('small', `시간 이동 RMSE ${num(a)} → ${num(b)} ${sync.physicalComparison.units[quantity]}${b > a ? ' · 증가' : ''}`);
        comparison.style.gridColumn = '1 / -1'; card.append(comparison);
      }
      $('metric-cards').append(card);
    });
    addStep('기존 real · sim 로그 유지', '동일 baseline 정책·명령의 기록 · 재시뮬레이션 없음', 1, false);
    addStep('관절 · 발 순서와 단위 확인', 'SDK FR·FL·RR·RL / hip·thigh·calf · q는 rad 그대로', 2, false);
    addStep('동일 전처리 · 기록 시간축', `tick 중복 ${trial.timing.duplicate_ticks_removed}개 제거 · 양쪽 15 Hz 필터 · 공통 200 Hz`, 3, false);
    addStep(`공통 시간 이동 ${num(sync.applied_lag_s * 1000, 1)} ms`, sync.passed ? '앞 구간 추정 · 뒤 구간 검증 · 모든 sim 채널에 동일 적용' : '검증 미통과 · 공통 PC 시각 유지', 4, sync.passed);
    if (state.mode === 'after') {
      addStep('배포 관측 좌표 변환', 'policy.json 관절 순서·기본각 → 실제 ONNX 고정 정규화 · 숫자 역변환 없음', 5, true);
      addStep('목표각 scale 검증', '기록 목표각 ↔ action 좌표 왕복 · 관측 q에 action_scale을 곱하지 않음', 6, true);
    }
    notes('전후 모두 같은 실제 시각·평가 샘플로 오차를 계산합니다. q/dq는 15 Hz 필터 값이며 ddq는 필터 dq의 미분입니다.');
    if (sync.phase_validation_performed && sync.config?.same_shift_all_channels) {
      const cal = sync.calibration, val = sync.validation;
      notes(`시행별 공통 시간 이동 ${num(sync.applied_lag_s * 1000, 1)} ms. 양수는 sim 파형을 왼쪽으로, 음수는 오른쪽으로 이동합니다. q/dq/ddq·토크·발 힘에 같은 이동을 적용합니다.`);
      const contactReference = sync.reference === 'contact_load_with_joint_consensus';
      const reference = contactReference ? '네 발 하중과 관절 추정 시각의 일치로' : '8개 thigh/calf 관절의 q/dq로';
      notes(`앞 ${cal.window_s.map(x => num(x, 2)).join('–')}초에서 ${reference} 추정하고, 뒤 ${val.window_s.map(x => num(x, 2)).join('–')}초에서 검증합니다. 추정 ${num(sync.estimated_lag_s * 1000, 1)} ms · 뒤 구간 ${val.cycle_anchored ? '같은 보행 주기 내' : '독립'} 추정 ${num(val.lag_s == null ? null : val.lag_s * 1000, 1)} ms.`);
      if (sync.search_expansion) notes('기본 ±250 ms 탐색에서 경계에 도달하여 ±500 ms로 확장했습니다. 앞 구간에서 반복 주기를 구분하고 뒤 구간에서 고정 이동량의 상관과 잔여 위상 차이를 검증합니다.');
      if (val.cycle_anchored && val.ambiguous) notes('뒤 구간에는 한 보행 주기만큼 떨어진 유사한 해가 있습니다. 앞 구간에서 선택한 주기를 유지한 파형 정렬값이며, 순수 통신 지연을 식별한 값은 아닙니다.');
      if (contactReference) notes('관절 파형만의 상관은 부족했지만, 네 발 하중이 앞뒤 구간에서 일치하고 관절의 추정 이동량과도 일치하여 공통 시간 이동을 적용했습니다.');
      if (!sync.passed) {
        const reasons = {insufficient_motion:'움직임 표본 부족', search_boundary:'탐색 범위 경계', ambiguous_period:'반복 주기 모호', low_correlation:'파형 상관 부족', lag_changed:'앞뒤 시간 차이 불일치', low_fixed_correlation:'검증 구간 상관 부족', no_improvement:'이동 효과 부족', better_cycle:'다른 보행 주기의 검증 상관이 더 높음'};
        notes(`시간 이동 미적용: ${[...new Set(sync.rejection_reasons.map(x => reasons[x.split(':')[1]] || x))].join(', ')}.`, true);
      }
      notes('파형 위상 정렬값이며 하드웨어 통신 지연 측정값은 아닙니다. 아래 RMSE 카드에는 같은 물리 단위로 시간 이동 전후 오차를 표시합니다.');
      const tau = sync.physicalComparison?.quantities.tau;
      if (tau && tau.aligned[state.joint].rmse > tau.clockOnly[state.joint].rmse) notes('선택 관절의 토크 RMSE는 공통 시간 이동 후 증가했습니다. 토크만 따로 이동하거나 진폭을 맞추지 않았습니다.', true);
    } else if (trial.timing.shared_pc_clock) {
      notes('공통 PC monotonic 수신 시각의 겹치는 구간을 비교합니다. sim 필터·미분은 sim_time을 사용합니다.');
    }
    if (trial.timing.shared_pc_clock) {
      notes('move_start/stop은 녹음 구간입니다. 시작 2초·끝 1초와 누락 주변을 평가에서 제외하며 정상 보행 여부를 자동 검증하지 않습니다.');
    } else notes(`위상 추정은 시작 구간, 검증은 끝 구간입니다. 끝 잔차 p90 ${num(sync.held_out_abs_residual_p90_s == null ? null : sync.held_out_abs_residual_p90_s * 1000, 1)} ms. 독립 보행의 위상 차이이며 하드웨어 지연을 뜻하지 않습니다.`);
    notes('발 힘은 실제 counts와 시뮬 양의 수직 하중(N)을 비교합니다. S3 시뮬 로그의 접촉 벡터는 z가 음수이므로 부호를 한 번 바꿔 표시합니다. z-score는 파형 비교용이며 센서 교정이 아닙니다.');
    notes(sync.reference === 'contact_load_with_joint_consensus'
      ? '이 시행은 네 발 하중을 기준으로 공통 이동을 검증했습니다. 발별 추가 이동은 없으며 센서 응답·접촉 파형 차이는 남을 수 있습니다.'
      : '시간 이동 검증은 관절 q/dq 기준입니다. 발 접촉 시점의 일치까지 검증한 값은 아니며 센서 응답·접촉 파형 차이가 남을 수 있습니다.');
    const dl = node('dl');
    const deployment = trial.mapping.deployment;
    if (deployment.recording_policy_hash_verified === false) notes('수집 NPZ에는 정책 해시가 없습니다. 데이터 설명의 baseline 배포 파일로 매핑하며, 해당 배포 파일의 해시/parity만 검증했습니다.');
    const policyJoint = deployment.policy_to_sdk.indexOf(state.joint);
    [['데이터', '기존 real / sim NPZ'], ['시간 정렬', `${num(sync.applied_lag_s * 1000, 1)} ms · 01/02 공통`], ['배포 관절 순서', 'FL·FR·RL·RR'],
      ['선택 관절 기본각', `${num(deployment.default_joint_pos[policyJoint], 3)} rad`],
      ['목표각 action scale', num(deployment.action_scale[policyJoint], 3)],
      ['02 q 배율', num(1 / deployment.normalization_denominator[11 + policyJoint], 4)],
      ['02 q 오프셋', num((-deployment.default_joint_pos[policyJoint] - deployment.normalization_mean[11 + policyJoint]) / deployment.normalization_denominator[11 + policyJoint], 4)],
      ['02 dq 배율', num(1 / deployment.normalization_denominator[23 + policyJoint], 4)], ['관절 관측 parity', `${deployment.joint_parity_frames} frames`],
      ['q 왕복 최대오차', `${num(Math.max(deployment.real_q_roundtrip_max_error_rad, deployment.sim_q_roundtrip_max_error_rad))} rad`],
      ['tau 매핑', '배포 규약 없음 · N·m 유지'], ['발 힘 매핑', '배포 규약 없음 · 유지'], ['재시뮬레이션', '없음']].forEach(([key, value]) => dl.append(node('dt', key), node('dd', value)));
    notes('02 q/dq는 배포 정책이 사용하는 정규화 관측(무차원)입니다. 01의 물리 단위 RMSE와 숫자를 직접 비교해 개선률로 해석하지 않습니다. ddq·tau는 정책 입력에 없으므로 물리값과 단위를 유지합니다. tau는 real 추정 모터 토크와 sim 액추에이터 토크를 비교하며, 배포 토크 스케일은 없습니다.');
    $('contract').append(dl);
    $('provenance').textContent = `${manifest.dataset || ''} · real ${manifest.trials.find(t => t.id === trial.trial).realSha256.slice(0, 12)} · ${trial.trial}`;
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
      ? `01 물리값: 공통 전처리 후 시간 이동 ${num(trial.synchronization.applied_lag_s * 1000, 1)} ms를 반영한 real·sim 로그입니다. 미검증 이동은 적용하지 않습니다.`
      : '02 보정 후: 같은 시간 정렬 데이터의 q/dq를 실제 배포 정책 입력으로 매핑해 그대로 표시합니다(무차원). ddq·tau·발 힘은 배포 매핑 규약이 없어 물리값을 유지합니다.';
    const review = !trial.synchronization.passed;
    $('status').classList.toggle('review', review);
    $('status').textContent = review ? '시간 이동 미적용 · 검토 필요' : `시간 이동 ${num(trial.synchronization.applied_lag_s * 1000, 1)} ms · 검증 통과`;
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
      trial = value;
      $('velocity-summary').hidden = !Number.isFinite(value.velocity);
      $('velocity').textContent = Number.isFinite(value.velocity) ? value.velocity.toFixed(1) : '';
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
      manifest.trials.forEach(entry => { const speed = Number.isFinite(entry.velocity) ? ` · ${entry.velocity.toFixed(1)} m/s` : ''; const option = node('option', `${entry.id}${speed}`); option.value = entry.id; $('trial').append(option); });
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
