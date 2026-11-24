/* Dependency-free UI logic smoke test. Uses a mock DOM, not browser rendering.
   runReplaySmoke(chartSource, appSource) returns a Promise with check results. */
async function runReplaySmoke(chartSource, appSource) {
  const elements = new Map(), all = [], events = {}, frames = [];
  class Element {
    constructor(tag = 'div') { this.tagName = tag.toUpperCase(); this.children = []; this.listeners = {}; this.textContent = ''; this.value = ''; this.hidden = false; this.attributes = {}; this.style = {}; this.classes = new Set(); this.classList = {toggle: (c, on) => { if (on) this.classes.add(c); else this.classes.delete(c); }}; all.push(this); }
    set id(value) { this._id = value; elements.set(value, this); } get id() { return this._id; }
    append(...items) { this.children.push(...items); }
    replaceChildren(...items) { this.children = items; }
    addEventListener(name, handler) { this.listeners[name] = handler; }
    setAttribute(name, value) { this.attributes[name] = value; }
    setPointerCapture() {}
    getBoundingClientRect() { return {left: 0, width: 700, height: 150}; }
    getContext() { return new Proxy({}, {get: (_, key) => () => {}, set: () => true}); }
  }
  const ids = ['trial','velocity','velocity-summary','before','after','status','mode-note','joint-title','prev','page','next','joints','sim-label','joint-charts','play','clock','seek','speed','window','force-raw','force-z','force-charts','scope','metric-cards','diagram','notes','contract','provenance','error','rmse-quantity','rmse-output','rmse-unit','rmse-canvas','force-rmse-cards','force-rmse-scope'];
  ids.forEach(id => { const e = new Element(id === 'rmse-canvas' ? 'canvas' : 'div'); e.id = id; });
  globalThis.document = {getElementById: id => elements.get(id), createElement: tag => new Element(tag)};
  globalThis.window = globalThis; globalThis.devicePixelRatio = 1;
  globalThis.addEventListener = (name, handler) => { events[name] = handler; };
  globalThis.requestAnimationFrame = callback => { frames.push(callback); return frames.length; };
  function series(first, last, kind) {
    const t = Array.from({length: 101}, (_, i) => first + (last - first) * i / 100);
    const real = {}, sim = {}, metrics = {};
    for (const q of ['q','dq','ddq','tau']) {
      real[q] = t.map(time => Array.from({length: 12}, (_, j) => Math.sin(time) + j));
      sim[q] = t.map(time => Array.from({length: 12}, (_, j) => Math.sin(time) + j + .01));
      metrics[q] = Array.from({length: 12}, () => ({n: 100, rmse: .01, mae: .01, bias: .01, correlation: .9}));
    }
    const quantities = {};
    for (const q of ['q','dq','ddq','tau']) quantities[q] = {joint:t.map(() => new Array(12).fill(.01)),combined:t.map(() => .01)};
    return {t, real, sim, metrics, units:kind==='after'?{q:'무차원',dq:'무차원',ddq:'rad/s²',tau:'N·m'}:{q:'rad',dq:'rad/s',ddq:'rad/s²',tau:'N·m'}, kind, suggestedTime: (first + last) / 2, errors:{windowSeconds:.5,quantities},
      force: {real_counts: t.map(() => [20,30,40,50]), sim_N: t.map(() => [10,20,30,40]), correlation: [.9,.9,.9,.9], zscoreRmse:[.4,.4,.4,.4], real_mean: [20,30,40,50], real_std: [1,1,1,1], sim_mean: [10,20,30,40], sim_std: [1,1,1,1]}};
  }
  const joints = ['FR','FL','RR','RL'].flatMap(leg => ['hip','thigh','calf'].map(j => `${leg}_${j}_joint`));
  const manifest = {dataset:'1118_RL_base', excludedTrials:[{id:'normal_07'}], joints, feet: ['FR','FL','RR','RL'], parity: {onnx_torch_max_error: 1e-6}, trials: [{id:'0.5_0.5_v1',file:'0.5_0.5_v1.json',velocity:null,realSha256:'abc123456789'}]};
  const trial = {trial: '0.5_0.5_v1', velocity: null, before: series(0, 10, 'baseline'), after: series(0,10,'after'), timing:{duplicate_ticks_removed:2,shared_pc_clock:true}, synchronization:{passed:true,applied_lag_s:.045,estimated_lag_s:.045,phase_validation_performed:true,config:{same_shift_all_channels:true},calibration:{window_s:[2.3,4.4]},validation:{window_s:[4.8,7.1],lag_s:.045},rejection_reasons:[]}, mapping:{deployment:{recording_policy_hash_verified:false,policy_to_sdk:[3,4,5,0,1,2,9,10,11,6,7,8],default_joint_pos:new Array(12).fill(0),action_scale:new Array(12).fill(.25),joint_parity_frames:100,normalization_denominator:new Array(47).fill(1),normalization_mean:new Array(47).fill(0),real_q_roundtrip_max_error_rad:0,sim_q_roundtrip_max_error_rad:0}}};
  trial.synchronization.physicalComparison = {units:{q:'rad',dq:'rad/s',ddq:'rad/s²',tau:'N·m'},quantities:{}};
  for (const q of ['q','dq','ddq','tau']) trial.synchronization.physicalComparison.quantities[q] = {clockOnly:new Array(12).fill({rmse:.02}),aligned:new Array(12).fill({rmse:q==='tau'?.03:.01})};
  globalThis.fetch = async path => ({ok:true,json:async () => path.endsWith('index.json') ? manifest : trial});
  eval(chartSource); eval(appSource);
  for (let i = 0; i < 30; i++) await Promise.resolve();
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  check(elements.get('joints').children.length === 12, '12 joint pages');
  check(all.filter(e => e.tagName === 'CANVAS').length === 9, 'four joint, one RMSE and four force charts');
  check(elements.get('force-rmse-cards').children.length === 4, 'four foot RMSE cards');
  check(elements.get('force-rmse-cards').children[0].children[1].textContent === '0.400', 'foot RMSE value');
  check(elements.get('velocity-summary').hidden && elements.get('velocity').textContent === '', 'missing velocity summary is hidden');
  check(elements.get('trial').children[0].textContent === '0.5_0.5_v1', 'trial label omits missing velocity text');
  check(elements.get('provenance').textContent.includes('1118_RL_base'), 'selected dataset shown');
  check(elements.get('notes').children.some(e => e.textContent.includes('normal_07')), 'empty trial exclusion visible');
  check(elements.get('status').textContent.includes('45.0 ms') && elements.get('status').textContent.includes('검증 통과'), 'applied common lag displayed');
  check(elements.get('notes').children.some(e => e.textContent.includes('토크 RMSE') && e.textContent.includes('증가')), 'torque degradation disclosed');
  check(elements.get('metric-cards').children[0].children.some(e => e.textContent.includes('시간 이동 RMSE')), 'clock baseline versus aligned physical metrics');
  trial.synchronization.passed = false; trial.synchronization.applied_lag_s = 0; trial.synchronization.rejection_reasons = ['validation:lag_changed']; ReplayApp.refresh();
  check(elements.get('status').textContent.includes('미적용') && elements.get('notes').children.some(e => e.textContent.includes('앞뒤 시간 차이 불일치')), 'failed validation preserves clock and explains rejection');
  trial.synchronization.passed = true; trial.synchronization.applied_lag_s = .045; trial.synchronization.rejection_reasons = []; ReplayApp.refresh();
  trial.synchronization.reference = 'contact_load_with_joint_consensus'; ReplayApp.refresh();
  check(elements.get('notes').children.some(e => e.textContent.includes('네 발 하중')), 'contact-supported reference disclosed');
  trial.synchronization.reference = 'joint_q_dq'; ReplayApp.refresh();
  trial.synchronization.search_expansion = {initial_max_lag_s:.25,expanded_max_lag_s:.5};
  trial.synchronization.validation.cycle_anchored = true; trial.synchronization.validation.ambiguous = true;
  ReplayApp.refresh();
  check(elements.get('notes').children.some(e => e.textContent.includes('±500 ms')), 'expanded lag search disclosed');
  check(elements.get('notes').children.some(e => e.textContent.includes('앞 구간에서 선택한 주기')), 'late periodic cycle ambiguity disclosed');
  delete trial.synchronization.search_expansion;
  delete trial.synchronization.validation.cycle_anchored; delete trial.synchronization.validation.ambiguous;
  ReplayApp.joint(11); check(ReplayApp.state.joint === 11, 'last joint');
  ReplayApp.joint(1); check(ReplayApp.state.joint === 0, 'joint page wraps');
  ReplayApp.selectMode('after'); check(ReplayApp.state.mode === 'after', 'mapping after mode');
  check(elements.get('sim-label').textContent.includes('기존'), 'same existing simulation in both modes');
  check(elements.get('joint-unit-q').textContent.includes('무차원'), 'mapped q unit is dimensionless');
  check(elements.get('joint-unit-ddq').textContent.includes('물리값 유지'), 'unmapped ddq is explicitly physical');
  elements.get('force-z').listeners.click(); check(ReplayApp.state.forceZ, 'force z-score mode');
  elements.get('rmse-quantity').listeners.change({target:{value:'dq'}});
  check(ReplayApp.state.rmseQuantity === 'dq', 'RMSE quantity selection');
  elements.get('rmse-quantity').listeners.change({target:{value:'tau'}});
  check(ReplayApp.state.rmseQuantity === 'tau' && elements.get('rmse-unit').textContent.includes('N·m'), 'torque RMSE selector and unit');
  elements.get('seek').listeners.input({target:{value:'500'}});
  check(Math.abs(ReplayApp.state.cursor - 5) < 1e-9, 'timeline seeks within same full record');
  elements.get('play').listeners.click(); check(ReplayApp.state.playing, 'play starts');
  elements.get('play').listeners.click(); check(!ReplayApp.state.playing, 'play pauses');
  ReplayApp.selectMode('before'); check(ReplayApp.state.mode === 'before', 'mapping before mode');
  check(elements.get('joint-unit-q').textContent === 'rad', 'physical unit restored in baseline view');
  check(elements.get('error').hidden && elements.get('error').textContent === '', 'no runtime error');
  return {checks:30, result:'PASS', scope:'mock DOM and real Canvas call paths; not a visual browser test'};
}
if (typeof module !== 'undefined') module.exports = runReplaySmoke;
globalThis.runReplaySmoke = runReplaySmoke;
