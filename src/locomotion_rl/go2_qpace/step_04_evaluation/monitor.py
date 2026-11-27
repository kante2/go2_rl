"""QPace 학습 모니터: 학습 로그(TensorBoard 이벤트·체크포인트·영상)를 읽어 웹 페이지로 보여준다.

실행 (호스트, tensorboard가 설치된 python):
  /home/kante/miniconda3/envs/env_isaaclab/bin/python \
    src/locomotion_rl/go2_qpace/step_04_evaluation/monitor.py --port 6007
브라우저: http://localhost:6007

- 15초마다 자동 갱신 (학습 중인 run도 실시간 반영).
- 상단 건강 카드: 최근 50 iteration 평균을 기준값과 비교 (정상/주의/위험).
  이전 학습(qpace_34997_full)의 붕괴 신호 — 속도 추종 악화, 접촉 수 증가(두드리기),
  행동 표준편차 증가, 목표 만족률 하락 — 를 바로 보이게 한 것.
- 표준 라이브러리 + tensorboard 만 사용한다.
"""

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# (태그, 표시 이름, 정상 한계, 주의 한계, 높을수록 나쁨?)  — 기준: baseline model_34997 측정값
HEALTH = [
  ("QPace/tracking_error_mps", "속도 추종 오차 [m/s] (게이트 0.20)", 0.20, 0.25, True),
  ("Metrics/twist/error_vel_xy", "속도 추종 지표 (mjlab)", 0.50, 0.65, True),
  ("QPace/contacts_per_s", "초당 접촉 (baseline 11)", 12.0, 14.0, True),
  ("QPace/rejected_frac", "탈락 접촉 비율", 0.55, 0.65, True),
  ("Policy/mean_std", "행동 표준편차", 0.50, 0.70, True),
  ("Loss/noise_satisfied_frac", "목표 만족 환경", 0.90, 0.75, False),
  ("Train/mean_episode_length", "에피소드 길이", 950.0, 900.0, False),
  ("Metrics/slip_velocity_mean", "발 미끄러짐", 0.10, 0.15, True),
]

PANELS = [
  ("소음 [dB]", [("QPace/robot_power_db", "로봇 소음"), ("Loss/target_min_db", "가장 엄격한 목표"),
                 ("QPace/target_db", "평균 목표")], None),
  ("제약 / λ", [("Loss/noise_satisfied_frac", "목표 만족 비율"), ("Loss/lagrange_multiplier", "λ 평균"),
                ("Loss/lagrange_multiplier_max", "λ 최대"), ("Loss/lambda_cap", "λ 상한")], None),
  ("속도 추종 [m/s] (점선: 목표 게이트)", [("QPace/tracking_error_mps", "xy 오차 [m/s]")], 0.20),
  ("속도 추종 (mjlab 지표)", [("Metrics/twist/error_vel_xy", "xy"), ("Metrics/twist/error_vel_yaw", "yaw")], 0.50),
  ("접촉 (두드리기 감시)", [("QPace/contacts_per_s", "초당 접촉"), ("QPace/touchdowns_per_s", "초당 착지")], 12.0),
  ("접촉 품질", [("QPace/rejected_frac", "탈락 비율"), ("QPace/floored_frac", "floor 비율"),
                 ("QPace/contact_speed", "접촉 속도 [m/s]")], None),
  ("정책 탐색", [("Policy/mean_std", "행동 표준편차")], 0.50),
  ("baseline 이탈 (앵커)", [("Loss/anchor_mse", "Σ(μ − μ_base)²")], None),
  ("목표 커리큘럼 진행 (게이트)", [("Loss/target_progress", "통과한 ramp iteration")], None),
  ("안정성", [("Train/mean_episode_length", "에피소드 길이")], None),
  ("보상", [("Train/mean_reward", "평균 보상")], None),
  ("미끄러짐", [("Metrics/slip_velocity_mean", "미끄러짐 속도")], 0.10),
]


class RunCache:
  def __init__(self):
    self.acc = {}

  def get(self, run_dir: Path) -> EventAccumulator:
    key = str(run_dir)
    if key not in self.acc:
      self.acc[key] = EventAccumulator(key, size_guidance={"scalars": 0})
    self.acc[key].Reload()  # 증분 읽기
    return self.acc[key]


def downsample(steps, values, n=500):
  if len(steps) <= n:
    return steps, values
  k = -(-len(steps) // n)
  s = [steps[i] for i in range(0, len(steps), k)]
  v = [sum(values[i:i + k]) / len(values[i:i + k]) for i in range(0, len(values), k)]
  return s, v


def yaml_int(text, key):
  m = re.search(rf"^\s*{key}:\s*(\d+)", text, re.M)
  return int(m.group(1)) if m else None


class Monitor:
  def __init__(self, logdir: Path):
    self.logdir = logdir
    self.cache = RunCache()

  def runs(self):
    dirs = [p for p in self.logdir.iterdir() if p.is_dir() and any(p.glob("events.out.tfevents*"))]
    dirs.sort(key=lambda p: max(f.stat().st_mtime for f in p.glob("events.out.tfevents*")), reverse=True)
    return [p.name for p in dirs]

  def run_data(self, name: str):
    run = self.logdir / name
    acc = self.cache.get(run)
    tags = set(acc.Tags()["scalars"])
    series = {}
    for tag in tags:
      ev = acc.Scalars(tag)
      s, v = downsample([e.step for e in ev], [e.value for e in ev])
      series[tag] = {"s": s, "v": v}

    agent = (run / "params" / "agent.yaml")
    agent_text = agent.read_text() if agent.exists() else ""
    max_it = yaml_int(agent_text, "max_iterations")
    steps_per_env = yaml_int(agent_text, "num_steps_per_env") or 24

    health = []
    for tag, label, ok, warn, high_bad in HEALTH:
      if tag not in tags:
        continue
      ev = acc.Scalars(tag)
      recent = [e.value for e in ev[-50:]]
      cur = sum(recent) / len(recent)
      old = [e.value for e in ev[-350:-300]] if len(ev) > 350 else None
      bad = (lambda x, t: x > t) if high_bad else (lambda x, t: x < t)
      level = "bad" if bad(cur, warn) else ("warn" if bad(cur, ok) else "ok")
      trend = cur - sum(old) / len(old) if old else None
      health.append({"label": label, "value": cur, "ok": ok, "warn": warn, "level": level,
                     "trend": trend, "high_bad": high_bad})

    progress = {"iteration": None, "max": max_it, "eta_s": None, "it_per_s": None, "running": False}
    ref = "Train/mean_reward" if "Train/mean_reward" in tags else next(iter(tags), None)
    if ref:
      ev = acc.Scalars(ref)
      last = ev[-1]
      progress["iteration"] = last.step
      if len(ev) > 20:
        a = ev[-min(len(ev), 200)]
        rate = (last.step - a.step) / max(last.wall_time - a.wall_time, 1e-6)
        progress["it_per_s"] = rate
        if max_it and rate > 0:
          progress["eta_s"] = max(max_it - 1 - last.step, 0) / rate
      progress["running"] = time.time() - last.wall_time < 120

    ckpts = sorted((int(m.group(1)) for p in run.glob("model_*.pt")
                    if (m := re.match(r"model_(\d+)\.pt", p.name))))
    videos = []
    vdir = run / "videos" / "train"
    if vdir.exists():
      for p in vdir.glob("*.mp4"):
        m = re.search(r"step-(\d+)", p.name)
        if m:
          videos.append({"file": p.name, "iteration": int(m.group(1)) // steps_per_env})
      videos.sort(key=lambda x: x["iteration"])

    return {"name": name, "series": series, "health": health, "progress": progress,
            "checkpoints": ckpts, "videos": videos,
            "panels": [{"title": t, "lines": l, "limit": lim} for t, l, lim in PANELS]}


def make_handler(mon: Monitor):
  class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
      pass

    def send(self, code, body, ctype):
      self.send_response(code)
      self.send_header("Content-Type", ctype)
      self.send_header("Content-Length", str(len(body)))
      self.send_header("Cache-Control", "no-store")
      self.end_headers()
      self.wfile.write(body)

    def do_GET(self):
      url = urlparse(self.path)
      try:
        if url.path == "/":
          self.send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif url.path == "/api/runs":
          self.send(200, json.dumps(mon.runs()).encode(), "application/json")
        elif url.path == "/api/run":
          name = parse_qs(url.query)["name"][0]
          if "/" in name or name.startswith("."):
            raise ValueError("bad run")
          self.send(200, json.dumps(mon.run_data(name)).encode(), "application/json")
        elif url.path.startswith("/video/"):
          _, _, run, file = url.path.split("/", 3)
          run, file = unquote(run), unquote(file)
          if "/" in file or ".." in run or ".." in file:
            raise ValueError("bad path")
          self.serve_file(mon.logdir / run / "videos" / "train" / file)
        else:
          self.send(404, b"not found", "text/plain")
      except Exception as exc:  # noqa: BLE001
        self.send(500, str(exc).encode(), "text/plain")

    def serve_file(self, path: Path):
      data = path.read_bytes()
      rng = self.headers.get("Range")
      if rng and (m := re.match(r"bytes=(\d+)-(\d*)", rng)):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else len(data) - 1
        chunk = data[start:end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)
      else:
        self.send(200, data, "video/mp4")

  return Handler


PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QPace 학습 모니터</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d2330;--mut:#6b7280;--line:#e5e7eb;--ok:#1a7f4b;--warn:#b7791f;--bad:#c53030}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;--fg:#e6e8ec;--mut:#9aa3b2;--line:#2b3240;--ok:#48bb78;--warn:#ecc94b;--bad:#fc8181}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:12px 16px;border-bottom:1px solid var(--line);background:var(--card);position:sticky;top:0;z-index:2}
h1{font-size:16px;margin:0}select{font:inherit;padding:4px 8px;max-width:100%}
.mut{color:var(--mut)}main{padding:16px;max-width:1500px;margin:auto}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden;min-width:160px;flex:1}.bar>div{height:100%;background:var(--ok)}
.grid{display:grid;gap:12px}.health{grid-template-columns:repeat(auto-fill,minmax(170px,1fr));margin-bottom:16px}
.charts{grid-template-columns:repeat(auto-fill,minmax(420px,1fr))}
@media (max-width:480px){.charts{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
.h .v{font-size:22px;font-weight:600}.h.ok{border-left:4px solid var(--ok)}.h.warn{border-left:4px solid var(--warn)}.h.bad{border-left:4px solid var(--bad)}
.h.bad .v{color:var(--bad)}.h.warn .v{color:var(--warn)}
h2{font-size:14px;margin:0 0 8px}canvas{width:100%!important;height:230px!important}
.videos{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}video{width:100%;border-radius:6px;background:#000}
.ck{display:flex;flex-wrap:wrap;gap:4px}.ck span{border:1px solid var(--line);border-radius:4px;padding:1px 6px;font-size:12px}
section{margin-top:20px}
</style></head><body>
<header><h1>QPace 학습 모니터</h1><select id="run"></select>
<span id="status" class="mut"></span><div class="bar"><div id="pbar" style="width:0"></div></div>
<label class="mut"><input type="checkbox" id="auto" checked> 15초 자동 갱신</label></header>
<main>
<div id="health" class="grid health"></div>
<div id="charts" class="grid charts"></div>
<section><h2>보행 영상</h2><div id="videos" class="videos"></div></section>
<section><h2>체크포인트</h2><div id="ckpts" class="ck card"></div></section>
</main>
<script>
const COLORS=['#2563eb','#dc2626','#16a34a','#9333ea','#ea580c'];
const charts={};let cur=null,lastVideos='';
const fmt=(x,d=3)=>x==null?'-':(Math.abs(x)>=100?x.toFixed(0):x.toPrecision(d));
const hms=s=>{s=Math.round(s);return `${Math.floor(s/3600)}:${String(Math.floor(s%3600/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`};
async function loadRuns(){
  const runs=await (await fetch('/api/runs')).json();const sel=document.getElementById('run');
  const keep=sel.value;sel.innerHTML=runs.map(r=>`<option>${r}</option>`).join('');
  sel.value=runs.includes(keep)?keep:runs[0];}
async function refresh(){
  const name=document.getElementById('run').value;if(!name)return;
  const d=await (await fetch('/api/run?name='+encodeURIComponent(name))).json();
  if(cur!==name){Object.values(charts).forEach(c=>c.destroy());for(const k in charts)delete charts[k];
    document.getElementById('charts').innerHTML='';lastVideos='';cur=name;}
  const p=d.progress;
  document.getElementById('status').textContent=`${p.running?'● 학습 중':'○ 정지'} · iteration ${p.iteration??'-'} / ${p.max??'?'}`+
    (p.it_per_s?` · ${(1/p.it_per_s).toFixed(2)} s/it`:'')+(p.running&&p.eta_s!=null?` · 남은 시간 ${hms(p.eta_s)}`:'');
  document.getElementById('pbar').style.width=(p.max?100*(p.iteration+1)/p.max:0)+'%';
  document.getElementById('health').innerHTML=d.health.map(h=>{
    const arrow=h.trend==null?'':(h.trend>0?'▲':'▼')+fmt(Math.abs(h.trend),2);
    const worse=h.trend!=null&&((h.trend>0)===h.high_bad)&&Math.abs(h.trend)>1e-3;
    return `<div class="card h ${h.level}"><div class="mut">${h.label}</div><div class="v">${fmt(h.value)}</div>
    <div class="mut">${h.high_bad?'정상 <':'정상 >'} ${h.ok} · 위험 ${h.high_bad?'>':'<'} ${h.warn}</div>
    <div class="mut" style="color:${worse?'var(--warn)':''}">300 it 대비 ${arrow||'-'}</div></div>`}).join('');
  const box=document.getElementById('charts');
  d.panels.forEach((pn,i)=>{
    const ds=pn.lines.filter(([t])=>d.series[t]).map(([t,l],j)=>({label:l,data:d.series[t].s.map((s,k)=>({x:s,y:d.series[t].v[k]})),
      borderColor:COLORS[j],backgroundColor:COLORS[j],borderWidth:1.5,pointRadius:0,tension:0}));
    if(!ds.length)return;
    if(pn.limit!=null){const xs=ds[0].data;ds.push({label:'정상 한계',data:[{x:xs[0].x,y:pn.limit},{x:xs[xs.length-1].x,y:pn.limit}],
      borderColor:'#9ca3af',borderDash:[5,4],borderWidth:1,pointRadius:0});}
    if(charts[i]){charts[i].data.datasets=ds;charts[i].update('none');return;}
    const c=document.createElement('div');c.className='card';c.innerHTML=`<h2>${pn.title}</h2><canvas></canvas>`;box.appendChild(c);
    charts[i]=new Chart(c.querySelector('canvas'),{type:'line',data:{datasets:ds},options:{animation:false,parsing:false,
      interaction:{mode:'nearest',axis:'x',intersect:false},scales:{x:{type:'linear',title:{display:true,text:'iteration'}}},
      plugins:{legend:{labels:{boxWidth:12}}}}});});
  const vk=JSON.stringify(d.videos);
  if(vk!==lastVideos){lastVideos=vk;document.getElementById('videos').innerHTML=d.videos.length?d.videos.slice().reverse().map(v=>
    `<div class="card"><h2>iteration ${v.iteration}</h2><video controls preload="metadata" src="/video/${encodeURIComponent(name)}/${encodeURIComponent(v.file)}"></video></div>`).join('')
    :'<div class="mut">영상 없음 (--video True 로 학습하면 저장됨)</div>';}
  document.getElementById('ckpts').innerHTML=d.checkpoints.length?d.checkpoints.map(c=>`<span>${c}</span>`).join(''):'<span class="mut">없음</span>';
}
document.getElementById('run').onchange=refresh;
(async()=>{await loadRuns();await refresh();setInterval(async()=>{if(document.getElementById('auto').checked){await loadRuns();await refresh();}},15000);})();
</script></body></html>
"""


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--logdir", default="logs/rsl_rl/go2_qpace")
  ap.add_argument("--host", default="127.0.0.1")
  ap.add_argument("--port", type=int, default=6007)
  args = ap.parse_args()
  mon = Monitor(Path(args.logdir).resolve())
  print(f"QPace monitor: http://localhost:{args.port}  (logdir {mon.logdir})")
  ThreadingHTTPServer((args.host, args.port), make_handler(mon)).serve_forever()


if __name__ == "__main__":
  main()
