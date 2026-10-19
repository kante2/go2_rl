"use strict";
const $ = id => document.getElementById(id);
const feet = ["FR", "FL", "RR", "RL"];
const colors = ["#62ddba", "#8bbcff", "#eaba7c", "#c6a6f7"];
const fmt = (v, places=2) => Number.isFinite(v) ? v.toFixed(places) : "—";
let samples = [], latest = null, generation = "", after = 0, paused = false, windowSeconds = 30;
$("landings").innerHTML = feet.map((name, i) => `<tr><td style="color:${colors[i]}">${name}</td><td id="impulse-${name}">—</td><td id="landing-peak-${name}">—</td><td id="count-${name}">—</td><td id="time-${name}">—</td></tr>`).join("");
$("force-legend").innerHTML = feet.map((name, i) => `<span style="--color:${colors[i]}">${name}</span>`).join("");

function chart(id, series, unit, positive=false) {
  const canvas = $(id), ctx = canvas.getContext("2d"), rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio);
  ctx.scale(ratio, ratio);
  const w = rect.width, h = rect.height, left = 49, right = w - 13, top = 22, bottom = h - 28;
  const end = latest ? latest.sim_time : windowSeconds, start = end - windowSeconds;
  const points = samples.filter(s => s.sim_time >= start && s.sim_time <= end);
  const values = series.flatMap(line => points.map(line.get)).filter(Number.isFinite);
  let low = positive ? 0 : Math.min(0, ...values), high = Math.max(positive ? 10 : .2, ...values);
  if (!positive) low = Math.min(low, -.05);
  high += (high-low) * .12;
  const x = t => left + (t-start)/windowSeconds*(right-left);
  const y = v => bottom - (v-low)/(high-low)*(bottom-top);
  ctx.font = '10px ui-monospace, monospace'; ctx.lineWidth = 1;
  for (let i=0; i<=4; i++) {
    const v = low + (high-low)*i/4, yy = y(v);
    ctx.strokeStyle = "#2a373d"; ctx.setLineDash([]); ctx.beginPath(); ctx.moveTo(left,yy); ctx.lineTo(right,yy); ctx.stroke();
    ctx.fillStyle="#879a9e"; ctx.textAlign="right"; ctx.fillText(v.toFixed(positive?0:2),left-8,yy+3);
  }
  ctx.textAlign="left"; ctx.fillText(unit,4,11);
  ctx.textAlign="center";
  for(let i=0;i<=4;i++) ctx.fillText(`${Math.round(-windowSeconds + windowSeconds*i/4)}s`,left+(right-left)*i/4,h-6);
  ctx.save(); ctx.beginPath(); ctx.rect(left,top,right-left,bottom-top); ctx.clip();
  for(const line of series) {
    ctx.strokeStyle=line.color; ctx.lineWidth=line.dash?1.2:1.8; ctx.setLineDash(line.dash?[5,5]:[]); ctx.beginPath();
    let previous=null;
    for(const point of points) {
      const value=line.get(point);
      if(!Number.isFinite(value)) { previous=null; continue; }
      // Missing telemetry is a gap, never a fabricated interpolation.
      if(previous===null || point.sequence-previous.sequence>1) ctx.moveTo(x(point.sim_time),y(value));
      else ctx.lineTo(x(point.sim_time),y(value));
      previous=point;
    }
    ctx.stroke();
  }
  ctx.restore(); ctx.setLineDash([]);
  if(!points.length) {ctx.fillStyle="#7e9495";ctx.textAlign="center";ctx.fillText("측정 데이터 대기 중",w/2,h/2);}
}

function render() {
  if(latest) {
    const s=latest;
    $("model").textContent=s.model; $("checkpoint").textContent=s.checkpoint;
    $("quietness").textContent=fmt(s.quietness); $("sim-time").textContent=`${fmt(s.sim_time,1)} s`;
    $("rtf").textContent=`${fmt(s.rtf)}×`; $("reset").textContent=s.reset_counter;
    $("vx").textContent=fmt(s.linear_velocity[0]); $("vy").textContent=fmt(s.linear_velocity[1]); $("wz").textContent=fmt(s.angular_velocity[2]);
    ["vx","vy","wz"].forEach((axis,i)=>$("cmd-"+axis).textContent=fmt(s.command[i]));
    const peaks=samples.filter(p=>p.sim_time>=s.sim_time-windowSeconds).flatMap(p=>p.peak_n);
    $("peak").textContent=peaks.length?fmt(Math.max(...peaks),1):"—";
    feet.forEach((name,i)=>{
      const foot=$("foot-"+name);
      foot.classList.toggle("contact",s.contact[i]);
      foot.querySelector("strong").innerHTML=`${fmt(s.force_n[i],1)} <small>N</small>`;
      foot.querySelector("span").textContent=s.contact[i]?"접지":"공중";
      $("impulse-"+name).textContent=fmt(s.landing_impulse_ns[i],3);
      $("landing-peak-"+name).textContent=fmt(s.landing_peak_n[i],1);
      $("count-"+name).textContent=s.landing_count[i];
      $("time-"+name).textContent=s.landing_time_s[i]===null?"착지 대기":`${fmt(s.sim_time-s.landing_time_s[i],1)}초 전`;
    });
  }
  chart("forces",feet.map((_,i)=>({color:colors[i],get:s=>s.peak_n[i]})),"N",true);
  chart("speeds",[0,1].flatMap(i=>[{color:colors[i],get:s=>s.linear_velocity[i]}, {color:colors[i],get:s=>s.command[i],dash:true}]),"m/s");
  chart("yaw",[{color:colors[2],get:s=>s.angular_velocity[2]},{color:colors[2],get:s=>s.command[2],dash:true}],"rad/s");
}

async function poll() {
  try {
    const response=await fetch(`/api/state?after=${after}`,{cache:"no-store",signal:AbortSignal.timeout(2000)});
    if(!response.ok) throw new Error("server");
    const data=await response.json();
    $("lamp").classList.toggle("live",data.connected);
    $("connection").textContent=data.connected?"연결됨":data.latest?"수신 중단":"데이터 대기 중";
    $("age").textContent=data.age_s===null?"MuJoCo 연결을 기다립니다":`마지막 수신 ${fmt(data.age_s,1)}초 전`;
    $("diagnostics").textContent=`물리 측정 ${data.latest?fmt(1/data.latest.physics_dt,0):"—"} Hz · UDP 누락 ${data.dropped} · 잘못된 패킷 ${data.invalid} · 읽기 전용`;
    $("notice").textContent=paused?"화면 표시가 일시정지되었습니다. 측정·CSV 기록·시뮬레이션은 계속 진행됩니다.":data.connected?"":data.latest?"수신이 중단되어 마지막 측정값을 표시합니다. 시뮬레이터 실행 상태를 확인하세요.":"시뮬레이터를 실행하면 자동으로 연결됩니다. 기존 시뮬레이터가 켜져 있다면 한 번 재실행하세요.";
    if(!paused) {
      const nextGeneration=JSON.stringify(data.generation);
      if(generation!==nextGeneration || (data.latest && data.latest.id<after)) {
        samples=[];generation=nextGeneration;after=0;
      }
      latest=data.latest;
      samples.push(...data.samples);
      if(data.samples.length) after=data.samples[data.samples.length-1].id;
      if(latest) samples=samples.filter(s=>s.sim_time>=latest.sim_time-60).slice(-3000);
      render();
    }
  } catch(error) {
    $("lamp").classList.remove("live"); $("connection").textContent="UI 서버 연결 끊김";
    $("notice").textContent="UI 서버와 연결할 수 없습니다. 모니터 패키지 실행 상태를 확인하세요. 마지막 표시값은 갱신되지 않습니다.";
  } finally {setTimeout(poll,100);}
}
$("pause").addEventListener("click",()=>{paused=!paused;document.body.classList.toggle("paused",paused);$("pause").textContent=paused?"실시간 표시 재개":"화면 일시정지";if(!paused){after=0;samples=[];}});
$("window").addEventListener("change",()=>{windowSeconds=Number($("window").value);render();});
window.addEventListener("resize",render);
render(); poll();
