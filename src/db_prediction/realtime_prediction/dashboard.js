const $ = id => document.getElementById(id);
const colors = { measured: '#65dfbd', predicted: '#ffbc79' };
let points = [], frozenPoints = [], cursor = 0, snapshot = null, connected = false, paused = false;
let lastResponse = 0, frozenNow = 0, clockOffset = 0;
const format = value => Number.isFinite(value) ? value.toFixed(1) : '—';

function serverNow() { return performance.now()/1000 + clockOffset; }
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
  $('horizon').textContent = robotFresh ? format(status.remaining_horizon_ms) : '—';
  $('audioState').textContent = !status.audio_enabled ? 'Disabled · Start with --audio-host' : audioFresh ? (measured?.clipping ? 'PCM stream active · Clipping detected' : 'PCM stream active · Estimated audio clock') : 'Audio unavailable or stale';
  $('robotState').textContent = robotFresh ? `LowState active · Source age ${format(status.low_age_ms)} ms` : 'LowState unavailable or stale';
  $('connection').textContent = !fresh ? 'Disconnected' : status.simulated ? 'Simulated preview' : robotFresh && audioFresh ? 'Live · Both inputs' : 'Waiting for inputs';
  $('connection').className = `badge ${fresh && robotFresh && audioFresh ? 'live' : 'stale'}`;
  $('demo').hidden = !status.simulated;
  $('model').textContent = status.model || '—';
  $('count').textContent = Number(status.predictions || 0).toLocaleString();
  $('message').textContent = !fresh ? 'No response from the prediction process. The page will reconnect automatically.' : status.audio_error || status.last_error || (!status.audio_enabled ? 'Measured noise requires --audio-host. Predictions continue using LowState only.' : 'Live comparison uses approximate audio timestamps. Match microphone gain and placement to the training recordings.');
}

function drawChart() {
  const canvas = $('chart'), rect = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
  const width = Math.max(260, rect.width), height = rect.height;
  if (canvas.width !== Math.round(width*ratio) || canvas.height !== Math.round(height*ratio)) {
    canvas.width = Math.round(width*ratio); canvas.height = Math.round(height*ratio);
  }
  const ctx = canvas.getContext('2d'); ctx.setTransform(ratio,0,0,ratio,0,0); ctx.clearRect(0,0,width,height);
  const now = paused ? frozenNow : serverNow(), span = Number($('window').value), left=56, right=width-16, top=26, bottom=height-32;
  const visible = (paused ? frozenPoints : points).filter(p => p.time >= now-span && p.time <= now+.05);
  const values = visible.map(p=>p.value);
  let low = Math.floor(Math.min(...values,-45)/10)*10, high = Math.ceil(Math.max(...values,-15)/10)*10;
  if (high-low < 20) high = low+20;
  const step = Math.max(5, Math.ceil((high-low)/60)*10);
  const x = t => left+(t-now+span)/span*(right-left), y = db => bottom-(db-low)/(high-low)*(bottom-top);
  ctx.font='11px system-ui'; ctx.fillStyle='#8fa4bd'; ctx.strokeStyle='#25354a'; ctx.lineWidth=1;
  ctx.fillText('dBFS',8,12);
  for (let db=low; db<=high; db+=step) {
    ctx.beginPath(); ctx.moveTo(left,y(db)); ctx.lineTo(right,y(db)); ctx.stroke(); ctx.fillText(String(db),10,y(db)+4);
  }
  for (let i=0;i<=4;i++) {
    const t=now-span+i*span/4, px=x(t);
    ctx.beginPath(); ctx.moveTo(px,top); ctx.lineTo(px,bottom); ctx.stroke();
    ctx.fillText(i===4?'Now':`−${format(span-i*span/4)}s`, Math.min(px,right-25),height-9);
  }
  ctx.save(); ctx.beginPath(); ctx.rect(left,top,right-left,bottom-top); ctx.clip();
  for (const kind of ['measured','predicted']) {
    const line = visible.filter(p=>p.kind===kind).sort((a,b)=>a.time-b.time);
    ctx.strokeStyle=colors[kind]; ctx.lineWidth=1.8; ctx.beginPath(); let previous;
    for (const p of line) {
      if (!previous || p.time-previous.time > .12) ctx.moveTo(x(p.time),y(p.value)); else ctx.lineTo(x(p.time),y(p.value));
      previous=p;
    }
    ctx.stroke();
    if (line.length) { const last=line.at(-1); ctx.fillStyle=colors[kind];ctx.beginPath();ctx.arc(x(last.time),y(last.value),3,0,Math.PI*2);ctx.fill(); }
  }
  ctx.restore();
  if (!visible.length) { ctx.fillStyle='#8fa4bd';ctx.textAlign='center';ctx.fillText('Waiting for measured or predicted noise…',(left+right)/2,height/2);ctx.textAlign='left'; }
  $('viewState').textContent = `${paused?'View paused · Data collection continues':'Live view'} · ${span} s window · ${visible.filter(p=>p.kind==='measured').length} measured / ${visible.filter(p=>p.kind==='predicted').length} predicted samples`;
}

async function poll() {
  const controller = new AbortController(), timeout = setTimeout(()=>controller.abort(),2000);
  try {
    const response = await fetch(`/api/state?since=${cursor}`, {cache:'no-store', signal:controller.signal});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (data.reset) points=[];
    snapshot=data; cursor=data.cursor; clockOffset=data.now-performance.now()/1000;
    points.push(...data.points.filter(p=>Number.isFinite(p.time) && Number.isFinite(p.value) && p.kind in colors));
    points=points.filter(p=>p.time>=data.now-60).slice(-8000);
    connected=true;lastResponse=performance.now();
  } catch { connected=false; }
  finally { clearTimeout(timeout);updateCards(); if(!paused)drawChart();setTimeout(poll,200); }
}
$('pause').onclick = () => {
  paused=!paused;
  if(paused) { frozenNow=serverNow();frozenPoints=points.slice(); }
  else frozenPoints=[];
  $('pause').textContent=paused?'Resume live':'Pause view';drawChart();
};
$('window').onchange=drawChart;
$('export').onclick = () => {
  const now=paused?frozenNow:serverNow(), span=Number($('window').value);
  const visible=(paused?frozenPoints:points).filter(p=>p.time>=now-span && p.time<=now+.05).sort((a,b)=>a.time-b.time);
  const csv='kind,time_monotonic_s,noise_dbfs\n'+visible.map(p=>`${p.kind},${p.time},${p.value}`).join('\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv'})), anchor=document.createElement('a');
  anchor.href=url;anchor.download='robot_noise_visible.csv';anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
window.addEventListener('resize',drawChart);
setInterval(updateCards,500);
drawChart();poll();
