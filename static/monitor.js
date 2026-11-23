const $=id=>document.getElementById(id);let points=[],socket,lastUpdate=0;
function draw(){const c=$('wave'),r=c.getBoundingClientRect(),d=window.devicePixelRatio||1;c.width=r.width*d;c.height=r.height*d;const g=c.getContext('2d');g.scale(d,d);const w=r.width,h=r.height,l=34,b=30,t=12,rp=8;g.font='12px system-ui';g.lineWidth=1;g.strokeStyle='#d1dbe5';g.fillStyle='#53637a';for(let i=0;i<=6;i++){const x=l+(w-l-rp)*i/6;g.beginPath();g.moveTo(x,t);g.lineTo(x,h-b);g.stroke();g.fillText(String(-12+i*2),x-8,h-10)}for(let i=0;i<5;i++){const y=t+(h-t-b)*i/4;g.beginPath();g.moveTo(l,y);g.lineTo(w-rp,y);g.stroke();g.fillText(String(2-i),3,y+4)}g.save();g.beginPath();g.rect(l,t,w-l-rp,h-t-b);g.clip();g.strokeStyle='#008e92';g.lineWidth=2;g.beginPath();points.forEach(([x,y],i)=>{const px=l+(x+12)/12*(w-l-rp),py=t+(2-y)/4*(h-t-b);i?g.lineTo(px,py):g.moveTo(px,py)});g.stroke();g.restore()}
function stressScoreBand(score) {
  if (score < 0.10) return 'Strongly baseline-like';
  if (score < 0.26) return 'Baseline-leaning';
  if (score <= 0.36) return 'Borderline';
  if (score < 0.80) return 'Stress-leaning';
  return 'Strongly stress-like';
}

function renderStress(s={}) {
  const fresh = Number.isFinite(s.score);
  const previous = s.previous_estimate;
  const visible = fresh ? s : previous;
  const modelLabels = {
    baseline_pattern: 'Baseline-like',
    elevated_stress_pattern: 'Elevated pattern',
  };
  const modelLabel = visible && modelLabels[visible.label];
  const headline = visible ? stressScoreBand(visible.score) : '—';
  $('stress-label').textContent = !fresh && previous ? `Previous · ${headline}` : headline;
  $('stress-model-label').hidden = !modelLabel;
  $('stress-model-label').textContent = modelLabel ? `Model output · ${modelLabel}` : '';
  $('stress-score').textContent = visible ? `${fresh ? 'Class score' : 'Previous class score'} ${visible.score.toFixed(3)} · threshold ${visible.threshold.toFixed(2)}` : 'Class score —';
  $('stress-state').textContent = previous && !fresh ? `Previous estimate · ${s.status || 'Waiting'}` : (s.status || 'Waiting');
  $('stress-state').dataset.state = (s.status || '').toLowerCase();
  const seconds = Math.max(0, Math.min(64, Number(s.collected_seconds) || 0));
  $('stress-progress').value = seconds;
  const acquiring = s.status === 'Acquiring pulse';
  $('stress-progress-label').textContent = acquiring
    ? `Checking pulse · ${seconds.toFixed(1)} s buffered`
    : `${seconds >= 64 ? 'Window ready' : 'Collecting'} ${seconds.toFixed(1)} / 64 seconds`;
  $('stress-timing').textContent = fresh
    ? `Estimate age ${Number(s.age_seconds).toFixed(0)} s · next window in ${Math.ceil(s.next_update_seconds || 0)} s`
    : (previous ? `Previous estimate age ${Math.floor(s.age_seconds || 0)} s · collecting the next usable window`
      : (seconds >= 64 ? '64 seconds collected · analyzing or waiting to retry this window.' : 'First estimate after 64 usable seconds; refreshes every 30 seconds.'));
  $('stress-reason').textContent = (s.reason || 'Waiting for a usable recording')
    + (!fresh && s.reset_count && s.last_reset_reason ? ` · Last recording restart: ${s.last_reset_reason}` : '');
  const d = s.signal_diagnostics || {};
  $('stress-prv').textContent = fresh
    ? (d.prv_valid ? `Pulse variability: RMSSD ${Number(d.rmssd_ms).toFixed(1)} ms · SDNN ${Number(d.sdnn_ms).toFixed(1)} ms (experimental PRV)`
      : 'Pulse variability unavailable: beat timing did not pass its separate checks.') : '';
  $('stress-history').hidden = !previous;
  $('stress-history').textContent = previous ? 'Showing the previous measurement, not a new reading. It will update when a usable window is available.' : '';
}
function offline(){points=[];draw();$('bpm').textContent='—';$('quality').textContent='Disconnected';$('score').textContent='Score —';$('fps').textContent='— fps';$('diagnostics').textContent='Waiting for camera diagnostics';$('status').textContent='No live data. Reconnecting to the local server…';$('start').disabled=true;$('stop').disabled=true;renderStress({status:'Disconnected',reason:'Connection lost; no current stress estimate'})}
function render(s){lastUpdate=Date.now();$('source').textContent={phone:'Phone USB',camera:'V4L2 camera',demo:'Demo'}[s.source]||s.source;$('demo').textContent=s.source==='demo'?'SIMULATED · 72 BPM':'';$('bpm').textContent=s.bpm==null?'—':Math.round(s.bpm);$('quality').textContent=s.quality;$('quality').dataset.level=(s.quality||'').toLowerCase();$('score').textContent='Score '+(Number.isFinite(s.signal_score)?s.signal_score:'—');$('fps').textContent=(Number(s.fps)||0).toFixed(1)+' fps';const channel=s.channel==='auto'?(s.channel_used?'auto → '+s.channel_used:'auto RGB'):s.channel;const d=s.diagnostics||{};const parts=[];if(Number.isFinite(d.template_quality))parts.push('shape correlation '+d.template_quality.toFixed(2));if(Number.isFinite(d.brightness))parts.push('light '+Math.round(d.brightness));if(Number.isFinite(d.clipping_percent))parts.push('clipped '+d.clipping_percent+'%');if(Number.isFinite(d.motion))parts.push('motion '+d.motion.toFixed(2));if(Number.isFinite(d.peak_bpm))parts.push('beat rate '+d.peak_bpm.toFixed(1)+' BPM');if(Number.isFinite(d.support_error_bpm))parts.push('cross-check difference '+d.support_error_bpm.toFixed(1)+' BPM');$('diagnostics').textContent=parts.join(' · ')||'Collecting camera diagnostics';const reason=(!s.phone_connected&&s.capture_error)||s.reason;$('status').textContent=reason+' · '+channel+' channel'+(s.transport?' · '+s.transport+' raw stream':'')+(s.running?' · '+Math.floor(s.elapsed)+' s':'');$('start').disabled=s.running;$('stop').disabled=!s.running;points=Array.isArray(s.waveform)?s.waveform:[];draw()}
function connect(){socket=new WebSocket(`ws://${location.host}/ws`);socket.onmessage=e=>{const s=JSON.parse(e.data);render(s);renderStress(s.stress)};socket.onclose=()=>{offline();setTimeout(connect,1500)};socket.onerror=()=>socket.close()}
async function control(running){try{const r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({running})});if(!r.ok)throw Error('Control request failed');const s=await r.json();render(s);renderStress(s.stress)}catch(e){$('status').textContent=e.message}}
$('start').onclick=()=>control(true);$('stop').onclick=()=>control(false);window.addEventListener('resize',draw);setInterval(()=>{if(lastUpdate&&Date.now()-lastUpdate>2500)offline()},1000);draw();connect();
