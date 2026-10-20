/* Fixed token states with actual attention weights and short phase playback. */
(() => {
 'use strict';
 const $=id=>document.getElementById(id),F=FlowGraph,events=F.events;
 const state={ready:false,hasInk:false,trace:null,pixels:new Float32Array(784),step:0,progress:0,playing:false,frame:null,lastTime:0,example:null,sampleCounts:Array(10).fill(0),drawing:false,detailsOpen:false};
 const drawing=$('drawing'),ctx=drawing.getContext('2d',{willReadFrequently:true});
 const offscreen=document.createElement('canvas');offscreen.width=offscreen.height=28;
 const offctx=offscreen.getContext('2d',{willReadFrequently:true}),thumbnail=$('thumbnail'),thumbctx=thumbnail.getContext('2d');
 const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)');
 const profileMode=new URLSearchParams(location.search).get('profile')==='1';
 let profileStep=-1,profileFrames=[],profilePaints=[];
 function publishProfile(){if(!profileMode)return;const deltas=profileFrames.slice(2).sort((a,b)=>a-b),paints=profilePaints.slice(2).sort((a,b)=>a-b),percentile=(a,p)=>a[Math.min(a.length-1,Math.floor(a.length*p))]||0;$('viz').dataset.profile=JSON.stringify({event:state.step,phase:events[state.step].phase,frames:deltas.length,medianMs:percentile(deltas,.5),p95Ms:percentile(deltas,.95),maxMs:percentile(deltas,1),over25ms:deltas.filter(t=>t>25).length,paintP95Ms:percentile(paints,.95)});}
 let model;
 $('viz').setAttribute('viewBox',`0 0 ${F.width} ${F.height}`);$('viz').innerHTML=F.markup();
 const nodes=[...$('viz').querySelectorAll('[data-node]')].map(el=>({el,l:Number(el.dataset.nodeLayer),i:Number(el.dataset.token),bars:el.querySelector('.feature-bars'),qkv:el.querySelector('.qkv-glyph'),heads:el.querySelector('.heads-glyph'),residual:el.querySelector('.residual-glyph'),ring:el.querySelector('.token-ring')}));

 const inputTokens=[...$('viz').querySelectorAll('[data-input-token]')],tiles=[...$('viz').querySelectorAll('[data-patch]')];
 const digits=[...$('viz').querySelectorAll('[data-digit]')].map(el=>({el,body:el.querySelector('.digit-body'),halo:el.querySelector('.digit-halo'),label:el.querySelector('.digit-label')}));
 // A small Canvas overlay draws cached glow sprites; SVG geometry remains fixed.
 const particleCanvas=$('particleCanvas'),particleCtx=particleCanvas.getContext('2d');
 const connectionsCanvas=$('connectionsCanvas'),connectionsCtx=connectionsCanvas.getContext('2d');
 // Rasterize the immutable 867 attention paths once per size, outside the changing SVG.
 const connectionPaths=[...$('viz').querySelectorAll('[data-connection]')].map(el=>{const p=new Path2D(el.getAttribute('d'));el.remove();return p;});
 connectionsCanvas.dataset.connectionCount=String(connectionPaths.length);
 const nodeLayers=Array.from({length:F.layers},(_,l)=>nodes.filter(node=>node.l===l));
 const progressFill=$('progressFill'),resultSummary=document.querySelector('.result-summary');
 let featureCache=[],diagnosticsAt=-Infinity;
 const glow=document.createElement('canvas');glow.width=glow.height=64;
 const glowCtx=glow.getContext('2d'),gradient=glowCtx.createRadialGradient(32,32,0,32,32,32);
 gradient.addColorStop(0,'rgba(249,255,253,1)');gradient.addColorStop(.12,'rgba(223,255,248,1)');gradient.addColorStop(.25,'rgba(132,255,224,.85)');gradient.addColorStop(.5,'rgba(70,235,200,.22)');gradient.addColorStop(1,'rgba(70,235,200,0)');
 glowCtx.fillStyle=gradient;glowCtx.fillRect(0,0,64,64);
 function buildTrack(x0,y0,x1,y1,x2,y2,x3,y3){
  const points=[],distance=[0];let total=0;
  for(let i=0;i<=64;i++){const u=i/64,v=1-u;const p={x:v*v*v*x0+3*v*v*u*x1+3*v*u*u*x2+u*u*u*x3,y:v*v*v*y0+3*v*v*u*y1+3*v*u*u*y2+u*u*u*y3};if(i){total+=Math.hypot(p.x-points[i-1].x,p.y-points[i-1].y);distance.push(total);}points.push(p);}
  return progress=>{const desired=clamp(progress)*total;let lo=0,hi=64;while(hi-lo>1){const mid=(lo+hi)>>1;if(distance[mid]<desired)lo=mid;else hi=mid;}const a=points[lo],b=points[hi],f=(desired-distance[lo])/(distance[hi]-distance[lo]||1);return {x:a.x+(b.x-a.x)*f,y:a.y+(b.y-a.y)*f};};
 }
 const tracks=Array.from({length:F.layers},(_,layer)=>Array.from({length:F.n},(_,source)=>Array.from({length:F.n},(_,target)=>{const a=layer?F.xs[layer-1]:F.rail,b=F.xs[layer];return buildTrack(a+16,F.y(source),a+(b-a)*.42,F.y(source),b-(b-a)*.42,F.y(target),b-16,F.y(target));})));
 const outputTracks=Array.from({length:10},(_,i)=>buildTrack(F.xs[F.layers-1]+15,115,1045,115,1040,129+i*55,1107,129+i*55));
 let canvasTransform=[1,0,0,1,0,0],weightTrace=null,weightCache=[],lastSourceClasses='',lastReceiveClasses='';
 function resizeParticles(){
  const box=$('viz').getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2),scale=Math.min(box.width/F.width,box.height/F.height);
  const width=Math.ceil(box.width*ratio),height=Math.ceil(box.height*ratio);
  if(particleCanvas.width===width&&particleCanvas.height===height)return;
  canvasTransform=[scale*ratio,0,0,scale*ratio,(box.width-F.width*scale)*ratio/2,(box.height-F.height*scale)*ratio/2];
  for(const canvas of [particleCanvas,connectionsCanvas]){canvas.width=width;canvas.height=height;canvas.style.width=box.width+'px';canvas.style.height=box.height+'px';}
  connectionsCtx.setTransform(...canvasTransform);connectionsCtx.strokeStyle='#5a7490';connectionsCtx.lineWidth=.7;connectionsCtx.globalAlpha=.065;
  connectionPaths.forEach(path=>connectionsCtx.stroke(path));
  paintProgress();
 }
 new ResizeObserver(resizeParticles).observe($('viz'));
 function drawGlow(track,progress,strength){
  const p=track(progress),radius=6.5;
  particleCtx.globalAlpha=strength;particleCtx.drawImage(glow,p.x-radius,p.y-radius,radius*2,radius*2);
  return p;
 }
 function clearParticles(){particleCtx.setTransform(1,0,0,1,0,0);particleCtx.clearRect(0,0,particleCanvas.width,particleCanvas.height);particleCtx.setTransform(...canvasTransform);particleCtx.globalCompositeOperation='lighter';}
 function cacheWeights(){if(weightTrace===state.trace)return;weightTrace=state.trace;weightCache=state.trace?Array.from({length:F.layers},(_,l)=>Array.from({length:F.n},(_,s)=>{const w=Array.from({length:F.n},(_,t)=>F.meanAttention(state.trace,l,s,t)),max=Math.max(...w);return w.map(v=>v>0?.16+.84*Math.sqrt(v/max):0);})):[];}
 const navPhases=[['image','图像'],['split','切分'],['receive','入层'],['norm','归一化'],['qkv','Q/K/V'],['attention','注意力'],['merge','多头合并'],['residual','残差①'],['norm2','归一化②'],['ffn','前馈'],['residual2','残差②'],['classify','分类'],['output','结果']];
 $('phaseNav').innerHTML=navPhases.map(([p,label])=>`<button data-phase="${p}" disabled>${label}</button>`).join('');
 function setText(id,value){if($(id).textContent!==String(value))$(id).textContent=String(value);}
 function finished(){return !!state.trace&&state.step===events.length-1&&state.progress>=1;}
 function stop(){state.playing=false;cancelAnimationFrame(state.frame);state.frame=null;state.lastTime=0;}
 function mix(a,b,t){return `rgb(${a.map((v,i)=>Math.round(v+(b[i]-v)*t)).join(',')})`;}
 function clamp(t){return Math.max(0,Math.min(1,t));}
 function paintGraph(){
  const e=events[state.step],active=!!state.trace;
  $('viz').dataset.mode=active?e.phase:'waiting';
  nodes.forEach(node=>{
   const current=active&&node.l===e.layer,complete=active&&(e.layer!==undefined?node.l<e.layer:['classify','output'].includes(e.phase));
   node.el.classList.toggle('is-current',current);node.el.classList.toggle('is-complete',complete);
   node.el.classList.remove('is-sending','is-receiving');node.el.dataset.phase=current?e.phase:complete?'complete':'waiting';
   node.el.setAttribute('aria-disabled',String(!active));node.el.setAttribute('aria-current',current?'step':'false');node.ring.style.opacity='';
  });
  $('viz').querySelectorAll('[data-layer]').forEach(el=>{
   const layer=Number(el.dataset.layer),current=active&&e.layer===layer;
   el.classList.toggle('is-current',current);
   el.querySelector('.layer-step').textContent=current?e.label:active&&(e.layer>layer||['classify','output'].includes(e.phase))?'本层完成':active&&e.phase==='receive'&&layer===0?'正在接收':'等待';
  });
  document.body.classList.toggle('paused',!state.playing);
  $('phaseNav').querySelectorAll('button').forEach(b=>{b.disabled=!active;b.classList.toggle('is-selected',active&&b.dataset.phase===e.phase);b.setAttribute('aria-pressed',String(active&&b.dataset.phase===e.phase));});
  if(e.layer!==undefined&&active)$('inspectLayer').value=String(e.layer);
  inputTokens.forEach(el=>el.classList.remove('is-sending'));
  lastSourceClasses='';lastReceiveClasses='';diagnosticsAt=-Infinity;cacheWeights();cacheFeatures();
  paintProgress();
 }
 function cacheFeatures(){
  const e=events[state.step];featureCache=[];
  if(!state.trace||e.layer===undefined||!['norm','norm2','ffn'].includes(e.phase))return;
  const block=state.trace.blocks[e.layer],before=e.phase==='norm'?block.input:e.phase==='norm2'?block.residual:block.norm2,after=e.phase==='norm'?block.norm1:e.phase==='norm2'?block.norm2:block.mlp,d=F.config.dim;
  featureCache=Array.from({length:F.n},(_,i)=>[-5,0,5].map((y,k)=>{let a=0,b=0;for(let j=k;j<d;j+=3){a+=Math.abs(before[i*d+j]);b+=Math.abs(after[i*d+j]);}return {y,start:4+8*Math.tanh(a/(d/3)),end:4+8*Math.tanh(b/(d/3))};}));
 }
 function featurePath(node,t){const phase=events[state.step].phase;return featureCache[node.i].map(({y,start,end})=>{const w=phase==='ffn'?(start+(end-start)*t)*(1+.65*Math.sin(Math.PI*t)):start+(10-start)*t;return `M${-w/2},${y} H${w/2}`;}).join(' ');}
 function paintProgress(){
  const active=!!state.trace,e=events[state.step],t=state.progress;
  progressFill.style.transform=`scaleX(${active?(state.step+t)/events.length:0})`;
  if(!active||['image','split'].includes(e.phase)){
   const split=active&&e.phase==='split'?t:0;
   tiles.forEach(tile=>tile.setAttribute('transform',`translate(${(Number(tile.dataset.px)-1.5)*6*split},${(Number(tile.dataset.py)-1.5)*6*split})`));
  }else if(t===0||!state.playing)tiles.forEach(tile=>tile.setAttribute('transform',`translate(${(Number(tile.dataset.px)-1.5)*6},${(Number(tile.dataset.py)-1.5)*6})`));
  if(!active||t===0||!state.playing)inputTokens.forEach(el=>{el.style.opacity=active&&state.step>=2?'1':'.23';});
  if(active&&e.layer!==undefined&&!['attention'].includes(e.phase))nodeLayers[e.layer].forEach(node=>{
   if(['norm','norm2','ffn'].includes(e.phase))node.bars.setAttribute('d',featurePath(node,t));
   if(e.phase==='qkv')node.qkv.setAttribute('transform',`scale(${.65+.35*Math.sin(t*Math.PI/2)})`);
   if(e.phase==='merge')node.heads.setAttribute('transform',`rotate(${t*90}) scale(${1-.25*t})`);
   if(['residual','residual2'].includes(e.phase)){node.residual.children[0].setAttribute('cx',String(-7*(1-t)));node.residual.children[1].setAttribute('cx',String(7*(1-t)));}
  });
  clearParticles();
  let packetCount=0,sending=[],receiving=new Set(),latest=-1,probe=null;
  if(active&&(e.phase==='attention'||e.phase==='receive')){
   const layer=e.phase==='receive'?0:e.layer;
   {
    const source=Math.min(F.n-1,Math.floor(t*F.n)),local=clamp(F.transferProgress(t,source));latest=source;
    sending.push(source);
    for(let target=0;target<F.n;target++){
     const strength=e.phase==='receive'?(target===source?1:0):weightCache[layer][source][target];if(!strength)continue;
     const point=drawGlow(tracks[layer][source][target],F.pathProgress(local,target,source),strength);packetCount++;
     if(!probe)probe={source,target,x:point.x,y:point.y,progress:local};
     if(local>.88)receiving.add(target);
    }
   }
   const sourceClass=sending.join(',');
   if(lastSourceClasses!==sourceClass){
    const previous=lastSourceClasses?lastSourceClasses.split(',').map(Number):[],sourceNodes=layer?nodeLayers[layer-1].map(node=>node.el):inputTokens;
    previous.forEach(source=>{if(!sending.includes(source))sourceNodes[source].classList.remove('is-sending');});
    sending.forEach(source=>{if(!previous.includes(source))sourceNodes[source].classList.add('is-sending');});
    lastSourceClasses=sourceClass;
   }
   const receiveClass=[...receiving].join(',');
   if(lastReceiveClasses!==receiveClass){const previous=lastReceiveClasses?lastReceiveClasses.split(',').map(Number):[];previous.forEach(target=>{if(!receiving.has(target))nodeLayers[layer][target].el.classList.remove('is-receiving');});receiving.forEach(target=>{if(!previous.includes(target))nodeLayers[layer][target].el.classList.add('is-receiving');});lastReceiveClasses=receiveClass;}

  }
  if(active&&['classify','output'].includes(e.phase))nodes[(F.layers-1)*F.n].el.classList.add('is-sending');
  const revealing=active&&e.phase==='output',classifying=active&&e.phase==='classify';
  if(classifying)outputTracks.forEach((track,target)=>{drawGlow(track,F.pathProgress(t,target),.85);packetCount++;});
  if(!state.playing||(profileMode&&state.lastTime-diagnosticsAt>=100)){particleCanvas.dataset.packetCount=String(packetCount);particleCanvas.dataset.sources=sending.join(',');particleCanvas.dataset.probe=probe?JSON.stringify(probe):'';$('viz').dataset.source=latest<0?'':String(latest);diagnosticsAt=state.lastTime;}
  // Result styling is static during the encoder; update only for reveal or a phase change.
  if(revealing||t===0||!state.playing){
   const eased=revealing?t*t*(3-2*t):0,probabilities=state.trace?.probabilities,max=probabilities?Math.max(...probabilities):1;
   digits.forEach((digit,i)=>{
    const strength=revealing?F.outputStrength(probabilities[i],max)*eased:0;
    digit.el.dataset.brightness=strength.toFixed(5);digit.el.classList.toggle('is-winner',revealing&&i===state.trace.prediction&&t>.05);
    digit.el.setAttribute('aria-label',`数字 ${i}${revealing&&i===state.trace.prediction&&t>=1?'，预测概率最高':' 的预测亮度'}`);
    digit.body.style.fill=mix([38,47,67],[240,218,255],strength);digit.body.style.stroke=mix([62,79,101],[250,231,255],strength);
    digit.halo.style.opacity=String(strength*.65);digit.label.style.fill=strength>.55?mix([230,222,250],[42,28,65],strength):mix([141,150,177],[226,215,250],Math.min(1,strength*3.2));
   });
  }
  const done=finished();resultSummary.classList.toggle('revealed',done);
  setText('predictedDigit',done?state.trace.prediction:'—');setText('resultLabel',done?'概率最高':revealing?'概率正在点亮':active?'完成后揭晓':'等待识别');
 }
 function renderDetails(){
  const active=!!state.trace,open=active&&state.detailsOpen;
  $('detailsToggle').disabled=!active;$('detailsToggle').setAttribute('aria-expanded',String(open));
  $('detailsToggle').textContent=open?'收起过程详解 ↑':'查看过程详解 ↓';
  $('processDetails').hidden=!open;
  if(!open)return;
  const detail=F.explain(events[state.step]);
  setText('detailsTitle',detail.title);setText('detailPurpose',detail.purpose);setText('detailInput',detail.input);setText('detailOutput',detail.output);
  const steps=detail.process.map(text=>{const li=document.createElement('li');li.textContent=text;return li;});
  $('detailProcess').replaceChildren(...steps);
  $('processDetails').dataset.phase=events[state.step].phase;
  $('processDetails').dataset.layer=events[state.step].layer===undefined?'':String(events[state.step].layer);
 }
 function render(){
  const active=!!state.trace,e=events[state.step],done=finished();
  $('prevButton').disabled=!active||state.step===0;$('nextButton').disabled=!active||state.step===events.length-1;$('playButton').disabled=!active;$('restartButton').disabled=!active;$('inspectLayer').disabled=!active;
  $('playButton').textContent=state.playing?'Ⅱ':'▶';$('playButton').setAttribute('aria-label',state.playing?'暂停':done?'重新播放':'播放');
  setText('sceneBadge',!active?'等待输入':done?'预测完成':e.phase==='attention'?'注意力传输':'节点状态');
  setText('graphStatus',active?`${e.layer!==undefined?'第 '+(e.layer+1)+' 层 · ':''}${e.label||e.title}`:'3 层 × 17 个真实 token · 固定视图');
  setText('phaseIcon',!active?'○':done?'✓':e.phase==='attention'?'→':'◉');setText('explainTag',!active?'观察路线':done?'预测完成':e.layer!==undefined?`第 ${e.layer+1} 层`:'输入 / 输出');
  setText('explainTitle',active?e.title:'51 个 token 节点，始终在原位');
  setText('explainBody',active?e.body:'先观察输入切分与入层，再跟随圆点内部状态的变化。注意力阶段按从上到下的顺序，让亮点沿连线流动，最后用十个数字的亮度展示预测。');
  setText('stepLabel',active?e.title:'准备就绪');setText('stepCount',active?`${String(state.step+1).padStart(2,'0')} / ${events.length}`:'— / —');
  $('viz').dataset.event=active?String(state.step):'waiting';$('viz').dataset.phase=active?e.phase:'waiting';$('viz').dataset.duration=String(e.duration/Number($('speed').value));
  if(active)setText('inputStatus',done?'预测完成，可以重新书写':state.playing?'正在演示 · 切分 1 秒，其余过程 2 秒':'已暂停，可单步或选择过程');
  paintGraph();renderDetails();
 }
 function frame(time){
  if(!state.playing)return;
  const elapsed=Math.max(0,time-state.lastTime);state.lastTime=time;state.progress=Math.min(1,state.progress+elapsed*Number($('speed').value)/events[state.step].duration);
  if(profileMode&&profileStep!==state.step){profileStep=state.step;profileFrames=[];profilePaints=[];}
  const paintStart=profileMode?performance.now():0;paintProgress();if(profileMode){profileFrames.push(elapsed);profilePaints.push(performance.now()-paintStart);}
  if(state.progress>=1){publishProfile();if(state.step===events.length-1){stop();render();return;}state.step++;state.progress=0;render();}
  state.frame=requestAnimationFrame(frame);
 }
 function play(){state.playing=true;state.lastTime=performance.now();render();state.frame=requestAnimationFrame(frame);}
 function jump(index){if(!state.trace)return;stop();state.step=Math.max(0,Math.min(events.length-1,index));state.progress=1;render();}
 function updateImage(){const url=thumbnail.toDataURL();$('viz').querySelectorAll('.input-image').forEach(el=>el.setAttribute('href',url));}
 function invalidate(message='笔画已更新，点击开始识别'){
  stop();state.trace=null;state.step=0;state.progress=0;state.detailsOpen=false;$('inputStatus').classList.remove('error');setText('inputStatus',message);$('recognizeButton').disabled=!state.ready||!state.hasInk;updateImage();render();
 }

  function grayImage(pixels, context, ink = [70, 58, 102], bg = [247, 245, 251]) {
    const image = context.createImageData(28, 28);
    pixels.forEach((value, i) => {
      for (let c = 0; c < 3; c++) image.data[i * 4 + c] = bg[c] * (1 - value) + ink[c] * value;
      image.data[i * 4 + 3] = 255;
    });
    context.putImageData(image, 0, 0);
  }
  function preprocess() {
    // Samples are fed unchanged, preserving their MNIST test preprocessing.
    if (state.example) { state.pixels = state.example.slice(); grayImage(state.pixels, thumbctx); return state.pixels; }
    const { data } = ctx.getImageData(0, 0, 280, 280);
    let left = 280, right = -1, top = 280, bottom = -1;
    for (let y = 0; y < 280; y++) for (let x = 0; x < 280; x++) {
      if (data[(y * 280 + x) * 4 + 3] > 18) { left = Math.min(left, x); right = Math.max(right, x); top = Math.min(top, y); bottom = Math.max(bottom, y); }
    }
    state.pixels = new Float32Array(784);
    if (right < left) { grayImage(state.pixels, thumbctx); return state.pixels; }
    const w = right - left + 1, h = bottom - top + 1, scale = 20 / Math.max(w, h);
    const nw = Math.max(1, Math.round(w * scale)), nh = Math.max(1, Math.round(h * scale));
    offctx.clearRect(0, 0, 28, 28);
    offctx.imageSmoothingEnabled = true; offctx.imageSmoothingQuality = 'high';
    offctx.drawImage(drawing, left, top, w, h, Math.floor((28 - nw) / 2), Math.floor((28 - nh) / 2), nw, nh);
    const alpha = offctx.getImageData(0, 0, 28, 28).data;
    let mass = 0, sumX = 0, sumY = 0;
    for (let i = 0; i < 784; i++) { const a = alpha[i * 4 + 3] / 255; mass += a; sumX += (i % 28) * a; sumY += Math.floor(i / 28) * a; }
    const shiftX = mass ? Math.round(13.5 - sumX / mass) : 0, shiftY = mass ? Math.round(13.5 - sumY / mass) : 0;
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
      const tx = x + shiftX, ty = y + shiftY;
      if (tx >= 0 && tx < 28 && ty >= 0 && ty < 28) state.pixels[ty * 28 + tx] = alpha[(y * 28 + x) * 4 + 3] / 255;
    }
    grayImage(state.pixels, thumbctx);
    return state.pixels;
  }
  function position(event) { const r = drawing.getBoundingClientRect(); return { x: (event.clientX - r.left) * 280 / r.width, y: (event.clientY - r.top) * 280 / r.height }; }
  let last;
  drawing.addEventListener('pointerdown', event => {
    if (event.pointerType === 'mouse' && event.button !== 0) return;
    event.preventDefault(); drawing.setPointerCapture(event.pointerId);
    if (state.example) { ctx.clearRect(0, 0, 280, 280); state.example = null; }
    state.drawing = true; state.hasInk = true; last = position(event);
    ctx.lineWidth = Number($('brush').value); ctx.lineCap = ctx.lineJoin = 'round'; ctx.strokeStyle = ctx.fillStyle = '#51466e';
    ctx.beginPath(); ctx.arc(last.x, last.y, ctx.lineWidth / 2, 0, Math.PI * 2); ctx.fill();
    $('drawHint').hidden = true;
    document.querySelectorAll('.sample-button').forEach(b => b.classList.remove('active'));
    invalidate('书写完成后，点击开始识别');
  });
  drawing.addEventListener('pointermove', event => {
    if (!state.drawing) return;
    const points = event.getCoalescedEvents ? event.getCoalescedEvents() : [event];
    for (const point of points.length ? points : [event]) {
      const next = position(point); ctx.beginPath(); ctx.moveTo(last.x, last.y); ctx.lineTo(next.x, next.y); ctx.stroke(); last = next;
    }
  });
  function endStroke() { if (!state.drawing) return; state.drawing = false; preprocess(); }
  drawing.addEventListener('pointerup', endStroke); drawing.addEventListener('pointercancel', endStroke); drawing.addEventListener('lostpointercapture', endStroke);
  $('clearButton').addEventListener('click', () => {
    ctx.clearRect(0, 0, 280, 280); state.hasInk = false; state.example = null; state.drawing = false;
    $('drawHint').hidden = false; state.pixels.fill(0); grayImage(state.pixels, thumbctx);
    document.querySelectorAll('.sample-button').forEach(b => b.classList.remove('active'));
    invalidate(state.ready ? '写好后，点击开始识别' : '模型尚未就绪');
  });
  function samplePixels(example) { return Float32Array.from(atob(example.pixels), c => c.charCodeAt(0) / 255); }
  function chooseSample(digit) {
    const examples = VIT_DATA.examples.filter(e => e.label === digit), count = state.sampleCounts[digit]++;
    state.example = samplePixels(examples[count % examples.length]); state.hasInk = true; state.drawing = false;
    grayImage(state.example, offctx, [81, 70, 110], [255, 255, 255]);
    // Transparent background is needed if the next pointer stroke starts a new input.
    const image = offctx.getImageData(0, 0, 28, 28);
    for (let i = 0; i < 784; i++) { image.data[i * 4] = 81; image.data[i * 4 + 1] = 70; image.data[i * 4 + 2] = 110; image.data[i * 4 + 3] = state.example[i] * 255; }
    offctx.putImageData(image, 0, 0); ctx.clearRect(0, 0, 280, 280); ctx.imageSmoothingEnabled = true; ctx.drawImage(offscreen, 0, 0, 280, 280);
    $('drawHint').hidden = true; preprocess();
    document.querySelectorAll('.sample-button').forEach(b => b.classList.toggle('active', Number(b.dataset.sample) === digit));
    invalidate(`已载入样例 ${digit}，点击开始识别`);
  }

  $('recognizeButton').addEventListener('click',()=>{
    if(!state.ready||!state.hasInk||state.drawing)return;
    preprocess();updateImage();
    if(state.pixels.reduce((a,b)=>a+b,0)<2){setText('inputStatus','笔画太少，请写大一些');$('inputStatus').classList.add('error');return;}
    stop();state.trace=model.predict(state.pixels);state.step=0;state.progress=0;state.detailsOpen=false;$('inputStatus').classList.remove('error');
    if(matchMedia('(max-width:1050px)').matches)document.querySelector('.explorer-panel').scrollIntoView({block:'start',behavior:reducedMotion.matches?'auto':'smooth'});
    if(reducedMotion.matches){render();}else play();
  });
  $('playButton').addEventListener('click',()=>{
    if(!state.trace)return;
    if(state.playing){stop();render();return;}
    if(finished()){state.step=0;state.progress=0;}
    // A manually inspected event is shown fully; replay it from its beginning.
    if(state.progress>=1)state.progress=0;
    play();
  });
  $('prevButton').addEventListener('click',()=>jump(state.step-1));
  $('nextButton').addEventListener('click',()=>jump(state.step+1));
  $('restartButton').addEventListener('click',()=>{if(!state.trace)return;stop();state.step=0;state.progress=0;render();});
  $('viz').addEventListener('click',event=>{
    const node=event.target.closest('[data-node]');
    if(node)jump(events.findIndex(e=>e.layer===Number(node.dataset.nodeLayer)&&e.phase==='norm'));
    if(event.target.closest('[data-digit]'))jump(events.length-1);
  });
  $('viz').addEventListener('keydown',event=>{
    if((event.key==='Enter'||event.key===' ')&&event.target.matches('[data-node],[data-digit]')){event.preventDefault();event.target.dispatchEvent(new MouseEvent('click',{bubbles:true}));}
  });
  $('samples').addEventListener('click',event=>{const target=event.target.closest('[data-sample]');if(target&&state.ready)chooseSample(Number(target.dataset.sample));});
  $('detailsToggle').addEventListener('click',()=>{
    if(!state.trace)return;
    stop();state.detailsOpen=!state.detailsOpen;render();
    if(state.detailsOpen){$('processDetails').scrollIntoView({block:'start',behavior:reducedMotion.matches?'auto':'smooth'});$('detailsTitle').focus({preventScroll:true});}
  });
  $('closeDetails').addEventListener('click',()=>{
    state.detailsOpen=false;render();$('detailsToggle').focus({preventScroll:true});$('detailsToggle').scrollIntoView({block:'nearest',behavior:reducedMotion.matches?'auto':'smooth'});
  });
  $('aboutButton').addEventListener('click',()=>{stop();render();$('aboutDialog').showModal();});
  $('closeAbout').addEventListener('click',()=>$('aboutDialog').close());
  $('aboutDialog').addEventListener('click',event=>{if(event.target===$('aboutDialog')){const r=event.target.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)event.target.close();}});
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&state.playing){stop();render();}});
  try{
    if(!globalThis.VIT_DATA)throw new Error('缺少模型文件');
    model=new TinyViT(VIT_DATA);state.ready=true;setText('inputStatus','写好后，点击开始识别');
    for(let digit=0;digit<10;digit++){
      const button=document.createElement('button');button.className='sample-button';button.dataset.sample=digit;button.title=`数字 ${digit} 样例，重复点击切换笔迹`;button.setAttribute('aria-label',`载入数字 ${digit} 样例`);
      const canvas=document.createElement('canvas');canvas.width=canvas.height=28;
      grayImage(samplePixels(VIT_DATA.examples.find(e=>e.label===digit)),canvas.getContext('2d'),[113,100,145],[252,252,254]);button.append(canvas);$('samples').append(button);
    }
    const meta=VIT_DATA.metadata;
    const facts=[['结构','3 层编码器 · 4 个注意力头'],['输入','28 × 28 灰度图 · 16 个图块 + CLS'],['参数量',meta.parameters.toLocaleString()],['独立测试',`${(meta.test_accuracy*100).toFixed(2)}% · 10,000 张 MNIST 测试图像`],['演示时长','切分 1 秒 · 其余 2 秒（1×） · 传输从上到下'],['运行方式','原生 JavaScript · 无远程请求']];
    $('modelFacts').innerHTML=facts.map(([label,value])=>`<dt>${label}</dt><dd>${value}</dd>`).join('');
  }catch(error){setText('inputStatus','模型加载失败，请检查 model 文件夹');$('inputStatus').classList.add('error');console.error(error);}
  $('phaseNav').addEventListener('click',event=>{const button=event.target.closest('[data-phase]');if(!button)return;const layer=Number($('inspectLayer').value);jump(events.findIndex(e=>e.phase===button.dataset.phase&&(e.layer===undefined||e.layer===layer)));});
  $('inspectLayer').addEventListener('change',()=>{const phase=events[state.step].phase,layer=Number($('inspectLayer').value);jump(events.findIndex(e=>e.layer===layer&&e.phase===(['image','split','receive','classify','output'].includes(phase)?'norm':phase)));});
  grayImage(state.pixels,thumbctx);updateImage();render();
})();
