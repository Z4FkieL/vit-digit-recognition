// Verify model-derived token counts, dense attention direction, and phase timing.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..');
fs.mkdirSync(path.join(root, 'work'), { recursive: true });require(path.join(root,'model/model-data.js'));
const F=require(path.join(root,'flow.js')),{TinyViT}=require(path.join(root,'engine.js'));
const model=new TinyViT(VIT_DATA),n=(VIT_DATA.config.image_size/VIT_DATA.config.patch_size)**2+1;
assert.equal(F.n,n);assert.equal(F.layers,VIT_DATA.config.layers);
const markup=F.markup();
assert.equal([...markup.matchAll(/data-node="/g)].length,n*F.layers);
assert.equal([...markup.matchAll(/data-input-token="/g)].length,n);
assert.equal([...markup.matchAll(/data-connection="/g)].length,n*n*F.layers);
assert.equal([...markup.matchAll(/data-digit="/g)].length,VIT_DATA.config.classes);
for(let l=0;l<F.layers;l++){
 assert.deepEqual(F.events.filter(e=>e.layer===l).map(e=>e.phase),['norm','qkv','attention','merge','residual','norm2','ffn','residual2']);
 for(let s=0;s<n;s++)for(let t=0;t<n;t++)assert.ok(markup.includes(`data-connection="${l}-${s}-${t}"`));
}
assert.ok(F.events.every(e=>e.duration===(e.phase==='split'?1000:2000)));
for(const event of F.events){
 const detail=F.explain(event);
 assert.equal(detail.title,event.title);
 for(const field of ['purpose','input','output'])assert.ok(typeof detail[field]==='string'&&detail[field].length>8,`Missing ${field}: ${event.phase}`);
 assert.ok(detail.process.length>=3&&detail.process.every(step=>step.length>8),`Missing process: ${event.phase}`);
}
const sweepMs=F.events.find(e=>e.phase==='attention').duration;
assert.equal(sweepMs,2000);
const launches=Array.from({length:n},(_,s)=>s/n);
for(let s=0;s<n;s++){
 assert.ok(Math.abs(F.transferProgress(launches[s],s))<1e-10);
 assert.ok(Math.abs(F.transferProgress(launches[s]+F.flightFraction,s)-1)<1e-10);
}
for(let frame=0;frame<2000;frame++){
 const progress=frame/2000,source=Math.min(n-1,Math.floor(progress*n));
 const local=F.transferProgress(progress,source);
 assert.ok(local>=0&&local<=1);
 if(source)assert.ok(F.transferProgress(progress,source-1)>=1-1e-12);
}
let examples=0,attentionChecks=0;
for(let source=0;source<n;source++)for(let target=0;target<n;target++){
 assert.equal(F.pathProgress(0,target,source),0);assert.ok(Math.abs(F.pathProgress(1,target,source)-1)<1e-12);
 let previous=-1;for(let i=0;i<=100;i++){const p=F.pathProgress(i/100,target,source);assert.ok(p>=0&&p<=1+1e-12&&p>previous,'Every single light must move forward without reversing');previous=p;}
}
for(const example of VIT_DATA.examples){
 const trace=model.predict(Float32Array.from(Buffer.from(example.pixels,'base64'),v=>v/255));
 const strengths=Array.from(trace.probabilities,p=>F.outputStrength(p,Math.max(...trace.probabilities)));
 assert.equal(strengths.indexOf(Math.max(...strengths)),trace.prediction);
 for(let i=0;i<10;i++)for(let j=0;j<10;j++)if(trace.probabilities[i]>trace.probabilities[j])assert.ok(strengths[i]>strengths[j]);
 for(let l=0;l<F.layers;l++)for(let target=0;target<n;target++){
  let sum=0;
  for(let source=0;source<n;source++){
   // Attention is indexed [query/receiver, key/value/sender], never transposed.
   const expected=trace.blocks[l].attentions.reduce((v,head)=>v+head[target*n+source],0)/VIT_DATA.config.heads;
   const actual=F.meanAttention(trace,l,source,target);assert.equal(actual,expected);sum+=actual;attentionChecks++;
  }
  assert.ok(Math.abs(sum-1)<1e-5);
 }
 examples++;
}
const report={passed:true,tokenNodes:n*F.layers,tokensPerLayer:n,digitNodes:10,denseConnections:F.layers*n*n,events:F.events.length,splitMs:1000,maxPhaseMs:Math.max(...F.events.map(e=>e.duration)),sweepMs,launchIntervalMs:(launches[1]-launches[0])*sweepMs,packetFlightMs:F.flightFraction*sweepMs,totalDemoMs:F.events.reduce((s,e)=>s+e.duration,0),attentionChecks,examples};
fs.writeFileSync(path.join(root,'work/flow-verification.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
