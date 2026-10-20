/* Each circle is one real token vector. Edges are attention dependencies. */
(function(scope){
 'use strict';
 const config=scope.VIT_DATA?.config||{image_size:28,patch_size:7,layers:3,dim:48,heads:4};
 const side=config.image_size/config.patch_size,n=side*side+1,layers=config.layers;
 const xs=Array.from({length:layers},(_,i)=>445+i*250),rail=235,y=i=>115+i*33;
 const steps=[
  ['norm','归一化','整理每个 token 内部的特征尺度。内部条纹逐渐对齐；不同 token 在这一步分别处理。',2000],
  ['qkv','生成 Q / K / V','每个 token 映射出三组特征：Q 用于查询，K 用于匹配，V 携带待汇聚的信息。',2000],
  ['attention','注意力信息汇聚','发送节点从上到下依次传输，前一组到达后才切换下一组。每条连线上只有一个光点，强弱来自本次真实注意力的四头平均值。',2000],
  ['merge','合并四个注意力头','四段光环代表四个注意力头：拼接各头结果，再映射回 token 的特征空间。',2000],
  ['residual','注意力残差相加','原有特征与注意力更新合并。两条光路在节点中相遇，保留原信息并加入新的信息。',2000],
  ['norm2','前馈前归一化','再次在各 token 内整理特征。节点位置不变，条纹对齐后进入前馈网络。',2000],
  ['ffn','前馈网络','每个 token 分别经过特征扩展、GELU 激活与收缩。内部条纹展开再收回，不在 token 之间交换信息。',2000],
  ['residual2','前馈残差相加','前馈更新加回原有表示，完成这一层。两个光环合并后，保留这一层的 token 状态。',2000]
 ];
 const events=[
  {phase:'image',title:'整理输入图像',body:'笔画裁剪、缩放并居中，得到模型实际接收的灰度图像。左侧小图与本次输入一致。',duration:2000},
  {phase:'split',title:`切成 ${n-1} 个图块`,body:`沿网格切开图像，得到 ${n-1} 块局部笔画。图块的空间位置将由位置编码保留。`,duration:1000},
  {phase:'receive',title:'图块变成 token，送入第一层',body:`图块经过线性嵌入，加入位置编码与 CLS。${n} 个 token 按从上到下的顺序进入第一层；CLS 是模型的汇总向量。`,duration:2000}
 ];
 for(let layer=0;layer<layers;layer++)for(const [phase,label,body,duration] of steps)events.push({phase,layer,label,title:`第 ${layer+1} 层 · ${label}`,body,duration});
 events.push({phase:'classify',title:'CLS 汇总特征连接十个类别',body:'取最后一层的 CLS，经过最终归一化与分类映射，产生 0–9 十个分类得分。其他图块 token 不直接接分类头。',duration:2000});
 events.push({phase:'output',title:'概率化作亮度，最亮的数字是预测结果',body:'Softmax 把分类得分转成概率。数字按本次真实概率逐渐亮起，概率最高的最亮。可暂停或单步回看任意过程。',duration:2000});
 function edgePath(layer,source,target){const a=layer?xs[layer-1]:rail,b=xs[layer];return `M${a+16},${y(source)} C${a+(b-a)*.42},${y(source)} ${b-(b-a)*.42},${y(target)} ${b-16},${y(target)}`;}
 // One sender at a time: every visible edge carries exactly one packet.
 const flightFraction=1/n;
 function transferProgress(t,source){return t*n-source;}
 // Separate dots on different edges visually; every edge still starts and ends together.
 function pathProgress(t,target,source=0){const offset=(((target*7+source*3)%n)/(n-1)-.5)*.5;return t+Math.sin(Math.PI*t)*offset;}
 function markup(){
  let out='<defs><radialGradient id="nodeAura"><stop offset="0" stop-color="#e0c4ff" stop-opacity=".95"/><stop offset=".5" stop-color="#ba8df7" stop-opacity=".55"/><stop offset="1" stop-color="#aa7be9" stop-opacity="0"/></radialGradient></defs>';
  out+='<text x="103" y="55" class="graph-column-label">输入图像</text><text x="235" y="55" class="graph-column-label">输入序列</text>';
  for(let l=0;l<layers;l++)out+=`<g class="token-layer" data-layer="${l}"><rect x="${xs[l]-65}" y="29" width="130" height="653" rx="16"/><text x="${xs[l]}" y="55" class="graph-column-label">第 ${l+1} 层</text><text x="${xs[l]}" y="76" class="graph-column-sub">${n} TOKENS · ${config.dim}D</text><text x="${xs[l]}" y="666" class="layer-step">等待输入</text></g>`;
  for(let l=0;l<layers;l++)for(let s=0;s<n;s++)for(let t=0;t<n;t++)out+=`<path class="attention-base" data-connection="${l}-${s}-${t}" data-edge-layer="${l}" d="${edgePath(l,s,t)}"/>`;
  for(let i=0;i<side*side;i++){let px=i%side,py=Math.floor(i/side);out+=`<g class="patch-tile" data-patch="${i}" data-px="${px}" data-py="${py}"><svg x="${39+px*32}" y="${280+py*32}" width="32" height="32" viewBox="${px*config.patch_size} ${py*config.patch_size} ${config.patch_size} ${config.patch_size}"><image class="input-image" width="${config.image_size}" height="${config.image_size}"/></svg></g>`;}
  out+='<rect class="image-outline" x="38" y="279" width="130" height="130" rx="3"/><text x="103" y="443" class="graph-column-sub">28 × 28 → 4 × 4 图块</text><text x="103" y="463" class="graph-column-sub">图块嵌入 + 位置编码</text>';
  for(let i=0;i<n;i++){out+=`<g class="input-token" data-input-token="${i}" transform="translate(${rail},${y(i)})"><rect x="-12" y="-12" width="24" height="24" rx="5"/><text x="-25" y="4" class="token-number">${i?'P'+String(i).padStart(2,'0'):'CLS'}</text>`;if(i){const p=i-1;out+=`<svg x="-9" y="-9" width="18" height="18" viewBox="${(p%side)*7} ${Math.floor(p/side)*7} 7 7"><image class="input-image" width="28" height="28"/></svg>`;}else out+='<text class="cls-mark" y="4">C</text>';out+='</g>';}
  for(let l=0;l<layers;l++)for(let i=0;i<n;i++)out+=`<g class="token-node ${i===0?'cls-token':''}" data-node="${l}-${i}" data-token="${i}" data-node-layer="${l}" transform="translate(${xs[l]},${y(i)})" tabindex="0" role="button" aria-label="第 ${l+1} 层 ${i?'图块 '+i:'CLS 汇总'} token，点击观察"><circle class="token-halo" r="19"/><circle class="token-body" r="14"/><circle class="token-ring" r="18"/><text x="-27" y="4" class="token-number">${i?String(i).padStart(2,'0'):'CLS'}</text><g class="token-glyph"><path class="feature-bars" d="M-6,-5 H5 M-6,0 H2 M-6,5 H6"/><g class="qkv-glyph"><circle cx="-6" cy="0" r="3"/><circle cy="0" r="3"/><circle cx="6" cy="0" r="3"/></g><g class="heads-glyph">${[0,90,180,270].map(a=>`<path d="M0,-9 A9,9 0 0 1 8,-4" transform="rotate(${a})"/>`).join('')}</g><g class="residual-glyph"><circle cx="-4" r="6"/><circle cx="4" r="6"/></g></g></g>`;
  out+='<text x="1130" y="55" class="graph-column-label">预测结果</text><text x="1130" y="76" class="graph-column-sub">概率 → 亮度</text>';
  for(let i=0;i<10;i++){const yy=129+i*55;out+=`<path class="output-base" d="M${xs[layers-1]+15},115 C1045,115 1040,${yy} 1107,${yy}"/><g class="digit-node" data-digit="${i}" tabindex="0" role="button" aria-label="数字 ${i} 的预测亮度"><circle class="digit-halo" cx="1130" cy="${yy}" r="25"/><circle class="digit-body" cx="1130" cy="${yy}" r="21"/><text x="1130" y="${yy+8}" class="digit-label">${i}</text></g>`;}
  return out+'<text x="620" y="715" class="graph-annotation">圆点 = 一个 token 向量 · 连线 = 对上一状态的注意力依赖 · 亮点强弱 = 四头平均权重</text><text x="620" y="738" class="graph-annotation">逐个发送为教学动画；真实模型采用批量计算。前馈与残差在各 token 内进行。</text>';
 }
 function meanAttention(trace,layer,source,target){const heads=trace.blocks[layer].attentions;return heads.reduce((sum,a)=>sum+a[target*n+source],0)/heads.length;}
 const dim=config.dim,heads=config.heads,headDim=dim/heads,hidden=config.hidden_dim||dim*2;
 const tokenShape=`${n} 个 token，每个 ${dim} 维`;
 const details={
  image:{
   purpose:'把不同大小、不同位置的笔画整理成模型熟悉的输入，减少书写位置和尺寸带来的影响。',
   process:['找到笔画边界，裁去多余空白。','保持长宽比例，将笔画最长边缩放到 20 像素。','放入 28 × 28 灰度图，再按笔画重心居中。内置 MNIST 样例保留原有预处理。'],
   input:'画板上的单个手写数字，或选中的 MNIST 数字样例。',
   output:'一张 28 × 28 的单通道灰度图。这也是左侧“模型看到的图像”中展示的内容。'
  },
  split:{
   purpose:'把整张图像变成一组局部片段，让 Transformer 能把不同区域作为序列元素来处理。',
   process:[`沿横向和纵向各分成 ${side} 份，得到 ${n-1} 个互不重叠的图块。`,`每块包含 ${config.patch_size} × ${config.patch_size} 个像素，按行展开为像素向量。`,'按从左到右、从上到下的固定顺序排列，并在送入嵌入映射时使用训练时的像素归一化。'],
   input:'预处理后的 28 × 28 灰度图。',
   output:`${n-1} 个图块向量，每个包含 ${config.patch_size**2} 个像素特征；此时它们还不是 ${dim} 维 token。`
  },
  receive:{
   purpose:'把图块转换为模型能计算的特征向量，并补上空间位置和全图汇总入口。',
   process:[`所有图块使用同一组学到的线性映射，变成 ${dim} 维向量。`,'在序列最前面加入可学习的 CLS 汇总 token。','给所有 token 加上可学习的位置编码，保留各图块的空间身份。','将整组 token 送入第一层；逐个移动的光点用于帮助观察传入顺序。'],
   input:`${n-1} 个归一化图块向量，以及模型学到的嵌入参数、CLS 和位置编码。`,
   output:`第一层的输入：${tokenShape}。CLS 在最前，其后是按空间顺序排列的图块 token。`
  },
  norm:{
   purpose:'在注意力计算前整理每个 token 的特征尺度，使后续匹配和特征变换更稳定。',
   process:[`分别计算每个 token 内 ${dim} 个特征的均值和方差。`,'先减去均值，再除以加入稳定项后的标准差。','使用本步学到的缩放和偏置调整结果。原输入另行保留，供后面的残差相加使用。'],
   input:`本层的输入表示：${tokenShape}。`,
   output:`归一化后的 ${tokenShape}，交给 Q/K/V 映射。不同 token 在这一步独立处理，不交换信息。`
  },
  qkv:{
   purpose:'让同一份 token 分别承担“发起查询”“提供匹配线索”和“携带内容”三种角色。',
   process:['用学到的线性投影处理归一化后的 token。','将投影结果拆分为 Q（查询）、K（键）和 V（值）。',`把各组特征划分给 ${heads} 个注意力头，每头使用 ${headDim} 维特征；此时还没有汇聚其他 token 的内容。`],
   input:`本层第一次归一化的输出：${tokenShape}。`,
   output:`Q、K、V 三组向量，每组都有 ${tokenShape}；随后分别进入 ${heads} 个注意力头。`
  },
  attention:{
   purpose:'让每个位置依据当前内容选择需要参考的其他位置，把局部笔画和全图信息联系起来。',
   process:[`每个头计算所有 Q 与所有 K 的匹配，再按每头特征维度的平方根缩放。`,'对每个接收 token 的匹配结果做 Softmax，得到它接收各发送 token 的权重。','按照权重对 V 加权求和，得到每个接收 token 的新内容。',`真实模型分别计算 ${heads} 个头。图中将四头平均权重用于光点强弱，逐个发送是教学展示顺序。`],
   input:`本层的 Q、K、V；每个注意力头处理 ${n} 个 ${headDim} 维向量。`,
   output:`${heads} 组注意力头结果，每组为 ${n} 个 ${headDim} 维向量；同时得到每头 ${n} × ${n} 的注意力权重。`
  },
  merge:{
   purpose:'把不同注意力头关注到的信息整合为每个 token 的一次完整更新。',
   process:[`对同一个 token，沿特征维度拼接 ${heads} 个头的结果。`,`拼接后恢复为 ${dim} 维，再经过学到的输出线性映射。`,'将映射结果交给注意力残差连接。'],
   input:`${heads} 个注意力头的结果，每个头对各 token 产生 ${headDim} 维内容。`,
   output:`注意力更新量：${tokenShape}。它还需要与本层原输入相加。`
  },
  residual:{
   purpose:'保留本层原有表示，同时加入注意力带来的更新，让信息沿主路径继续流动。',
   process:['取出本层第一次归一化之前保留的原输入。','把多头合并后的注意力更新，与原输入按相同 token、相同特征位置逐项相加。','将相加后的表示交给第二次归一化，并保留它供下一次残差使用。'],
   input:`本层原输入和注意力更新量；两者都是${tokenShape}。`,
   output:`加入注意力更新后的 ${tokenShape}。这里相加的是原输入与注意力更新，不是两个归一化结果。`
  },
  norm2:{
   purpose:'在进入前馈网络前，再次整理 token 内的特征尺度。',
   process:['对注意力残差相加后的各 token，分别计算特征均值和方差。','完成标准化，再应用本步独立学到的缩放和偏置。','把结果送入前馈网络；归一化前的表示继续保留在残差主路径上。'],
   input:`第一次残差相加的输出：${tokenShape}。`,
   output:`前馈网络的输入：${tokenShape}。这一步仍在每个 token 内独立进行。`
  },
  ffn:{
   purpose:'用非线性变换进一步加工每个 token 已汇聚的特征，组合出更有用的表示。',
   process:[`第一层线性映射将特征从 ${dim} 维扩展到 ${hidden} 维。`,'经过 GELU 激活，对特征进行非线性调整。',`第二层线性映射将特征从 ${hidden} 维收缩回 ${dim} 维。`,'所有 token 使用同一套网络参数，但分别计算；这一步不在 token 之间传输数据。'],
   input:`第二次归一化的输出：${tokenShape}。`,
   output:`前馈更新量：${tokenShape}。中间扩展的特征维度已收缩，随后进入第二次残差相加。`
  },
  residual2:{
   purpose:'把前馈网络产生的更新加入主路径，完成这一层编码器。',
   process:['取出注意力残差相加后保留的表示。','将前馈更新与它按相同 token、相同特征位置逐项相加。','把完整的本层输出送往下一层；最后一层则交给分类头。'],
   input:`注意力残差相加的输出，以及前馈网络的更新量；两者都是${tokenShape}。`,
   output:`本层最终的 ${tokenShape}。其中 CLS 也经过了同样的更新，持续汇总数字相关信息。`
  },
  classify:{
   purpose:'把经过三层加工的全图汇总特征，转换成对十个数字类别的判断依据。',
   process:['对最后一层的 token 做最终归一化。','取序列最前面的 CLS 汇总 token；图块 token 不直接连接分类头。',`通过学到的分类线性映射，把 CLS 的 ${dim} 维特征映射成十个类别得分。`],
   input:`最后一层输出的 ${tokenShape}，以及最终归一化与分类映射参数。`,
   output:'对应数字 0–9 的十个分类得分（logits）。此时它们还不是概率。'
  },
  output:{
   purpose:'把分类得分转成可比较的预测概率，并用数字亮度直观展示模型的倾向。',
   process:['为数值稳定，先从各类别得分中减去最高得分。','对得分取指数，再除以所有指数值之和，得到总和为 1 的概率。','使用单调映射把概率转成数字亮度，概率越高越亮。','选择概率最高的数字作为本次预测。'],
   input:'分类映射产生的十个类别得分（logits）。',
   output:'数字 0–9 的十个预测概率、对应的亮度，以及概率最高的预测数字。亮度展示相对倾向，不显示具体概率值。'
  }
 };
 function explain(event){const detail=details[event.phase];return {...detail,title:event.title};}
 const api={config,n,layers,xs,rail,y,events,markup,edgePath,meanAttention,transferProgress,pathProgress,explain,flightFraction,outputStrength:(p,max)=>.06+.94*Math.sqrt(Math.max(0,p)/Math.max(max,1e-12)),width:1200,height:760};
 scope.FlowGraph=api;if(typeof module!=='undefined')module.exports=api;
})(globalThis);
