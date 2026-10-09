/* ============================================================================
   APP.JS — KNN INTERACTIVO 2D/3D + EJEMPLO IRIS

   LEYENDA DEL CÓDIGO:
   A. Datos, escalas y estado
   B. Matemáticas de KNN (idéntico algoritmo en 2D y 3D)
   C. Creación de ejemplos y datos reales Iris
   D. Dibujo 2D: regiones, frontera y vecinos
   E. Dibujo 3D: proyección y frontera (corte o superficie voxel)
   F. Explicación en 3 pasos y controles
   G. Eventos del ratón/teclado, inicialización

   Sin dependencias: utiliza Canvas 2D y JavaScript estándar.
   IMPORTANTE: la vista 3D representa coordenadas x,y,z de verdad.
   La distancia usa también z (no es un mero efecto visual).
   ============================================================================ */
(() => {
  'use strict';

  // A1. Configuración general y nombres de las clases.
  const $ = (id) => document.getElementById(id);
  const COLORS = ['#3677f6','#ef755d','#9066dd','#10a886','#d69a28','#d6539e'];
  const FREE_NAMES = ['Azul','Coral','Violeta','Verde','Ámbar','Rosa'];
  const IRIS_NAMES = ['Iris setosa','Iris versicolor','Iris virginica'];
  const CENTER = {
    2: [[22,48,30],[78,52,75]],
    3: [[22,29,27],[76,30,75],[49,79,51]],
    4: [[22,22,24],[77,23,74],[23,78,76],[78,77,25]],
    5: [[20,21,22],[80,22,77],[22,80,76],[79,80,25],[51,49,51]],
    6: [[18,21,21],[81,22,78],[20,79,79],[81,78,22],[49,48,30],[54,49,78]]
  };

  // A2. Filas reales del dataset Iris (15 observaciones por especie = 45).
  // Cada fila = [longitud_pétalo_cm, anchura_pétalo_cm, longitud_sépalo_cm, especie].
  // El valor de la especie es 0=setosa, 1=versicolor, 2=virginica.
  // Extraído de sklearn.datasets.load_iris() (datos clásicos de Fisher/Anderson).
  const IRIS_ROWS = [[1.4,0.2,5.1,0],[1.5,0.2,4.6,0],[1.4,0.3,4.6,0],[1.5,0.1,4.9,0],[1.4,0.1,4.8,0],[1.5,0.4,5.7,0],[1.7,0.3,5.7,0],[1.5,0.4,5.1,0],[1.9,0.2,4.8,0],[1.5,0.2,5.2,0],[1.6,0.2,4.8,0],[1.4,0.2,5.5,0],[1.3,0.2,5.5,0],[1.5,0.2,5.1,0],[1.3,0.2,4.4,0],[4.7,1.4,7.0,1],[4.0,1.3,5.5,1],[4.7,1.6,6.3,1],[3.9,1.4,5.2,1],[4.0,1.0,6.0,1],[4.4,1.4,6.7,1],[4.5,1.5,6.2,1],[4.0,1.3,6.1,1],[4.3,1.3,6.4,1],[5.0,1.7,6.7,1],[3.8,1.1,5.5,1],[5.1,1.6,6.0,1],[4.7,1.5,6.7,1],[4.0,1.3,5.5,1],[4.0,1.2,5.8,1],[6.0,2.5,6.3,2],[5.6,1.8,6.3,2],[4.5,1.7,4.9,2],[6.1,2.5,7.2,2],[5.5,2.1,6.8,2],[5.3,2.3,6.4,2],[6.9,2.3,7.7,2],[4.9,2.0,5.6,2],[5.7,2.1,6.7,2],[4.9,1.8,6.1,2],[6.1,1.9,7.4,2],[5.1,1.5,6.3,2],[5.6,2.4,6.3,2],[5.4,2.1,6.9,2],[5.1,1.9,5.8,2]];
  const IRIS_RANGES = [[1,6.9],[.1,2.5],[4.3,7.9]];
  const clamp = (v,lo,hi) => Math.max(lo,Math.min(hi,v));
  const normalize = (value,i) => (value-IRIS_RANGES[i][0])*100/(IRIS_RANGES[i][1]-IRIS_RANGES[i][0]);
  const denormalize = (value,i) => IRIS_RANGES[i][0]+value*(IRIS_RANGES[i][1]-IRIS_RANGES[i][0])/100;
  const hexRGBA = (hex,alpha) => {
    const m = hex.match(/\w\w/g); return `rgba(${parseInt(m[0],16)},${parseInt(m[1],16)},${parseInt(m[2],16)},${alpha})`;
  };
  const seedRand = (seed) => { let x = seed >>> 0; return () => ((x = (1664525*x+1013904223)>>>0)/4294967296); };

  // A3. Estado: fuente única de verdad para todas las interacciones.
  const state = {
    scenario:'free', mode:'classification', dims:2, classCount:3,
    k:5, points:[], query:{x:52,y:49,z:50}, step:0,
    showAreas:true, showBorders:true, boundary3D:'slice', sliceZ:50,
    tool:'move', selectedClass:0, seed:3, hovered:-1, dragging:null,
    yaw:-.74, pitch:.43, zoom:1
  };
  const canvas=$('knnCanvas'), ctx=canvas.getContext('2d');
  const screen={w:1,h:1,dpr:1};
  let twoDCache=null, volumeCache=null, sliceCache=null, raf=false;
  const names=()=>state.scenario==='iris'?IRIS_NAMES:FREE_NAMES;
  function invalidate(){twoDCache=null;volumeCache=null;sliceCache=null;}

  // B1. Distancia euclídea entre dos vectores; sólo usa z cuando se elige 3D.
  // En Iris los valores se han transformado con min-max (0...100) para evitar
  // que una variable domine por sus unidades o por su rango.
  function distance(a,b) {
    const dx=a.x-b.x,dy=a.y-b.y;
    return Math.sqrt(dx*dx+dy*dy+(state.dims===3?(a.z-b.z)**2:0));
  }
  // B2. Ordenamos ejemplos por cercanía (y por índice para desempates estables).
  function sortedNeighbors(query){
    return state.points.map((p,index)=>({point:p,index,d:distance(query,p)}))
      .sort((a,b)=>a.d-b.d||a.index-b.index);
  }
  // B3. Predicción: mayoría de votos (clasificación) o promedio (regresión).
  // Empate en votos => menor suma de distancias; empate restante => menor índice.
  function predict(query){
    const nearest=sortedNeighbors(query).slice(0,Math.min(state.k,state.points.length));
    if(!nearest.length) return {nearest,winner:0,value:0,votes:[]};
    if(state.mode==='regression'){
      const value=nearest.reduce((s,n)=>s+n.point.v,0)/nearest.length;
      return {nearest,value};
    }
    const votes=Array(state.classCount).fill(0),distances=Array(state.classCount).fill(0);
    for(const n of nearest){votes[n.point.c]++;distances[n.point.c]+=n.d;}
    let winner=0;
    for(let i=1;i<state.classCount;i++){
      if(votes[i]>votes[winner]||(votes[i]===votes[winner]&&distances[i]<distances[winner])) winner=i;
    }
    return {nearest,winner,votes,distances};
  }

  // C1. Datos SINTÉTICOS didácticos para el laboratorio libre.
  // Los grupos se generan en x,y,z (cada clase tiene siete muestras).
  function makeFree(){
    const rand=seedRand(state.seed+state.classCount*2371);
    state.points=[];
    CENTER[state.classCount].forEach(([x,y,z],c)=>{
      for(let i=0;i<7;i++){
        const theta=(i/7)*Math.PI*2+rand()*.5, r=5+rand()*14;
        const px=clamp(x+Math.cos(theta)*r,3,97),py=clamp(y+Math.sin(theta)*r,3,97);
        const pz=clamp(z+(rand()-.5)*26,3,97);
        const v=Math.round(clamp(1+px*.043+py*.031+pz*.012+(rand()-.5),1,10)*10)/10;
        state.points.push({x:px,y:py,z:pz,c,v,custom:false});
      }
    });
    invalidate();
  }
  // C2. Caso Iris: convertimos medidas físicas a coordenadas comparables.
  function makeIris(){
    state.points=IRIS_ROWS.map(([pl,pw,sl,c])=>({
      x:normalize(pl,0),y:normalize(pw,1),z:normalize(sl,2),c,
      raw:[pl,pw,sl],v:0,custom:false
    }));
    invalidate();
  }
  function resetQuery(){
    state.query=state.scenario==='iris'
      ?{x:normalize(4.7,0),y:normalize(1.4,1),z:normalize(6.1,2)}
      :{x:52,y:49,z:50};
    state.sliceZ=Math.round(state.query.z);
  }

  // D1. Correspondencia entre coordenadas matemáticas y píxeles (2D).
  function rect(){return {x:48,y:20,w:Math.max(10,screen.w-75),h:Math.max(10,screen.h-77)};}
  function to2D(p){let a=rect();return {x:a.x+p.x*a.w/100,y:a.y+(100-p.y)*a.h/100};}
  function from2D(px,py){let a=rect();return {x:clamp((px-a.x)*100/a.w,0,100),y:clamp(100-(py-a.y)*100/a.h,0,100)};}
  function line(a,b,color,width=1,dash=[]){ctx.save();ctx.beginPath();ctx.setLineDash(dash);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();ctx.restore();}
  function polygon(p,color,stroke=null,sw=1){ctx.beginPath();ctx.moveTo(p[0].x,p[0].y);for(let i=1;i<p.length;i++)ctx.lineTo(p[i].x,p[i].y);ctx.closePath();if(color){ctx.fillStyle=color;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=sw;ctx.stroke();}}
  function regressionColor(v,alpha=1){let t=clamp((v-1)/9,0,1),r=Math.round(17+120*t),g=Math.round(168-69*t),b=Math.round(143+83*t);return `rgba(${r},${g},${b},${alpha})`;}

  // D2. Fronteras en 2D = cambios de etiqueta en una cuadrícula.
  // Esta técnica se basa en calcular KNN en cada celda, no en dibujar curvas a mano.
  function build2DMap(){
    if(twoDCache) return twoDCache;
    const a=rect(),step=6,nx=Math.ceil(a.w/step),ny=Math.ceil(a.h/step);
    const surface=document.createElement('canvas');surface.width=Math.ceil(a.w);surface.height=Math.ceil(a.h);
    const g=surface.getContext('2d'), labels=[];
    for(let row=0;row<ny;row++)for(let col=0;col<nx;col++){
      const q={x:clamp((col+.5)*step*100/a.w,0,100),y:clamp(100-(row+.5)*step*100/a.h,0,100),z:state.query.z};
      const result=predict(q), v=state.mode==='classification'?result.winner:result.value;
      labels[row*nx+col]=v;
      if(state.showAreas){g.fillStyle=state.mode==='classification'?hexRGBA(COLORS[v],.16):regressionColor(v,.16);g.fillRect(col*step,row*step,step+.5,step+.5);}
    }
    if(state.showBorders&&state.mode==='classification'){
      g.beginPath();for(let row=0;row<ny;row++)for(let col=0;col<nx;col++){
        let i=row*nx+col,x=(col+1)*step,y=(row+1)*step;
        if(col<nx-1&&labels[i]!==labels[i+1]){g.moveTo(x,row*step);g.lineTo(x,Math.min(y,a.h));}
        if(row<ny-1&&labels[i]!==labels[i+nx]){g.moveTo(col*step,y);g.lineTo(Math.min(x,a.w),y);}
      }
      g.strokeStyle='rgba(255,255,255,.92)';g.lineWidth=4;g.stroke();g.strokeStyle='#263957';g.lineWidth=1.7;g.stroke();
    }
    twoDCache=surface;return surface;
  }
  function draw2DAxes(){
    const a=rect();ctx.save();ctx.strokeStyle='#e9eef7';ctx.fillStyle='#8897b1';ctx.font='11px system-ui';ctx.lineWidth=1;
    for(let t=0;t<=100;t+=20){let x=a.x+t*a.w/100,y=a.y+(100-t)*a.h/100;
      ctx.beginPath();ctx.moveTo(x,a.y);ctx.lineTo(x,a.y+a.h);ctx.moveTo(a.x,y);ctx.lineTo(a.x+a.w,y);ctx.stroke();
      if(t%40===0||t===100){ctx.textAlign='center';ctx.fillText(state.scenario==='iris'?denormalize(t,0).toFixed(1):t,x,a.y+a.h+17);ctx.textAlign='right';ctx.fillText(state.scenario==='iris'?denormalize(t,1).toFixed(1):t,a.x-10,y+4);}
    }
    ctx.strokeStyle='#cfdcec';ctx.strokeRect(a.x,a.y,a.w,a.h);
    ctx.fillStyle='#7185a3';ctx.font='bold 11px system-ui';ctx.textAlign='center';
    ctx.fillText(state.scenario==='iris'?'Longitud pétalo (cm) →':'Característica X →',a.x+a.w/2,a.y+a.h+40);
    ctx.save();ctx.translate(13,a.y+a.h/2);ctx.rotate(-Math.PI/2);ctx.fillText(state.scenario==='iris'?'Anchura pétalo (cm) →':'Característica Y →',0,0);ctx.restore();ctx.restore();
  }
  function drawPoint(p,color,selected=false,isQuery=false,label=''){
    ctx.save();let r=isQuery?15:8.7;
    if(selected){ctx.beginPath();ctx.arc(p.x,p.y,r+7,0,Math.PI*2);ctx.fillStyle=hexRGBA(color,.16);ctx.fill();}
    ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fillStyle=isQuery?(state.step===3?color:'#fff'):color;ctx.strokeStyle=isQuery?color:'#fff';ctx.lineWidth=isQuery?3:2.4;ctx.shadowColor='#12345c33';ctx.shadowBlur=6;ctx.fill();ctx.shadowBlur=0;ctx.stroke();
    if(isQuery){ctx.fillStyle=state.step===3?'#fff':'#273a5a';ctx.font='900 15px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(state.step===3?'✓':'?',p.x,p.y+.5);ctx.fillStyle='#526582';ctx.font='800 10px system-ui';ctx.fillText('NUEVO',p.x,p.y-29);}
    else if(label){ctx.fillStyle='#fff';ctx.font='bold 9px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(label,p.x,p.y);}
    ctx.restore();
  }
  function draw2D(){
    const a=rect(),q=to2D(state.query),neighbors=sortedNeighbors(state.query),pred=predict(state.query);
    if(state.showAreas||(state.showBorders&&state.mode==='classification')){
      ctx.save();ctx.beginPath();ctx.rect(a.x,a.y,a.w,a.h);ctx.clip();ctx.drawImage(build2DMap(),a.x,a.y);ctx.restore();
    }
    draw2DAxes();
    if(state.step===1)neighbors.forEach(n=>line(q,to2D(n.point),'#9cadc977',1,[4,5]));
    if(state.step>=2)pred.nearest.forEach(n=>line(q,to2D(n.point),state.mode==='classification'?hexRGBA(COLORS[n.point.c],.76):'#6589af',2.3));
    const selection=new Set(pred.nearest.map(n=>n.index));
    state.points.forEach((p,i)=>drawPoint(to2D(p),state.mode==='classification'?COLORS[p.c]:regressionColor(p.v),state.step>=2&&selection.has(i),false,state.mode==='regression'?p.v.toFixed(0):''));
    drawPoint(q,state.step===3?(state.mode==='classification'?COLORS[pred.winner]:'#10a886'):'#2b3d5b',true,true);
  }

  // E1. Proyección 3D manual: rotamos el espacio con yaw y pitch y proyectamos
  // sobre el mismo canvas. "depth" permite ordenar objetos del fondo al frente.
  function project(p){
    let x=p.x-50,y=p.y-50,z=p.z-50,cy=Math.cos(state.yaw),sy=Math.sin(state.yaw);
    const xx=x*cy+z*sy,zz=-x*sy+z*cy,cp=Math.cos(state.pitch),sp=Math.sin(state.pitch);
    const yy=y*cp+zz*sp,depth=-y*sp+zz*cp;
    const scale=Math.min(screen.w*.0057,screen.h*.0064)*state.zoom;
    return {x:screen.w*.50+xx*scale,y:screen.h*.51-yy*scale,depth};
  }
  function corners(){return [[0,0,0],[100,0,0],[0,100,0],[100,100,0],[0,0,100],[100,0,100],[0,100,100],[100,100,100]].map(([x,y,z])=>project({x,y,z}));}
  function draw3DBox(){
    const c=corners(),edge=[[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];
    for(let [a,b] of edge)line(c[a],c[b],'#bfcde299',1.3,[4,5]);
    const origin=project({x:0,y:0,z:0});
    const ends=[['X',{x:112,y:0,z:0},'#527be1'],['Y',{x:0,y:112,z:0},'#16a082'],['Z',{x:0,y:0,z:112},'#9e71e2']];
    for(const [name,d,col] of ends){let e=project(d);line(origin,e,col,2);ctx.fillStyle=col;ctx.font='bold 13px system-ui';ctx.fillText(name,e.x+5,e.y-5);}
  }
  // E2. Grilla 3D en el plano Z elegido: la distancia usa las TRES coordenadas.
  function buildSlice(){
    if(sliceCache)return sliceCache;
    const N=26, cells=[], labels=[];
    for(let y=0;y<N;y++)for(let x=0;x<N;x++){
      let q={x:(x+.5)*100/N,y:(y+.5)*100/N,z:state.sliceZ},p=predict(q);
      labels[y*N+x]=state.mode==='classification'?p.winner:p.value;
    }
    for(let y=0;y<N;y++)for(let x=0;x<N;x++){
      const x0=x*100/N,x1=(x+1)*100/N,y0=y*100/N,y1=(y+1)*100/N;
      const label=labels[y*N+x];
      const v=(xx,yy)=>({x:xx,y:yy,z:state.sliceZ});
      const quad=[v(x0,y0),v(x1,y0),v(x1,y1),v(x0,y1)];
      const segments=[];
      if(state.showBorders&&state.mode==='classification'){
        // Segmento hacia la derecha o arriba cuando cambia la clase predicha.
        if(x<N-1&&labels[y*N+x+1]!==label)segments.push([v(x1,y0),v(x1,y1)]);
        if(y<N-1&&labels[(y+1)*N+x]!==label)segments.push([v(x0,y1),v(x1,y1)]);
      }
      cells.push({quad,label,segments});
    }
    sliceCache=cells;return cells;
  }
  // E3. Superficie aproximada: clasificamos voxeles y extraemos caras entre
  // vecinos de etiqueta distinta. Es una aproximación DISCRETA, no un mesh exacto.
  function buildVolume(){
    if(volumeCache)return volumeCache;
    if(state.mode==='regression')return [];
    const N=14,V=[],faces=[];
    const id=(x,y,z)=>z*N*N+y*N+x;
    for(let z=0;z<N;z++)for(let y=0;y<N;y++)for(let x=0;x<N;x++){
      V[id(x,y,z)]=predict({x:(x+.5)*100/N,y:(y+.5)*100/N,z:(z+.5)*100/N}).winner;
    }
    // Sólo superficies internas donde dos voxeles adyacentes cambian de clase.
    for(let z=0;z<N;z++)for(let y=0;y<N;y++)for(let x=0;x<N;x++){
      const c=V[id(x,y,z)],x0=x*100/N,x1=(x+1)*100/N,y0=y*100/N,y1=(y+1)*100/N,z0=z*100/N,z1=(z+1)*100/N;
      if(x<N-1&&V[id(x+1,y,z)]!==c)faces.push({quad:[{x:x1,y:y0,z:z0},{x:x1,y:y1,z:z0},{x:x1,y:y1,z:z1},{x:x1,y:y0,z:z1}],label:c});
      if(y<N-1&&V[id(x,y+1,z)]!==c)faces.push({quad:[{x:x0,y:y1,z:z0},{x:x1,y:y1,z:z0},{x:x1,y:y1,z:z1},{x:x0,y:y1,z:z1}],label:c});
      if(z<N-1&&V[id(x,y,z+1)]!==c)faces.push({quad:[{x:x0,y:y0,z:z1},{x:x1,y:y0,z:z1},{x:x1,y:y1,z:z1},{x:x0,y:y1,z:z1}],label:c});
    }
    volumeCache=faces;return faces;
  }
  // E4. Pintor por profundidad: los objetos más lejanos se dibujan primero.
  // Así se pueden ver los puntos y las fronteras de forma rotatoria sin WebGL.
  function draw3D(){
    const things=[],q=project(state.query),prediction=predict(state.query),selection=new Set(prediction.nearest.map(n=>n.index));
    if(state.showAreas||state.showBorders){
      if(state.boundary3D==='volume'&&state.mode==='classification'){
        for(const cell of buildVolume()){
          const pp=cell.quad.map(project),d=pp.reduce((s,p)=>s+p.depth,0)/4;
          things.push({depth:d,draw:()=>polygon(pp,state.showAreas?hexRGBA(COLORS[cell.label],.23):null,state.showBorders?'#33445b35':null,.6)});
        }
      }else{
        for(const cell of buildSlice()){
          const pp=cell.quad.map(project),d=pp.reduce((s,p)=>s+p.depth,0)/4;
          things.push({depth:d,draw:()=>{
            if(state.showAreas)polygon(pp,state.mode==='classification'?hexRGBA(COLORS[cell.label],.26):regressionColor(cell.label,.28));
            if(state.showBorders)cell.segments.forEach(seg=>{const a=project(seg[0]),b=project(seg[1]);line(a,b,'#fff',4);line(a,b,'#243a56',2.3);});
          }});
        }
      }
    }
    // Distancias en 3D: líneas que unen las proyecciones de los puntos 3D.
    const connections=state.step===1?sortedNeighbors(state.query):state.step>=2?prediction.nearest:[];
    for(const n of connections){const p=project(n.point);const col=state.step===1?'#9aacca77':COLORS[n.point.c]+'bb';
      things.push({depth:(p.depth+q.depth)/2,draw:()=>line(q,p,col,state.step===1?1:2,state.step===1?[4,5]:[])});
    }
    state.points.forEach((p,i)=>{
      const s=project(p),color=state.mode==='classification'?COLORS[p.c]:regressionColor(p.v);
      things.push({depth:s.depth+.01,draw:()=>drawPoint(s,color,state.step>=2&&selection.has(i),false,state.mode==='regression'?p.v.toFixed(0):'')});
    });
    things.push({depth:q.depth+.1,draw:()=>drawPoint(q,state.step===3?(state.mode==='classification'?COLORS[prediction.winner]:'#10a886'):'#2b3d5b',true,true)});
    ctx.font='bold 11px system-ui';draw3DBox();
    things.sort((a,b)=>a.depth-b.depth).forEach(a=>a.draw());
    ctx.save();ctx.font='bold 12px system-ui';ctx.fillStyle='#7083a0';ctx.textAlign='left';
    ctx.fillText('X, Y, Z representan características (no el valor predicho)',14,screen.h-13);ctx.restore();
  }
  function queueDraw(){if(raf)return;raf=true;requestAnimationFrame(()=>{raf=false;if(screen.w<20)return;ctx.clearRect(0,0,screen.w,screen.h);state.dims===2?draw2D():draw3D();});}

  // F1. Leyenda, coordenadas, y datos de la explicación de los tres pasos.
  function renderLegend(){
    $('legend').innerHTML=(state.mode==='classification'?names().slice(0,state.classCount).map((n,i)=>`<span class="legend-item"><i class="legend-dot" style="--dot-color:${COLORS[i]}"></i>${n}</span>`):'<span class="legend-item">Valor bajo → valor alto</span>')+'<span class="legend-item"><i class="legend-query">?</i>Flor / dato nuevo</span>';
  }
  function renderPicker(){
    $('classPicker').innerHTML=names().slice(0,state.classCount).map((n,i)=>`<button type="button" class="class-chip ${state.selectedClass===i?'active':''}" data-c="${i}"><i class="mini-dot" style="--chip-color:${COLORS[i]}"></i>${state.scenario==='iris'?n:'Clase '+(i+1)}</button>`).join('');
    $('classPicker').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{state.selectedClass=+b.dataset.c;renderPicker();}));
  }
  function valueText(v,i){return state.scenario==='iris'?`${denormalize(v,i).toFixed(2)} cm`:Math.round(v).toString();}
  function renderCoords(){
    ['x','y','z'].forEach((axis,i)=>{const name=axis.toUpperCase();$( 'query'+name ).value=state.query[axis];$('query'+name+'Value').textContent=valueText(state.query[axis],i);});
    $('sliceZ').value=state.sliceZ;$('sliceZValue').textContent=valueText(state.sliceZ,2);
    $('addZValue').textContent=$('addZ').value;
  }
  function renderSteps(){
    let pred=predict(state.query),near=pred.nearest,labels=['PUNTO DE PARTIDA','PASO 1 · DISTANCIAS','PASO 2 · VECINOS','PASO 3 · RESULTADO'];
    const descriptions=[
      'Tenemos ejemplos conocidos y un dato nuevo. ¿De qué clase será?',
      `Medimos la distancia euclídea en ${state.dims}D del dato nuevo a cada uno de los ${state.points.length} ejemplos.`,
      `Ordenamos las distancias y nos quedamos con los ${near.length} vecinos más próximos.`,
      state.mode==='classification'?'Cada vecino vota. Gana la clase con más votos.':'Promediamos los valores de los vecinos seleccionados.'
    ];
    const titles=['¿Quiénes serán sus vecinos?','Calculamos las distancias',`Seleccionamos K = ${state.k}`,'¡Predicción completada!'];
    $('stepCounter').textContent=`${state.step} / 3`;$('stepLabel').textContent=labels[state.step];
    $('stepTitle').textContent=titles[state.step];$('stepDescription').textContent=descriptions[state.step];
    $('stepProgress').querySelectorAll('span').forEach((el,i)=>el.classList.toggle('done',state.step>i));
    if(state.step===0)$('stepDetail').innerHTML='<div class="empty-detail">Pulsa «Mostrar distancias» para comenzar.</div>';
    if(state.step===1)$('stepDetail').innerHTML='<div class="detail-label">LOS TRES EJEMPLOS MÁS CERCANOS</div>'+sortedNeighbors(state.query).slice(0,3).map((n,i)=>`<div class="dist-row"><span>${i+1}. ${state.mode==='classification'?names()[n.point.c]:'Valor '+n.point.v.toFixed(1)}</span><strong>d = ${n.d.toFixed(2)}</strong></div>`).join('')+'<p class="hint">Las distancias se muestran en unidades normalizadas (0–100).</p>';
    if(state.step===2)$('stepDetail').innerHTML='<div class="detail-label">VECINOS ELEGIDOS EN ORDEN</div>'+near.map((n,i)=>`<div class="dist-row"><span>${i+1}. ${state.mode==='classification'?names()[n.point.c]:'Valor '+n.point.v.toFixed(1)}</span><strong>${n.d.toFixed(2)}</strong></div>`).join('');
    if(state.step===3){
      if(state.mode==='classification'){
        let votes=pred.votes.map((n,i)=>`<div class="vote-row"><span>${names()[i]}</span><div class="vote-track"><div class="vote-fill" style="--vote-width:${100*n/near.length}%;--vote-color:${COLORS[i]}"></div></div><b>${n}</b></div>`).join('');
        $('stepDetail').innerHTML=`<div class="result-card"><span class="result-dot" style="--result-color:${COLORS[pred.winner]}"></span><div><div class="result-label">PREDICCIÓN POR MAYORÍA</div><div class="result-text">${names()[pred.winner]}</div></div></div>`+votes;
      }else $('stepDetail').innerHTML=`<div class="result-card"><span class="result-dot" style="--result-color:#11a88f"></span><div><div class="result-label">PROMEDIO DE LOS ${near.length} VECINOS</div><div class="result-text">${pred.value.toFixed(2)} / 10</div></div></div><p class="hint">(${near.map(n=>n.point.v.toFixed(1)).join(' + ')}) / ${near.length} = ${pred.value.toFixed(2)}</p>`;
    }
    $('nextStep').textContent=['Mostrar distancias →','Seleccionar vecinos →','Mostrar predicción →','↻ Volver a empezar'][state.step];
    $('prevStep').disabled=state.step===0;
  }
  function renderUI(){
    const iris=state.scenario==='iris',three=state.dims===3,cls=state.mode==='classification';
    $('freeBtn').classList.toggle('active',!iris);$('irisBtn').classList.toggle('active',iris);
    $('freeBtn').setAttribute('aria-pressed',String(!iris));$('irisBtn').setAttribute('aria-pressed',String(iris));
    $('caseDescription').hidden=!iris;
    $('view2DBtn').classList.toggle('selected',!three);$('view3DBtn').classList.toggle('selected',three);
    $('view2DBtn').setAttribute('aria-pressed',String(!three));$('view3DBtn').setAttribute('aria-pressed',String(three));
    $('classificationBtn').classList.toggle('selected',cls);$('regressionBtn').classList.toggle('selected',!cls);
    $('classificationBtn').setAttribute('aria-pressed',String(cls));$('regressionBtn').setAttribute('aria-pressed',String(!cls));
    $('regressionBtn').disabled=iris;
    $('regressionNote').textContent=iris?'Iris es un problema de clasificación (3 especies).':cls?'Cada vecino vota por su clase.':'Se calcula la media de los valores conocidos.';
    $('classControls').hidden=!cls;$('classSelect').disabled=iris;$('classSelect').value=state.classCount;
    $('classHint').textContent=iris?'Iris contiene exactamente 3 especies. Para elegir entre 2 y 6 clases, utiliza el laboratorio libre.':'Selecciona 2–6 grupos; se regeneran los ejemplos de cada grupo.';
    $('kValue').textContent=state.k;$('kSlider').max=Math.min(15,state.points.length);$('kSlider').value=state.k;
    $('kHint').textContent=state.k<=3?'K pequeño: fronteras detalladas y sensibilidad al ruido.':state.k>=10?'K grande: las predicciones tienden a ser menos locales.':'K intermedio: varios vecinos comparten la decisión.';
    $('areasToggle').checked=state.showAreas;$('boundaryToggle').checked=state.showBorders;$('boundaryToggle').disabled=!cls;
    $('areasLabel').textContent=cls?'Colorear regiones predichas':'Colorear valores estimados';
    $('boundaryIndicator').hidden=!cls||!state.showBorders;
    $('boundary3DControls').hidden=!three;$('boundary3DMode').value=state.boundary3D;
    $('boundary3DMode').disabled=!cls;
    $('sliceZRow').hidden=!(three&&(state.boundary3D==='slice'||!cls));
    $('sliceHint').textContent=state.boundary3D==='slice'?'En 3D, la frontera es una superficie. Se muestra su corte con el plano Z.':'Frontera volumétrica: superficies entre voxeles con clases distintas (aproximación discreta).';
    $('orbitTools').hidden=!three;$('queryZRow').hidden=!three;
    $('dimensionHint').textContent=three?'En 3D usamos d = √(Δx² + Δy² + Δz²). La Z sí influye en KNN.':'En 2D usamos d = √(Δx² + Δy²). La Z no interviene.';
    $('editSection').hidden=iris;$('classPickerContainer').hidden=state.tool!=='add'||!cls;
    $('addZRow').hidden=state.tool!=='add'||!three;
    $('moveTool').classList.toggle('active',state.tool==='move');$('addTool').classList.toggle('active',state.tool==='add');
    $('moveTool').setAttribute('aria-pressed',String(state.tool==='move'));$('addTool').setAttribute('aria-pressed',String(state.tool==='add'));
    $('undoButton').disabled=!state.points.some(p=>p.custom);
    $('canvasHint').textContent=three?'Arrastra para rotar · usa los controles X, Y, Z':'Arrastra ? o los ejemplos · clic para mover ?';
    $('queryHint').textContent=iris?'Los controles representan centímetros reales; internamente se normalizan.':three?'En 3D cambia X, Y y Z con los deslizadores.':'En 2D también puedes mover ? directamente con el ratón.';
    $('chartStatus').textContent=`● ${cls?state.classCount+' clases':'Regresión'} · ${state.dims}D`;
    $('axisInfo').textContent=iris?'X = longitud pétalo (cm) · Y = anchura pétalo (cm) · Z = longitud sépalo (cm)':'X = característica 1 · Y = característica 2 · Z = característica 3';
    $('formulaExplanation').innerHTML=three?'En <strong>3D</strong>: d = √(Δx² + Δy² + Δz²). La tercera característica afecta a los vecinos.':'En <strong>2D</strong>: d = √(Δx² + Δy²). Sólo comparamos dos características.';
    $('teacherTip').innerHTML=iris?'<strong>Ejemplo para clase:</strong> tenemos una flor nueva con medidas 4,7 cm de pétalo, 1,4 cm de anchura y 6,1 cm de sépalo. Prueba K = 1 y K = 9, activa 3D y descubre si la especie predicha cambia.':'<strong>Pregunta para clase:</strong> compara K = 1 con K = 11. ¿Cómo cambian las regiones? Activa 3D y mueve Z: ¿podría cambiar la etiqueta al añadir una tercera característica?';
    renderCoords();renderLegend();renderPicker();renderSteps();queueDraw();
  }

  // G1. Cambio de escenario, dimensionalidad y modos.
  function scenario(which){
    state.scenario=which;state.step=0;state.tool='move';
    if(which==='iris'){state.mode='classification';state.classCount=3;state.k=5;makeIris();}
    else{state.classCount=3;state.k=5;makeFree();}
    resetQuery();renderUI();
  }
  function dims(d){state.dims=d;state.step=0;invalidate();renderUI();}
  function changeK(k){state.k=clamp(k,1,Math.min(15,state.points.length));invalidate();renderUI();}
  function reset(){Object.assign(state,{scenario:'free',mode:'classification',dims:2,classCount:3,k:5,step:0,showAreas:true,showBorders:true,boundary3D:'slice',tool:'move',selectedClass:0,seed:3,yaw:-.74,pitch:.43,zoom:1});resetQuery();makeFree();renderUI();}

  // G2. Ratón / táctil. En 2D se pueden mover puntos; en 3D se gira la cámara.
  function mouse(e){const r=canvas.getBoundingClientRect();return {x:e.clientX-r.left,y:e.clientY-r.top};}
  function hitTest(pos){
    const q=to2D(state.query);if(Math.hypot(pos.x-q.x,pos.y-q.y)<24)return {kind:'query'};
    let best=21,index=-1;state.points.forEach((p,i)=>{let pp=to2D(p),d=Math.hypot(pp.x-pos.x,pp.y-pos.y);if(d<best){best=d;index=i;}});
    return index<0?null:{kind:'point',index};
  }
  function move2D(pos){const p=from2D(pos.x,pos.y);if(!state.dragging)return;
    if(state.dragging.kind==='query')Object.assign(state.query,p);
    if(state.dragging.kind==='point'){const sample=state.points[state.dragging.index];sample.x=p.x;sample.y=p.y;invalidate();}
    if(state.dragging.kind==='query'){/* La superficie no depende de la posición del punto ?. */}
    renderCoords();renderSteps();queueDraw();
  }
  canvas.addEventListener('pointerdown',e=>{
    const pos=mouse(e);$('tooltip').style.display='none';state.hovered=-1;
    if(state.dims===3){state.dragging={kind:'orbit',last:pos};}
    else if(state.scenario==='iris'){state.dragging={kind:'query'};move2D(pos);}
    else{
      const hit=hitTest(pos);
      if(hit)state.dragging=hit;
      else if(state.tool==='add'){
        const p=from2D(pos.x,pos.y),z=+$('addZ').value,v=Math.round(clamp(1+p.x*.043+p.y*.031+z*.012,1,10)*10)/10;
        state.points.push({...p,z,c:state.selectedClass,v,custom:true});
        state.dragging={kind:'point',index:state.points.length-1};invalidate();renderUI();
      }else state.dragging={kind:'query'};
      move2D(pos);
    }
    canvas.setPointerCapture(e.pointerId);queueDraw();
  });
  canvas.addEventListener('pointermove',e=>{
    const pos=mouse(e);
    if(state.dragging){
      if(state.dragging.kind==='orbit'){
        const last=state.dragging.last;state.yaw+=(pos.x-last.x)*.012;state.pitch=clamp(state.pitch-(pos.y-last.y)*.010,-1.3,1.3);state.dragging.last=pos;queueDraw();
      }else move2D(pos);
      return;
    }
    if(state.dims===3){canvas.style.cursor='grab';return;}
    const hit=hitTest(pos);state.hovered=hit?.kind==='point'?hit.index:-1;
    canvas.style.cursor=hit?'grab':state.tool==='add'?'copy':'crosshair';
    const tip=$('tooltip');if(state.hovered>=0){const p=state.points[state.hovered];tip.textContent=state.scenario==='iris'?`${names()[p.c]} · pétalo ${p.raw[0]} × ${p.raw[1]} cm · sépalo ${p.raw[2]} cm`:state.mode==='classification'?`Clase ${p.c+1} · ${names()[p.c]}`:`Valor: ${p.v.toFixed(1)}`;
      tip.style.left=`${clamp(pos.x+10,3,screen.w-230)}px`;tip.style.top=`${clamp(pos.y-39,3,screen.h-40)}px`;tip.style.display='block';}
    else tip.style.display='none';queueDraw();
  });
  function stop(e){if(!state.dragging)return;state.dragging=null;try{canvas.releasePointerCapture(e.pointerId)}catch{}renderUI();}
  canvas.addEventListener('pointerup',stop);canvas.addEventListener('pointercancel',stop);
  canvas.addEventListener('wheel',e=>{if(state.dims!==3)return;e.preventDefault();state.zoom=clamp(state.zoom*(e.deltaY>0?.91:1.09),.55,1.8);queueDraw();},{passive:false});
  canvas.addEventListener('keydown',e=>{const key=e.key;const a=key==='ArrowLeft'?'x':key==='ArrowRight'?'x':key==='ArrowUp'?'y':key==='ArrowDown'?'y':null;if(!a)return;e.preventDefault();let step=e.shiftKey?5:2;state.query[a]=clamp(state.query[a]+((key==='ArrowLeft'||key==='ArrowDown')?-step:step),0,100);renderCoords();renderSteps();queueDraw();});

  // G3. Enlazamos los controles visibles con funciones de estado.
  $('freeBtn').addEventListener('click',()=>scenario('free'));
  $('irisBtn').addEventListener('click',()=>scenario('iris'));
  $('view2DBtn').addEventListener('click',()=>dims(2));
  $('view3DBtn').addEventListener('click',()=>dims(3));
  $('classificationBtn').addEventListener('click',()=>{state.mode='classification';state.step=0;invalidate();renderUI();});
  $('regressionBtn').addEventListener('click',()=>{if(state.scenario==='iris')return;state.mode='regression';state.step=0;invalidate();renderUI();});
  $('classSelect').addEventListener('change',e=>{if(state.scenario==='iris')return;state.classCount=+e.target.value;state.step=0;state.selectedClass=0;makeFree();renderUI();});
  $('kSlider').addEventListener('input',e=>changeK(+e.target.value));
  $('areasToggle').addEventListener('change',e=>{state.showAreas=e.target.checked;invalidate();renderUI();});
  $('boundaryToggle').addEventListener('change',e=>{state.showBorders=e.target.checked;invalidate();renderUI();});
  $('boundary3DMode').addEventListener('change',e=>{state.boundary3D=e.target.value;invalidate();renderUI();});
  $('sliceZ').addEventListener('input',e=>{state.sliceZ=+e.target.value;sliceCache=null;renderCoords();queueDraw();});
  ['X','Y','Z'].forEach(axis=>{$('query'+axis).addEventListener('input',e=>{state.query[axis.toLowerCase()]=+e.target.value;renderCoords();renderSteps();queueDraw();});});
  $('addZ').addEventListener('input',e=>{$('addZValue').textContent=e.target.value;});
  $('moveTool').addEventListener('click',()=>{state.tool='move';renderUI();});
  $('addTool').addEventListener('click',()=>{state.tool='add';renderUI();});
  $('undoButton').addEventListener('click',()=>{let i=state.points.findLastIndex(p=>p.custom);if(i>=0)state.points.splice(i,1);invalidate();renderUI();});
  $('shuffleButton').addEventListener('click',()=>{state.seed++;state.step=0;makeFree();renderUI();});
  $('nextStep').addEventListener('click',()=>{state.step=state.step===3?0:state.step+1;renderUI();});
  $('prevStep').addEventListener('click',()=>{state.step=Math.max(0,state.step-1);renderUI();});
  $('resetButton').addEventListener('click',reset);
  $('resetCameraButton').addEventListener('click',()=>{state.yaw=-.74;state.pitch=.43;state.zoom=1;queueDraw();});

  // G4. Responsive: el canvas tiene resolución adecuada en pantallas retina.
  function resize(){const b=canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);
    screen.w=b.width;screen.h=b.height;screen.dpr=dpr;canvas.width=Math.max(1,Math.round(b.width*dpr));canvas.height=Math.max(1,Math.round(b.height*dpr));ctx.setTransform(dpr,0,0,dpr,0,0);twoDCache=null;queueDraw();}
  resetQuery();makeFree();renderUI();
  if(window.ResizeObserver)new ResizeObserver(resize).observe(canvas);else window.addEventListener('resize',resize);
  resize();
  // API para demostrar y probar el algoritmo en consola del navegador.
  window.KNN_DEMO={state,predict,sortedNeighbors,changeK,dims,scenario,reset};
})();
