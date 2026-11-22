// Open the report preview in an isolated VS Code profile before running this check.
const fs=require('node:fs');
const {chromium}=require('playwright');
async function main(){
  const [port,report,out]=process.argv.slice(2);
  const endpoint=`http://127.0.0.1:${port}`;
  const version=await(await fetch(`${endpoint}/json/version`)).json();
  const socket=new WebSocket(version.webSocketDebuggerUrl);
  await new Promise(r=>socket.addEventListener('open',r,{once:true}));
  let next=0;const pending=new Map();
  socket.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){pending.get(m.id)?.(m);pending.delete(m.id)}};
  const send=(method,params={},sessionId)=>new Promise((resolve,reject)=>{
    const id=++next;const timer=setTimeout(()=>reject(Error('CDP timeout')),10000);
    pending.set(id,m=>{clearTimeout(timer);m.error?reject(Error(JSON.stringify(m.error))):resolve(m.result)});
    socket.send(JSON.stringify({id,method,params,sessionId}));
  });
  const browser=await chromium.connectOverCDP(endpoint);
  try {
    const pages=browser.contexts().flatMap(c=>c.pages());
    if(pages.length!==1 || !pages[0].url().startsWith('vscode-file:')) throw Error('Expected isolated VS Code test window');
    const {targetInfos}=await send('Target.getTargets');
    if(!targetInfos.some(x=>x.type==='page'&&x.title.includes(require('node:path').basename(report)))) throw Error('Final report must be open');
    const targets=targetInfos.filter(x=>x.type==='iframe'&&x.url.includes('vscode.markdown-language-features'));
    let images=[];
    for(const target of targets){
      const {sessionId}=await send('Target.attachToTarget',{targetId:target.targetId,flatten:true});
      const {result}=await send('Runtime.evaluate',{expression:`JSON.stringify([...document.querySelectorAll('iframe')].flatMap(f=>[...f.contentDocument.images].map(i=>({alt:i.alt,width:i.naturalWidth,height:i.naturalHeight,complete:i.complete,src:i.src}))))`,returnByValue:true},sessionId);
      const found=JSON.parse(result.value);
      if(found.length===4){images=found;break;}
    }
    if(images.length!==4 || images.some(x=>!x.complete || !x.width || !x.src.includes('/docs/reports/assets/learning_curve_20260921/'))) throw Error('Final report figures failed: '+JSON.stringify(images));
    await pages[0].screenshot({path:out.replace(/\.json$/,'.png')});
    fs.writeFileSync(out,JSON.stringify({checkedAt:new Date().toISOString(),report,application:'Visual Studio Code',isolatedProfile:true,images,passed:true},null,2));
    console.log(JSON.stringify({passed:true,images:images.map(x=>({alt:x.alt,width:x.width,height:x.height}))}));
  } finally {socket.close();await browser.close();}
}
main().catch(e=>{console.error(e);process.exit(1)});
