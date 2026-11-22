const fs=require('node:fs');
const {chromium}=require('playwright');
async function main(){
  const [url,out]=process.argv.slice(2);
  const browser=await chromium.launch({headless:true,channel:'chrome'});
  try {
    const page=await browser.newPage();
    const errors=[]; page.on('pageerror',e=>errors.push(String(e)));
    const evidence={url,checkedAt:new Date().toISOString(),views:[]};
    for(const width of [1280,390]){
      await page.setViewportSize({width,height:900}); await page.goto(url);
      await page.locator('img').last().waitFor();
      await page.evaluate(()=>Promise.all([...document.images].map(x=>x.decode())));
      const info=await page.evaluate(()=>({
        headings:[...document.querySelectorAll('h2')].map(x=>x.textContent),
        images:[...document.images].map(x=>({alt:x.alt,width:x.naturalWidth,embedded:x.src.startsWith('data:image/png;base64,')})),
        overflow:document.documentElement.scrollWidth>innerWidth,
        title:document.title
      }));
      if(info.images.length!==4 || info.images.some(x=>!x.width||!x.embedded) || info.headings.length!==6 || info.overflow)
        throw Error('Render validation failed: '+JSON.stringify(info));
      await page.screenshot({path:out.replace(/\.json$/,`-${width}.png`)});
      evidence.views.push({width,...info});
    }
    if(errors.length) throw Error(errors.join('\n'));
    evidence.passed=true;fs.writeFileSync(out,JSON.stringify(evidence,null,2));
    console.log(JSON.stringify(evidence));
  } finally {await browser.close();}
}
main().catch(e=>{console.error(e);process.exit(1)});
