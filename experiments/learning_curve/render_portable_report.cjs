// NODE_PATH=<runtime node_modules> node render_portable_report.cjs report.md
const fs = require('node:fs');
const path = require('node:path');
const {marked} = require('marked');
const input = path.resolve(process.argv[2]);
const output = input.replace(/\.md$/, '.html');
let text = fs.readFileSync(input, 'utf8');
let images = [];
text = text.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_,alt,url)=>{
  if (/^[a-z]+:/i.test(url)) throw Error('Only inspected local images allowed');
  const file=path.resolve(path.dirname(input),decodeURI(url));
  const bytes=fs.readFileSync(file);
  if (!bytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10]))) throw Error('Expected PNG');
  images.push({url,bytes:bytes.length});
  return `![${alt}](data:image/png;base64,${bytes.toString('base64')})`;
});
if(images.length!==4) throw Error('Expected four report figures');
const html=`<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>데이터 규모별 모델 성능 보고서</title><style>
body{font-family:-apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo','Malgun Gothic',sans-serif;color:#17212b;background:#fff;max-width:1180px;margin:32px auto;padding:0 24px;line-height:1.75}h1{font-size:30px}h2{font-size:24px;margin-top:50px;border-bottom:2px solid #dce4eb;padding-bottom:12px}h3{margin-top:30px}img{display:block;width:100%;height:auto;margin:24px 0}table{display:block;overflow:auto;border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th,td{border:1px solid #d5dfe7;padding:9px 12px;white-space:nowrap}th{background:#edf3f8}a{color:#005ca8}code{font-size:13px;background:#f3f5f7}pre{padding:16px;overflow:auto;background:#f3f5f7}strong{color:#17212b}@media print{body{max-width:none;padding:0;font-size:10pt}img,table{break-inside:avoid}h2{break-after:avoid}}
</style></head><body>${marked.parse(text)}</body></html>`;
fs.writeFileSync(output,html);
console.log(JSON.stringify({output,images,embedded:true}));
