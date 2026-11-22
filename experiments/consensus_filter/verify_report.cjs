// Usage: NODE_PATH=<bundled node_modules> node verify_report.cjs <run directory>
const {chromium} = require('playwright');
const fs = require('fs');
const path = require('path');
const {pathToFileURL, fileURLToPath} = require('url');
const assert = require('assert/strict');

(async () => {
  const out = path.resolve(process.argv[2]);
  const root = path.resolve(__dirname, '../..');
  const name = path.basename(out);
  const report = path.join(root, 'docs/reports', name + '_report.html');
  const review = path.join(root, 'docs/research/results', name, 'review.html');
  const summary = JSON.parse(fs.readFileSync(path.join(out, 'summary.json')));
  const protocol = JSON.parse(fs.readFileSync(path.join(out, 'protocol.json')));
  const dest = path.join(out, 'report_qa');
  fs.mkdirSync(dest, {recursive:true});
  const browser = await chromium.launch({headless:true, executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
  const page = await browser.newPage({viewport:{width:1440,height:1050}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(pathToFileURL(report).href);
  const images = await page.locator('img').evaluateAll(async imgs => {
    await Promise.all(imgs.map(img => { img.loading='eager'; return img.decode(); }));
    return imgs.map(img => ({src:img.src,width:img.naturalWidth,height:img.naturalHeight}));
  });
  assert.equal(images.length, 9);
  assert.ok(images.every(x => x.width > 0));
  const links = await page.locator('a').evaluateAll(as => as.map(a=>a.href));
  const missing = links.filter(x => x.startsWith('file:')).filter(x=>!fs.existsSync(fileURLToPath(x)));
  assert.deepEqual(missing, []);
  await page.screenshot({path:path.join(dest,'report_desktop.png')});
  await page.locator('img[alt="performance"]').screenshot({path:path.join(dest,'performance_in_report.png')});
  await page.setViewportSize({width:390,height:844});
  await page.evaluate(()=>window.scrollTo(0,0));
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth+1));
  await page.screenshot({path:path.join(dest,'report_mobile.png')});
  await page.goto(pathToFileURL(review).href);
  assert.ok((await page.locator('#count').innerText()).startsWith(summary.common_all_six_records+'개'));
  await page.locator('#scope').selectOption('all');
  assert.ok((await page.locator('#count').innerText()).startsWith(protocol.records+'개'));
  await page.locator('#label').selectOption(protocol.labels[0]);
  const truthCount = await page.evaluate(()=>DATA.filter(r=>r.truth_label===LABELS[0]).length);
  assert.ok((await page.locator('#count').innerText()).startsWith(truthCount+'개'));
  await page.locator('#label').selectOption('');
  await page.locator('#search').fill(protocol.subjects[0]);
  assert.ok((await page.locator('#count').innerText()).endsWith('1명'));
  await page.locator('summary').first().click();
  assert.equal(await page.locator('details').first().getAttribute('open'), '');
  await page.locator('#search').fill('');
  await page.locator('#scope').selectOption('common_all_six');
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth+1));
  await page.screenshot({path:path.join(dest,'review_mobile.png')});
  await page.setViewportSize({width:1440,height:1050});
  await page.screenshot({path:path.join(dest,'review_desktop.png')});
  if (summary.common_all_six_records > 20) {
    await page.locator('#next').click();
    assert.ok((await page.locator('#page').innerText()).startsWith('2 /'));
  }
  const rawLink = await page.locator('article a').first().getAttribute('href');
  assert.ok(fs.existsSync(fileURLToPath(new URL(rawLink, pathToFileURL(review)))));
  await page.goto(pathToFileURL(path.join(out, 'review.html')).href);
  const back = await page.locator('main > a').first().evaluate(a=>a.href);
  assert.ok(fs.existsSync(fileURLToPath(back)));
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(out,'browser_verification.json'),JSON.stringify({checked_at:new Date().toISOString(),
    images:images.length,local_report_links:links.filter(x=>x.startsWith('file:')).length,missing_links:missing,
    desktop_and_mobile_rendered:true,review_filters_search_details_pagination:true,
    compatibility_alias_links_valid:true,page_errors:errors},null,2)+'\n');
  await browser.close();
  console.log('Browser report and review checks passed');
})().catch(error=>{console.error(error);process.exit(1)});
