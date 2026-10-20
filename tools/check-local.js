// Verify that the file-open delivery is self-contained without running a browser.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const cp = require('node:child_process');
const root = path.resolve(__dirname, '..');
fs.mkdirSync(path.join(root, 'work'), { recursive: true });
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const refs = [...html.matchAll(/(?:src|href)="([^"]+)"/g)].map(m => m[1]).filter(x => !x.startsWith('data:') && !x.startsWith('#'));
for (const ref of refs) {
  assert.ok(!/^\w+:/.test(ref), `External dependency: ${ref}`);
  assert.ok(fs.statSync(path.join(root, ref)).isFile(), `Missing local file: ${ref}`);
}
assert.ok(!/<script[^>]+type="module"/.test(html), 'File-open entry must use classic scripts');
const scripts = ['app.js', 'flow.js', 'engine.js', 'model/model-data.js'];
for (const file of scripts) {
  cp.execFileSync(process.execPath, ['--check', path.join(root, file)]);
  const source = fs.readFileSync(path.join(root, file), 'utf8');
  assert.ok(!/\bfetch\s*\(|\bXMLHttpRequest\b|\bWebSocket\s*\(|\bimport\s*\(/.test(source), `Network/module runtime dependency: ${file}`);
}
const css = fs.readFileSync(path.join(root, 'style.css'), 'utf8');
assert.ok(!/@import\b|url\(\s*['"]?https?:/i.test(css), 'External CSS dependency');
const launcher = fs.readFileSync(path.join(root, '打开实验台.cmd'), 'utf8');
assert.ok(launcher.includes('%~dp0index.html'), 'Launcher must open the adjacent HTML');
require(path.join(root, 'model/model-data.js'));
assert.equal(VIT_DATA.examples.length, 30);
const productionFiles = ['index.html', 'style.css', 'app.js', 'flow.js', 'engine.js', 'model/model-data.js', 'README.md', '打开实验台.cmd'];
const total = productionFiles.reduce((n, name) => n + fs.statSync(path.join(root, name)).size, 0);
const report = { passed: true, localReferences: [...new Set(refs)], classicScripts: true, noRuntimeNetworkCalls: true,
  syntaxChecked: scripts, productionFiles, totalBytes: total, modelParameters: VIT_DATA.metadata.parameters,
  browserFileProtocol: 'NOT_EXECUTED: browser tool rejects file:// URLs; structural checks only' };
fs.writeFileSync(path.join(root, 'work/static-acceptance.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
