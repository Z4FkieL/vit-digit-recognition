/* Compare the browser implementation with independently computed PyTorch traces. */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
fs.mkdirSync(path.join(root, 'work'), { recursive: true });
require(path.join(root, 'model/model-data.js'));
const { TinyViT } = require(path.join(root, 'engine.js'));
const model = new TinyViT(globalThis.VIT_DATA);
const fixtures = JSON.parse(require('node:zlib').gunzipSync(fs.readFileSync(path.join(root, 'tests/fixtures/pytorch-reference.json.gz'))).toString('utf8'));
const metrics = { logits: 0, tokens: 0, attention: 0, layerOutputs: 0, predictions: 0, samplePredictions: 0 };
function compare(a, b, key) {
  assert.equal(a.length, b.length, `${key}: shape mismatch`);
  for (let i = 0; i < a.length; i++) {
    assert.ok(Number.isFinite(a[i]), `${key}: non-finite value`);
    metrics[key] = Math.max(metrics[key], Math.abs(a[i] - b[i]));
  }
}
const start = performance.now();
for (const fixture of fixtures) {
  const result = model.predict(Float32Array.from(fixture.pixels, x => x / 255));
  compare(result.logits, fixture.logits, 'logits');
  compare(result.tokens, fixture.tokens.flat(), 'tokens');
  for (let layer = 0; layer < 3; layer++) {
    compare(result.blocks[layer].output, fixture.outputs[layer].flat(), 'layerOutputs');
    for (let head = 0; head < 4; head++) {
      const a = result.blocks[layer].attentions[head];
      compare(a, fixture.attention[layer][head].flat(), 'attention');
      for (let row = 0; row < 17; row++) {
        const sum = a.subarray(row * 17, (row + 1) * 17).reduce((x, y) => x + y, 0);
        assert.ok(Math.abs(sum - 1) < 1e-5, 'Attention rows must sum to one');
      }
    }
  }
  assert.equal(result.prediction, fixture.logits.indexOf(Math.max(...fixture.logits)), 'Predictions differ from PyTorch');
  metrics.predictions++;
}
for (const example of VIT_DATA.examples) {
  const pixels = Float32Array.from(Buffer.from(example.pixels, 'base64'), v => v / 255);
  const result = model.predict(pixels);
  metrics.samplePredictions += Number(result.prediction === example.label);
  assert.ok(Math.abs(result.probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-5);
}
assert.ok(metrics.logits < 1e-3, 'Logit mismatch');
assert.ok(metrics.tokens < 1e-4, 'Embedding mismatch');
assert.ok(metrics.attention < 1e-4, 'Attention mismatch');
assert.ok(metrics.layerOutputs < 1e-3, 'Layer output mismatch');
const report = { passed: true, fixtures: fixtures.length, examples: VIT_DATA.examples.length, maxAbsoluteErrors: metrics, elapsedMs: Math.round(performance.now() - start), modelTestAccuracy: VIT_DATA.metadata.test_accuracy };
fs.writeFileSync(path.join(root, 'work/verification.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
