/* Small, auditable CPU ViT inference. No framework, network, or build step. */
(function (scope) {
  'use strict';
  function dense(x, rows, input, output, w, b) {
    const out = new Float32Array(rows * output);
    for (let r = 0; r < rows; r++) {
      for (let o = 0; o < output; o++) {
        let sum = b ? b[o] : 0;
        for (let i = 0; i < input; i++) sum += x[r * input + i] * w[o * input + i];
        out[r * output + o] = sum;
      }
    }
    return out;
  }
  function norm(x, rows, dim, weight, bias, eps) {
    const out = new Float32Array(x.length);
    for (let r = 0; r < rows; r++) {
      const offset = r * dim;
      let mean = 0, variance = 0;
      for (let j = 0; j < dim; j++) mean += x[offset + j];
      mean /= dim;
      for (let j = 0; j < dim; j++) variance += (x[offset + j] - mean) ** 2;
      const inv = 1 / Math.sqrt(variance / dim + eps);
      for (let j = 0; j < dim; j++) out[offset + j] = (x[offset + j] - mean) * inv * weight[j] + bias[j];
    }
    return out;
  }
  function add(a, b) { return Float32Array.from(a, (v, i) => v + b[i]); }
  function softmax(values) {
    const max = Math.max(...values);
    const out = Float32Array.from(values, v => Math.exp(v - max));
    const sum = out.reduce((a, b) => a + b, 0);
    return out.map(v => v / sum);
  }
  class TinyViT {
    constructor(data) {
      this.config = data.config;
      this.metadata = data.metadata;
      const binary = typeof atob === 'function' ? atob(data.weights) : Buffer.from(data.weights, 'base64').toString('binary');
      const bytes = Uint8Array.from(binary, c => c.charCodeAt(0));
      const numbers = new Float32Array(bytes.buffer);
      this.weights = {};
      for (const [name, entry] of Object.entries(data.manifest)) this.weights[name] = numbers.subarray(entry.offset, entry.offset + entry.length);
    }
    predict(pixels) {
      if (pixels.length !== 784) throw new Error('Expected a 28 × 28 image.');
      const { dim: d, heads: h, layers, hidden_dim: hidden, eps } = this.config;
      const n = 17, headDim = d / h, w = this.weights;
      const patches = new Float32Array(16 * 49);
      for (let py = 0; py < 4; py++) for (let px = 0; px < 4; px++) {
        const patch = py * 4 + px;
        for (let y = 0; y < 7; y++) for (let x = 0; x < 7; x++) {
          patches[patch * 49 + y * 7 + x] = (pixels[(py * 7 + y) * 28 + px * 7 + x] - 0.1307) / 0.3081;
        }
      }
      const embedded = dense(patches, 16, 49, d, w['patch.weight'], w['patch.bias']);
      let tokens = new Float32Array(n * d);
      tokens.set(w.cls, 0); tokens.set(embedded, d);
      tokens = add(tokens, w.pos);
      let x = tokens;
      const blocks = [];
      for (let layer = 0; layer < layers; layer++) {
        const p = `blocks.${layer}.`, input = x;
        const norm1 = norm(x, n, d, w[p + 'norm1.weight'], w[p + 'norm1.bias'], eps);
        const qkv = dense(norm1, n, d, d * 3, w[p + 'qkv.weight'], w[p + 'qkv.bias']);
        const q = new Float32Array(n * d), k = new Float32Array(n * d), v = new Float32Array(n * d);
        for (let i = 0; i < n; i++) {
          q.set(qkv.subarray(i * 3 * d, i * 3 * d + d), i * d);
          k.set(qkv.subarray(i * 3 * d + d, i * 3 * d + 2 * d), i * d);
          v.set(qkv.subarray(i * 3 * d + 2 * d, (i + 1) * 3 * d), i * d);
        }
        const attentions = [], context = new Float32Array(n * d);
        for (let head = 0; head < h; head++) {
          const attention = new Float32Array(n * n);
          for (let i = 0; i < n; i++) {
            const scores = new Float32Array(n);
            for (let j = 0; j < n; j++) {
              let dot = 0;
              for (let t = 0; t < headDim; t++) dot += q[i * d + head * headDim + t] * k[j * d + head * headDim + t];
              scores[j] = dot / Math.sqrt(headDim);
            }
            const probabilities = softmax(scores);
            attention.set(probabilities, i * n);
            for (let t = 0; t < headDim; t++) {
              let sum = 0;
              for (let j = 0; j < n; j++) sum += probabilities[j] * v[j * d + head * headDim + t];
              context[i * d + head * headDim + t] = sum;
            }
          }
          attentions.push(attention);
        }
        const projected = dense(context, n, d, d, w[p + 'proj.weight'], w[p + 'proj.bias']);
        const residual = add(x, projected);
        const norm2 = norm(residual, n, d, w[p + 'norm2.weight'], w[p + 'norm2.bias'], eps);
        const preActivation = dense(norm2, n, d, hidden, w[p + 'fc1.weight'], w[p + 'fc1.bias']);
        const activation = preActivation.map(a => 0.5 * a * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (a + 0.044715 * a * a * a))));
        const mlp = dense(activation, n, hidden, d, w[p + 'fc2.weight'], w[p + 'fc2.bias']);
        x = add(residual, mlp);
        blocks.push({ input, norm1, q, k, v, attentions, context, projected, residual, norm2, hidden: activation, mlp, output: x });
      }
      const normalized = norm(x, n, d, w['norm.weight'], w['norm.bias'], eps);
      const logits = dense(normalized.subarray(0, d), 1, d, 10, w['head.weight'], w['head.bias']);
      const probabilities = softmax(logits);
      const prediction = probabilities.indexOf(Math.max(...probabilities));
      return { prediction, probabilities, logits, patches, embedded, tokens, blocks, normalized };
    }
  }
  scope.TinyViT = TinyViT;
  if (typeof module !== 'undefined') module.exports = { TinyViT };
})(globalThis);
