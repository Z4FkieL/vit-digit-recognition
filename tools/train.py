"""Train the small ViT used by the local, dependency-free browser demo."""
import argparse
import base64
import gzip
import json
import math
from pathlib import Path
import random
import struct
import time
import urllib.request

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
CONFIG = dict(image_size=28, patch_size=7, dim=48, heads=4, layers=3, hidden_dim=96, classes=10, eps=1e-5)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        d = CONFIG['dim']
        self.norm1 = nn.LayerNorm(d, eps=CONFIG['eps'])
        self.qkv = nn.Linear(d, 3*d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d, eps=CONFIG['eps'])
        self.fc1 = nn.Linear(d, CONFIG['hidden_dim'])
        self.fc2 = nn.Linear(CONFIG['hidden_dim'], d)

    def forward(self, x, trace=False):
        b, n, d = x.shape
        z = self.norm1(x)
        q, k, v = self.qkv(z).reshape(b, n, 3, CONFIG['heads'], d//CONFIG['heads']).permute(2, 0, 3, 1, 4).unbind(0)
        attention = (q @ k.transpose(-2, -1) / math.sqrt(d//CONFIG['heads'])).softmax(-1)
        context = (attention @ v).transpose(1, 2).reshape(b, n, d)
        projected = self.proj(context)
        residual = x + projected
        normalized = self.norm2(residual)
        hidden = F.gelu(self.fc1(normalized), approximate='tanh')
        mlp = self.fc2(hidden)
        output = residual + mlp
        if trace:
            return output, dict(norm1=z, q=q, k=k, v=v, attention=attention, context=context,
                                projected=projected, residual=residual, norm2=normalized, hidden=hidden, mlp=mlp, output=output)
        return output


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        d = CONFIG['dim']
        self.patch = nn.Linear(CONFIG['patch_size']**2, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, 17, d)*0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(CONFIG['layers'])])
        self.norm = nn.LayerNorm(d, eps=CONFIG['eps'])
        self.head = nn.Linear(d, 10)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, images, trace=False):
        patches = F.unfold((images-0.1307)/0.3081, kernel_size=7, stride=7).transpose(1, 2)
        embedded = self.patch(patches)
        tokens = torch.cat([self.cls.expand(images.shape[0], -1, -1), embedded], 1) + self.pos
        x = tokens
        traces = []
        for block in self.blocks:
            if trace:
                x, t = block(x, True)
                traces.append(t)
            else:
                x = block(x)
        norm = self.norm(x)
        logits = self.head(norm[:, 0])
        if trace:
            return logits, dict(patches=patches, embedded=embedded, tokens=tokens, blocks=traces, norm=norm)
        return logits


def dataset(split):
    folder = ROOT/'work'/'data'
    folder.mkdir(parents=True, exist_ok=True)
    prefix = 'train' if split == 'train' else 't10k'
    out = []
    for suffix in ['images-idx3-ubyte', 'labels-idx1-ubyte']:
        filename = f'{prefix}-{suffix}.gz'
        path = folder/filename
        if not path.exists():
            print(f'Downloading {filename}', flush=True)
            urllib.request.urlretrieve('https://storage.googleapis.com/cvdf-datasets/mnist/'+filename, path)
        raw = gzip.decompress(path.read_bytes())
        magic, n = struct.unpack('>II', raw[:8])
        if magic == 2051:
            array = np.frombuffer(raw, np.uint8, offset=16).copy().reshape(n, 1, 28, 28)
            out.append(torch.from_numpy(array).float()/255)
        else:
            out.append(torch.from_numpy(np.frombuffer(raw, np.uint8, offset=8).copy()).long())
    return out


def augment(x):
    n = len(x)
    angle = (torch.rand(n)-0.5)*0.40
    scale = 0.9+torch.rand(n)*0.2
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = torch.cos(angle)*scale
    theta[:, 1, 1] = torch.cos(angle)*scale
    theta[:, 0, 1] = -torch.sin(angle)*scale
    theta[:, 1, 0] = torch.sin(angle)*scale
    theta[:, :, 2] = (torch.rand(n, 2)-0.5)*0.14
    return F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    correct = 0
    for start in range(0, len(x), 512):
        correct += int((model(x[start:start+512]).argmax(1) == y[start:start+512]).sum())
    return correct/len(x)


def export(model, test_x, test_y, metadata):
    destination = ROOT/'model'
    destination.mkdir(exist_ok=True)
    state = model.state_dict()
    manifest, chunks, offset = {}, [], 0
    for name, tensor in state.items():
        arr = tensor.detach().cpu().numpy().astype('<f4')
        manifest[name] = dict(shape=list(arr.shape), offset=offset, length=arr.size)
        chunks.append(arr.tobytes())
        offset += arr.size
    weights = b''.join(chunks)
    examples = []
    model.eval()
    for digit in range(10):
        indices = torch.where(test_y == digit)[0][:3]
        for idx in indices:
            pixels = (test_x[idx, 0].numpy()*255).round().astype(np.uint8)
            examples.append(dict(label=digit, pixels=base64.b64encode(pixels.tobytes()).decode()))
    data = dict(config=CONFIG, manifest=manifest, metadata=metadata, examples=examples,
                weights=base64.b64encode(weights).decode())
    (destination/'model-data.js').write_text('/* Local trained ViT. See README.md and tools/train.py. */\n'+
                                          'globalThis.VIT_DATA = '+json.dumps(data, separators=(',', ':'))+';\n', encoding='utf-8')
    (destination/'weights.bin').write_bytes(weights)
    (destination/'manifest.json').write_text(json.dumps(dict(config=CONFIG, manifest=manifest, metadata=metadata), indent=2), encoding='utf-8')
    fixtures = []
    with torch.no_grad():
        for idx in range(12):
            logits, trace = model(test_x[idx:idx+1], True)
            fixtures.append(dict(pixels=(test_x[idx, 0].numpy()*255).round().astype(np.uint8).reshape(-1).tolist(),
                                 logits=logits[0].tolist(), tokens=trace['tokens'][0].tolist(),
                                 attention=[b['attention'][0].tolist() for b in trace['blocks']],
                                 outputs=[b['output'][0].tolist() for b in trace['blocks']],
                                 label=int(test_y[idx])))
    fixture_folder=ROOT/'tests'/'fixtures'
    fixture_folder.mkdir(parents=True, exist_ok=True)
    (fixture_folder/'pytorch-reference.json.gz').write_bytes(gzip.compress(json.dumps(fixtures).encode('utf-8'), mtime=0))
    print(f'Exported {offset:,} parameters, {len(weights):,} bytes of weights', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    (ROOT/'work').mkdir(exist_ok=True)
    (ROOT/'docs').mkdir(exist_ok=True)
    checkpoint = ROOT/'work'/'best-model.pt'
    model = TinyViT()
    test_x, test_y = dataset('test')
    if args.export_only:
        saved = torch.load(checkpoint, weights_only=True)
        model.load_state_dict(saved['state'])
        export(model, test_x, test_y, saved['metadata'])
        return
    x, y = dataset('train')
    order = torch.randperm(len(x))
    train_x, train_y = x[order[:55000]], y[order[:55000]]
    val_x, val_y = x[order[55000:]], y[order[55000:]]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00008)
    best, history = 0, []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        indices = torch.randperm(len(train_x))
        total_loss, train_correct = 0, 0
        epoch_start = time.time()
        for start in range(0, len(indices), 256):
            batch = indices[start:start+256]
            image, labels = augment(train_x[batch]), train_y[batch]
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.03)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())*len(batch)
            train_correct += int((logits.argmax(1) == labels).sum())
        scheduler.step()
        val = accuracy(model, val_x, val_y)
        record = dict(epoch=epoch+1, loss=total_loss/len(train_x), train_accuracy=train_correct/len(train_x), validation_accuracy=val)
        history.append(record)
        print(f'Epoch {epoch+1}/{args.epochs} loss={record["loss"]:.4f} train={record["train_accuracy"]:.4%} val={val:.4%} seconds={time.time()-epoch_start:.1f}', flush=True)
        if val > best:
            best = val
            torch.save(dict(state=model.state_dict(), metadata={}), checkpoint)
        (ROOT/'docs'/'training-history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    model.load_state_dict(torch.load(checkpoint, weights_only=True)['state'])
    test = accuracy(model, test_x, test_y)
    metadata = dict(dataset='MNIST', training_samples=55000, validation_samples=5000, test_samples=10000,
                    test_accuracy=test, validation_accuracy=best, epochs=args.epochs, seed=42,
                    parameters=sum(p.numel() for p in model.parameters()), training_seconds=round(time.time()-started, 1),
                    normalization=dict(mean=0.1307, std=0.3081), augmentation='rotation +/-11.5 deg, translation +/-1.96 px, scale 0.9-1.1',
                    dataset_source='https://github.com/cvdfoundation/mnist',
                    architecture='Pre-LN ViT; 16 image patches + CLS; 3 blocks; 4 heads; dim 48; MLP 96; GELU(tanh)')
    torch.save(dict(state=model.state_dict(), metadata=metadata), checkpoint)
    export(model, test_x, test_y, metadata)
    print(f'FINAL test accuracy {test:.4%}; total {time.time()-started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
