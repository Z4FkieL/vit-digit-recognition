# PyTorch reference fixture

`pytorch-reference.json.gz` contains 12 MNIST test images and their independently computed PyTorch results: token embeddings, per-head attention matrices, every encoder block output, and classification logits.

- Images: the first 12 entries of the original MNIST test split, from the [CVDF MNIST mirror](https://github.com/cvdfoundation/mnist).
- Model configuration and training metadata: `model/manifest.json`.
- Generator: `tools/train.py`, which exports this fixture along with the model.
- Consumer: `node tools/verify.js`, using Node's built-in gzip support.

The fixture is committed so validation works immediately after cloning, without downloading the dataset or installing Python/PyTorch. Full datasets and local training checkpoints remain in ignored `work/`.
