# gpt-model

Tiny GPT decoder written in raw PyTorch — ai-stack **Phase 2**.
Structurally identical to production LLMs, sized to train on a single
consumer GPU: 6 layers, 6 heads, 384 embedding dims.

Part of the [ai-stack](../ai-stack/README.md) project. Tokenization comes
from [bpe-tokenizer](../bpe-tokenizer/README.md); the custom CUDA kernel
reuses work from
[gpu_programming](https://github.com/joseraulperezrodriguez/gpu_programming).

## Design

| Component | Choice |
|---|---|
| Attention | hand-written causal MHSA (scores, mask, softmax, weighted sum as explicit steps) |
| Blocks | 6 × pre-LN residual blocks |
| MLP | 4× expansion, GELU |
| Positions | learned `wpe` embedding |
| Head | weight-tied `lm_head` ↔ `wte` |
| Context | 256 tokens (config: `block_size`) |
| Vocab | ~32k BPE variant (config: `vocab_size`) |

All hyperparameters live in `gpt_model/config.py` (`GPTConfig`); the model
code contains no hardcoded values. `layernorm_impl` on the config selects
between `"torch"` and the planned custom CUDA kernel (parity < 1e-5 fp32).

## Usage

```python
from gpt_model import GPT, GPTConfig

config = GPTConfig()
model = GPT(config)
logits, loss = model(idx, targets)   # idx: (B, T) int64, targets: (B, T)
out = model.generate(idx, max_new_tokens=64, temperature=0.8, top_k=50)
```

## Data

`gpt_model/data.py` reads flat token files (`data/train.bin`,
`data/val.bin`) written by the bpe-tokenizer export mode: little-endian
uint32 header `(n_tokens, vocab_size_hint)` followed by uint32 token ids.
The files are gitignored and regenerable from the corpus in one command
(spec: [`bpe-tokenizer/docs/data-bridge-spec.md`](../bpe-tokenizer/docs/data-bridge-spec.md)):

```
bpe-tokenizer --model models/wiki-en-100k.bin --export <wikiextractor-text-dir> data --val-ratio 10 --skip-tags
```

## Status

Implemented in level-descending iterations — see the [iteration ladder in
model-spec.md](model-spec.md) for the full plan and its deltas vs. the spec.

- [x] Model skeleton: config, attention, MLP, blocks, tied head, sampling
- [x] Unit tests: shapes, causality, param count, init loss ≈ ln(vocab) — 8/8
- [x] Data bridge: bpe-tokenizer `--export` mode → `data/train.bin` / `data/val.bin`
      (verified end-to-end: export → loader → real-data forward through the model)
- [ ] **Iteration 1 (L0, "make it real"):** green tests, regression
      harness, M1 gradcheck, M2 real-data forward, M3 throwaway training run,
      M5 greedy sampling
- [ ] Iteration 2 (L1, "own the math"): hand-written LN/softmax/GELU/linears
- [ ] Iteration 3 (L2, "own the gradients"): layernorm `autograd.Function`
- [ ] Iteration 4 (L3, "own the hardware"): custom CUDA layer (M4)
- [ ] `.venv` here is CPU-only torch; CUDA kernel work needs a GPU host
