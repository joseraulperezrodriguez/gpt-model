# Phase 2 Spec — Model Implementation (Transformer Decoder)

**Scope:** Tiny GPT decoder in raw PyTorch — no `nn.TransformerDecoder`, no `nn.MultiheadAttention`. All attention math hand-written with `@` and softmax. Target architecture: **6 layers, 6 heads, 384 dim** (decided; see plan.md).

**Context:** Input is the `.bin` token file produced in Phase 1 (English Wikipedia subset, BPE tokenizer).

---

## Architecture

| Component | Detail |
|---|---|
| Token embedding | `nn.Embedding(vocab_size, 384)` — vocab size comes from the Phase 1 tokenizer |
| Positional encoding | Learned `nn.Parameter` (simplest; positional sinusoids optional comparison) |
| Block × 6 | Each: LN → MHA → residual → LN → MLP → residual (pre-LN, as in nanoGPT) |
| Multi-head attention | 6 heads × 64 dim/head; causal mask; hand-written softmax(@) math |
| MLP | 384 → 1536 → 384 (4× expansion), GELU |
| Final layernorm + LM head | Weight-tied with token embedding (optional; decide during implementation) |
| Dropout | 0.1 during training (tunable via config) |

Params ≈ ~10–15M depending on vocab size — comfortably fits an 8GB GPU.

## Hand-written attention (no torch built-ins)

- `att = (q @ k.transpose(-2,-1)) * (1/sqrt(head_dim))`
- Causal mask: set upper-triangle positions to `-inf` before softmax
- `y = att @ v`, then concatenate heads, project with `Wo`

## Custom CUDA kernel (required, pick one)

- **Attention** (recommended: batched matmuls + softmax, builds on prior matmul-kernel work) or **LayerNorm**
- Acceptance: max abs diff < 1e-5 vs. PyTorch in fp32, on random tensors and inside the full model

## Milestones

1. **M1 — Single block forward pass.** Shapes correct; manual gradient check on a tiny input (`torch.autograd.gradcheck` on a few params).
2. **M2 — Full model forward.** Random init, forward a batch from the real `.bin`; output logits shape `(B, T, vocab)`; loss computable.
3. **M3 — Sanity training run.** A few hundred steps of a quick training loop (borrow Phase 3 loop minimally); loss drops below ~4 and keeps decreasing.
4. **M4 — Custom CUDA kernel.** Matches PyTorch within 1e-5; swapped into the model with no loss divergence; measured speedup/slowdown recorded.
5. **M5 — Sample generation (greedy).** Coherent token-level samples on Shakespeare-like or Wikipedia text.

## Exit criteria (Phase 2 done)

- Loss decreases steadily on a tiny run with coherent samples
- Custom kernel within tolerance (max abs diff < 1e-5, fp32)
- Model config (layers/heads/dim/vocab) lives in a config file, not hardcoded
- Everything importable as a module the Phase 3 training loop can consume

## Open questions to decide while implementing

- Weight tying on/off (measure val loss both ways)
- Learned vs. sinusoidal positional encodings
- Kernel target: attention vs. layernorm (attention recommended)
- Context length (block size) for training: 256 to start?

## References

- nanoGPT, llm.c, minbpe (Karpathy) — implement first, compare after
- Original plan: `custom-stack.txt` (project library)
- Review amendments: `custom-stack-amendments.md`

---

## Iteration ladder — how the milestones get implemented

The milestones above are milestone-ordered; they are *implemented* in four
iterations, each at a lower abstraction level than the last. Lower levels are
added behind per-component config flags (`*_impl`); nothing is deleted, so
every level doubles as a reference oracle for the next.

| Iteration | Level | Components | Covers |
|---|---|---|---|
| 1 — "make it real" | `nn`/`F` primitives; attention recipe hand-written (spec floor) | whole model | M1, M2, M3, M5 + regression harness |
| 2 — "own the math" | raw `Parameter` + `@`; hand-written LN / softmax / GELU; embeddings as indexing | one swap per commit | regression vs iteration 1 goldens |
| 3 — "own the gradients" | `autograd.Function` with hand-derived fwd + bwd | layernorm only | gradcheck (fp64), grad parity vs autograd |
| 4 — "own the hardware" | CUDA kernel via `torch.utils.cpp_extension` | layernorm (recommended) or attention | M4 |

### Iteration 1 — expanded plan

Goal: the whole pipeline (data → forward → train → sample) works end to end
at the highest reasonable level, and is captured in a regression harness that
every later iteration is checked against. No low-level rewrites before this
exists — descending without a known-good oracle is how kernel and tensor-math
bugs survive for days.

Tasks:

1. **Data bridge** (in `bpe-tokenizer`): a bulk-encode mode that walks the
   wikiextractor text tree and writes `data/train.bin` + `data/val.bin` in the
   format `gpt_model/data.py` reads (LE u32 header `[n_tokens, vocab_hint]`,
   then u32 token ids — u32 because vocab can exceed uint16). One command,
   seeded 90/10 split by file. M2 needs this; the scaffold deferred it.
2. **Green test baseline:** resolve the two known test-expectation issues
   (param-count convention excludes `wpe`; the init-loss window must account
   for the weight-tying logit boost) → 8/8 before touching anything else.
3. **Regression harness:** fixed seed, fixed data batch; save goldens with
   `torch.save` — init loss, logits for the batch, a ~50-step loss curve.
   A pytest regenerates deterministically and compares (exact for same-seed
   runs; tolerance only where reductions reorder). Document the golden
   regeneration command in the README.
4. **M1 — single block:** forward shapes, then `torch.autograd.gradcheck`
   on a double-precision input. Keep every component dtype-generic so
   hand-written ops (iteration 2/3) survive gradcheck too.
5. **M2 — full model on real data:** forward a real `.bin` batch, assert
   logits `(B, T, vocab)`, compute cross-entropy.
6. **M3 — throwaway training loop:** ~300 steps; `TokenDataset` batching,
   AdamW, fixed seed, CSV log of (step, loss). Acceptance per spec: loss
   below ~4 and still decreasing. Borrow nothing more from Phase 3.
7. **M5 — greedy sampling:** small script, a handful of fixed prompts; save
   outputs (seed of the Phase 3.5 prompt set).
8. **Record the open-question decisions** (spec section above):
   - Weight tying: ON to start; the A/B is a one-line toggle during M3.
   - Positional: learned (spec default); sinusoidal is an iteration-2
     comparison candidate.
   - Dropout: 0.1 during training, always config-driven.
   - Context: 256.
   - Vocab: pin before the M2 loss baseline. Note the spec's "~10–15M params"
     implies ~16k vocab; at 32k with tying it is ~23M, at 100k ~49M.

Acceptance (iteration 1 done) = M1 + M2 + M3 + M5 pass, regression harness
operational, data regenerable by one command.

### Deltas vs. the original milestone list

- **M5 (sampling) moves earlier** — iteration 1 instead of after M4. Greedy
  generation is trivial at L0 and gives all later iterations (and Phase 3.5)
  the eyeball tooling they need.
- **M4 (kernel) moves last** — it is the only milestone that *needs* the
  reference model as its parity oracle, so the ladder schedules it after
  one exists.
- All other milestones map 1:1, with their acceptance criteria unchanged —
  the ladder is purely a scheduling and validation strategy.
- Kernel recommendation differs from the spec's: layernorm over attention
  (simpler parity story, memory-bound win over eager PyTorch). The decision
  stays open until iteration 4, as the spec says.
