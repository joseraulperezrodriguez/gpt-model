import math

import numpy as np
import pytest
import torch

from gpt_model.config import GPTConfig
from gpt_model.data import TokenDataset
from gpt_model.model import GPT


@pytest.fixture
def tiny_config() -> GPTConfig:
    # small dims, different from the defaults, to catch hardcoded values
    return GPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=4, n_embd=24)


@pytest.fixture
def tiny_model(tiny_config: GPTConfig) -> GPT:
    torch.manual_seed(0)
    return GPT(tiny_config)


def random_tokens(config: GPTConfig, batch: int = 2) -> torch.Tensor:
    return torch.randint(0, config.vocab_size, (batch, config.block_size))


def test_forward_shapes(tiny_model: GPT, tiny_config: GPTConfig):
    idx = random_tokens(tiny_config)
    logits, loss = tiny_model(idx, idx)
    assert logits.shape == (2, tiny_config.block_size, tiny_config.vocab_size)
    assert loss is not None and loss.ndim == 0
    _, no_loss = tiny_model(idx)
    assert no_loss is None


def test_causality_perturbing_future_leaves_past_logits_unchanged(
    tiny_model: GPT, tiny_config: GPTConfig
):
    torch.manual_seed(1)
    idx = random_tokens(tiny_config, batch=1)
    changed = idx.clone()
    changed[:, -4:] = torch.randint(0, tiny_config.vocab_size, (1, 4))

    logits_before, _ = tiny_model(idx)
    logits_after, _ = tiny_model(changed)

    torch.testing.assert_close(logits_before[:, :-4], logits_after[:, :-4])
    # the perturbed positions themselves must see the change
    assert not torch.allclose(logits_before[:, -4:], logits_after[:, -4:])


def test_param_count_matches_formula():
    config = GPTConfig()
    model = GPT(config)
    l, h, V, B = config.n_layer, config.n_embd, config.vocab_size, config.block_size
    # 12*h^2 per block (attn 4h^2 + mlp 8h^2), LN weights (2l+1)*h,
    # tied wte counted once; num_parameters(non_embedding=True) excludes wpe
    expected = 12 * l * h**2 + (2 * l + 1) * h + V * h
    assert model.num_parameters(non_embedding=True) == expected
    assert model.lm_head.weight is model.transformer.wte.weight  # tied


def test_initial_loss_is_near_ln_vocab():
    torch.manual_seed(2)
    config = GPTConfig(vocab_size=1000, block_size=32, n_layer=2, n_head=4, n_embd=64)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (4, config.block_size))
    _, loss = model(idx, idx)
    # Near-uniform predictions at init. Weight tying lets the current token's
    # own embedding leak a small positive boost into its logit, so the sane
    # window is ln(V) - 1 .. ln(V) + 0.25 rather than a tight equality.
    ln_v = math.log(config.vocab_size)
    assert ln_v - 1.0 < loss.item() < ln_v + 0.25


def test_generate_shapes_token_range_and_greedy_determinism(tiny_model: GPT, tiny_config: GPTConfig):
    idx = random_tokens(tiny_config, batch=2)[:, :4]
    for use_greedy, kwargs in [(True, {"temperature": 0.0}), (False, {"temperature": 1.0, "top_k": 10})]:
        torch.manual_seed(3)
        out = tiny_model.generate(idx, max_new_tokens=8, **kwargs)
        assert out.shape == (2, 4 + 8)
        assert out.min() >= 0 and out.max() < tiny_config.vocab_size
        assert torch.allclose(out[:, :4], idx)  # prompt preserved


def test_generate_crops_context_to_block_size():
    torch.manual_seed(4)
    config = GPTConfig(vocab_size=50, block_size=8, n_layer=1, n_head=2, n_embd=8)
    model = GPT(config)
    prompt = torch.randint(0, config.vocab_size, (1, config.block_size))
    out = model.generate(prompt, max_new_tokens=4, temperature=0.0)
    assert out.shape == (1, config.block_size + 4)


def write_bin_file(path, tokens: np.ndarray, vocab_hint: int) -> None:
    header = np.array([tokens.size, vocab_hint], dtype="<u4")
    np.concatenate([header, tokens.astype("<u4")]).tofile(path)


def test_token_dataset_reads_bin_and_shifts_targets(tmp_path):
    block_size = 4
    tokens = np.arange(20, dtype=np.uint32)
    path = str(tmp_path / "train.bin")
    write_bin_file(path, tokens, vocab_hint=100)

    ds = TokenDataset(path, block_size=block_size)
    assert len(ds) == tokens.size - block_size - 1
    x, y = ds[0]
    torch.testing.assert_close(x, torch.arange(0, 4, dtype=torch.int64))
    torch.testing.assert_close(y, torch.arange(1, 5, dtype=torch.int64))


def test_token_dataset_rejects_malformed_header(tmp_path):
    path = str(tmp_path / "bad.bin")
    np.array([10, 5], dtype="<u4").tofile(path)  # header claims 10, body has 0
    with pytest.raises(ValueError, match="header says"):
        TokenDataset(path, block_size=4)
