"""GPT model hyperparameters.

Every knob of the model lives here; model code contains no hardcoded
hyperparameters (a Phase 3 habit that starts now, not later).
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 32000  # ~32k BPE variant (see ai-stack Phase 2 plan)
    block_size: int = 256  # maximum context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384  # must be divisible by n_head
    dropout: float = 0.0  # raise (e.g. 0.1) for regularized real-data runs
    bias: bool = False  # no bias in Linear/LayerNorm, following GPT-2
    layernorm_impl: str = "torch"  # "torch" | "kernel" (custom CUDA, Phase 2)
