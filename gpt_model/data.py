"""Flat uint32 memmap token datasets, produced by the bpe-tokenizer bridge.

File format (little-endian):
    - header: 8 bytes = two uint32: [n_tokens, vocab_size_hint]
    - body: n_tokens * 4 bytes of uint32 token ids, native token id space
      of the BPE model that produced them (uint32 because vocab > 65535 is
      possible with this tokenizer).

The bpe-tokenizer side (still to be added) writes train.bin and val.bin with
exactly this header, so the files are self-describing regardless of vocab.
"""

import numpy as np
import torch

HEADER_N_TOKENS = 0
HEADER_VOCAB_HINT = 1
HEADER_UINT32_COUNT = 2  # header size in uint32 units


def read_bin_file(path: str) -> np.ndarray:
    """Load a whole token .bin file into a uint32 numpy array."""
    data = np.fromfile(path, dtype="<u4")
    if data.size < HEADER_UINT32_COUNT:
        raise ValueError(f"{path}: too short to contain a header")
    n_tokens = int(data[HEADER_N_TOKENS])
    body = data[HEADER_UINT32_COUNT:]
    if body.size != n_tokens:
        raise ValueError(f"{path}: header says {n_tokens} tokens, body has {body.size}")
    return body


class TokenDataset(torch.utils.data.Dataset):
    """memmap-backed dataset; get(i) returns a (block_size+1,) token window.

    Index i is the window start; x = tokens[i : i+block_size],
    y = tokens[i+1 : i+block_size+1] (next-token targets).
    """

    def __init__(self, path: str, block_size: int):
        self.tokens = read_bin_file(path)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, self.tokens.size - self.block_size - 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(
            self.tokens[index : index + self.block_size].astype(np.int64)
        )
        y = torch.from_numpy(
            self.tokens[index + 1 : index + self.block_size + 1].astype(np.int64)
        )
        return x, y
