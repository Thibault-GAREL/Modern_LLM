"""Datasets: raw bytes for ablations, packed token ids for real training.

The ablations in ``docs/ablations.md`` run at byte level on purpose, so no
tokenizer choice contaminates an architecture comparison. Training an actual
model needs the opposite: a tokenizer chosen for the languages involved, and
the tokens packed once into a flat file rather than re-encoded every epoch.

``TokenDataset`` reads that file with ``numpy.memmap``, so a corpus larger than
RAM costs nothing to open and the OS page cache does the work. This is the
layout nanoGPT popularized, and it is hard to improve on for a single machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

# uint16 holds any vocabulary up to 65536 and halves the file size against
# uint32. Vocabularies above that must switch, which prepare_data records.
DEFAULT_DTYPE = np.uint16


class ByteDataset:
    """Raw bytes of a file, or deterministic noise when none is given.

    Byte level means no tokenizer, so an ablation measures the architecture
    rather than a vocabulary choice. ``vocab_size`` is 256 by construction.
    """

    VOCAB_SIZE = 256

    def __init__(self, path: Path | None, seq_len: int, device: torch.device) -> None:
        self.seq_len = seq_len
        self.device = device
        if path is not None and path.exists():
            # copy, since frombuffer would alias a read-only bytes object
            data = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8)
            self.source = f"{path} ({len(data):,} bytes)"
        else:
            # a learnable synthetic task, so the loss curve means something
            g = torch.Generator().manual_seed(0)
            base = torch.randint(0, 64, (1 << 16,), generator=g, dtype=torch.uint8)
            data = (base + torch.arange(len(base)) % 4).to(torch.uint8)
            self.source = "synthetic bytes (no --data given)"
        self.data = data.long()

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        hi = len(self.data) - self.seq_len - 1
        starts = torch.randint(0, hi, (batch_size,))
        x = torch.stack([self.data[s : s + self.seq_len] for s in starts])
        y = torch.stack([self.data[s + 1 : s + 1 + self.seq_len] for s in starts])
        return x.to(self.device), y.to(self.device)


class TokenDataset:
    """A flat ``.bin`` of token ids, read through a memory map.

    Expects the layout written by ``scripts/prepare_data.py``::

        <dir>/train.bin   token ids, dtype from meta.json
        <dir>/val.bin
        <dir>/meta.json   vocab_size, dtype, tokenizer, counts, mixture

    Args:
        directory: folder holding the two ``.bin`` files and ``meta.json``.
        split: ``"train"`` or ``"val"``.
        seq_len: context length of one sample.
        device: where batches are moved to.
        seed: sampling seed. Fixing it on the validation split is what makes
            two runs comparable.
    """

    def __init__(
        self,
        directory: Path,
        split: str,
        seq_len: int,
        device: torch.device,
        *,
        seed: int | None = None,
    ) -> None:
        directory = Path(directory)
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found. Run scripts/prepare_data.py first."
            )
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.dtype = np.dtype(self.meta.get("dtype", "uint16"))
        self.vocab_size = int(self.meta["vocab_size"])

        path = directory / f"{split}.bin"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        self.path = path
        self.split = split
        self.seq_len = seq_len
        self.device = device
        # mode="r" keeps it read-only, so two processes can share the pages
        self.tokens = np.memmap(path, dtype=self.dtype, mode="r")
        if len(self.tokens) <= seq_len + 1:
            raise ValueError(
                f"{path} holds {len(self.tokens)} tokens, too few for seq_len={seq_len}"
            )
        # An unseeded torch.Generator carries a fixed default seed, so leaving
        # it alone would make every dataset draw the same windows. Deriving the
        # seed from the global RNG instead means set_determinism still controls
        # the run, while two datasets built in sequence differ.
        self.generator = torch.Generator()
        self.generator.manual_seed(
            seed if seed is not None else int(torch.randint(0, 2**31 - 1, (1,)).item())
        )

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @property
    def source(self) -> str:
        mixture = self.meta.get("mixture")
        detail = f", mixture {mixture}" if mixture else ""
        return (
            f"{self.path.name}: {self.n_tokens:,} tokens, vocab {self.vocab_size:,}"
            f"{detail}"
        )

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``batch_size`` windows uniformly at random.

        Random offsets rather than a sequential cursor, which is what lets a
        run resume mid-corpus without tracking a position, and keeps every
        batch independent.
        """
        hi = self.n_tokens - self.seq_len - 1
        starts = torch.randint(0, hi, (batch_size,), generator=self.generator)
        # the copy is needed: torch cannot own memmap-backed memory
        x = torch.stack(
            [
                torch.from_numpy(
                    self.tokens[int(s) : int(s) + self.seq_len].astype(np.int64)
                )
                for s in starts
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    self.tokens[int(s) + 1 : int(s) + 1 + self.seq_len].astype(np.int64)
                )
                for s in starts
            ]
        )
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def build_dataset(
    data_dir: Path | None,
    data_file: Path | None,
    seq_len: int,
    device: torch.device,
    *,
    split: str = "train",
    seed: int | None = None,
):
    """Pick the dataset implied by the arguments.

    A directory means prepared tokens, a file means raw bytes, neither means
    the synthetic byte stream used by the smoke tests.
    """
    if data_dir is not None:
        return TokenDataset(data_dir, split, seq_len, device, seed=seed)
    return ByteDataset(data_file, seq_len, device)
