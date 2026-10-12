"""Load a trained checkpoint and its tokenizer, exactly the way training saw them.

Shared by ``scripts/generate.py``, the web demo ``scripts/app.py`` and the
benchmarks in ``bench/eval_model.py``, so a prompt is encoded one way only.

**A prompt starts with ``</s>``, not ``<s>``.** ``scripts/prepare_data.py``
tokenizes every document without special tokens and appends the end of
document id, so the token the model always saw just before a document starts
is ``</s>``. The tokenizer's default adds ``<s>`` instead, an id that never
appeared once in the six billion training tokens. Measured on the step 9999
checkpoint, over four English and French paragraphs::

    first token        mean loss
    <s>  (default)       2.775
    </s> (as trained)    2.716
    none                 2.917
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from mllm.config import Config
from mllm.model import Transformer
from mllm.utils.numerics import pick_device

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER = "croissantllm/CroissantLLMBase"
EOS_CANDIDATES = ("</s>", "<|endoftext|>", "<eos>")


class Tokenizer:
    """Encode and decode, through whichever backend is installed.

    ``transformers`` is tried first because it is what tokenized the training
    corpus. It is a heavy dependency that also refuses to expose its torch
    bridge below torch 2.5, so the much lighter ``tokenizers`` is enough as a
    fallback: the model only needs ids in and ids out. That fallback downloads
    the 32 kB ``tokenizer.json`` once into ``.cache/``.
    """

    def __init__(self, name: str = DEFAULT_TOKENIZER) -> None:
        self.name = name
        try:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(name)
            self.backend = "transformers"
            self.eos_id = self._tok.eos_token_id
            self._encode = lambda text: self._tok(text, add_special_tokens=False)["input_ids"]
            return
        except ImportError:
            pass

        try:
            from tokenizers import Tokenizer as FastTokenizer
        except ImportError as exc:  # noqa: TRY003
            raise SystemExit(
                "Neither transformers nor tokenizers is installed. "
                "The lighter of the two is enough:\n"
                "    pip install tokenizers"
            ) from exc

        path = REPO / ".cache" / f"{name.replace('/', '_')}_tokenizer.json"
        if not path.exists():
            import requests

            url = f"https://huggingface.co/{name}/resolve/main/tokenizer.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(requests.get(url, timeout=60).content)
            print(f"tokenizer  : downloaded to {path}")

        self._tok = FastTokenizer.from_file(str(path))
        self.backend = "tokenizers"
        self.eos_id = next(
            (i for t in EOS_CANDIDATES if (i := self._tok.token_to_id(t)) is not None), None
        )
        self._encode = lambda text: self._tok.encode(text, add_special_tokens=False).ids

    def encode(self, text: str) -> list[int]:
        """Token ids of ``text`` alone, with no special token added."""
        return list(self._encode(text))

    def encode_prompt(self, text: str) -> list[int]:
        """Ids of a prompt as a document start, ``</s>`` first like in training."""
        ids = self.encode(text)
        return [self.eos_id, *ids] if self.eos_id is not None else ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    def token_str(self, token_id: int) -> str:
        """A readable form of one token, spaces made visible."""
        raw = self.decode([token_id])
        return repr(raw) if raw.strip() != raw or not raw else raw


@dataclass
class LoadedModel:
    """A checkpoint ready for inference, with what is worth showing about it."""

    model: Transformer
    cfg: Config
    device: torch.device
    path: Path
    step: int

    @property
    def tokens_seen(self) -> int:
        t = self.cfg.train
        return (self.step + 1) * t.micro_batch_size * t.grad_accum_steps * t.seq_len

    @property
    def params_deployed(self) -> int:
        """Parameters used at inference, the MTP heads being training-only.

        The MTP modules hold a reference to the tied embedding, which the
        trunk keeps using, so only the parameters they own are taken out.
        """
        trunk = {
            id(p) for name, p in self.model.named_parameters() if not name.startswith("mtp.")
        }
        return sum(p.numel() for p in self.model.parameters() if id(p) in trunk)

    @property
    def max_len(self) -> int:
        return self.cfg.model.max_seq_len

    def describe(self) -> str:
        return (
            f"{self.path.name}, step {self.step:,}, "
            f"{self.tokens_seen / 1e9:.2f}B tokens seen, "
            f"{self.params_deployed / 1e6:.0f}M parameters deployed"
        )


def load_model(path: Path | str, device: torch.device | None = None) -> LoadedModel:
    """Rebuild the model from the config stored in the checkpoint, then load it.

    ``map_location`` keeps a checkpoint written on a GPU pod loadable on a
    CPU-only machine.
    """
    path = Path(path)
    device = device or pick_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.model_validate(ckpt["config"])
    model = Transformer(cfg.model).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return LoadedModel(model=model, cfg=cfg, device=device, path=path, step=int(ckpt["step"]))
