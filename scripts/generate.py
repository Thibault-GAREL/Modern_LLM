"""Run a trained checkpoint: continue a prompt, or inspect the next token.

    # continue a prompt
    python scripts/generate.py --ckpt outputs/models/<run>/final_model.pt \\
        --prompt "La capitale de la France est" --tokens 60

    # show what the model thinks comes next, with probabilities
    python scripts/generate.py --ckpt outputs/models/<run>/final_model.pt \\
        --prompt "La capitale de la France est" --next

Without ``--prompt`` it runs one English and one French prompt, which is the
quickest way to see whether a bilingual run learned both sides.

The tokenizer loads through ``transformers`` when it is installed, and falls
back to the much lighter ``tokenizers`` package otherwise, downloading the
32 kB ``tokenizer.json`` once into ``.cache/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mllm.config import Config
from mllm.generate import SamplingConfig, generate
from mllm.model import Transformer
from mllm.utils.numerics import pick_device

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = [
    ("EN", "The history of the city is"),
    ("FR", "L'histoire de la ville est"),
]


class Tokenizer:
    """Encode and decode, through whichever backend is installed.

    ``transformers`` is the usual path but it is a heavy dependency that also
    refuses to expose its torch bridge below torch 2.5. ``tokenizers`` alone is
    enough here: the model only needs ids in and ids out.
    """

    def __init__(self, name: str) -> None:
        try:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(name)
            self.backend = "transformers"
            self.encode = lambda text: self._tok(text)["input_ids"]
            self.decode = lambda ids: self._tok.decode(ids, skip_special_tokens=True)
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
        self.encode = lambda text: self._tok.encode(text).ids
        self.decode = lambda ids: self._tok.decode(ids, skip_special_tokens=True)

    def token_str(self, token_id: int) -> str:
        """A readable form of one token, spaces made visible."""
        raw = self.decode([token_id])
        return repr(raw) if raw.strip() != raw or not raw else raw


def load(ckpt_path: Path, device) -> tuple[Transformer, Config]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config.model_validate(ckpt["config"])
    model = Transformer(cfg.model).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"checkpoint : {ckpt_path.name}, step {ckpt['step']:,}")
    return model, cfg


@torch.no_grad()
def show_next_token(model, tokenizer, prompt: str, device, top_k: int) -> None:
    """Print the distribution over the next token, which is all the model does."""
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    logits, _, _ = model(ids)
    probs = logits[0, -1].float().softmax(dim=-1)
    top = torch.topk(probs, top_k)

    print(f"\nprompt : {prompt!r}")
    print(f"  {len(ids[0])} tokens in, {top_k} most likely continuations:\n")
    print(f"  {'token':<24} {'probability':>12}")
    print(f"  {'-' * 24} {'-' * 12}")
    for prob, idx in zip(top.values.tolist(), top.indices.tolist(), strict=True):
        bar = "#" * max(1, round(prob * 40))
        print(f"  {tokenizer.token_str(idx):<24} {prob:>11.2%}  {bar}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--tokenizer", type=str, default="croissantllm/CroissantLLMBase")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument(
        "--next",
        dest="show_next",
        action="store_true",
        help="show the next-token distribution instead of generating",
    )
    p.add_argument("--top-k-show", type=int, default=10, help="rows printed by --next")
    p.add_argument("--tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.15,
        help="1.0 disables it, and this model loops without it",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = pick_device()
    tokenizer = Tokenizer(args.tokenizer)
    model, _ = load(args.ckpt, device)
    print(f"device     : {device}, {model.n_params():,} parameters")
    print(f"tokenizer  : {args.tokenizer} via {tokenizer.backend}")

    prompts = [("--", args.prompt)] if args.prompt else DEFAULT_PROMPTS

    if args.show_next:
        for _, prompt in prompts:
            show_next_token(model, tokenizer, prompt, device, args.top_k_show)
        return

    cfg = SamplingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    # the generator must live on the same device as the logits, torch refuses
    # to mix them rather than moving one silently
    gen = torch.Generator(device=device).manual_seed(args.seed)
    for lang, prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out = generate(model, ids, args.tokens, cfg, generator=gen)
        print(f"\n[{lang}] {tokenizer.decode(out[0].tolist())}")


if __name__ == "__main__":
    main()

