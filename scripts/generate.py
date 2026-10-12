"""Run a trained checkpoint: continue a prompt, or inspect the next token.

    # continue a prompt with M.A.I v1
    python scripts/generate.py --ckpt outputs/models/mai-v1/mai-v1.pt \\
        --prompt "La capitale de la France est" --tokens 60

    # show what the model thinks comes next, with probabilities
    python scripts/generate.py --ckpt outputs/models/mai-v1/mai-v1.pt \\
        --prompt "La capitale de la France est" --next

Without ``--prompt`` it runs one English and one French prompt, which is the
quickest way to see whether a bilingual run learned both sides. For a web page
instead of a terminal, see ``scripts/app.py``.

Loading and tokenization live in ``mllm.inference``, shared with the demo and
the benchmarks, so a prompt is encoded the same way everywhere: ``</s>`` first,
like every document the model was trained on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mllm.generate import SamplingConfig, generate
from mllm.inference import DEFAULT_TOKENIZER, Tokenizer, load_model

DEFAULT_PROMPTS = [
    ("EN", "The history of the city is"),
    ("FR", "L'histoire de la ville est"),
]


@torch.no_grad()
def next_token_distribution(model, tokenizer, prompt: str, device, top_k: int):
    """The ``top_k`` most likely next tokens, as ``(token, probability)`` pairs."""
    ids = torch.tensor([tokenizer.encode_prompt(prompt)], dtype=torch.long, device=device)
    logits, _, _ = model(ids)
    probs = logits[0, -1].float().softmax(dim=-1)
    top = torch.topk(probs, top_k)
    pairs = [
        (tokenizer.token_str(idx), prob)
        for prob, idx in zip(top.values.tolist(), top.indices.tolist(), strict=True)
    ]
    return len(ids[0]), pairs


def show_next_token(model, tokenizer, prompt: str, device, top_k: int) -> None:
    """Print the distribution over the next token, which is all the model does."""
    n_in, pairs = next_token_distribution(model, tokenizer, prompt, device, top_k)
    print(f"\nprompt : {prompt!r}")
    print(f"  {n_in} tokens in, {top_k} most likely continuations:\n")
    print(f"  {'token':<24} {'probability':>12}")
    print(f"  {'-' * 24} {'-' * 12}")
    for token, prob in pairs:
        bar = "#" * max(1, round(prob * 40))
        print(f"  {token:<24} {prob:>11.2%}  {bar}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
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

    tokenizer = Tokenizer(args.tokenizer)
    loaded = load_model(args.ckpt)
    model, device = loaded.model, loaded.device
    print(f"checkpoint : {loaded.describe()}")
    print(f"device     : {device}")
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
        ids = torch.tensor([tokenizer.encode_prompt(prompt)], dtype=torch.long, device=device)
        out = generate(model, ids, args.tokens, cfg, generator=gen, eos_id=tokenizer.eos_id)
        print(f"\n[{lang}] {tokenizer.decode(out[0].tolist())}")


if __name__ == "__main__":
    main()
