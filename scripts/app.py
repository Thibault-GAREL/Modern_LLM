"""A web page to try M.A.I, or any checkpoint of this library, built with Gradio.

Two tabs, the same two things ``scripts/generate.py`` does in a terminal:

  **Continue a text**, streamed token by token, with the sampling knobs.
  **Next token**, the probability of each candidate, which is the only thing
  a language model actually computes.

    python scripts/app.py --ckpt outputs/models/mai-v1/mai-v1.pt
    python scripts/app.py --ckpt ... --share      # temporary public link
    python scripts/app.py --ckpt ... --port 7861

Then open http://127.0.0.1:7860. M.A.I needs about 0.5 GB, so it runs on any
GPU and on a CPU, where a 100 token answer takes a few seconds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mllm.demo import THEME, build_demo
from mllm.inference import DEFAULT_TOKENIZER, Tokenizer, load_model

__all__ = ["build_demo", "main"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="temporary public gradio.live link")
    args = p.parse_args()

    tokenizer = Tokenizer(args.tokenizer)
    loaded = load_model(args.ckpt)  # loaded once, reused by every request
    print(f"checkpoint : {loaded.describe()}")
    print(f"device     : {loaded.device}, tokenizer via {tokenizer.backend}")

    demo = build_demo(loaded, tokenizer)
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=THEME,
    )


if __name__ == "__main__":
    main()
