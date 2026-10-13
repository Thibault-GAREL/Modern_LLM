"""Hugging Face Space of M.A.I, the Gradio page of Modern LLM on weights from the Hub.

The page itself is ``mllm.demo.build_demo``, installed from the GitHub
repository (see requirements.txt), so the Space and ``scripts/app.py`` show
exactly the same thing. Only the weights come from elsewhere: the model repo
named by ``MAI_REPO``, downloaded once at start.
"""

from __future__ import annotations

import os

# set before mllm.inference is imported, it reads the variable at import
os.environ.setdefault("MLLM_CACHE", "/tmp/mllm-cache")

import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from mllm.demo import THEME, build_demo  # noqa: E402
from mllm.inference import Tokenizer, load_model  # noqa: E402

REPO_ID = os.environ.get("MAI_REPO", "Thibault-GAREL/mai-v2")
FILENAME = os.environ.get("MAI_FILE", "mai-v2.pt")

weights = hf_hub_download(REPO_ID, FILENAME)
loaded = load_model(weights, torch.device("cpu"))  # the free tier has no GPU
tokenizer = Tokenizer()
demo = build_demo(loaded, tokenizer)

if __name__ == "__main__":
    demo.queue().launch(theme=THEME)
