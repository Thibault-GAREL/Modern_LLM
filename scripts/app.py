"""A web page to try a trained checkpoint, built with Gradio.

Two tabs, the same two things ``scripts/generate.py`` does in a terminal:

  **Continue a text**, streamed token by token, with the sampling knobs.
  **Next token**, the probability of each candidate, which is the only thing
  a language model actually computes.

    python scripts/app.py --ckpt outputs/models/<run>/final_model.pt
    python scripts/app.py --ckpt ... --share      # temporary public link
    python scripts/app.py --ckpt ... --port 7861

Then open http://127.0.0.1:7860. The 111M model needs about 0.5 GB, so it runs
on any GPU and on a CPU, where a 100 token answer takes a few seconds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import torch

from mllm.generate import SamplingConfig, stream
from mllm.inference import DEFAULT_TOKENIZER, LoadedModel, Tokenizer, load_model

EXAMPLES = [
    ["La capitale de la France est"],
    ["The history of the city is"],
    ["Il était une fois, dans un petit village de montagne,"],
    ["Photosynthesis is the process by which"],
    ["Le système solaire est composé"],
    ["The best way to learn a new language is"],
]

INTRO = """\
# 🧬 Modern LLM, the bilingual {params}M model

{describe}. A **base model**: it continues a text in English or French, it does
not follow instructions or answer questions reliably. Write the beginning of a
sentence, not a question.
"""


def build_demo(loaded: LoadedModel, tokenizer: Tokenizer) -> gr.Blocks:
    model, device = loaded.model, loaded.device

    def continue_text(prompt, max_tokens, temperature, top_p, repetition_penalty, seed):
        """Yield the text as it grows, prompt and continuation highlighted apart."""
        if not prompt.strip():
            yield [("Write the beginning of a sentence first.", None)]
            return
        ids = tokenizer.encode_prompt(prompt)
        # stay inside the context length the positions were trained on
        max_tokens = int(min(max_tokens, loaded.max_len - 1))
        ids = ids[-(loaded.max_len - max_tokens):]
        x = torch.tensor([ids], dtype=torch.long, device=device)

        cfg = SamplingConfig(
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            greedy=float(temperature) == 0.0,
        )
        gen = torch.Generator(device=device).manual_seed(int(seed))
        # decoding the prompt and the continuation together keeps the space
        # a SentencePiece token carries in front of it
        prompt_text = tokenizer.decode(ids)
        new: list[int] = []
        for token in stream(model, x, max_tokens, cfg, generator=gen, eos_id=tokenizer.eos_id):
            token_id = int(token[0, 0])
            if token_id == tokenizer.eos_id:
                break
            new.append(token_id)
            text = tokenizer.decode(ids + new)
            yield [(prompt, None), (text[len(prompt_text):], "model")]

    @torch.no_grad()
    def next_token(prompt, top_k):
        if not prompt.strip():
            return {}
        x = torch.tensor([tokenizer.encode_prompt(prompt)], dtype=torch.long, device=device)
        logits, _, _ = model(x[:, -loaded.max_len:])
        probs = logits[0, -1].float().softmax(dim=-1)
        top = torch.topk(probs, int(top_k))
        shown: dict[str, float] = {}
        for prob, idx in zip(top.values.tolist(), top.indices.tolist(), strict=True):
            label = tokenizer.token_str(idx)
            # two ids can print alike, a label must stay unique
            shown[label if label not in shown else f"{label} (id {idx})"] = prob
        return shown

    with gr.Blocks(title="Modern LLM") as demo:
        gr.Markdown(
            INTRO.format(
                params=round(loaded.params_deployed / 1e6), describe=loaded.describe()
            )
        )

        with gr.Tab("✍️ Continue a text"):
            with gr.Row():
                with gr.Column(scale=3):
                    prompt = gr.Textbox(
                        label="Beginning of the text",
                        value="La capitale de la France est",
                        lines=3,
                    )
                    run = gr.Button("Continue", variant="primary")
                    output = gr.HighlightedText(
                        label="Prompt, then what the model wrote",
                        color_map={"model": "#d946ef"},
                        combine_adjacent=True,
                        show_legend=False,
                    )
                with gr.Column(scale=2):
                    max_tokens = gr.Slider(10, 400, value=80, step=10, label="New tokens")
                    temperature = gr.Slider(
                        0.0, 1.5, value=0.9, step=0.05,
                        label="Temperature (0 always takes the most likely token)",
                    )
                    top_p = gr.Slider(0.1, 1.0, value=0.92, step=0.01, label="Top-p")
                    repetition_penalty = gr.Slider(
                        1.0, 1.5, value=1.15, step=0.01,
                        label="Repetition penalty (at 1.0 this model loops)",
                    )
                    seed = gr.Number(value=0, precision=0, label="Seed, same seed same text")
            gr.Examples(EXAMPLES, inputs=[prompt])
            inputs = [prompt, max_tokens, temperature, top_p, repetition_penalty, seed]
            run.click(continue_text, inputs=inputs, outputs=output)
            prompt.submit(continue_text, inputs=inputs, outputs=output)

        with gr.Tab("🎲 Next token"):
            gr.Markdown(
                "A language model computes one thing: a probability for every token "
                "of its 32,000 token vocabulary. Here are the most likely ones."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    next_prompt = gr.Textbox(
                        label="Text so far", value="La capitale de la France est", lines=2
                    )
                    show = gr.Button("Show the next token", variant="primary")
                with gr.Column(scale=1):
                    top_k = gr.Slider(3, 20, value=10, step=1, label="Candidates shown")
            distribution = gr.Label(label="Next token probabilities")
            gr.Examples(EXAMPLES, inputs=[next_prompt])
            show.click(next_token, inputs=[next_prompt, top_k], outputs=distribution)
            next_prompt.submit(next_token, inputs=[next_prompt, top_k], outputs=distribution)

    return demo


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
        theme=gr.themes.Soft(primary_hue="fuchsia"),
    )


if __name__ == "__main__":
    main()
