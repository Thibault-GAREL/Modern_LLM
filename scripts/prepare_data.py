"""Download, tokenize and mix a bilingual corpus into flat token files.

Produces the layout ``mllm.data.TokenDataset`` expects::

    <out>/train.bin   token ids as uint16
    <out>/val.bin
    <out>/meta.json   vocab size, tokenizer, per-language counts, mixture

Defaults target a French and English model:

  **English** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  ``sample/10BT``, Common Crawl filtered by an educational-quality classifier.

  **French** [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
  ``fra_Latn``, the same pipeline with per-language filters and stopwords.

  **Tokenizer** [CroissantLLM](https://huggingface.co/croissantllm/CroissantLLMBase),
  32k, trained for a French and English model. An English-centric tokenizer
  spends 30 to 50% more tokens on the same French text, which is 30 to 50% more
  compute for the same content.

**Why shards and not streaming.** Measured on a RunPod RTX 4090 pod:
streaming document by document sustains about 10k tokens/s, which is 167 hours
for 6B tokens. Downloading the parquet shards directly runs at 25 MiB/s and
batch tokenization at 3.2M tokens/s, which brings the same job under an hour.
The bottleneck was never the tokenizer.

    python scripts/prepare_data.py --out /workspace/data/bilingual --tokens 6e9
    python scripts/prepare_data.py --out /tmp/tiny --tokens 2e6   # smoke test
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

DTYPE = np.uint16  # any vocabulary below 65536 fits, and it halves the file size

SOURCES = {
    "en": {"repo": "HuggingFaceFW/fineweb-edu", "prefix": "sample/10BT/"},
    "fr": {"repo": "HuggingFaceFW/fineweb-2", "prefix": "data/fra_Latn/train/"},
}


def build_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.vocab_size >= 2**16:
        raise SystemExit(
            f"{name} has a {tok.vocab_size} vocabulary, which does not fit in uint16."
        )
    if tok.eos_token_id is None:
        raise SystemExit(f"{name} has no eos token, cannot mark document boundaries")
    return tok


def list_shards(lang: str) -> list[str]:
    from huggingface_hub import list_repo_files

    src = SOURCES[lang]
    files = [
        f
        for f in list_repo_files(src["repo"], repo_type="dataset")
        if f.startswith(src["prefix"]) and f.endswith(".parquet")
    ]
    return sorted(files)


def fetch(lang: str, remote: str, scratch: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            SOURCES[lang]["repo"],
            remote,
            repo_type="dataset",
            local_dir=str(scratch / lang),
        )
    )


def tokenize_shard(
    path: Path, tokenizer, budget: int | None = None, batch: int = 1000
) -> np.ndarray:
    """Read one parquet shard and return its token ids, up to ``budget``.

    Two things this must not do, both learned the hard way on the pod. It must
    not read the whole shard into Python (``to_pylist`` on a 2 GB parquet plus
    714M accumulated tokens got the process killed by the OOM killer), and it
    must not tokenize a whole shard to keep a fraction of it. Reading row
    batches and stopping at the budget fixes both.

    Batching the tokenizer is a 4.6x speedup over one call per document, since
    the fast tokenizer parallelizes inside Rust.
    """
    import pyarrow.parquet as pq

    eos = tokenizer.eos_token_id
    chunks: list[np.ndarray] = []
    produced = 0

    parquet = pq.ParquetFile(path)
    for record_batch in parquet.iter_batches(batch_size=batch, columns=["text"]):
        window = [t for t in record_batch.column("text").to_pylist() if t]
        if not window:
            continue
        for ids in tokenizer(window, add_special_tokens=False)["input_ids"]:
            ids.append(eos)  # a document boundary the model can learn to respect
            chunks.append(np.asarray(ids, dtype=DTYPE))
            produced += len(ids)
        if budget is not None and produced >= budget:
            break

    if not chunks:
        return np.empty(0, dtype=DTYPE)
    tokens = np.concatenate(chunks)
    return tokens[:budget] if budget is not None else tokens


def write_split(
    tokenizer,
    budgets: dict[str, int],
    out_dir: Path,
    split: str,
    scratch: Path,
    *,
    workers: int = 4,
) -> dict[str, int]:
    """Fill one ``.bin`` by alternating languages, shard by shard.

    Interleaving happens at shard granularity rather than per document, which
    is enough because ``TokenDataset`` samples windows at uniformly random
    offsets: a batch mixes languages whatever the file order. Concatenating
    would only matter for a reader that walks the file sequentially.

    Shards are deleted as soon as they are tokenized, so peak disk stays at a
    couple of shards rather than the whole corpus.
    """
    remaining = {k: v for k, v in budgets.items() if v > 0}
    queues = {lang: list_shards(lang) for lang in remaining}
    counts = {lang: 0 for lang in remaining}
    path = out_dir / f"{split}.bin"

    bar = tqdm(
        total=sum(remaining.values()), unit="tok", unit_scale=True, desc=f"{split:<6}"
    )
    with path.open("wb") as fh, ThreadPoolExecutor(max_workers=workers) as pool:
        prefetch: dict[str, object] = {}
        while remaining:
            for lang in list(remaining):
                if lang not in prefetch and queues[lang]:
                    prefetch[lang] = pool.submit(fetch, lang, queues[lang].pop(0), scratch)

            for lang in list(remaining):
                if not queues[lang] and lang not in prefetch:
                    print(f"[warning] {lang}: ran out of shards, {remaining[lang]:,} short")
                    remaining.pop(lang)
                    continue
                future = prefetch.pop(lang, None)
                if future is None:
                    continue
                shard = future.result()
                # never tokenize more than we are going to keep
                keep = tokenize_shard(shard, tokenizer, budget=remaining[lang])
                shard.unlink(missing_ok=True)

                fh.write(keep.tobytes())
                counts[lang] += len(keep)
                remaining[lang] -= len(keep)
                bar.update(len(keep))
                if remaining[lang] <= 0:
                    remaining.pop(lang)

        # a language can finish while its next shard is still downloading, and
        # that file would otherwise sit on disk until the pod dies
        for future in prefetch.values():
            try:
                future.result().unlink(missing_ok=True)
            except Exception:  # noqa: BLE001 - a failed download has nothing to clean
                pass
    bar.close()
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokens", type=float, default=6e9, help="training tokens to produce")
    p.add_argument(
        "--en-ratio", type=float, default=0.7, help="share of English, the rest is French"
    )
    p.add_argument("--val-tokens", type=float, default=5e6)
    p.add_argument("--tokenizer", type=str, default="croissantllm/CroissantLLMBase")
    p.add_argument(
        "--scratch",
        type=Path,
        default=Path("/tmp/mt_shards"),
        help="where parquet shards land, deleted as they are consumed. Point this "
        "at the big ephemeral disk, not at the persistent volume",
    )
    p.add_argument("--workers", type=int, default=4, help="parallel shard downloads")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not 0.0 <= args.en_ratio <= 1.0:
        raise SystemExit("--en-ratio must lie in [0, 1]")

    out, scratch = args.out, args.scratch
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(args.tokenizer)

    total, val_total = int(args.tokens), int(args.val_tokens)
    plan = {
        "val": {
            "en": int(val_total * args.en_ratio),
            "fr": val_total - int(val_total * args.en_ratio),
        },
        "train": {
            "en": int(total * args.en_ratio),
            "fr": total - int(total * args.en_ratio),
        },
    }

    print(f"tokenizer : {args.tokenizer}, vocab {tokenizer.vocab_size:,}")
    print(f"mixture   : {args.en_ratio:.0%} English, {1 - args.en_ratio:.0%} French")
    print(f"scratch   : {scratch}")
    print(f"output    : {out.resolve()}\n")

    t0 = time.perf_counter()
    counts = {
        split: write_split(tokenizer, budgets, out, split, scratch, workers=args.workers)
        for split, budgets in plan.items()
    }
    shutil.rmtree(scratch, ignore_errors=True)

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "dtype": str(np.dtype(DTYPE)),
        "tokenizer": args.tokenizer,
        "mixture": {"en": args.en_ratio, "fr": round(1 - args.en_ratio, 4)},
        "sources": SOURCES,
        "counts": counts,
        "seed": args.seed,
        "prepared_s": round(time.perf_counter() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\ndone in {meta['prepared_s'] / 60:.0f} min")
    for split, c in counts.items():
        size = (out / f"{split}.bin").stat().st_size / 2**30
        print(f"  {split:<6} {sum(c.values()):>14,} tokens  {size:>6.1f} GiB  {c}")


if __name__ == "__main__":
    main()
