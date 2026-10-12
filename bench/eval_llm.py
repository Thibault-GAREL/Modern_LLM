"""Benchmark a model that answers in text, such as Claude, on the same items.

A chat model cannot be scored by log-likelihood from outside, so it reads each
question with lettered choices and writes its answer. The protocol keeps the
answers honest:

  1. ``export`` writes the questions and, into a separate file, the expected
     answers. It prints counts only, never a key.
  2. The answerer reads ``questions.jsonl`` and nothing else: no key file, no
     dataset, no web, no code, no other model. It writes one line per item to
     an answers file, ``{"uid": ..., "answer": "B"}``.
  3. ``score`` hashes the answers file first, then compares it to the keys.
     The hash goes into the results, so the answers cannot be edited after
     the score without it showing.

    python bench/eval_llm.py export --limit 100
    python bench/eval_llm.py score --name claude-opus-5-5 \\
        --answers outputs/results/evals/llm/claude-opus-5-5_answers.jsonl

The first 100 items of each task are the head of the seeded order of
``eval_tasks.load_task``, so a base model scored by ``eval_model.py`` on every
item can be compared on exactly these 100.

**Contamination.** These benchmarks are public since 2018 to 2023, so a large
model trained on the web has almost certainly seen some of them. A high score
here is an upper bound, which is fine for an order of magnitude and nothing
more.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from eval_tasks import CACHE, LANGUAGE, LETTERS, REPO, load_task, parse_tasks

QUESTIONS_DIR = REPO / "outputs" / "results" / "evals" / "llm"
KEYS = CACHE / "keys" / "keys.jsonl"  # gitignored, never shown to the answerer
RESULTS = REPO / "outputs" / "results" / "evals"


def export(tasks: list[str], limit: int) -> None:
    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    KEYS.parent.mkdir(parents=True, exist_ok=True)
    questions = QUESTIONS_DIR / "questions.jsonl"
    with questions.open("w", encoding="utf-8") as q, KEYS.open("w", encoding="utf-8") as k:
        for task in tasks:
            for item in load_task(task, limit):
                q.write(json.dumps({"uid": item.uid, "task": task, "question": item.question},
                                   ensure_ascii=False) + "\n")
                key = item.choices[0].strip() if item.is_cloze else LETTERS[item.gold]
                k.write(json.dumps({"uid": item.uid, "task": task, "key": key},
                                   ensure_ascii=False) + "\n")
            print(f"  {task:<17} {limit} questions")
    print(f"questions : {questions}")
    print(f"keys      : {KEYS}  (not for the answerer)")


def normalize_word(text: str) -> str:
    """LAMBADA answers compare as words: case, accents and punctuation aside."""
    text = unicodedata.normalize("NFKD", text.strip().lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^\w']", "", text)


def normalize_letter(text: str) -> str:
    match = re.search(r"\b([A-E])\b", text.strip().upper())
    return match.group(1) if match else text.strip().upper()[:1]


def score(answers_path: Path, name: str) -> None:
    raw = answers_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    print(f"answers   : {answers_path}  sha256 {digest[:16]}")

    answers = {}
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            answers[row["uid"]] = str(row["answer"])
    keys = [json.loads(line) for line in KEYS.read_text(encoding="utf-8").splitlines() if line]

    by_task: dict[str, list[bool]] = {}
    missing = 0
    for row in keys:
        given = answers.get(row["uid"])
        if given is None:
            missing += 1
            ok = False
        elif row["task"].startswith("lambada"):
            ok = normalize_word(given) == normalize_word(row["key"])
        else:
            ok = normalize_letter(given) == row["key"]
        by_task.setdefault(row["task"], []).append(ok)

    report = {
        "model": name,
        "describe": f"{name}, answering lettered questions in text, zero-shot",
        "params": None,
        "date": f"{datetime.now():%Y-%m-%d %H:%M}",
        "answers_sha256": digest,
        "missing": missing,
        "tasks": {},
    }
    for task, flags in by_task.items():
        n = len(flags)
        p = sum(flags) / n
        summary = {"value": p, "stderr": math.sqrt(p * (1 - p) / n), "n": n}
        report["tasks"][task] = {
            "language": LANGUAGE[task],
            "acc": summary,
            "acc_norm": summary,
            "correct": "".join("1" if f else "0" for f in flags),
        }
        print(f"  {task:<17} {p:6.1%}  (n={n})")
    if missing:
        print(f"missing   : {missing} items had no answer, counted wrong")
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"results   : {out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="write the questions, and the keys apart")
    e.add_argument("--tasks", type=str, default="all")
    e.add_argument("--limit", type=int, default=100)
    s = sub.add_parser("score", help="compare an answers file to the keys")
    s.add_argument("--answers", type=Path, required=True)
    s.add_argument("--name", type=str, required=True)
    args = p.parse_args()

    if args.cmd == "export":
        export(parse_tasks(args.tasks), args.limit)
    else:
        score(args.answers, args.name)


if __name__ == "__main__":
    main()
