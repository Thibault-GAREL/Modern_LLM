"""Gather every benchmark result into one markdown table.

Reads ``outputs/results/evals/*.json``, written by ``eval_model.py`` (base
models, by log-likelihood) and ``eval_llm.py score`` (models answering in
text). Every column is measured on the same first 100 items of each task, the
ones the text answerers saw, so a row can be compared with any other.

    python bench/eval_report.py
    python bench/eval_report.py --out docs/benchmark_table.md

The score shown is ``acc_norm`` for the multiple choice tasks of a base model,
``acc`` for LAMBADA, and plain accuracy for a text answerer, which has no
length bias to remove.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_tasks import REPO, TASKS

RESULTS = REPO / "outputs" / "results" / "evals"
SUBSET = 100
CHANCE = {task: ("~0%" if task.startswith("lambada") else "25%") for task in TASKS}
SHORT = {
    "hellaswag_en": "HellaSwag EN",
    "hellaswag_fr": "HellaSwag FR",
    "arc_easy": "ARC-E EN",
    "arc_challenge": "ARC-C EN",
    "arc_challenge_fr": "ARC-C FR",
    "belebele_en": "Belebele EN",
    "belebele_fr": "Belebele FR",
    "lambada_en": "LAMBADA EN",
    "lambada_fr": "LAMBADA FR",
}


def subset_score(result: dict, task: str) -> float | None:
    """Accuracy on the first ``SUBSET`` items, recomputed from the stored flags."""
    entry = result["tasks"].get(task)
    if entry is None:
        return None
    flags = entry.get("correct_norm") or entry["correct"]
    if task.startswith("lambada"):
        flags = entry["correct"]
    head = flags[:SUBSET]
    return head.count("1") / len(head) if head else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(RESULTS.glob("*.json"))
    ]
    if not results:
        raise SystemExit(f"no results in {RESULTS}, run bench/eval_model.py first")

    header = "| Model | " + " | ".join(SHORT[t] for t in TASKS) + " | Mean |"
    rule = "|---|" + "---|" * (len(TASKS) + 1)
    rows = [header, rule, "| *chance* | " + " | ".join(CHANCE[t] for t in TASKS) + " | |"]
    for result in results:
        scores = [subset_score(result, t) for t in TASKS]
        known = [s for s in scores if s is not None]
        cells = [f"{s:.0%}" if s is not None else "-" for s in scores]
        mean = f"**{sum(known) / len(known):.0%}**" if known else "-"
        rows.append(f"| {result['model']} | " + " | ".join(cells) + f" | {mean} |")

    table = "\n".join(rows)
    table += f"\n\n*{SUBSET} items per task, so one standard error is about 4 to 5 points.*"

    # base models scored beyond the shared head get a second, tighter table
    full = [r for r in results if any(t["acc"]["n"] > SUBSET for t in r["tasks"].values())]
    if full:
        rows = [header.replace("| Model |", "| Model, all items |"), rule]
        for result in full:
            cells = []
            for task in TASKS:
                entry = result["tasks"].get(task)
                if entry is None:
                    cells.append("-")
                    continue
                metric = "acc" if task.startswith("lambada") else "acc_norm"
                cells.append(f"{entry[metric]['value']:.1%} (n={entry[metric]['n']})")
            rows.append(f"| {result['model']} | " + " | ".join(cells) + " | |")
        table += "\n\n" + "\n".join(rows)

    print(table)
    if args.out:
        args.out.write_text(table + "\n", encoding="utf-8")
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
