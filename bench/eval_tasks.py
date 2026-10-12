"""The benchmark questions, loaded once and shared by every way of answering them.

Nine tasks, chosen so that English and French face the same test:

    task              language  what it measures                         chance
    hellaswag_en      EN        commonsense, the plausible next sentence    25%
    hellaswag_fr      FR        same questions, machine translated           25%
    arc_easy          EN        grade-school science, easy split            25%
    arc_challenge     EN        grade-school science, hard split            25%
    arc_challenge_fr  FR        same questions, machine translated           25%
    belebele_en       EN        reading comprehension of a short passage    25%
    belebele_fr       FR        the same 900 passages and questions, human  25%
                                translated (FLORES)
    lambada_en        EN        guess the last word of a story               ~0%
    lambada_fr        FR        same stories, machine translated             ~0%

Each item is one ``Item``. A base model answers it by scoring every choice as
a continuation of ``context`` (``bench/eval_model.py``). An instruction model,
such as Claude, reads ``question`` instead, where the choices are lettered
(``bench/eval_llm.py``). Both read the same items, drawn in the same seeded
order, so the first 100 items of a task are the same 100 for everyone.

The files are fetched once from the Hugging Face Hub into ``.cache/evals/``.
"""

from __future__ import annotations

import ast
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / ".cache" / "evals"
LETTERS = "ABCDE"
ORDER_SEED = 0

TASKS = {
    "hellaswag_en": ("Rowan/hellaswag", "data/validation-00000-of-00001.parquet"),
    "hellaswag_fr": ("manu/french_bench_hellaswag", "data/validation-00000-of-00001.parquet"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy/test-00000-of-00001.parquet"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet"),
    "arc_challenge_fr": ("manu/french_bench_arc_challenge", "data/test-00000-of-00001.parquet"),
    "belebele_en": ("facebook/belebele", "data/eng_Latn.jsonl"),
    "belebele_fr": ("facebook/belebele", "data/fra_Latn.jsonl"),
    "lambada_en": ("EleutherAI/lambada_openai", "data/lambada_test_en.jsonl"),
    "lambada_fr": ("EleutherAI/lambada_openai", "data/lambada_test_fr.jsonl"),
}
LANGUAGE = {name: "FR" if name.endswith("_fr") else "EN" for name in TASKS}


@dataclass
class Item:
    """One question.

    Attributes:
        context: the text a base model continues, ending right before the answer.
        choices: the candidate continuations, each starting with its space.
            LAMBADA has a single one, the expected last word.
        gold: index of the right choice.
        question: the same item written for a model that follows instructions.
    """

    task: str
    uid: str
    context: str
    choices: list[str]
    gold: int
    question: str

    @property
    def is_cloze(self) -> bool:
        """LAMBADA: one expected word, scored by greedy match, not by ranking."""
        return len(self.choices) == 1


# ---------------------------------------------------------------------------
# Raw files
# ---------------------------------------------------------------------------


def _fetch(task: str) -> Path:
    from huggingface_hub import hf_hub_download

    repo, name = TASKS[task]
    local = CACHE / repo.replace("/", "__")
    path = local / name
    if not path.exists():
        hf_hub_download(repo, name, repo_type="dataset", local_dir=str(local))
    return path


def _rows(task: str) -> list[dict]:
    path = _fetch(task)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------------------
# One builder per format
# ---------------------------------------------------------------------------


def _clean_wikihow(text: str) -> str:
    """HellaSwag carries WikiHow markup, removed the way lm-evaluation-harness does."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ").strip()


def _lettered(stem: str, options: list[str], instruction: str) -> str:
    lines = [stem, ""]
    lines += [f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options)]
    return "\n".join(lines + ["", instruction])


def _hellaswag(task: str, rows: list[dict]) -> list[Item]:
    items = []
    fr = task.endswith("_fr")
    for n, row in enumerate(rows):
        endings = row["endings"]
        if isinstance(endings, str):  # some exports store the list as its repr
            endings = ast.literal_eval(endings)
        ctx = _clean_wikihow(f"{row['ctx_a']} {row['ctx_b'].capitalize()}")
        context = f"{row['activity_label']}: {ctx}"
        choices = [" " + _clean_wikihow(e) for e in endings]
        instruction = (
            "Quelle suite est la plus plausible ? Réponds uniquement par la lettre."
            if fr
            else "Which continuation is the most plausible? Answer with the letter only."
        )
        items.append(
            Item(
                task=task,
                uid=f"{task}-{n}-{row['ind']}",  # "ind" alone repeats across rows
                context=context,
                choices=choices,
                gold=int(row["label"]),
                question=_lettered(context, [c.strip() for c in choices], instruction),
            )
        )
    return items


def _arc(task: str, rows: list[dict]) -> list[Item]:
    items = []
    fr = task.endswith("_fr")
    for row in rows:
        if isinstance(row["choices"], dict):  # allenai/ai2_arc
            texts, labels = row["choices"]["text"], row["choices"]["label"]
            gold = labels.index(row["answerKey"])
        else:  # the French translation keeps the texts and letter keys only
            texts = list(row["choices"])
            gold = LETTERS.index(row["answerKey"])
        context = f"Question: {row['question']}\n{'Réponse' if fr else 'Answer'}:"
        instruction = (
            "Réponds uniquement par la lettre de la bonne réponse."
            if fr
            else "Answer with the letter of the correct answer only."
        )
        items.append(
            Item(
                task=task,
                uid=f"{task}-{row['id']}",
                context=context,
                choices=[" " + t for t in texts],
                gold=gold,
                question=_lettered(row["question"], texts, instruction),
            )
        )
    return items


def _belebele(task: str, rows: list[dict]) -> list[Item]:
    """Scored as a cloze on the answer text, not on a letter.

    lm-evaluation-harness asks for the letter, which no 100M model can do, so
    every small model lands at chance and the task says nothing. Scoring the
    text of each answer at least leaves room to read the passage.
    """
    items = []
    fr = task.endswith("_fr")
    for n, row in enumerate(rows):
        options = [row[f"mc_answer{i}"] for i in range(1, 5)]
        q, a = ("Question", "Réponse") if fr else ("Question", "Answer")
        context = f"{row['flores_passage']}\n{q}: {row['question']}\n{a}:"
        instruction = (
            "Réponds uniquement par la lettre de la bonne réponse."
            if fr
            else "Answer with the letter of the correct answer only."
        )
        stem = f"{row['flores_passage']}\n\n{q}: {row['question']}"
        items.append(
            Item(
                task=task,
                # the row number keeps two articles both named "Introduction" apart
                uid=f"{task}-{n}-{row['link'].rsplit('/', 1)[-1]}-{row['question_number']}",
                context=context,
                choices=[" " + o for o in options],
                gold=int(row["correct_answer_num"]) - 1,
                question=_lettered(stem, options, instruction),
            )
        )
    return items


def _lambada(task: str, rows: list[dict]) -> list[Item]:
    items = []
    fr = task.endswith("_fr")
    for i, row in enumerate(rows):
        context, word = row["text"].rsplit(" ", 1)
        instruction = (
            "Le dernier mot de ce texte a été retiré. Quel est-il ? "
            "Réponds uniquement par ce mot."
            if fr
            else "The last word of this passage was removed. What is it? "
            "Answer with that single word only."
        )
        items.append(
            Item(
                task=task,
                uid=f"{task}-{i}",
                context=context,
                choices=[" " + word],
                gold=0,
                question=f"{context} ____\n\n{instruction}",
            )
        )
    return items


BUILDERS = {
    "hellaswag": _hellaswag,
    "arc": _arc,
    "belebele": _belebele,
    "lambada": _lambada,
}


def load_task(task: str, limit: int | None = None) -> list[Item]:
    """The items of ``task`` in a fixed shuffled order, the first ``limit`` of them.

    The order is seeded and independent of ``limit``, so a 100 item sample is
    exactly the head of a 1000 item one. That is what lets a model scored on
    every item and a person answering 100 be compared on the same questions.
    """
    if task not in TASKS:
        raise SystemExit(f"unknown task {task!r}, known: {', '.join(TASKS)}")
    family = task.split("_")[0]
    items = BUILDERS[family](task, _rows(task))
    uids = [item.uid for item in items]
    if len(set(uids)) != len(uids):
        raise ValueError(f"{task}: duplicate item ids, answers would be scored on the wrong item")
    random.Random(f"{ORDER_SEED}-{task}").shuffle(items)
    return items[:limit] if limit else items


def parse_tasks(spec: str) -> list[str]:
    """``all``, or a comma separated list of task names."""
    return list(TASKS) if spec == "all" else [t.strip() for t in spec.split(",") if t.strip()]
