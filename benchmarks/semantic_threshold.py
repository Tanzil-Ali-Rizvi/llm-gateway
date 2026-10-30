"""How safe is a semantic cache? Measured on Quora Question Pairs (GLUE QQP, validation split).

A semantic cache answers a new question with the cached answer to a *similar* earlier question. On a labelled
set of question pairs we can measure, for each similarity threshold:

* useful hit rate: share of true duplicate pairs that would be served from cache (higher = more savings)
* wrong-answer rate: share of non-duplicate pairs that would ALSO be served from cache (a different
  question answered with someone else's answer; this is the risk)
* precision: of all cache hits, the share that were genuinely the same question

Usage (needs network for the dataset and embedding model; runs in GitHub Actions):
    pip install -e ".[semantic]" datasets
    python benchmarks/semantic_threshold.py --out benchmarks/results
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

THRESHOLDS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    a = ap.parse_args()

    from datasets import load_dataset
    from fastembed import TextEmbedding

    ds = load_dataset("nyu-mll/glue", "qqp", split="validation")
    idx = random.Random(0).sample(range(len(ds)), a.pairs)
    rows = [ds[i] for i in idx]
    q1 = [r["question1"] for r in rows]
    q2 = [r["question2"] for r in rows]
    labels = np.array([r["label"] for r in rows])

    model = TextEmbedding(a.model)

    def emb(texts):
        v = np.asarray(list(model.embed(texts)), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    sims = (emb(q1) * emb(q2)).sum(axis=1)
    dup, non = sims[labels == 1], sims[labels == 0]
    table = []
    for t in THRESHOLDS:
        tp, fp = int((dup >= t).sum()), int((non >= t).sum())
        table.append(
            {
                "threshold": t,
                "useful_hit_rate": round(100 * tp / len(dup), 1),
                "wrong_answer_rate": round(100 * fp / len(non), 2),
                "precision": round(100 * tp / (tp + fp), 1) if tp + fp else None,
            }
        )
    res = {
        "dataset": "GLUE QQP validation",
        "pairs": a.pairs,
        "duplicates": int(len(dup)),
        "non_duplicates": int(len(non)),
        "embedding_model": a.model,
        "table": table,
        "examples_of_wrong_hits": [
            {"q1": q1[i], "q2": q2[i], "similarity": round(float(sims[i]), 3)}
            for i in np.argsort(-np.where(labels == 0, sims, -1))[:5]
        ],
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "semantic_threshold.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    md = [
        f"# Semantic cache: hit rate vs wrong answers ({a.model}, {a.pairs:,} QQP pairs)",
        "",
        "| Threshold | Duplicates served from cache | Different questions wrongly served | Precision of hits |",
        "|---|---|---|---|",
    ]
    md += [
        f"| {r['threshold']:.2f} | {r['useful_hit_rate']}% | {r['wrong_answer_rate']}% | {r['precision']}% |"
        for r in table
    ]
    md += ["", "Most similar *non-duplicate* pairs (what a too-low threshold would get wrong):", ""]
    md += [f'- {e["similarity"]}: "{e["q1"]}" vs "{e["q2"]}"' for e in res["examples_of_wrong_hits"]]
    (a.out / "semantic_threshold.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
