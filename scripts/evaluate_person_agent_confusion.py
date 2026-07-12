#!/usr/bin/env python3
"""Measure PERSON/AGENT/O model confusions on frozen development suites."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_model_consensus import (  # noqa: E402
    LABELS,
    argmax_rows,
    load_rows,
    predict,
    spans,
)


def overlap(left: tuple[int, int, str], right: tuple[int, int, str]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def classify_gold(gold: tuple[int, int, str], predicted: set[tuple[int, int, str]]) -> str:
    matches = [candidate for candidate in predicted if overlap(gold, candidate)]
    if not matches:
        return "O"
    matches.sort(key=lambda candidate: (overlap(gold, candidate), candidate[2] == gold[2]), reverse=True)
    return matches[0][2]


def evaluate_model(model_name: str, probabilities: dict, datasets: dict) -> dict:
    tags = argmax_rows(probabilities)
    output = {}
    disagreements = []
    for dataset_name, dataset in datasets.items():
        matrix = Counter()
        exact_matrix = Counter()
        total_gold = Counter()
        retained_gold = Counter()
        for row, predicted_tags in zip(dataset["rows"], tags[dataset_name]):
            length = len(predicted_tags)
            gold_spans = spans(row["bio_tags"])
            predicted_spans = spans(predicted_tags)
            for gold in gold_spans:
                if gold[2] not in {"PERSON", "AGENT"}:
                    continue
                total_gold[gold[2]] += 1
                if gold[1] > length:
                    matrix[(gold[2], "TRUNCATED")] += 1
                    continue
                retained_gold[gold[2]] += 1
                predicted_type = classify_gold(gold, predicted_spans)
                matrix[(gold[2], predicted_type)] += 1
                exact_type = gold[2] if gold in predicted_spans else predicted_type
                exact_matrix[(gold[2], exact_type)] += 1
                if (gold[2], predicted_type) in {
                    ("PERSON", "AGENT"),
                    ("AGENT", "PERSON"),
                    ("PERSON", "O"),
                    ("AGENT", "O"),
                }:
                    start, end, gold_type = gold
                    disagreements.append(
                        {
                            "candidate_id": f"{model_name}:{dataset_name}:{row['story_id']}:{start}:{end}",
                            "model": model_name,
                            "dataset": dataset_name,
                            "scene_id": row["story_id"],
                            "surface": " ".join(row["tokens"][start:end]),
                            "local_context": " ".join(row["tokens"][max(0,start-45):min(len(row['tokens']),end+45)]),
                            "gold_label": gold_type,
                            "predicted_label": predicted_type,
                            "gold_start": start,
                            "gold_end": end,
                        }
                    )
        output[dataset_name] = {
            "gold_mentions": dict(total_gold),
            "retained_gold_mentions": dict(retained_gold),
            "overlap_type_matrix": [
                {"gold": gold, "predicted": predicted, "count": count}
                for (gold, predicted), count in sorted(matrix.items())
            ],
            "required_confusions": {
                "gold_PERSON_predicted_AGENT": matrix[("PERSON", "AGENT")],
                "gold_AGENT_predicted_PERSON": matrix[("AGENT", "PERSON")],
                "gold_AGENT_predicted_O": matrix[("AGENT", "O")],
                "gold_PERSON_predicted_O": matrix[("PERSON", "O")],
                "gold_PERSON_truncated": matrix[("PERSON", "TRUNCATED")],
                "gold_AGENT_truncated": matrix[("AGENT", "TRUNCATED")],
            },
        }
    return {"datasets": output, "disagreements": disagreements}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--dataset", action="append", required=True, help="NAME=PATH:MAX_LENGTH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    models = dict(item.split("=", 1) for item in args.model)
    datasets = {}
    for item in args.dataset:
        name, value = item.split("=", 1)
        path, max_length = value.rsplit(":", 1)
        datasets[name] = {
            "path": path,
            "max_length": int(max_length),
            "rows": load_rows(Path(path)),
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "development_policy": "Previously tuned-on suites; model predictions are evidence, not ontology adjudication",
        "models": models,
        "datasets": {
            name: {"path": value["path"], "rows": len(value["rows"]), "max_length": value["max_length"]}
            for name, value in datasets.items()
        },
        "results": {},
    }
    all_disagreements = []
    for model_name, model_path in models.items():
        probabilities = predict(model_path, datasets, device)
        result = evaluate_model(model_name, probabilities, datasets)
        report["results"][model_name] = result["datasets"]
        all_disagreements.extend(result["disagreements"])
    write_jsonl(args.output_dir / "development_model_disagreements.jsonl", all_disagreements)
    (args.output_dir / "development_confusion_matrices.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
