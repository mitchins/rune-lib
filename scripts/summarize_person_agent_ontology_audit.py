#!/usr/bin/env python3
"""Summarize completed blind ontology-v1 adjudication sheets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


VALID_ONTOLOGY = {"PERSON", "AGENT", "O", "LOCATION", "REVIEW"}
VALID_ASSESSMENTS = {"DEFENSIBLE", "CLEARLY_WRONG", "AMBIGUOUS"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wilson(successes: int, total: int, z: float = 1.96) -> dict:
    if not total:
        return {"estimate": None, "lower": None, "upper": None}
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return {"estimate": p, "lower": max(0, center - margin), "upper": min(1, center + margin)}


def summarize(rows: list[dict]) -> dict:
    missing = [
        row["candidate_id"]
        for row in rows
        if row.get("ontology_v1_label") not in VALID_ONTOLOGY
        or row.get("current_label_assessment") not in VALID_ASSESSMENTS
        or not row.get("adjudicator")
    ]
    if missing:
        raise ValueError(f"{len(missing)} candidates lack valid manual adjudication; first={missing[:10]}")

    transitions = Counter((row["current_label"], row["ontology_v1_label"]) for row in rows)
    assessment = Counter(row["current_label_assessment"] for row in rows)
    clear_wrong = assessment["CLEARLY_WRONG"]
    by_category = defaultdict(lambda: Counter(total=0, clearly_wrong=0, ambiguous=0))
    for row in rows:
        for category in row.get("candidate_categories", []):
            by_category[category]["total"] += 1
            if row["current_label_assessment"] == "CLEARLY_WRONG":
                by_category[category]["clearly_wrong"] += 1
            if row["current_label_assessment"] == "AMBIGUOUS":
                by_category[category]["ambiguous"] += 1

    return {
        "adjudicated_mentions": len(rows),
        "assessment_counts": dict(assessment),
        "affected_fraction_95ci": wilson(clear_wrong, len(rows)),
        "person_to_agent": transitions[("PERSON", "AGENT")],
        "agent_to_person": transitions[("AGENT", "PERSON")],
        "agent_to_o": transitions[("AGENT", "O")],
        "person_to_o": transitions[("PERSON", "O")],
        "agent_to_review": transitions[("AGENT", "REVIEW")],
        "person_to_review": transitions[("PERSON", "REVIEW")],
        "transitions": [
            {"current": current, "ontology_v1": target, "count": count}
            for (current, target), count in sorted(transitions.items())
        ],
        "category_patterns": [
            {"category": category, **counts}
            for category, counts in sorted(by_category.items())
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjudicated", action="append", required=True, help="CORPUS=PATH")
    parser.add_argument("--repeated-conflicts", action="append", default=[], help="CORPUS=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    corpora = {}
    all_rows = {}
    for item in args.adjudicated:
        name, path = item.split("=", 1)
        rows = read_jsonl(Path(path))
        all_rows[name] = rows
        corpora[name] = summarize(rows)

    repeated = {}
    for item in args.repeated_conflicts:
        name, path = item.split("=", 1)
        rows = read_jsonl(Path(path))
        repeated[name] = {
            "conflicting_surfaces": len(rows),
            "person_example_mentions": sum(len(row.get("person_examples", [])) for row in rows),
            "agent_example_mentions": sum(len(row.get("agent_examples", [])) for row in rows),
        }

    report = {
        "ontology_version": "rune_ner_ontology.v1",
        "manual_adjudication_required": True,
        "corpora": corpora,
        "repeated_surface_inconsistency": repeated,
        "labels_modified": False,
        "interpretation_note": "Model confusion matrices are reported separately and are not treated as proof of gold corruption.",
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
