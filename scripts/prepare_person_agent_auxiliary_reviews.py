#!/usr/bin/env python3
"""Prepare blinded development and repeated-surface auxiliary review packets."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


ROOT = Path("person_agent_ontology_audit_20260712")
OUT = ROOT / "auxiliary_packets"
SEED = 20260713


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    rng = random.Random(SEED)
    dev_specs = [
        ("pd1000_eval", ROOT / "pd1000_eval_ontology_candidates.jsonl", 200, 100),
        ("modern", ROOT / "modern_ontology_candidates.jsonl", None, 100),
    ]
    dev_rows = []
    sealed = []
    for corpus, path, agent_n, person_n in dev_specs:
        rows = read_jsonl(path)
        agents = [r for r in rows if r["current_label"] == "AGENT"]
        people = [r for r in rows if r["current_label"] == "PERSON"]
        rng.shuffle(agents)
        rng.shuffle(people)
        selected = agents[:agent_n] if agent_n is not None else agents
        selected += people[:person_n]
        for row in selected:
            anonymous_id = "dev-" + hashlib.sha256(
                f"{corpus}\0{row['candidate_id']}".encode()
            ).hexdigest()[:20]
            dev_rows.append({
                "record_id": anonymous_id,
                "surface": row["surface"],
                "context": row["local_context"],
                "review_role": "DEVELOPMENT_AUXILIARY",
            })
            sealed.append({
                "record_id": anonymous_id,
                "candidate_id": row["candidate_id"],
                "corpus": corpus,
                "current_label": row["current_label"],
                "candidate_categories": row.get("candidate_categories", []),
            })
    rng.shuffle(dev_rows)
    write_jsonl(OUT / "development_primary_a.jsonl", dev_rows[::2])
    write_jsonl(OUT / "development_primary_b.jsonl", dev_rows[1::2])
    write_jsonl(ROOT / "development_auxiliary_sealed_lookup.jsonl", sealed)

    conflict_rows = []
    conflict_sealed = []
    for corpus, filename in (
        ("synthetic", "synthetic_repeated_surface_conflicts.jsonl"),
        ("pd1000", "pd1000_repeated_surface_conflicts.jsonl"),
        ("pd1000_eval", "pd1000_eval_repeated_surface_conflicts.jsonl"),
        ("modern", "modern_repeated_surface_conflicts.jsonl"),
    ):
        for index, row in enumerate(read_jsonl(ROOT / filename)):
            record_id = "conflict-" + hashlib.sha256(
                f"{corpus}\0{row['normalized_surface']}\0{index}".encode()
            ).hexdigest()[:20]
            contexts = []
            hidden = []
            examples = list(row.get("person_examples", [])) + list(row.get("agent_examples", []))
            rng.shuffle(examples)
            for context_index, example in enumerate(examples):
                context_id = f"c{context_index + 1}"
                contexts.append({
                    "context_id": context_id,
                    "surface": example["surface"],
                    "context": example["local_context"],
                })
                hidden.append({
                    "context_id": context_id,
                    "current_label": example["current_label"],
                    "candidate_id": example["candidate_id"],
                })
            conflict_rows.append({
                "record_id": record_id,
                "normalized_surface": row["normalized_surface"],
                "contexts": contexts,
            })
            conflict_sealed.append({
                "record_id": record_id,
                "corpus": corpus,
                "normalized_surface": row["normalized_surface"],
                "contexts": hidden,
            })
    rng.shuffle(conflict_rows)
    for index in range(3):
        write_jsonl(OUT / f"conflicting_surfaces_{index + 1}.jsonl", conflict_rows[index::3])
    write_jsonl(ROOT / "conflicting_surface_sealed_lookup.jsonl", conflict_sealed)

    manifest = {
        "seed": SEED,
        "development_records": len(dev_rows),
        "conflicting_surfaces": len(conflict_rows),
        "development_packet_counts": [len(dev_rows[::2]), len(dev_rows[1::2])],
        "conflict_packet_counts": [len(conflict_rows[i::3]) for i in range(3)],
    }
    (ROOT / "auxiliary_packet_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
