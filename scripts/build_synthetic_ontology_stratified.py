#!/usr/bin/env python3
"""Build subtype-balanced ontology audit mentions from the original synthetic export."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import msgspec

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_person_agent_ontology_audit import categories, normalized_surface, reservoir_add


def context_for(text: str, surface: str, radius: int = 300) -> tuple[str, int, int] | None:
    match = re.search(re.escape(surface), text, re.I)
    if not match:
        return None
    return text[max(0, match.start() - radius):min(len(text), match.end() + radius)], match.start(), match.end()


def make_item(row: dict, entity: dict, context: str, start: int, end: int, subtype: str | None) -> dict:
    story_id = str(row.get("_meta", {}).get("story_id") or "unknown")
    surface = str(entity["text"])
    label = str(entity["type"])
    return {
        "candidate_id": hashlib.sha256(f"synthetic_raw\0{story_id}\0{start}\0{end}\0{label}".encode()).hexdigest()[:20],
        "corpus_source": "synthetic_original_export",
        "source_path": None,
        "book_id": story_id,
        "scene_id": story_id,
        "char_start": start,
        "char_end": end,
        "surface": surface,
        "normalized_surface": normalized_surface(surface),
        "local_context": context,
        "current_label": label,
        "source_agent_subtype": subtype,
        "candidate_categories": categories(surface, context, label) + ([f"source_subtype:{subtype}"] if subtype else []),
        "ontology_v1_label": None,
        "error_category": None,
        "current_label_assessment": None,
        "adjudicator": None,
        "adjudication_notes": None,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--agents-per-subtype", type=int, default=20)
    parser.add_argument("--person-edge-target", type=int, default=100)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    agents: dict[str, list[dict]] = defaultdict(list)
    agent_seen = Counter()
    people: list[dict] = []
    people_seen = 0
    surface_labels: dict[str, set[str]] = defaultdict(set)
    examples: dict[tuple[str, str], list[dict]] = defaultdict(list)

    with args.source.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = msgspec.json.decode(line)
            text = str(row.get("text") or "")
            for entity in row.get("entities", []):
                label = entity.get("type")
                if label not in {"PERSON", "AGENT"} or not entity.get("text"):
                    continue
                located = context_for(text, str(entity["text"]))
                if not located:
                    continue
                context, start, end = located
                subtype = str(entity.get("subtype")) if label == "AGENT" and entity.get("subtype") else None
                item = make_item(row, entity, context, start, end, subtype)
                norm = item["normalized_surface"]
                surface_labels[norm].add(label)
                if len(examples[(norm, label)]) < 3:
                    examples[(norm, label)].append(item)
                if label == "AGENT":
                    subtype = subtype or "MISSING"
                    agent_seen[subtype] += 1
                    reservoir_add(agents[subtype], item, agent_seen[subtype], args.agents_per_subtype, rng)
                elif any(value in item["candidate_categories"] for value in (
                    "nonhuman_or_personified_individual", "organization_or_institution_term", "collective_or_species_term"
                )):
                    people_seen += 1
                    reservoir_add(people, item, people_seen, args.person_edge_target, rng)

    selected_agents = [item for subtype in sorted(agents) for item in agents[subtype]]
    rng.shuffle(selected_agents)
    conflicts = []
    for norm, labels in surface_labels.items():
        if labels == {"PERSON", "AGENT"}:
            conflicts.append({
                "corpus_source": "synthetic_original_export",
                "normalized_surface": norm,
                "person_examples": examples[(norm, "PERSON")],
                "agent_examples": examples[(norm, "AGENT")],
            })
    conflicts.sort(key=lambda row: row["normalized_surface"])
    write_jsonl(args.output_dir / "synthetic_manual_audit_candidates.jsonl", selected_agents + people)
    write_jsonl(args.output_dir / "synthetic_repeated_surface_conflicts.jsonl", conflicts)
    summary = {
        "source": str(args.source),
        "seed": args.seed,
        "agent_subtype_population": dict(agent_seen),
        "agent_sample_by_subtype": {key: len(value) for key, value in sorted(agents.items())},
        "sampled_agents": len(selected_agents),
        "sampled_person_edges": len(people),
        "conflicting_person_agent_surfaces": len(conflicts),
    }
    (args.output_dir / "synthetic_stratified_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
