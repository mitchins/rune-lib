#!/usr/bin/env python3
"""Build reproducible PERSON/AGENT ontology audit slices without relabelling data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

import msgspec


ORG_TERMS = re.compile(
    r"\b(?:army|armies|association|brotherhood|church|company|conclave|council|court|crew|"
    r"department|embassy|empire|faction|federation|foundation|government|guild|hegemony|"
    r"institution|ministry|order|parliament|police|republic|society|staff|team|templars|unit)\b",
    re.I,
)
INDIVIDUAL_EDGE_TERMS = re.compile(
    r"\b(?:ai|android|angel|animal|beast|construct|deity|demon|dragon|goddess?|machine mind|"
    r"oracle|robot|sentient|ship|spirit|supernatural|wolf)\b",
    re.I,
)
COLLECTIVE_TERMS = re.compile(
    r"\b(?:aliens?|army|clan|collective|crew|family|guards?|people|peoples|race|species|tribe|villagers)\b",
    re.I,
)


def spans(tags: list[str]) -> list[tuple[int, int, str]]:
    found = []
    start = None
    label = None
    for index, tag in enumerate(tags + ["O"]):
        if tag == "O":
            if start is not None:
                found.append((start, index, label))
            start = label = None
            continue
        prefix, current = tag.split("-", 1)
        if prefix == "B" or start is None or current != label:
            if start is not None:
                found.append((start, index, label))
            start, label = index, current
    return found


def normalized_surface(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def categories(surface: str, context: str, label: str) -> list[str]:
    value = f"{surface} {context}"
    found = []
    if ORG_TERMS.search(value):
        found.append("organization_or_institution_term")
    if INDIVIDUAL_EDGE_TERMS.search(value):
        found.append("nonhuman_or_personified_individual")
    if COLLECTIVE_TERMS.search(value):
        found.append("collective_or_species_term")
    if label == "PERSON" and found:
        found.append("person_edge_case")
    if label == "AGENT" and not found:
        found.append("untyped_agent_random")
    return found


def reservoir_add(bucket: list[dict], item: dict, seen: int, limit: int, rng: random.Random) -> None:
    if len(bucket) < limit:
        bucket.append(item)
        return
    replacement = rng.randrange(seen)
    if replacement < limit:
        bucket[replacement] = item


def scan(path: Path, corpus: str, agent_target: int | None, person_target: int | None, seed: int) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(seed)
    agent_edge: list[dict] = []
    agent_random: list[dict] = []
    person_edge: list[dict] = []
    seen = Counter()
    surface_labels: dict[str, set[str]] = defaultdict(set)
    surface_examples: dict[tuple[str, str], list[dict]] = defaultdict(list)
    totals = Counter()

    with path.open("rb") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = msgspec.json.decode(line)
            tokens = row.get("tokens", [])
            tags = row.get("bio_tags", [])
            if len(tokens) != len(tags):
                raise ValueError(f"{path}:{row_index}: token/tag mismatch")
            story_id = str(row.get("story_id") or f"row-{row_index}")
            parent_id = str(row.get("parent_story_id") or story_id)
            for start, end, label in spans(tags):
                if label not in {"PERSON", "AGENT"}:
                    continue
                surface = " ".join(tokens[start:end])
                left, right = max(0, start - 45), min(len(tokens), end + 45)
                context = " ".join(tokens[left:right])
                cats = categories(surface, context, label)
                norm = normalized_surface(surface)
                candidate_id = hashlib.sha256(
                    f"{corpus}\0{story_id}\0{start}\0{end}\0{label}".encode("utf-8")
                ).hexdigest()[:20]
                item = {
                    "candidate_id": candidate_id,
                    "corpus_source": corpus,
                    "source_path": str(path),
                    "row_index": row_index,
                    "book_id": parent_id,
                    "scene_id": story_id,
                    "token_start": start,
                    "token_end": end,
                    "surface": surface,
                    "normalized_surface": norm,
                    "local_context": context,
                    "current_label": label,
                    "candidate_categories": cats,
                    "ontology_v1_label": None,
                    "error_category": None,
                    "current_label_assessment": None,
                    "adjudicator": None,
                    "adjudication_notes": None,
                }
                totals[label] += 1
                surface_labels[norm].add(label)
                key = (norm, label)
                if len(surface_examples[key]) < 3:
                    surface_examples[key].append(item)
                if label == "AGENT":
                    target = agent_edge if any(cat != "untyped_agent_random" for cat in cats) else agent_random
                    kind = "agent_edge" if target is agent_edge else "agent_random"
                    seen[kind] += 1
                    if agent_target is None:
                        target.append(item)
                    else:
                        reservoir_add(target, item, seen[kind], agent_target, rng)
                elif cats:
                    seen["person_edge"] += 1
                    if person_target is None:
                        person_edge.append(item)
                    else:
                        reservoir_add(person_edge, item, seen["person_edge"], person_target, rng)

    # Prefer ontology-sensitive AGENTs, then fill the fixed quota with a true random reservoir.
    rng.shuffle(agent_edge)
    selected_agents = agent_edge if agent_target is None else agent_edge[:agent_target]
    selected_ids = {row["candidate_id"] for row in selected_agents}
    for row in agent_random:
        if agent_target is not None and len(selected_agents) >= agent_target:
            break
        if row["candidate_id"] not in selected_ids:
            selected_agents.append(row)
            selected_ids.add(row["candidate_id"])

    conflicts = []
    for norm, labels in surface_labels.items():
        if labels == {"PERSON", "AGENT"}:
            conflicts.append(
                {
                    "corpus_source": corpus,
                    "normalized_surface": norm,
                    "person_examples": surface_examples[(norm, "PERSON")],
                    "agent_examples": surface_examples[(norm, "AGENT")],
                }
            )
    conflicts.sort(key=lambda row: row["normalized_surface"])
    summary = {
        "path": str(path),
        "corpus": corpus,
        "gold_person_mentions": totals["PERSON"],
        "gold_agent_mentions": totals["AGENT"],
        "sampled_agents": len(selected_agents),
        "sampled_person_edges": len(person_edge),
        "conflicting_person_agent_surfaces": len(conflicts),
    }
    return selected_agents + person_edge, conflicts, summary


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--pd1000", type=Path, required=True)
    parser.add_argument("--pd1000-eval", type=Path)
    parser.add_argument("--modern-development-pretokenized", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    synthetic, synthetic_conflicts, synthetic_summary = scan(
        args.synthetic, "synthetic_phase_a", 200, 100, args.seed
    )
    pd1000, pd_conflicts, pd_summary = scan(
        args.pd1000, "pd1000_train", 200, 100, args.seed + 1
    )
    write_jsonl(args.output_dir / "synthetic_manual_audit_candidates.jsonl", synthetic)
    write_jsonl(args.output_dir / "pd1000_manual_audit_candidates.jsonl", pd1000)
    write_jsonl(args.output_dir / "synthetic_repeated_surface_conflicts.jsonl", synthetic_conflicts)
    write_jsonl(args.output_dir / "pd1000_repeated_surface_conflicts.jsonl", pd_conflicts)
    development = {}
    if args.pd1000_eval:
        rows, conflicts, item_summary = scan(
            args.pd1000_eval, "pd1000_eval_development", None, None, args.seed + 2
        )
        write_jsonl(args.output_dir / "pd1000_eval_ontology_candidates.jsonl", rows)
        write_jsonl(args.output_dir / "pd1000_eval_repeated_surface_conflicts.jsonl", conflicts)
        development["pd1000_eval"] = item_summary
    if args.modern_development_pretokenized:
        rows, conflicts, item_summary = scan(
            args.modern_development_pretokenized, "modern_development", None, None, args.seed + 3
        )
        write_jsonl(args.output_dir / "modern_development_ontology_candidates.jsonl", rows)
        write_jsonl(args.output_dir / "modern_development_repeated_surface_conflicts.jsonl", conflicts)
        development["modern"] = item_summary
    summary = {
        "seed": args.seed,
        "ontology_version": "rune_ner_ontology.v1",
        "synthetic": synthetic_summary,
        "pd1000": pd_summary,
        "development": development,
        "adjudication_status": "PENDING_BLIND_MANUAL_REVIEW",
        "labels_modified": False,
    }
    (args.output_dir / "candidate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
