#!/usr/bin/env python3
"""Validate initial blinded reviews and build deterministic follow-up routes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("person_agent_ontology_audit_20260712")
REVIEWERS = ("reviewer_a", "reviewer_b", "reviewer_c")
DECISIONS = {
    "PERSON", "AGENT", "NEITHER", "AMBIGUOUS_CONTEXT", "BAD_BOUNDARY",
    "INSUFFICIENT_CONTEXT",
}
SPECIAL = {"AMBIGUOUS_CONTEXT", "BAD_BOUNDARY", "INSUFFICIENT_CONTEXT"}
BOUNDARIES = {"NONE", "EXPAND", "CONTRACT"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def validate(row: dict, path: Path, line_no: int) -> None:
    required = {
        "record_id", "review_role", "decision", "confidence", "rationale",
        "ontology_subtype", "boundary_action", "context_dependent_surface",
    }
    missing = required - row.keys()
    if missing:
        raise ValueError(f"{path}:{line_no}: missing {sorted(missing)}")
    if row["decision"] not in DECISIONS:
        raise ValueError(f"{path}:{line_no}: invalid decision {row['decision']}")
    if row["review_role"] not in {"PRIMARY", "SECONDARY_OVERLAP"}:
        raise ValueError(f"{path}:{line_no}: invalid role")
    if not isinstance(row["confidence"], (int, float)) or not 0 <= row["confidence"] <= 1:
        raise ValueError(f"{path}:{line_no}: invalid confidence")
    if row["boundary_action"] not in BOUNDARIES:
        raise ValueError(f"{path}:{line_no}: invalid boundary action")
    if not isinstance(row["context_dependent_surface"], bool):
        raise ValueError(f"{path}:{line_no}: invalid context-dependent flag")
    if not isinstance(row["rationale"], str) or not row["rationale"].strip():
        raise ValueError(f"{path}:{line_no}: empty rationale")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    packet_rows = {}
    for reviewer in REVIEWERS:
        for role in ("primary", "secondary"):
            path = args.root / "review_packets" / f"{reviewer}_{role}.jsonl"
            for row in read_jsonl(path):
                packet_rows[(reviewer, row["record_id"])] = row

    reviews_by_id = defaultdict(list)
    validation = {}
    for reviewer in REVIEWERS:
        path = args.root / "review_outputs" / f"{reviewer}.jsonl"
        rows = read_jsonl(path)
        seen = set()
        for line_no, row in enumerate(rows, 1):
            validate(row, path, line_no)
            key = (reviewer, row["record_id"])
            if key not in packet_rows:
                raise ValueError(f"{path}:{line_no}: unassigned record {row['record_id']}")
            if row["record_id"] in seen:
                raise ValueError(f"{path}:{line_no}: duplicate record")
            expected = packet_rows[key]
            if row["review_role"] != expected["review_role"]:
                raise ValueError(f"{path}:{line_no}: role mismatch")
            seen.add(row["record_id"])
            reviews_by_id[row["record_id"]].append({"reviewer": reviewer, **row})
        expected_ids = {rid for who, rid in packet_rows if who == reviewer}
        if seen != expected_ids:
            raise ValueError(
                f"{path}: missing={len(expected_ids-seen)} unexpected={len(seen-expected_ids)}"
            )
        validation[reviewer] = len(rows)

    candidates = {}
    for name in ("synthetic", "pd1000"):
        path = args.root / f"{name}_manual_audit_candidates.jsonl"
        candidates.update({r["candidate_id"]: r for r in read_jsonl(path)})

    second_review = []
    tie_break = []
    expanded = []
    provisional = []
    route_counts = Counter()
    for record_id, candidate in candidates.items():
        reviews = reviews_by_id[record_id]
        primary = next(r for r in reviews if r["review_role"] == "PRIMARY")
        secondary = next((r for r in reviews if r["review_role"] == "SECONDARY_OVERLAP"), None)
        packet = {
            "record_id": record_id,
            "surface": candidate["surface"],
            "context": candidate["local_context"],
            "review_role": "FOLLOWUP",
        }
        if secondary:
            if primary["decision"] in SPECIAL or secondary["decision"] in SPECIAL:
                expanded.append(packet)
                route_counts["overlap_special_to_expanded"] += 1
            elif primary["decision"] != secondary["decision"]:
                tie_break.append(packet)
                route_counts["overlap_disagreement_to_tiebreak"] += 1
            else:
                provisional.append({"record_id": record_id, "status": "OVERLAP_CONSENSUS"})
                if min(primary["confidence"], secondary["confidence"]) < 0.80:
                    route_counts["overlap_consensus_low_confidence"] += 1
                else:
                    route_counts["overlap_consensus_high_confidence"] += 1
        elif primary["decision"] in SPECIAL or primary["confidence"] < 0.65:
            expanded.append(packet)
            route_counts["nonoverlap_to_expanded"] += 1
        elif primary["confidence"] < 0.85:
            second_review.append(packet)
            route_counts["nonoverlap_to_second_review"] += 1
        else:
            provisional.append({"record_id": record_id, "status": "PRIMARY_HIGH_CONFIDENCE"})
            route_counts["nonoverlap_primary_accepted"] += 1

    out = args.root / "followup_packets"
    write_jsonl(out / "second_review.jsonl", second_review)
    write_jsonl(out / "tie_break.jsonl", tie_break)
    write_jsonl(out / "expanded_context_needed.jsonl", expanded)
    write_jsonl(args.root / "provisional_routes.jsonl", provisional)
    report = {
        "validated_output_rows": validation,
        "unique_records": len(reviews_by_id),
        "review_count_distribution": dict(Counter(len(v) for v in reviews_by_id.values())),
        "route_counts": dict(route_counts),
        "second_review_records": len(second_review),
        "tie_break_records": len(tie_break),
        "expanded_context_records": len(expanded),
        "provisional_records": len(provisional),
    }
    (args.root / "initial_review_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
