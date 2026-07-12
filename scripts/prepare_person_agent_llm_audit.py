#!/usr/bin/env python3
"""Prepare deterministic, blinded PERSON/AGENT ontology review packets."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("person_agent_ontology_audit_20260712")
PACKETS = ROOT / "review_packets"
SEED = 20260712
REVIEWERS = ("reviewer_a", "reviewer_b", "reviewer_c")
RISKY = ("artifact", "ai", "beast", "deity", "spirit", "ship")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def blind(row: dict, role: str) -> dict:
    # The role supports accounting but discloses no corpus, label, or selection stratum.
    return {
        "record_id": row["candidate_id"],
        "surface": row["surface"],
        "context": row["local_context"],
        "review_role": role,
    }


def choose_overlap(rows: list[dict], corpus: str, rng: random.Random) -> list[dict]:
    agents = [row for row in rows if row["current_label"] == "AGENT"]
    people = [row for row in rows if row["current_label"] == "PERSON"]
    selected: list[dict] = []

    if corpus == "synthetic":
        # Six of each 20-record risky subtype gives the required >=30% overlap.
        for subtype in RISKY:
            bucket = [row for row in agents if row.get("source_agent_subtype") == subtype]
            selected.extend(rng.sample(bucket, 6))
        remainder = [row for row in agents if row not in selected]
        selected.extend(rng.sample(remainder, 50 - len(selected)))
    else:
        # PD1000 lacks declared fine subtypes; oversample its available risk proxies.
        risk = [
            row
            for row in agents
            if "nonhuman_or_personified_individual" in row.get("candidate_categories", [])
            or "collective_or_species_term" in row.get("candidate_categories", [])
        ]
        selected.extend(rng.sample(risk, min(43, len(risk))))
        remainder = [row for row in agents if row not in selected]
        selected.extend(rng.sample(remainder, 50 - len(selected)))

    selected.extend(rng.sample(people, 25))
    assert len(selected) == 75 and len({r["candidate_id"] for r in selected}) == 75
    return selected


def main() -> None:
    PACKETS.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    corpora = {
        "synthetic": read_jsonl(ROOT / "synthetic_manual_audit_candidates.jsonl"),
        "pd1000": read_jsonl(ROOT / "pd1000_manual_audit_candidates.jsonl"),
    }
    for rows in corpora.values():
        assert len(rows) == 300

    overlap = []
    for corpus, rows in corpora.items():
        overlap.extend(choose_overlap(rows, corpus, rng))
    overlap_ids = {row["candidate_id"] for row in overlap}

    # Stratified round-robin makes each reviewer primary for exactly 100 records/corpus.
    primary: dict[str, list[dict]] = defaultdict(list)
    primary_owner: dict[str, str] = {}
    for corpus, rows in corpora.items():
        strata = defaultdict(list)
        for row in rows:
            strata[row["current_label"]].append(row)
        for label in sorted(strata):
            rng.shuffle(strata[label])
            for index, row in enumerate(strata[label]):
                reviewer = REVIEWERS[index % len(REVIEWERS)]
                primary[reviewer].append(row)
                primary_owner[row["candidate_id"]] = reviewer

    # Assign each overlap record to one non-primary reviewer, balancing reviewer and corpus.
    secondary: dict[str, list[dict]] = defaultdict(list)
    secondary_owner: dict[str, str] = {}
    overlap_by_corpus = defaultdict(list)
    source_for_id = {}
    for corpus, rows in corpora.items():
        for row in rows:
            source_for_id[row["candidate_id"]] = corpus
    for row in overlap:
        overlap_by_corpus[source_for_id[row["candidate_id"]]].append(row)
    for corpus in sorted(overlap_by_corpus):
        rows = overlap_by_corpus[corpus]
        rng.shuffle(rows)
        for row in rows:
            eligible = [r for r in REVIEWERS if r != primary_owner[row["candidate_id"]]]
            reviewer = min(
                eligible,
                key=lambda r: (
                    sum(source_for_id[x["candidate_id"]] == corpus for x in secondary[r]),
                    len(secondary[r]),
                    r,
                ),
            )
            secondary[reviewer].append(row)
            secondary_owner[row["candidate_id"]] = reviewer

    files = []
    for reviewer in REVIEWERS:
        rng.shuffle(primary[reviewer])
        rng.shuffle(secondary[reviewer])
        p1 = PACKETS / f"{reviewer}_primary.jsonl"
        p2 = PACKETS / f"{reviewer}_secondary.jsonl"
        write_jsonl(p1, [blind(row, "PRIMARY") for row in primary[reviewer]])
        write_jsonl(p2, [blind(row, "SECONDARY_OVERLAP") for row in secondary[reviewer]])
        files.extend([p1, p2])

    assignment = []
    for row_id in sorted(overlap_ids):
        assignment.append(
            {
                "record_id": row_id,
                "primary_reviewer": primary_owner[row_id],
                "secondary_reviewer": secondary_owner[row_id],
            }
        )
    assignment_path = ROOT / "sealed_overlap_assignments.jsonl"
    write_jsonl(assignment_path, assignment)
    files.append(assignment_path)

    primary_counts = {
        reviewer: Counter(source_for_id[r["candidate_id"]] for r in primary[reviewer])
        for reviewer in REVIEWERS
    }
    secondary_counts = {
        reviewer: Counter(source_for_id[r["candidate_id"]] for r in secondary[reviewer])
        for reviewer in REVIEWERS
    }
    manifest = {
        "audit_id": "person_agent_ontology_audit_20260712",
        "seed": SEED,
        "total_records": 600,
        "overlap_records": len(overlap_ids),
        "overlap_fraction": len(overlap_ids) / 600,
        "reviewers": list(REVIEWERS),
        "primary_counts": {r: dict(primary_counts[r]) for r in REVIEWERS},
        "secondary_counts": {r: dict(secondary_counts[r]) for r in REVIEWERS},
        "synthetic_risky_overlap": Counter(
            row.get("source_agent_subtype")
            for row in overlap
            if source_for_id[row["candidate_id"]] == "synthetic"
            and row.get("source_agent_subtype") in RISKY
        ),
        "files": {},
    }
    manifest_path = ROOT / "review_packet_manifest.json"
    for path in files:
        manifest["files"][str(path)] = sha256(path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
