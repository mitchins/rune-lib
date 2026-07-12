#!/usr/bin/env python3
"""Build blinded expanded-context packets for routed ontology audit records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path("person_agent_ontology_audit_20260712")
SYNTHETIC_SOURCE = Path("ner_training_agent_rich_training_ultra_short_agent_rich_20260703_143554.jsonl")
PD_SOURCE = Path("/mnt/storage/gold-books-pd1000/splits/pd1000_train_windows_2048.jsonl")
PD_TEXTS = Path("/mnt/storage/gold-books-pd1000/texts")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def paragraph_context(text: str, mention_start: int) -> str:
    matches = list(re.finditer(r"(?:^|\n\s*\n)(.*?)(?=\n\s*\n|$)", text, re.S))
    if not matches:
        return text[:12000]
    index = next((i for i, match in enumerate(matches) if match.start(1) <= mention_start <= match.end(1)), 0)
    parts = []
    labels = ("PRECEDING PARAGRAPH", "MENTION PARAGRAPH", "FOLLOWING PARAGRAPH")
    for label, i in zip(labels, (index - 1, index, index + 1)):
        value = matches[i].group(1).strip() if 0 <= i < len(matches) else "[not available]"
        parts.append(f"[{label}]\n{value}")
    return "\n\n".join(parts)[:16000]


def repeated_windows(text: str, surface: str, radius: int = 220, limit: int = 5) -> list[str]:
    values = []
    for match in re.finditer(re.escape(surface), text, re.I):
        values.append(text[max(0, match.start() - radius):min(len(text), match.end() + radius)])
        if len(values) >= limit:
            break
    return values


def raw_pd_context(row: dict, candidate: dict) -> tuple[str, list[str]]:
    book_id = candidate["book_id"]
    matches = list(PD_TEXTS.glob(f"*__{book_id}.txt"))
    if len(matches) != 1:
        return row["text"][:16000], repeated_windows(row["text"], candidate["surface"])
    raw = matches[0].read_text(encoding="utf-8", errors="replace")
    tokens = row["tokens"]
    left = max(0, candidate["token_start"] - 10)
    right = min(len(tokens), candidate["token_end"] + 10)
    anchor_words = [re.sub(r"\W+", "", t.casefold()) for t in tokens[left:right]]
    anchor_words = [word for word in anchor_words if word]
    raw_matches = list(re.finditer(r"\w+", raw, re.UNICODE))
    raw_words = [match.group().casefold() for match in raw_matches]
    anchor = tuple(anchor_words)
    found = None
    if anchor:
        for index in range(max(0, len(raw_words) - len(anchor) + 1)):
            if tuple(raw_words[index:index + len(anchor)]) == anchor:
                found = raw_matches[index].start()
                break
    if found is None:
        # The 2048-token source window still supplies substantially more context than the audit slice.
        return row["text"][:16000], repeated_windows(row["text"], candidate["surface"])
    return paragraph_context(raw, found), repeated_windows(raw, candidate["surface"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wanted = {row["record_id"] for row in read_jsonl(args.ids)}
    candidates = {}
    for corpus in ("synthetic", "pd1000"):
        for row in read_jsonl(ROOT / f"{corpus}_manual_audit_candidates.jsonl"):
            if row["candidate_id"] in wanted:
                candidates[row["candidate_id"]] = row

    synthetic_story_ids = {
        row["scene_id"] for row in candidates.values() if row["corpus_source"] == "synthetic_original_export"
    }
    synthetic_texts = {}
    if synthetic_story_ids:
        with SYNTHETIC_SOURCE.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                story_id = str(row.get("_meta", {}).get("story_id"))
                if story_id in synthetic_story_ids:
                    synthetic_texts[story_id] = row.get("text", "")

    pd_indices = {
        row["row_index"] for row in candidates.values() if row["corpus_source"] == "pd1000_train"
    }
    pd_rows = {}
    if pd_indices:
        with PD_SOURCE.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index in pd_indices:
                    pd_rows[index] = json.loads(line)
                if len(pd_rows) == len(pd_indices):
                    break

    output = []
    for record_id in sorted(wanted):
        candidate = candidates[record_id]
        if candidate["corpus_source"] == "synthetic_original_export":
            text = synthetic_texts[candidate["scene_id"]]
            expanded = paragraph_context(text, candidate["char_start"])
            repeats = repeated_windows(text, candidate["surface"])
        else:
            expanded, repeats = raw_pd_context(pd_rows[candidate["row_index"]], candidate)
        repeat_text = "\n\n".join(f"[REPEATED MENTION {i + 1}]\n{x}" for i, x in enumerate(repeats))
        output.append({
            "record_id": record_id,
            "surface": candidate["surface"],
            "context": expanded + ("\n\n" + repeat_text if repeat_text else ""),
            "review_role": "EXPANDED_CONTEXT",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"requested": len(wanted), "written": len(output)}, indent=2))


if __name__ == "__main__":
    main()
