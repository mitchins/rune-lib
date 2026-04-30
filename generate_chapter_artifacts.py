#!/usr/bin/env python3
"""
Generate chapter/window artifacts from downloaded raw books for annotation.

Produces ≥100 annotation-ready artifacts in gold_sources/chapter_artifacts/:
  - Individual .txt and .json files per artifact
  - A JSONL manifest (artifacts_manifest.jsonl)

Selection strategy:
  1. Try chapter-level splits via regex patterns
  2. Fall back to 4096-word sliding windows for texts without clear chapters
  3. Score each candidate via narrative/entity-density heuristic
  4. Select top 100 artifacts (allowing multiple per source)
  5. Deduplicate by content hash
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RAW_DIR = Path("gold_sources/raw")
OUT_DIR = Path("gold_sources/chapter_artifacts")

WORD_TARGET = 4096
WORD_FLOOR = 400         # minimum words to keep a chapter/window artifact
WINDOW_STEP = 1024       # stride for sliding-window fallback
MAX_ARTIFACTS_TOTAL = 100
# How many artifacts to generate per source before scoring (upper bound)
MAX_CANDS_PER_SOURCE = 30
# For large books, split generously; score will filter down to top N
MAX_CHAPTER_CHAR = 800_000   # skip chapter-split for a single section >this
# How many top artifacts to keep per source in the final selection
MAX_PER_SOURCE = 12

# Chapter-heading patterns ordered by specificity (most specific first).
# Each pattern should match a full heading line.
CHAPTER_PATTERNS = [
    # "CHAPTER I", "CHAPTER IV.", "CHAPTER 1.", "CHAPTER ONE"
    r"^(?:CHAPTER)\s+(?:[IVXLCDM]+\.?|[0-9]+\.?|[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)(?:\s*[.:\-—].*)?$",
    # "Chapter I", "Chapter 1", "Chapter One"
    r"^(?:Chapter)\s+(?:[IVXLCDM]+\.?|[0-9]+\.?|[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)(?:\s*[.:\-—].*)?$",
    # Standalone roman numeral: "I.", "II.", "XXI." (at least 2 chars, own line)
    r"^[IVXLCDM]{2,}\.?\s*$",
    # "PART I", "PART 1"
    r"^(?:PART|Part)\s+(?:[IVXLCDM]+|[0-9]+)(?:\s.*)?$",
    # "ACT I", "ACT II", "SCENE I"
    r"^(?:ACT|Act|SCENE|Scene)\s+(?:[IVXLCDM]+|[0-9]+)(?:\s.*)?$",
]

# Minimum reasonable chapter count after split (below this → use windows)
MIN_CHAPTERS = 4
# Minimum average chapter word count to trust a chapter split
MIN_AVG_CHAPTER_WORDS = 500


# --------------------------------------------------------------------------- #
# Utilities (mirrors download_gold_sources.py)
# --------------------------------------------------------------------------- #

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-") or "artifact"


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def score_text(text: str) -> float:
    """Narrative / entity-density heuristic (from download_gold_sources.py)."""
    words = tokenize_words(text)
    if not words:
        return float("-inf")
    word_count = len(words)
    if word_count < 120:
        return float("-inf")

    quote_score = text.count('"') + text.count("\u201c") + text.count("\u201d")
    dialogue_lines = sum(
        1 for line in text.splitlines()
        if line.lstrip().startswith('"') or line.lstrip().startswith("\u201c")
    )
    cap_words = 0
    skip = {"the", "and", "or", "but", "a", "an", "of", "to", "in"}
    for tok in words:
        clean = re.sub(r"^[^\w]+|[^\w]+$", "", tok)
        if clean and clean[0].isupper() and clean.lower() not in skip:
            cap_words += 1

    target = 1400
    length_score = max(0.0, 1.0 - abs(word_count - target) / target)
    capital_density = cap_words / max(1, word_count)
    dialogue_density = dialogue_lines / max(1, word_count / 50)
    quote_density = quote_score / max(1, word_count / 80)
    return (length_score * 3.0) + (capital_density * 80.0) + (dialogue_density * 2.0) + (quote_density * 1.5)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Chapter splitting
# --------------------------------------------------------------------------- #

def _split_by_pattern(text: str, pattern: str) -> list[tuple[str, str]]:
    """
    Split `text` on lines matching `pattern`.
    Returns list of (heading, content) pairs.
    The content does NOT include the heading line.
    """
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    current_heading = "preamble"
    current_lines: list[str] = []
    compiled = re.compile(pattern)
    for line in lines:
        if compiled.match(line.strip()):
            content = "\n".join(current_lines).strip()
            if content:
                sections.append((current_heading, content))
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    # flush last section
    content = "\n".join(current_lines).strip()
    if content:
        sections.append((current_heading, content))
    return sections


def _quality_score_split(sections: list[tuple[str, str]]) -> float:
    """Score a chapter split: more chapters with reasonable size → higher score."""
    word_counts = [len(tokenize_words(c)) for _, c in sections if c.strip()]
    if not word_counts:
        return 0.0
    n = len(word_counts)
    if n < MIN_CHAPTERS:
        return 0.0
    avg = sum(word_counts) / n
    if avg < MIN_AVG_CHAPTER_WORDS:
        return 0.0
    # Penalise very uneven splits
    min_wc = min(word_counts)
    ratio = min_wc / avg if avg > 0 else 0
    return n * avg * ratio


def try_chapter_split(text: str) -> list[tuple[str, str]] | None:
    """
    Try each pattern in CHAPTER_PATTERNS and return the best split,
    or None if no pattern gives a quality split.
    The preamble section (before the first heading) is excluded.
    """
    best_sections = None
    best_quality = 0.0

    for pattern in CHAPTER_PATTERNS:
        sections = _split_by_pattern(text, pattern)
        # Remove 'preamble' section (first one if heading == 'preamble')
        body = [(h, c) for h, c in sections if h != "preamble"]
        quality = _quality_score_split(body)
        if quality > best_quality:
            best_quality = quality
            best_sections = body

    return best_sections if best_sections else None


# --------------------------------------------------------------------------- #
# Window fallback
# --------------------------------------------------------------------------- #

def generate_windows(
    text: str,
    max_words: int = WORD_TARGET,
    min_words: int = WORD_FLOOR,
    step: int = WINDOW_STEP,
) -> list[tuple[str, dict[str, Any]]]:
    """Generate scored sliding windows over `text`."""
    words = tokenize_words(text)
    if len(words) <= max_words:
        return [(text.strip(), {"start_word": 0, "end_word": len(words)})]

    windows = []
    for start in range(0, max(1, len(words) - min_words + 1), step):
        end = min(len(words), start + max_words)
        chunk = " ".join(words[start:end]).strip()
        if len(tokenize_words(chunk)) < min_words:
            continue
        windows.append((chunk, {"start_word": start, "end_word": end}))
    return windows


# --------------------------------------------------------------------------- #
# Per-source candidate generation
# --------------------------------------------------------------------------- #

def candidates_for_source(
    raw_path: Path,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return a list of scored artifact candidates for one source."""
    text = raw_path.read_text(encoding="utf-8")
    candidate_meta = meta.get("candidate", {})
    title = candidate_meta.get("title", raw_path.stem)
    author = candidate_meta.get("author", "Unknown")
    category = candidate_meta.get("category", "unknown")

    chapters = try_chapter_split(text)
    candidates: list[dict[str, Any]] = []

    if chapters:
        mode = "chapter"
        for i, (heading, content) in enumerate(chapters):
            wc = len(tokenize_words(content))
            if wc < WORD_FLOOR:
                continue
            if len(content) > MAX_CHAPTER_CHAR:
                # Oversized single chapter — take a window from it
                window_text, winfo = generate_windows(content, max_words=WORD_TARGET)[0]
                candidates.append({
                    "title": title,
                    "author": author,
                    "category": category,
                    "source_raw_path": str(raw_path),
                    "chapter_label": heading,
                    "chapter_index": i,
                    "selection_mode": "chapter_window",
                    "text": window_text,
                    "word_count": len(tokenize_words(window_text)),
                    "score": score_text(window_text),
                })
            else:
                candidates.append({
                    "title": title,
                    "author": author,
                    "category": category,
                    "source_raw_path": str(raw_path),
                    "chapter_label": heading,
                    "chapter_index": i,
                    "selection_mode": mode,
                    "text": content.strip(),
                    "word_count": wc,
                    "score": score_text(content),
                })
    else:
        # Window fallback
        windows = generate_windows(text, max_words=WORD_TARGET, step=WINDOW_STEP)
        for j, (chunk, winfo) in enumerate(windows):
            candidates.append({
                "title": title,
                "author": author,
                "category": category,
                "source_raw_path": str(raw_path),
                "chapter_label": f"window-{j}",
                "chapter_index": j,
                "selection_mode": "window",
                "text": chunk,
                "word_count": len(tokenize_words(chunk)),
                "score": score_text(chunk),
            })

    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:MAX_CANDS_PER_SOURCE]


# --------------------------------------------------------------------------- #
# Global selection
# --------------------------------------------------------------------------- #

def select_top_artifacts(
    all_candidates: list[dict[str, Any]],
    n: int = MAX_ARTIFACTS_TOTAL,
    max_per_source: int = MAX_PER_SOURCE,
) -> list[dict[str, Any]]:
    """
    Select top-N artifacts globally by score, with deduplication by content
    hash and a per-source cap.

    Strategy:
      1. Guarantee at least 1 artifact from every source (if it has any
         qualifying candidates) — ensures full source coverage.
      2. Fill remaining slots greedily by score (subject to per-source cap).
    """
    # Group candidates by source key, each list pre-sorted by score descending
    by_source: dict[str, list[dict[str, Any]]] = {}
    for cand in all_candidates:
        key = slugify(f"{cand['author']}-{cand['title']}")
        by_source.setdefault(key, []).append(cand)
    for lst in by_source.values():
        lst.sort(key=lambda x: x["score"], reverse=True)

    seen_hashes: set[str] = set()
    per_source_count: dict[str, int] = {}
    selected: list[dict[str, Any]] = []

    def _try_add(cand: dict[str, Any]) -> bool:
        digest = sha256_text(cand["text"])
        if digest in seen_hashes:
            return False
        key = slugify(f"{cand['author']}-{cand['title']}")
        if per_source_count.get(key, 0) >= max_per_source:
            return False
        seen_hashes.add(digest)
        per_source_count[key] = per_source_count.get(key, 0) + 1
        cand = dict(cand)
        cand["content_hash"] = digest
        selected.append(cand)
        return True

    # --- Phase 1: guarantee one artifact per source ---
    for key, lst in sorted(by_source.items()):
        for cand in lst:
            if _try_add(cand):
                break  # one guaranteed per source
        if len(selected) >= n:
            break

    # --- Phase 2: fill remaining slots greedily by score ---
    remaining = [c for lst in by_source.values() for c in lst]
    remaining.sort(key=lambda x: x["score"], reverse=True)
    for cand in remaining:
        if len(selected) >= n:
            break
        _try_add(cand)

    return selected


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def write_artifacts(artifacts: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "artifacts_manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as mf:
        for idx, art in enumerate(artifacts):
            source_slug = slugify(f"{art['author']}-{art['title']}")
            chapter_slug = slugify(art["chapter_label"])[:60]
            artifact_id = f"{source_slug}__{chapter_slug}__{idx:04d}"

            txt_path = out_dir / f"{artifact_id}.txt"
            json_path = out_dir / f"{artifact_id}.json"

            write_text(txt_path, art["text"])

            meta = {
                "artifact_id": artifact_id,
                "title": art["title"],
                "author": art["author"],
                "category": art["category"],
                "source_raw_path": art["source_raw_path"],
                "chapter_label": art["chapter_label"],
                "chapter_index": art["chapter_index"],
                "selection_mode": art["selection_mode"],
                "word_count": art["word_count"],
                "score": art["score"],
                "content_hash": art["content_hash"],
                "text_path": str(txt_path),
            }
            write_json(json_path, meta)

            manifest_record = {**meta, "text": art["text"]}
            mf.write(json.dumps(manifest_record, ensure_ascii=True) + "\n")

    print(f"\n✓ Wrote {len(artifacts)} artifacts to {out_dir}/")
    print(f"  Manifest: {manifest_path}")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def print_report(artifacts: list[dict[str, Any]]) -> None:
    by_source: dict[str, list] = {}
    by_mode: dict[str, int] = {}
    by_category: dict[str, int] = {}

    for art in artifacts:
        key = f"{art['author']} — {art['title']}"
        by_source.setdefault(key, []).append(art)
        mode = art["selection_mode"]
        by_mode[mode] = by_mode.get(mode, 0) + 1
        cat = art["category"]
        by_category[cat] = by_category.get(cat, 0) + 1

    print(f"\n{'='*60}")
    print(f"ARTIFACT GENERATION REPORT")
    print(f"{'='*60}")
    print(f"Total artifacts: {len(artifacts)}")
    print(f"\nBy selection mode:")
    for mode, count in sorted(by_mode.items()):
        print(f"  {mode:20s}: {count}")
    print(f"\nBy category:")
    for cat, count in sorted(by_category.items()):
        print(f"  {cat:30s}: {count}")
    print(f"\nPer-source coverage ({len(by_source)} sources):")
    for src, arts in sorted(by_source.items(), key=lambda x: -len(x[1])):
        modes = sorted({a["selection_mode"] for a in arts})
        print(f"  [{len(arts):2d}] {src}  ({', '.join(modes)})")
    print(f"{'='*60}\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    raw_dir = RAW_DIR
    out_dir = OUT_DIR

    txt_files = sorted(raw_dir.glob("*.txt"))
    if not txt_files:
        print(f"ERROR: No .txt files found in {raw_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(txt_files)} raw source files in {raw_dir}")

    all_candidates: list[dict[str, Any]] = []
    fallback_sources: list[str] = []
    chapter_sources: list[str] = []

    for raw_path in txt_files:
        meta_path = raw_path.with_suffix(".json")
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        cand_meta = meta.get("candidate", {})
        title = cand_meta.get("title", raw_path.stem)
        author = cand_meta.get("author", "Unknown")
        print(f"  Processing: {author} — {title}", end=" ... ", flush=True)

        try:
            cands = candidates_for_source(raw_path, meta)
            modes = {c["selection_mode"] for c in cands}
            print(f"{len(cands)} candidates ({', '.join(sorted(modes))})")
            if any(m.startswith("chapter") for m in modes):
                chapter_sources.append(title)
            else:
                fallback_sources.append(title)
            all_candidates.extend(cands)
        except Exception as exc:
            print(f"FAILED: {exc}")

    print(f"\nTotal candidates before selection: {len(all_candidates)}")
    print(f"Chapter splits succeeded: {len(chapter_sources)}")
    print(f"Window fallbacks used: {len(fallback_sources)}")
    if fallback_sources:
        print(f"  Fallback sources: {fallback_sources}")

    artifacts = select_top_artifacts(all_candidates, n=MAX_ARTIFACTS_TOTAL)
    print_report(artifacts)
    write_artifacts(artifacts, out_dir)


if __name__ == "__main__":
    main()
