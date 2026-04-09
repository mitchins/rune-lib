#!/usr/bin/env python3
"""
Regression / visual eval harness for augment_mlm_context.py

Runs MLM augmentation on a handful of stories and produces a human-readable
report that lets you verify:

  1. Entity preservation  – every entity appears unchanged in the output
  2. No stray <mask> tokens
  3. Lexical change rate   – what fraction of non-entity words were replaced
  4. Context windows       – 5-word windows around each entity, before vs after
  5. Sentence-level diff   – whole prose paragraphs side-by-side with changed
                             words capitalised for easy scanning

Usage:
  python3 eval_mlm_augmentation.py \
      --input ner_training_ultra_short_agent_training_agent_20260406_193605.jsonl \
      --n-stories 6 --story-ids 0,2,7,8,13,22 \
      --copies 2 --window 3 --peak-prob 0.55 --temperature 1.2 --seed 99
"""

import re
import sys
import json
import random
import argparse
import difflib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ---------------------------------------------------------------------------
# Inline import of augment helpers (avoids duplicating the logic)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from augment_mlm_context import (
    augment_story,
    find_entity_char_spans,
    get_word_spans,
)

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------
BOLD  = "\033[1m"
RED   = "\033[31m"
GREEN = "\033[32m"
CYAN  = "\033[36m"
YEL   = "\033[33m"
DIM   = "\033[2m"
RESET = "\033[0m"

def cprint(text: str, colour: str = "") -> None:
    print(colour + text + RESET)

def header(title: str, width: int = 78) -> None:
    line = "─" * width
    print(f"\n{BOLD}{CYAN}{line}")
    print(f"  {title}")
    print(f"{line}{RESET}")

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def entity_names(story: Dict) -> List[str]:
    """All canonical entity strings from entities + characters fields."""
    names = []
    for e in story.get("entities", []):
        n = e.get("text") or e.get("name", "")
        if n and len(n) >= 2:
            names.append(n)
    for c in story.get("characters", []):
        n = c.get("name", "")
        if n and len(n) >= 2:
            names.append(n)
    return names


def check_entity_preservation(orig_text: str, aug_text: str,
                               entities: List[str]) -> List[Dict]:
    """
    For every entity name, verify it occurs at least once in the augmented text.
    Returns list of dicts with: name, in_orig, in_aug, ok.
    """
    results = []
    for name in entities:
        in_orig = bool(re.search(re.escape(name), orig_text, re.IGNORECASE))
        in_aug  = bool(re.search(re.escape(name), aug_text,  re.IGNORECASE))
        results.append({"name": name, "in_orig": in_orig, "in_aug": in_aug,
                         "ok": (not in_orig) or in_aug})
    return results


def changed_word_ratio(orig_text: str, aug_text: str,
                        entity_spans: List[Tuple[int, int]]) -> Dict:
    """
    Split both texts into words; count how many non-entity words changed.
    Uses difflib SequenceMatcher to align words properly (zip drifts on
    long texts where fill-mask may insert/delete tokens).
    Returns dict: total_words, changed_words, entity_words, change_rate.
    """
    orig_words = re.findall(r"\S+", orig_text)
    aug_words  = re.findall(r"\S+", aug_text)

    # Build entity word set (lower) to exclude from change stats
    entity_words: set = set()
    for es, ee in entity_spans:
        for w in re.findall(r"\S+", orig_text[es:ee]):
            entity_words.add(w.lower())

    sm = difflib.SequenceMatcher(
        None,
        [w.lower() for w in orig_words],
        [w.lower() for w in aug_words],
        autojunk=False,
    )
    changed = 0
    non_entity = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for w in orig_words[i1:i2]:
                if w.lower() not in entity_words:
                    non_entity += 1
        elif op in ("replace", "delete"):
            for w in orig_words[i1:i2]:
                if w.lower() not in entity_words:
                    non_entity += 1
                    changed += 1

    return {
        "total_words":      max(len(orig_words), len(aug_words)),
        "non_entity_words": non_entity,
        "changed_words":    changed,
        "change_rate":      changed / max(non_entity, 1),
    }


def context_windows(orig_text: str, aug_text: str, entity_name: str,
                     window: int = 5) -> List[Tuple[str, str]]:
    """
    For each occurrence of entity_name, extract a ±window-word context
    from both original and augmented text.
    Returns list of (orig_window, aug_window) string pairs.
    """
    results = []
    orig_words = re.findall(r"\S+", orig_text)
    aug_words  = re.findall(r"\S+", aug_text)

    for m in re.finditer(re.escape(entity_name), orig_text, re.IGNORECASE):
        # find word index of match
        pre = orig_text[:m.start()]
        wi = len(re.findall(r"\S+", pre))
        ew = len(re.findall(r"\S+", m.group()))
        lo = max(0, wi - window)
        hi = min(len(orig_words), wi + ew + window)
        orig_ctx = " ".join(orig_words[lo:hi])

        # same window in aug text
        aug_ctx = " ".join(aug_words[lo:hi]) if lo < len(aug_words) else ""
        results.append((orig_ctx, aug_ctx))
    return results


def highlight_changes(orig: str, aug: str) -> str:
    """
    Inline diff of two strings at word level.
    Replaced words in aug are shown in GREEN, removed words in RED (strikethrough).
    Result is the augmented text with changed words highlighted.
    """
    orig_words = re.findall(r"\S+|\s+", orig)
    aug_words  = re.findall(r"\S+|\s+", aug)

    sm = difflib.SequenceMatcher(None,
                                  [w.lower() for w in orig_words],
                                  [w.lower() for w in aug_words],
                                  autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out.append("".join(aug_words[j1:j2]))
        elif op == "replace":
            out.append(RED   + "".join(orig_words[i1:i2]) + RESET)
            out.append(GREEN + "".join(aug_words[j1:j2])  + RESET)
        elif op == "insert":
            out.append(GREEN + "".join(aug_words[j1:j2])  + RESET)
        elif op == "delete":
            out.append(RED   + "".join(orig_words[i1:i2]) + RESET)
    return "".join(out)


def first_n_sentences(text: str, n: int = 4) -> str:
    """Return first n prose sentences (skipping ### headers)."""
    lines = [l for l in text.split("\n") if l.strip() and not l.startswith("#")]
    prose = " ".join(lines)
    sents = re.split(r"(?<=[.!?])\s+", prose)
    return " ".join(sents[:n])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report_story(orig: Dict, augmented_copies: List[Dict], copy_idx: int = 0) -> Dict:
    """
    Print a detailed report for one story / one augmented copy.
    Returns a summary dict for aggregate stats.
    """
    aug = augmented_copies[copy_idx]
    orig_text = orig["text"]
    aug_text  = aug["text"]
    ents      = entity_names(orig)
    n_masks   = aug.get("augmentation", {}).get("mlm_masks", 0)
    ent_spans = find_entity_char_spans(orig_text, orig.get("entities", [])
                                       + [{"text": c["name"]} for c in orig.get("characters", [])
                                          if c.get("name")])

    preservation = check_entity_preservation(orig_text, aug_text, ents)
    stats        = changed_word_ratio(orig_text, aug_text, ent_spans)
    remaining    = aug_text.count("<mask>")

    # ── Header ─────────────────────────────────────────────────────────────
    title = re.search(r"###\s*Scene\s*\d+[:\s]*(.*)", orig_text)
    story_title = title.group(1).strip() if title else orig.get("title", "Story")[:60]
    print(f"\n{BOLD}{'─'*78}{RESET}")
    cprint(f"  {story_title}", BOLD + CYAN)
    print(f"  Entities: "
          + ", ".join(f"{BOLD}{e['text']}{RESET}{DIM}({e['type']}){RESET}"
                      for e in orig.get("entities", [])[:8]))
    print(f"  Copy {copy_idx+1} │ {n_masks} masks filled │ "
          f"change rate {stats['change_rate']:.1%} "
          f"({stats['changed_words']}/{stats['non_entity_words']} non-entity words)")

    # ── Entity preservation ─────────────────────────────────────────────────
    failures = [p for p in preservation if not p["ok"]]
    if failures:
        cprint(f"\n  ⚠ ENTITY PRESERVATION FAILURES:", RED + BOLD)
        for f in failures:
            cprint(f"    ✗ '{f['name']}' lost!", RED)
    else:
        cprint(f"\n  ✓ All {len(preservation)} entities preserved", GREEN)

    if remaining:
        cprint(f"  ✗ {remaining} <mask> token(s) remain in output!", RED + BOLD)
    else:
        cprint(f"  ✓ No stray <mask> tokens", GREEN)

    # ── Prose paragraph diff ─────────────────────────────────────────────────
    orig_prose = first_n_sentences(orig_text, 4)
    aug_prose  = first_n_sentences(aug_text,  4)

    print(f"\n  {BOLD}ORIGINAL:{RESET}")
    print(f"  {DIM}{orig_prose[:400]}{RESET}")
    print(f"\n  {BOLD}AUGMENTED{RESET} {DIM}(red=removed, green=added):{RESET}")
    highlighted = highlight_changes(orig_prose, aug_prose)
    # Indent each line
    for line in highlighted.split("\n"):
        print(f"  {line}")

    # ── Context windows around entities ─────────────────────────────────────
    print(f"\n  {BOLD}ENTITY CONTEXT WINDOWS (±5 words):{RESET}")
    shown = 0
    for ent in orig.get("entities", []):
        name = ent.get("text") or ent.get("name", "")
        etype = ent.get("type", "?")
        windows = context_windows(orig_text, aug_text, name, window=5)
        if not windows:
            continue
        orig_w, aug_w = windows[0]
        changed = orig_w.lower() != aug_w.lower()
        marker = "~" if changed else "="
        print(f"\n  {DIM}[{etype}]{RESET} {BOLD}{name}{RESET}")
        print(f"    {DIM}orig:{RESET}  ...{orig_w}...")
        if changed:
            print(f"    {GREEN}aug:{RESET}   ...{aug_w}...")
        else:
            print(f"    {DIM}aug:{RESET}   (unchanged context)")
        shown += 1
        if shown >= 6:
            break

    return {
        "story_title":   story_title,
        "n_masks":       n_masks,
        "change_rate":   stats["change_rate"],
        "entities_ok":   len(preservation) - len(failures),
        "entities_fail": len(failures),
        "mask_remain":   remaining,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_stories(path: str, story_ids: Optional[List[int]] = None,
                 n: int = 6) -> List[Dict]:
    stories = []
    with open(path) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if story_ids is not None:
                if idx in story_ids:
                    stories.append(s)
            else:
                types = {e["type"] for e in s.get("entities", [])}
                prose = re.sub(r"###.*?\n", "", s.get("text", ""))
                if len(prose) > 500 and types >= {"PERSON", "LOCATION", "AGENT"}:
                    stories.append(s)
            if story_ids is None and len(stories) >= n:
                break
    return stories


def main():
    ap = argparse.ArgumentParser(description="Visual eval for MLM augmentation")
    ap.add_argument("--input",       required=True,           help="Raw story JSONL")
    ap.add_argument("--model",       default="roberta-base",  help="Fill-mask model")
    ap.add_argument("--n-stories",   type=int,   default=6,   help="Stories to evaluate")
    ap.add_argument("--story-ids",   default=None,            help="Comma-separated story line indices")
    ap.add_argument("--copies",      type=int,   default=2,   help="Augmented copies per story")
    ap.add_argument("--window",      type=int,   default=3)
    ap.add_argument("--peak-prob",   type=float, default=0.55)
    ap.add_argument("--base-prob",   type=float, default=0.05)
    ap.add_argument("--top-k",       type=int,   default=10)
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--output",      default=None, help="Optional JSONL to write augmented copies")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    story_ids = [int(x) for x in args.story_ids.split(",")] if args.story_ids else None

    print(f"🤖 Loading {args.model}…")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
    model.eval()
    print(f"✅ Model loaded on {device}\n")

    stories = load_stories(args.input, story_ids, args.n_stories)
    print(f"📖 Loaded {len(stories)} stories for evaluation")

    header("MLM AUGMENTATION EVALUATION REPORT", 78)
    print(f"  Input:       {args.input}")
    print(f"  Model:       {args.model}")
    print(f"  window={args.window}  peak={args.peak_prob}  base={args.base_prob}  "
          f"top_k={args.top_k}  temp={args.temperature}  copies={args.copies}")

    all_augmented = []
    summaries     = []

    for si, story in enumerate(stories):
        copies = augment_story(
            story, model, tokenizer,
            args.window, args.peak_prob, args.base_prob,
            args.top_k, args.temperature, args.copies, device, rng,
        )
        all_augmented.extend(copies)

        for ci in range(min(args.copies, len(copies))):
            summary = report_story(story, copies, ci)
            summaries.append(summary)

    # ── Aggregate summary ───────────────────────────────────────────────────
    header("AGGREGATE SUMMARY", 78)
    total    = len(summaries)
    ent_ok   = sum(s["entities_ok"]   for s in summaries)
    ent_fail = sum(s["entities_fail"] for s in summaries)
    masks_remain = sum(s["mask_remain"] for s in summaries)
    avg_change   = sum(s["change_rate"] for s in summaries) / max(total, 1)
    avg_masks    = sum(s["n_masks"]     for s in summaries) / max(total, 1)

    print(f"\n  Stories evaluated   : {len(stories)}")
    print(f"  Copies per story    : {args.copies}")
    print(f"  Total augmented docs: {total}")
    print()

    ok_str  = f"{GREEN}✓{RESET}" if ent_fail == 0 else f"{RED}✗{RESET}"
    print(f"  {ok_str} Entity preservation : {ent_ok} ok, {ent_fail} FAILED")

    mk_str  = f"{GREEN}✓{RESET}" if masks_remain == 0 else f"{RED}✗{RESET}"
    print(f"  {mk_str} Stray <mask> tokens : {masks_remain}")

    print(f"  ≈ Avg masks / doc   : {avg_masks:.1f}")
    print(f"  ≈ Avg change rate   : {avg_change:.1%}  (non-entity words)")
    print()

    if ent_fail == 0 and masks_remain == 0:
        cprint("  ✅ All checks PASSED", GREEN + BOLD)
    else:
        cprint("  ❌ Some checks FAILED — review output above", RED + BOLD)

    # ── Optional JSONL output ────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for doc in all_augmented:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"\n  💾 Augmented docs written to: {args.output}")

    print()


if __name__ == "__main__":
    main()
