#!/usr/bin/env python3
"""
MLM Context Augmentation

Produces n-gram diversity around entity spans by:
  1. Finding all entity character positions in raw text
  2. Masking nearby non-entity words with probability that decays outward
     from span boundaries (peak at distance 1, tapering to background rate)
  3. Infilling with roberta-base (one forward pass, top-k sampling per mask)
  4. Writing augmented raw-text JSONL — same entities list, new surrounding text

Output is raw JSONL suitable for preprocess_extended_training.py.

Usage:
  python3 augment_mlm_context.py \\
    --input ner_training_ultra_short_agent_training_agent_20260406_193605.jsonl \\
    --output augmented_mlm.jsonl \\
    --copies 2 --window 3 --peak-prob 0.55 --base-prob 0.05 --top-k 10
"""

import re
import json
import random
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ---------------------------------------------------------------------------
# Span utilities
# ---------------------------------------------------------------------------

def find_entity_char_spans(text: str, entities: List[Dict]) -> List[Tuple[int, int]]:
    """Return character (start, end) for every entity occurrence in text."""
    spans = []
    for ent in entities:
        name = ent.get("text") or ent.get("name", "")
        if not name or len(name) < 2:
            continue
        for m in re.finditer(re.escape(name), text, re.IGNORECASE):
            spans.append((m.start(), m.end()))
    # Merge overlapping spans
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def get_word_spans(text: str) -> List[Tuple[int, int]]:
    """Return (start, end) for every whitespace-separated word token."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def word_distance_to_entity(word_spans: List[Tuple[int, int]],
                             entity_char_spans: List[Tuple[int, int]]) -> List[float]:
    """
    For each word, compute its word-index distance to the nearest entity word.
    Words overlapping an entity span have distance 0 (and are never masked).
    """
    n = len(word_spans)
    in_entity = [False] * n
    for i, (ws, we) in enumerate(word_spans):
        for es, ee in entity_char_spans:
            if ws < ee and we > es:   # any overlap
                in_entity[i] = True
                break

    distances = [float("inf")] * n
    for i in range(n):
        if in_entity[i]:
            distances[i] = 0.0
            continue
        for j in range(n):
            if in_entity[j]:
                distances[i] = min(distances[i], abs(i - j))
    return distances, in_entity


def mask_probabilities(distances: List[float], in_entity: List[bool],
                        window: int, peak: float, base: float) -> List[float]:
    """
    Graduated masking probability that decays outward from entity spans.

      dist=0  (entity token)  → 0   (never masked)
      dist=1                  → peak
      dist=2..window          → linear decay from peak → base
      dist>window             → base * 0.3  (low background)
    """
    probs = []
    for dist, is_ent in zip(distances, in_entity):
        if is_ent or dist == 0.0:
            probs.append(0.0)
        elif dist == float("inf"):
            probs.append(base * 0.3)
        elif dist <= window:
            # Linear decay: dist=1 → peak, dist=window → base
            t = (dist - 1) / max(window - 1, 1)
            probs.append(peak * (1 - t) + base * t)
        else:
            probs.append(base)
    return probs


# ---------------------------------------------------------------------------
# Masking and infilling
# ---------------------------------------------------------------------------

MASK_TOKEN = "<mask>"   # RoBERTa
_ARTICLES = {"the", "a", "an"}

def mask_text(text: str, entities: List[Dict],
              window: int, peak: float, base: float,
              rng: random.Random) -> Tuple[str, int]:
    """
    Insert <mask> in-place for sampled non-entity word positions.
    Returns (masked_text, n_masks).
    Works on character positions — entity text is never touched.
    Articles (the/a/an) directly preceding an entity span are protected:
    they're structural glue that helps the fill-mask model understand context.
    """
    entity_spans = find_entity_char_spans(text, entities)
    word_spans   = get_word_spans(text)
    if not word_spans:
        return text, 0

    distances, in_entity = word_distance_to_entity(word_spans, entity_spans)
    probs = mask_probabilities(distances, in_entity, window, peak, base)

    # Protect articles at distance=1 that immediately precede an entity span
    for i, (ws, we) in enumerate(word_spans):
        if (not in_entity[i]
                and distances[i] == 1.0
                and text[ws:we].lower().rstrip(".,") in _ARTICLES):
            next_i = i + 1
            if next_i < len(word_spans) and in_entity[next_i]:
                probs[i] = 0.0

    to_mask = [
        (ws, we)
        for i, (ws, we) in enumerate(word_spans)
        if not in_entity[i] and rng.random() < probs[i]
    ]
    if not to_mask:
        return text, 0

    result = text
    for ws, we in sorted(to_mask, reverse=True):
        result = result[:ws] + MASK_TOKEN + result[we:]

    return result, len(to_mask)


def fill_masks(masked_text: str, original_text: str, model, tokenizer,
               top_k: int, temperature: float, device,
               chunk_size: int = 480) -> str:
    """
    Fill all <mask> tokens in masked_text via top-k sampling.

    For texts longer than chunk_size tokens, processes in overlapping windows
    so no mask is left unfilled due to truncation.

    Any mask that still can't be filled (e.g., empty prediction) is replaced
    with the original word from the source text.
    """
    # Build a fallback map: char position of each <mask> → original word
    # by aligning masks back to original text character positions
    fallback = _build_fallback_map(masked_text, original_text)

    # Tokenise to check total length
    all_tokens = tokenizer.encode(masked_text, add_special_tokens=False)
    if len(all_tokens) <= chunk_size:
        result = _fill_chunk(masked_text, model, tokenizer, top_k, temperature, device)
    else:
        result = _fill_in_chunks(masked_text, model, tokenizer,
                                  top_k, temperature, device, chunk_size)

    # Replace any remaining <mask> tokens with original word fallback
    result = _apply_fallbacks(result, fallback)
    return result


def _build_fallback_map(masked_text: str, original_text: str) -> Dict[int, str]:
    """
    Map each <mask> char-position in masked_text to the original word it replaced.
    Aligns by counting non-mask words in order.
    """
    # Split both into word spans preserving positions
    orig_words = re.finditer(r"\S+", original_text)
    mask_parts = re.finditer(r"<mask>|\S+", masked_text)

    fallback = {}
    orig_iter = iter(orig_words)
    for m in mask_parts:
        try:
            orig_m = next(orig_iter)
        except StopIteration:
            break
        if m.group() == MASK_TOKEN:
            fallback[m.start()] = orig_m.group()
        # else: non-mask token — already aligned, advance orig
    return fallback


def _apply_fallbacks(text: str, fallback: Dict[int, str]) -> str:
    """Replace any remaining <mask> in text using fallback map or empty string."""
    if MASK_TOKEN not in text:
        return text
    # Rebuild by scanning — positions shift as we replace, so go right-to-left
    positions = [m.start() for m in re.finditer(re.escape(MASK_TOKEN), text)]
    for pos in reversed(positions):
        word = fallback.get(pos, "")
        text = text[:pos] + word + text[pos + len(MASK_TOKEN):]
    return text


def _fill_chunk(text: str, model, tokenizer,
                top_k: int, temperature: float, device) -> str:
    """Fill all masks in a single chunk (fits within model max length)."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512,
    ).to(device)

    mask_id = tokenizer.mask_token_id
    mask_positions = (inputs.input_ids == mask_id).nonzero(as_tuple=True)[1]
    if len(mask_positions) == 0:
        return text

    with torch.no_grad():
        logits = model(**inputs).logits

    replacements: List[str] = []
    for tok_pos in mask_positions:
        scaled = logits[0, tok_pos] / max(temperature, 1e-6)
        top_vals, top_ids = torch.topk(scaled, k=min(top_k, scaled.shape[-1]))
        probs = torch.softmax(top_vals, dim=-1)
        sampled_id = top_ids[torch.multinomial(probs, 1).item()].item()
        word = tokenizer.decode([sampled_id]).strip()
        replacements.append(word if word else "")

    result = text
    for replacement in replacements:
        idx = result.find(MASK_TOKEN)
        if idx == -1:
            break
        result = result[:idx] + replacement + result[idx + len(MASK_TOKEN):]
    return result


def _fill_in_chunks(text: str, model, tokenizer,
                     top_k: int, temperature: float, device,
                     chunk_size: int) -> str:
    """
    Split text into overlapping token-length chunks, fill masks in each.
    Chunks are split on sentence boundaries where possible.
    """
    # Split into sentences (rough: split on ". " or "\n")
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(tokenizer.encode(sent, add_special_tokens=False))
        if current_len + sent_len > chunk_size and current:
            chunks.append(" ".join(current))
            current = [sent]
            current_len = sent_len
        else:
            current.append(sent)
            current_len += sent_len

    if current:
        chunks.append(" ".join(current))

    filled_chunks = [_fill_chunk(c, model, tokenizer, top_k, temperature, device)
                     for c in chunks]
    return " ".join(filled_chunks)


# ---------------------------------------------------------------------------
# Story augmentation
# ---------------------------------------------------------------------------

def augment_story(story: Dict[str, Any], model, tokenizer,
                  window: int, peak: float, base: float,
                  top_k: int, temperature: float,
                  n_copies: int, device, rng: random.Random) -> List[Dict]:
    """
    Produce n_copies augmented versions of a story.
    Returns list of augmented story dicts with same entities, new text.
    """
    text     = story.get("text", "")
    entities = story.get("entities", [])
    char_ents = [{"text": c["name"]} for c in story.get("characters", []) if c.get("name")]
    all_ents  = entities + char_ents

    results = []
    for _ in range(n_copies):
        masked, n_masks = mask_text(text, all_ents, window, peak, base, rng)
        if n_masks == 0:
            results.append(story)
            continue
        filled = fill_masks(masked, text, model, tokenizer, top_k, temperature, device)
        aug = dict(story)
        aug["text"] = filled
        aug["augmentation"] = {"mlm_masks": n_masks, "window": window}
        results.append(aug)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MLM context augmentation for NER training data")
    parser.add_argument("--input",       required=True,             help="Raw story JSONL (text + entities)")
    parser.add_argument("--output",      required=True,             help="Output JSONL path")
    parser.add_argument("--model",       default="roberta-base",    help="HuggingFace fill-mask model")
    parser.add_argument("--copies",      type=int,   default=2,     help="Augmented copies per story")
    parser.add_argument("--window",      type=int,   default=3,     help="Word-distance horizon from entity spans")
    parser.add_argument("--peak-prob",   type=float, default=0.55,  help="Masking probability at distance 1")
    parser.add_argument("--base-prob",   type=float, default=0.05,  help="Background masking probability")
    parser.add_argument("--top-k",       type=int,   default=10,    help="Top-k sampling for fill-mask")
    parser.add_argument("--temperature", type=float, default=1.0,   help="Sampling temperature (higher=more diversity)")
    parser.add_argument("--max-stories", type=int,   default=None,  help="Cap stories processed (for testing)")
    parser.add_argument("--seed",        type=int,   default=42,    help="Random seed")
    parser.add_argument("--include-original", action="store_true",  help="Include original story alongside augmented copies")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🤖 Loading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
    model.eval()
    print(f"✅ Model loaded")

    in_path  = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_in = total_out = total_masks = 0
    skipped = 0

    print(f"\n📂 Input:  {in_path}")
    print(f"📂 Output: {out_path}")
    print(f"⚙️  window={args.window}  peak={args.peak_prob}  base={args.base_prob}  "
          f"top_k={args.top_k}  temp={args.temperature}  copies={args.copies}")
    print()

    with open(in_path) as fin, open(out_path, "w") as fout:
        for lineno, line in enumerate(fin):
            if args.max_stories and total_in >= args.max_stories:
                break
            line = line.strip()
            if not line:
                continue

            try:
                story = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            # Write original if requested
            if args.include_original:
                fout.write(json.dumps(story, ensure_ascii=False) + "\n")
                total_out += 1

            augmented = augment_story(
                story, model, tokenizer,
                args.window, args.peak_prob, args.base_prob,
                args.top_k, args.temperature,
                args.copies, device, rng,
            )
            for aug in augmented:
                fout.write(json.dumps(aug, ensure_ascii=False) + "\n")
                total_out += 1
                total_masks += aug.get("augmentation", {}).get("mlm_masks", 0)

            total_in += 1
            if total_in % 500 == 0:
                avg_masks = total_masks / max(total_out, 1)
                print(f"  {total_in:,} stories → {total_out:,} augmented  "
                      f"(avg {avg_masks:.1f} masks/story)", end="\r", flush=True)

    print(f"\n✅ Done: {total_in:,} input → {total_out:,} output stories")
    print(f"   Total masks filled: {total_masks:,}  (avg {total_masks/max(total_out,1):.1f}/story)")
    if skipped:
        print(f"   Skipped (parse errors): {skipped}")


if __name__ == "__main__":
    main()
