#!/usr/bin/env python3
"""
MLM Context Augmentation

Produces n-gram diversity around entity spans by:
  1. Finding all entity character positions in raw text
  2. Masking nearby non-entity words with probability that decays outward
     from span boundaries (peak at distance 1, tapering to background rate)
  3. Infilling with roberta-base (one forward pass, top-k sampling per mask)
  4. Writing augmented raw-text JSONL — same entities list, new surrounding text

Protection policy (what is NEVER masked):
  - Entity spans (full strings from entities/characters fields)
  - All title-cased sub-spans of multi-word entities (e.g. "Odeceixe" from
    "Odeceixe River Beach") — prevents partial-name leakage
  - Structural spans: markdown headers, Scene/Chapter/Part labels
  - Articles (the/a/an) immediately preceding an entity span
  - Words within 3 positions of a sensory verb (saw/heard/felt/smelled/tasted…)
    to avoid cross-modal substitutions like "color" → "sound"

Post-fill validation: any augmented copy that drops a protected entity string
is silently discarded (falls back to original story).

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
from typing import List, Tuple, Dict, Any, Optional, Set

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ---------------------------------------------------------------------------
# Fix 1 – Subspan / alias expansion
# ---------------------------------------------------------------------------

_COMMON_WORDS: Set[str] = {
    "The", "A", "An", "Of", "In", "On", "At", "To", "For", "By", "With",
    "And", "Or", "But", "Not", "Is", "Was", "Are", "Were", "Be", "Been",
    "Has", "Had", "Have", "Do", "Did", "Does", "Up", "Down", "Out", "From",
    "Into", "Over", "Under", "Again", "New", "Old", "High", "Low",
    "My", "His", "Her", "Its", "Our", "Your", "Their",
    "River", "Sea", "Lake", "Mountain", "City", "Town", "Street", "Road",
    "Beach", "Forest", "Valley", "Hill", "Plains", "Desert", "Island",
    "North", "South", "East", "West", "Central", "Upper", "Lower",
    "Lord", "Lady", "King", "Queen", "Prince", "Princess", "Sir", "Master",
}


def expand_protected_strings(entities: List[Dict]) -> List[str]:
    """
    For every entity, return the full string PLUS any proper-noun sub-spans
    that are sufficiently distinctive (not common English words, ≥ 5 chars for
    single words so that generic nouns like 'Beach' don't carpet-protect the
    whole document via the proximity gradient).

    Multi-word sub-spans are allowed at ≥ 3 chars because phrase rarity is
    higher than single-word rarity.
    """
    all_strings: Set[str] = set()
    for ent in entities:
        name = (ent.get("text") or ent.get("name", "")).strip()
        if not name or len(name) < 2:
            continue
        all_strings.add(name)
        words = name.split()
        if len(words) <= 1:
            continue
        for start in range(len(words)):
            for end in range(start + 1, len(words) + 1):
                if start == 0 and end == len(words):
                    continue  # already added full span
                sub = " ".join(words[start:end])
                if not sub[0].isupper():
                    continue
                sub_words = words[start:end]
                if len(sub_words) == 1:
                    # Single-word sub-span: require ≥5 chars AND not a common word
                    if len(sub) >= 5 and sub not in _COMMON_WORDS:
                        all_strings.add(sub)
                else:
                    # Multi-word sub-span: ≥3 chars total, first word not a bare article
                    if len(sub) >= 3 and sub_words[0] not in {"The", "A", "An"}:
                        all_strings.add(sub)
    return list(all_strings)


# ---------------------------------------------------------------------------
# Fix 2 – Structural span protection
# ---------------------------------------------------------------------------

_STRUCTURAL_RE = re.compile(
    r"(?m)^(?:"
    r"#{1,6}\s+[^\n]+"          # ## Markdown headers
    r"|Scene\s+\d+[^\n]*"       # Scene 1: The Dark Hour
    r"|Chapter\s+\w+[^\n]*"     # Chapter One / Chapter 12
    r"|Part\s+[IVXivx\d]+[^\n]*"# Part II / Part 3
    r"|\*{3,}[^\n]*"            # *** separators
    r"|---[^\n]*"               # --- separators
    r")",
)

def find_structural_char_spans(text: str) -> List[Tuple[int, int]]:
    """Return character spans of structural markers (headers, scene labels, etc.)."""
    return [(m.start(), m.end()) for m in _STRUCTURAL_RE.finditer(text)]


# ---------------------------------------------------------------------------
# Fix 3 – Sensory verb no-fly zone
# ---------------------------------------------------------------------------

_SENSORY_VERBS: Set[str] = {
    "saw", "see", "sees", "seen",
    "watched", "watch", "watches",
    "looked", "look", "looks",
    "heard", "hear", "hears",
    "felt", "feel", "feels",
    "smelled", "smell", "smells", "smelt",
    "tasted", "taste", "tastes",
    "noticed", "notice", "notices",
    "glimpsed", "glimpse",
    "spotted", "spot",
    "sensed", "sense",
    "observed", "observe",
}

def sensory_neighbor_word_indices(word_spans: List[Tuple[int, int]],
                                   text: str,
                                   sensory_range: int = 3) -> Set[int]:
    """
    Return word indices that are within sensory_range positions of a sensory verb.
    These words get their mask probability capped at base rate to avoid
    cross-modal substitutions (e.g. "color" → "sound" near "eyes").
    """
    neighbors: Set[int] = set()
    for i, (ws, we) in enumerate(word_spans):
        word = text[ws:we].lower().rstrip(".,;:!?\"'")
        if word in _SENSORY_VERBS:
            lo = max(0, i - sensory_range)
            hi = min(len(word_spans), i + sensory_range + 1)
            for j in range(lo, hi):
                if j != i:
                    neighbors.add(j)
    return neighbors


# ---------------------------------------------------------------------------
# Span utilities
# ---------------------------------------------------------------------------

def find_entity_char_spans(text: str, entities: List[Dict]) -> List[Tuple[int, int]]:
    """
    Return merged character (start, end) spans for all entity occurrences
    (full names AND title-cased sub-spans).

    NOTE: Structural spans are handled separately in mask_text — they are
    protected from masking but do NOT radiate masking probability outward.
    """
    protected = expand_protected_strings(entities)
    spans = []
    for name in protected:
        for m in re.finditer(re.escape(name), text, re.IGNORECASE):
            spans.append((m.start(), m.end()))
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


# ---------------------------------------------------------------------------
# Fix 4 – Post-fill validation
# ---------------------------------------------------------------------------

def validate_augmented(orig_text: str, aug_text: str,
                        protected_strings: List[str]) -> Tuple[bool, str]:
    """
    Hard check: every protected string that appeared in orig_text must still
    appear in aug_text.  Returns (ok, reason).
    """
    for s in protected_strings:
        if re.search(re.escape(s), orig_text, re.IGNORECASE):
            if not re.search(re.escape(s), aug_text, re.IGNORECASE):
                return False, f"lost protected string: '{s}'"
    return True, ""


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

    Protection layers applied in order:
      1. Entity spans (full + sub-spans via find_entity_char_spans)
      2. Structural spans (headers, scene labels — inside find_entity_char_spans)
      3. Articles directly preceding an entity span
      4. Words within 3 positions of a sensory verb (capped at base rate)
    """
    entity_spans     = find_entity_char_spans(text, entities)
    structural_spans = find_structural_char_spans(text)
    word_spans       = get_word_spans(text)
    if not word_spans:
        return text, 0

    distances, in_entity = word_distance_to_entity(word_spans, entity_spans)
    probs = mask_probabilities(distances, in_entity, window, peak, base)

    # Fix 2: zero out structural tokens (headers, scene labels) independently
    # of the proximity gradient — they're protected but don't radiate outward.
    for i, (ws, we) in enumerate(word_spans):
        if probs[i] == 0.0:
            continue
        for ss, se in structural_spans:
            if ws < se and we > ss:   # word overlaps structural span
                probs[i] = 0.0
                break

    # Fix 3: cap masking probability near sensory verbs to background rate
    sensory_nbrs = sensory_neighbor_word_indices(word_spans, text)
    for i in sensory_nbrs:
        if probs[i] > base:
            probs[i] = base

    # Protect articles at distance=1 that immediately precede an entity span.
    # Strip both leading AND trailing punctuation so '"The' / '"a' are caught.
    _STRIP_PUNCT = str.maketrans("", "", '\"\'"\u2018\u2019\u201c\u201d.,!?;:()-')
    for i, (ws, we) in enumerate(word_spans):
        if (not in_entity[i]
                and distances[i] == 1.0
                and text[ws:we].lower().translate(_STRIP_PUNCT) in _ARTICLES):
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


def _split_into_chunks(text: str, tokenizer, chunk_size: int) -> List[str]:
    """Split text on sentence boundaries so each chunk fits within chunk_size tokens."""
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
    return chunks if chunks else [text]


def _decode_masks(text: str, logits_seq: torch.Tensor, mask_positions: torch.Tensor,
                  top_k: int, temperature: float,
                  special_ids: Set[int], tokenizer) -> str:
    """Replace <mask> tokens in text using sampled predictions from logits_seq."""
    result = text
    for tok_pos in mask_positions:
        scaled = logits_seq[tok_pos].clone() / max(temperature, 1e-6)
        for sid in special_ids:
            if sid < scaled.shape[0]:
                scaled[sid] = float('-inf')
        top_vals, top_ids = torch.topk(scaled, k=min(top_k, scaled.shape[-1]))
        probs = torch.softmax(top_vals, dim=-1)
        sampled_id = top_ids[torch.multinomial(probs, 1).item()].item()
        word = tokenizer.decode([sampled_id]).strip()
        idx = result.find(MASK_TOKEN)
        if idx == -1:
            break
        result = result[:idx] + (word if word else "") + result[idx + len(MASK_TOKEN):]
    return result


def _fill_texts_batched(texts: List[str], model, tokenizer,
                         top_k: int, temperature: float, device,
                         batch_size: int = 64) -> List[str]:
    """
    Fill <mask> tokens for a list of texts using batched GPU inference.
    All texts are padded to the longest in each mini-batch — one forward pass
    per batch_size texts instead of one per text.
    Returns filled texts in the same order as input.
    """
    special_ids: Set[int] = set(tokenizer.all_special_ids)
    mask_id = tokenizer.mask_token_id
    results = list(texts)

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start: batch_start + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt",
            truncation=True, max_length=512, padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits  # [B, seq_len, vocab]

        for bi, text in enumerate(batch):
            mask_positions = (inputs.input_ids[bi] == mask_id).nonzero(as_tuple=True)[0]
            if len(mask_positions) == 0:
                continue
            results[batch_start + bi] = _decode_masks(
                text, logits[bi], mask_positions,
                top_k, temperature, special_ids, tokenizer,
            )

    return results


# Legacy single-text wrappers — used by eval harness and augment_story
def _fill_chunk(text: str, model, tokenizer,
                top_k: int, temperature: float, device) -> str:
    return _fill_texts_batched([text], model, tokenizer,
                                top_k, temperature, device, batch_size=1)[0]


def _fill_in_chunks(text: str, model, tokenizer,
                     top_k: int, temperature: float, device,
                     chunk_size: int) -> str:
    chunks = _split_into_chunks(text, tokenizer, chunk_size)
    filled = _fill_texts_batched(chunks, model, tokenizer,
                                  top_k, temperature, device, batch_size=len(chunks))
    return " ".join(filled)


# ---------------------------------------------------------------------------
# Story augmentation
# ---------------------------------------------------------------------------

def augment_story(story: Dict[str, Any], model, tokenizer,
                  window: int, peak: float, base: float,
                  top_k: int, temperature: float,
                  n_copies: int, device, rng: random.Random,
                  max_retries: int = 2) -> List[Dict]:
    """
    Produce n_copies augmented versions of a story.
    Returns list of augmented story dicts with same entities, new text.

    Fix 4: any copy that fails post-fill validation (drops a protected entity
    string) is retried up to max_retries times, then falls back to original.
    """
    text      = story.get("text", "")
    entities  = story.get("entities", [])
    char_ents = [{"text": c["name"]} for c in story.get("characters", []) if c.get("name")]
    all_ents  = entities + char_ents
    protected = expand_protected_strings(all_ents)

    results = []
    for _ in range(n_copies):
        aug_text = None
        n_masks  = 0
        for attempt in range(max_retries + 1):
            masked, n_masks = mask_text(text, all_ents, window, peak, base, rng)
            if n_masks == 0:
                aug_text = text
                break
            filled = fill_masks(masked, text, model, tokenizer, top_k, temperature, device)
            ok, reason = validate_augmented(text, filled, protected)
            if ok:
                aug_text = filled
                break
            # validation failed — retry with fresh mask sampling

        if aug_text is None or aug_text == text:
            results.append(story)
            continue

        aug = dict(story)
        aug["text"] = aug_text
        aug["augmentation"] = {"mlm_masks": n_masks, "window": window}
        results.append(aug)

    return results


def augment_stories_batched(
    stories: List[Dict[str, Any]],
    model, tokenizer,
    window: int, peak: float, base: float,
    top_k: int, temperature: float,
    n_copies: int, device, rng: random.Random,
    gpu_batch_size: int = 64,
    chunk_size: int = 480,
    max_retries: int = 2,
) -> List[List[Dict[str, Any]]]:
    """
    Process a list of stories in one batched GPU pass per gpu_batch_size texts.

    Returns a list (one per input story) of lists of augmented story dicts.
    The masking step (CPU) runs for all stories first, then all masked texts
    are pushed through the model together — maximising GPU utilisation.
    """
    # ---- Phase 1: CPU masking for every story × copy ----
    # job_meta[i] = (story_idx, copy_idx, fallback_map, protected, n_masks, original_text)
    # flat_texts[i] = masked text for that job
    job_meta: List[tuple] = []
    flat_texts: List[str] = []
    # also track which jobs need chunking: chunk_map[(si,ci)] = (start_in_flat, n_chunks)
    chunk_map: Dict[tuple, tuple] = {}

    for si, story in enumerate(stories):
        text     = story.get("text", "")
        entities = story.get("entities", [])
        char_ents = [{"text": c["name"]} for c in story.get("characters", []) if c.get("name")]
        all_ents  = entities + char_ents
        protected = expand_protected_strings(all_ents)

        for ci in range(n_copies):
            masked, n_masks = mask_text(text, all_ents, window, peak, base, rng)
            fallback = _build_fallback_map(masked, text)
            if n_masks == 0:
                # No masks: record as identity, no GPU work needed
                job_meta.append((si, ci, fallback, protected, 0, text))
                flat_texts.append(masked)   # will not be batched (n_masks==0)
                chunk_map[(si, ci)] = (len(flat_texts) - 1, 1)
            else:
                toks = tokenizer.encode(masked, add_special_tokens=False)
                if len(toks) <= chunk_size:
                    start = len(flat_texts)
                    flat_texts.append(masked)
                    chunk_map[(si, ci)] = (start, 1)
                    job_meta.append((si, ci, fallback, protected, n_masks, text))
                else:
                    chunks = _split_into_chunks(masked, tokenizer, chunk_size)
                    start = len(flat_texts)
                    flat_texts.extend(chunks)
                    chunk_map[(si, ci)] = (start, len(chunks))
                    job_meta.append((si, ci, fallback, protected, n_masks, text))

    # ---- Phase 2: Batch GPU fill (skip identity texts) ----
    need_fill = [i for i, jm in enumerate(job_meta) if jm[4] > 0]
    # Build a flat list of texts that actually need filling
    fill_slots: List[int] = []   # index into flat_texts
    for job_i in need_fill:
        si, ci, _, _, _, _ = job_meta[job_i]
        start, n = chunk_map[(si, ci)]
        fill_slots.extend(range(start, start + n))

    if fill_slots:
        texts_to_fill = [flat_texts[i] for i in fill_slots]
        filled = _fill_texts_batched(
            texts_to_fill, model, tokenizer, top_k, temperature, device, gpu_batch_size
        )
        for slot_idx, flat_idx in enumerate(fill_slots):
            flat_texts[flat_idx] = filled[slot_idx]

    # ---- Phase 3: Reassemble, apply fallbacks, validate ----
    # per_story_results[(si, ci)] = filled_text or None
    per_job_filled: Dict[tuple, str] = {}
    for si, ci, fallback, protected, n_masks, orig_text in job_meta:
        start, n = chunk_map[(si, ci)]
        chunks = flat_texts[start: start + n]
        filled_text = " ".join(chunks) if n > 1 else chunks[0]
        filled_text = _apply_fallbacks(filled_text, fallback)
        ok, _ = validate_augmented(orig_text, filled_text, protected)
        per_job_filled[(si, ci)] = filled_text if ok else None

    # ---- Phase 4: Handle validation failures with single-story retry ----
    retry_keys = [(si, ci) for (si, ci), v in per_job_filled.items() if v is None]
    for si, ci in retry_keys:
        story = stories[si]
        text  = story.get("text", "")
        entities = story.get("entities", [])
        char_ents = [{"text": c["name"]} for c in story.get("characters", []) if c.get("name")]
        all_ents  = entities + char_ents
        protected = expand_protected_strings(all_ents)
        for _ in range(max_retries):
            masked, n_masks = mask_text(text, all_ents, window, peak, base, rng)
            if n_masks == 0:
                per_job_filled[(si, ci)] = text
                break
            filled = fill_masks(masked, text, model, tokenizer, top_k, temperature, device)
            ok, _ = validate_augmented(text, filled, protected)
            if ok:
                per_job_filled[(si, ci)] = filled
                break

    # ---- Phase 5: Build output ----
    results: List[List[Dict]] = []
    for si, story in enumerate(stories):
        story_results = []
        for ci in range(n_copies):
            filled_text = per_job_filled.get((si, ci))
            if not filled_text or filled_text == story.get("text", ""):
                story_results.append(story)
                continue
            _, _, _, _, n_masks, _ = next(
                jm for jm in job_meta if jm[0] == si and jm[1] == ci
            )
            aug = dict(story)
            aug["text"] = filled_text
            aug["augmentation"] = {"mlm_masks": n_masks, "window": window}
            story_results.append(aug)
        results.append(story_results)
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
    parser.add_argument("--batch-size",  type=int,   default=64,
                        help="Stories per GPU batch (higher = more GPU utilisation, default: 64)")
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
          f"top_k={args.top_k}  temp={args.temperature}  copies={args.copies}  "
          f"batch_size={args.batch_size}")
    print()

    story_buffer: List[Dict] = []

    def flush(fout):
        nonlocal total_in, total_out, total_masks
        if not story_buffer:
            return
        batch_results = augment_stories_batched(
            story_buffer, model, tokenizer,
            args.window, args.peak_prob, args.base_prob,
            args.top_k, args.temperature,
            args.copies, device, rng,
            gpu_batch_size=args.batch_size,
        )
        for story, story_aug_list in zip(story_buffer, batch_results):
            if args.include_original:
                fout.write(json.dumps(story, ensure_ascii=False) + "\n")
                total_out += 1
            for aug in story_aug_list:
                fout.write(json.dumps(aug, ensure_ascii=False) + "\n")
                total_out += 1
                total_masks += aug.get("augmentation", {}).get("mlm_masks", 0)
        total_in += len(story_buffer)
        story_buffer.clear()

    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            if args.max_stories and total_in + len(story_buffer) >= args.max_stories:
                break
            line = line.strip()
            if not line:
                continue
            try:
                story = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            story_buffer.append(story)

            if len(story_buffer) >= args.batch_size:
                flush(fout)
                avg_masks = total_masks / max(total_out, 1)
                print(f"  {total_in:,} stories → {total_out:,} docs  "
                      f"(avg {avg_masks:.1f} masks/story)", end="\r", flush=True)

        flush(fout)  # final partial batch

    print(f"\n✅ Done: {total_in:,} input → {total_out:,} output docs")
    print(f"   Total masks filled: {total_masks:,}  (avg {total_masks/max(total_out,1):.1f}/story)")
    if skipped:
        print(f"   Skipped (parse errors): {skipped}")


if __name__ == "__main__":
    main()
