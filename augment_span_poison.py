#!/usr/bin/env python3
"""
augment_span_poison.py — Span-poisoning augmentation for story NER data.

For each story (or N copies of each story), every entity name is replaced
with an invented string that:
  - is NOT a real name (forcing the model to learn from context, not memory)
  - matches the word-count of the original
  - is consistent within a story (same entity → same replacement every time)
  - is separate from MLM augmentation (do not combine outputs)

The two augmentation streams are kept separate:
  - augment_mlm_context.py  : same entities, varied surrounding n-grams
  - augment_span_poison.py  : same context n-grams, swapped entity strings

Usage:
    python3 augment_span_poison.py \\
        --input  ner_training_ultra_short_agent_training_agent_20260406_193605.jsonl \\
        --output augmented_span_poison.jsonl \\
        --copies 2 \\
        --include-original \\
        --seed 42
"""

import json
import re
import random
import argparse
import sys
from typing import List, Dict, Optional, Tuple
from pathlib import Path


# ---------------------------------------------------------------------------
# Name generation pools
# ---------------------------------------------------------------------------
# Goal: invented names that are pronounceable but clearly not real words.
# PERSON: first + last syllable mashups
# LOCATION: invented place-words ± geographic suffix
# AGENT: "The" (optional) + strange adj + strange noun  (org/faction feel)
# When absurdity is wanted, AGENT can also draw from ABSURD_COMBOS.

_PERSON_FIRSTS = [
    "Vel", "Tor", "Kael", "Morn", "Zeth", "Dray", "Slov", "Wren",
    "Fynd", "Grex", "Ael", "Brix", "Ceth", "Dhov", "Elv", "Frel",
    "Gash", "Holk", "Ivel", "Jurk", "Kelb", "Lorn", "Mrex", "Nolv",
    "Osk", "Prel", "Rask", "Salv", "Trev", "Vrex", "Xael", "Zorv",
    "Bael", "Crisp", "Drolt", "Sphen", "Threk", "Ulven", "Wolk", "Yolm",
    "Qelda", "Tavish", "Plorn", "Snev", "Truvax", "Hulken", "Gavrix", "Norv",
    "Delvak", "Selvin", "Fersk", "Molvey", "Ondrak", "Plaven", "Quilv", "Reveth",
]

_PERSON_LASTS = [
    "ax", "orn", "eld", "rix", "usk", "eth", "and", "ith", "ock",
    "ral", "ash", "olt", "erx", "aen", "uck", "iven", "olm", "avin",
    "reen", "urst", "avix", "orsk", "aven", "elth", "enth", "avel",
    "orvex", "eldun", "askren", "olvey", "elven", "axton", "orven", "uldrek",
    "emvax", "isfren", "orvath", "undrel", "elwick", "oshkeld", "arvath", "imvel",
]

_LOC_ROOTS = [
    "Brelvox", "Skarven", "Droveth", "Molgrin", "Wexhold", "Freldmoor",
    "Sholveth", "Torcald", "Bluvmire", "Aekford", "Grelthorn", "Neskvale",
    "Voldwick", "Urkmere", "Threlwick", "Dolbrace", "Skevmere", "Croltford",
    "Halvbreck", "Grimwald", "Spelwick", "Turkmere", "Woldhaven", "Dreskport",
    "Bolveth", "Snarvack", "Theldwick", "Ormspire", "Grevmire", "Croldhaven",
    "Skelmoor", "Vreldgate", "Ushkborn", "Dolvemere", "Threskwick", "Blornhold",
    "Quevmere", "Joldwick", "Helvmoor", "Snelford", "Bladvale", "Frelwick",
    "Groltmere", "Wurmhaven", "Thelkwick", "Veldrox", "Neskholm", "Drevspire",
]

_LOC_SUFFIXES = [
    "", "", "", "", "",  # bare root is common
    "mire", "vale", "hold", "ford", "mere", "wick", "ton",
    "ridge", "dell", "hollow", "pass", "croft", "haven", "shore",
    "fell", "moor", "gate", "spire", "holm",
]

_LOC_MODIFIERS = [
    "Dark", "Salt", "Grey", "Ash", "Black", "Iron", "Old", "Deep",
    "Outer", "Upper", "Lower", "Far", "Crook", "Lost", "Bitter",
    "Pale", "Hollow", "Broken", "Sunken", "Crooked", "Grim", "Murky",
]

_LOC_TYPES = [
    "Reach", "Crossing", "Flats", "Narrows", "Heights", "Depths",
    "Shore", "Ridge", "Dell", "Warren", "Expanse", "Wastes", "Fringe",
    "Quarter", "Strand", "Basin", "Tract",
]

_AGENT_ADJS = [
    "Hollow", "Ashen", "Broken", "Crooked", "Grim", "Pale", "Fetid",
    "Blunt", "Scoured", "Molten", "Slack", "Whetted", "Knotted", "Gaunt",
    "Shorn", "Veiled", "Sunken", "Bound", "Salted", "Hooked", "Braided",
    "Mended", "Halved", "Spent", "Drawn", "Folded", "Rusted", "Cleaved",
    "Notched", "Withered", "Burnished", "Flensed", "Pitted", "Wrung", "Staved",
]

_AGENT_NOUNS = [
    "Ledger", "Tallow", "Anvil", "Flue", "Gusset", "Tine", "Gimbal",
    "Kelp", "Spigot", "Welt", "Cinder", "Mallet", "Clasp", "Bolus",
    "Ferrule", "Swivel", "Plinth", "Quoin", "Spall", "Gambrel", "Trowel",
    "Gudgeon", "Strake", "Lintel", "Corbel", "Knurl", "Froe", "Collet",
    "Drawplate", "Sprue", "Mandrel", "Rabbet", "Scarf", "Tenon", "Hasp",
]

_AGENT_PREFIXES = [
    "The", "The", "The", "The", "Order of the", "Brotherhood of the",
    "Council of the", "Lodge of the", "Circle of the", "Sect of the",
    "Hand of the", "Eye of the",
]


# ---------------------------------------------------------------------------
# Name generation
# ---------------------------------------------------------------------------

def _gen_person(rng: random.Random, n_words: int) -> str:
    first = rng.choice(_PERSON_FIRSTS) + rng.choice(_PERSON_LASTS)
    first = first.capitalize()
    if n_words == 1:
        return first
    last = rng.choice(_PERSON_FIRSTS) + rng.choice(_PERSON_LASTS)
    last = last.capitalize()
    if n_words == 2:
        return f"{first} {last}"
    # 3-word: add a middle initial or an invented middle name
    mid = rng.choice(_PERSON_FIRSTS).capitalize()
    return f"{first} {mid} {last}"


def _gen_location(rng: random.Random, n_words: int) -> str:
    root = rng.choice(_LOC_ROOTS)
    suffix = rng.choice(_LOC_SUFFIXES)
    base = root + suffix if suffix and not root.endswith(suffix[:3]) else root
    base = base.title()

    if n_words == 1:
        return base
    if n_words == 2:
        if rng.random() < 0.5:
            return f"{rng.choice(_LOC_MODIFIERS)} {base}"
        else:
            return f"{base} {rng.choice(_LOC_TYPES)}"
    if n_words == 3:
        return f"{rng.choice(_LOC_MODIFIERS)} {base} {rng.choice(_LOC_TYPES)}"
    if n_words == 4:
        r2 = rng.choice(_LOC_ROOTS).title()
        return f"The {rng.choice(_LOC_MODIFIERS)} {base} {rng.choice(_LOC_TYPES)}"
    # 5+
    parts = [rng.choice(_LOC_MODIFIERS), base, rng.choice(_LOC_TYPES),
             rng.choice(_LOC_MODIFIERS).lower(), rng.choice(_LOC_ROOTS).lower()]
    return " ".join(parts[:n_words]).title()


def _gen_agent(rng: random.Random, n_words: int) -> str:
    adj  = rng.choice(_AGENT_ADJS)
    noun = rng.choice(_AGENT_NOUNS)

    if n_words == 1:
        # Single-word faction: just the noun (e.g. "Spigot")
        return noun
    if n_words == 2:
        return f"{adj} {noun}"
    if n_words == 3:
        prefix = rng.choice(["The", "The", "Order of"])
        if prefix == "Order of":
            return f"Order of {noun}"
        return f"The {adj} {noun}"
    if n_words == 4:
        adj2 = rng.choice(_AGENT_ADJS)
        return f"The {adj} {noun} {adj2}"
    # 5+: e.g. "The Brotherhood of the Hollow Ledger"
    return f"The {rng.choice(['Brotherhood', 'Order', 'Council', 'Lodge'])} of the {adj} {noun}"


def gen_name(entity_type: str, original: str, rng: random.Random) -> str:
    """Generate an invented replacement name matching the word count of original."""
    n_words = len(original.split())
    etype = entity_type.upper()
    if etype == "PERSON":
        return _gen_person(rng, n_words)
    if etype == "LOCATION":
        return _gen_location(rng, n_words)
    if etype == "AGENT":
        return _gen_agent(rng, n_words)
    # Fallback
    return _gen_person(rng, n_words)


# ---------------------------------------------------------------------------
# Text replacement
# ---------------------------------------------------------------------------

def _replace_all(text: str, orig: str, repl: str) -> str:
    """
    Replace all occurrences of orig with repl in text (case-insensitive).
    Handles possessives (Orig's → Repl's).
    Preserves the replacement in the canonical capitalisation of repl.
    """
    escaped = re.escape(orig)
    # Possessives first: "Name's" → "Repl's"
    text = re.sub(escaped + r"'s\b", repl + "'s", text, flags=re.IGNORECASE)
    text = re.sub(escaped + r"\b", repl, text, flags=re.IGNORECASE)
    return text


def poison_story(story: Dict, rng: random.Random) -> Dict:
    """
    Return a new story dict with all entity names swapped to invented strings.
    Entity metadata is updated to match new names.

    Also handles common partial references:
    - PERSON "First Last" → also replaces "First" with first word of new name
    - AGENT "The X Y" → also replaces "X Y" (article-stripped form)
    """
    entities = story.get("entities", [])
    text = story["text"]

    # Build old→new mapping.  Sort by length descending so longer multi-word
    # entities are replaced before any shared sub-strings accidentally get hit.
    mapping: Dict[str, str] = {}
    for ent in entities:
        orig = ent.get("text", "").strip()
        if not orig or orig in mapping:
            continue
        new_name = gen_name(ent.get("type", "PERSON"), orig, rng)
        # Ensure uniqueness within this story
        existing_new = set(mapping.values())
        attempts = 0
        while new_name in existing_new and attempts < 20:
            new_name = gen_name(ent.get("type", "PERSON"), orig, rng)
            attempts += 1
        mapping[orig] = new_name

    # Build alias mappings for common partial references
    alias_mapping: Dict[str, str] = {}
    for orig, new in mapping.items():
        ent_type = next(
            (e.get("type", "") for e in entities if e.get("text", "").strip() == orig),
            ""
        )
        orig_words = orig.split()
        new_words = new.split()

        if ent_type == "PERSON" and len(orig_words) >= 2:
            first = orig_words[0]
            # Only alias first names that are ≥4 chars and look like proper names
            if len(first) >= 4 and first[0].isupper() and first not in mapping:
                new_first = new_words[0] if new_words else new
                alias_mapping[first] = new_first

        elif ent_type == "AGENT":
            # "The X Y" → also replace "X Y" without "The"
            if orig_words[0].lower() == "the" and len(orig_words) > 1:
                stripped = " ".join(orig_words[1:])
                if stripped not in mapping and stripped not in alias_mapping:
                    new_stripped = " ".join(new_words[1:]) if new_words[0].lower() == "the" else new
                    alias_mapping[stripped] = new_stripped

    # Apply replacements: full names first (longest first), then aliases
    all_replacements = sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)
    all_replacements += sorted(alias_mapping.items(), key=lambda x: len(x[0]), reverse=True)

    for orig, repl in all_replacements:
        text = _replace_all(text, orig, repl)

    # Update entity metadata
    new_entities = []
    for ent in entities:
        orig = ent.get("text", "").strip()
        new_ent = dict(ent)
        if orig in mapping:
            new_ent["text"] = mapping[orig]
            new_ent["_original_text"] = orig
        new_entities.append(new_ent)

    new_story = dict(story)
    new_story["text"] = text
    new_story["entities"] = new_entities
    new_story["augmentation"] = {
        "method": "span_poison",
        "name_mapping": mapping,
        "alias_mapping": alias_mapping,
    }
    return new_story


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",  required=True, help="Source JSONL file")
    ap.add_argument("--output", required=True, help="Output JSONL file")
    ap.add_argument("--copies", type=int, default=2,
                    help="Poisoned copies per story (default: 2)")
    ap.add_argument("--include-original", action="store_true",
                    help="Also write the original story to output")
    ap.add_argument("--max-stories", type=int, default=None,
                    help="Cap for testing")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    in_path  = Path(args.input)
    out_path = Path(args.output)

    written = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for i, line in enumerate(fin):
            if args.max_stories and i >= args.max_stories:
                break
            story = json.loads(line)

            if args.include_original:
                fout.write(json.dumps(story, ensure_ascii=False) + "\n")
                written += 1

            for _ in range(args.copies):
                poisoned = poison_story(story, rng)
                fout.write(json.dumps(poisoned, ensure_ascii=False) + "\n")
                written += 1

            if (i + 1) % 1000 == 0:
                print(f"  processed {i+1} stories → {written} docs written",
                      file=sys.stderr)

    print(f"Done. {written} documents written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
