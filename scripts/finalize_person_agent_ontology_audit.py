#!/usr/bin/env python3
"""Freeze blinded decisions, reveal labels, and publish ontology audit reports."""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("person_agent_ontology_audit_20260712")
OUT = ROOT / "review_outputs"
SPECIAL = {"AMBIGUOUS_CONTEXT", "BAD_BOUNDARY", "INSUFFICIENT_CONTEXT"}
SEMANTIC = {"PERSON", "AGENT", "NEITHER"}
SEED = 20260715


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


def bootstrap_rate(values: list[int], iterations: int = 10000) -> dict:
    if not values:
        return {"n": 0, "rate": None, "ci95": [None, None]}
    n = len(values)
    point = sum(values) / n
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (point + z * z / (2 * n)) / denominator
    half = z * math.sqrt(point * (1 - point) / n + z * z / (4 * n * n)) / denominator
    return {
        "n": n,
        "rate": point,
        "ci95": [max(0.0, center - half), min(1.0, center + half)],
        "interval_method": "Wilson score",
    }


def weighted_synthetic_bootstrap(rows: list[dict], weights: dict[str, int], iterations: int = 10000) -> dict:
    by_subtype = defaultdict(list)
    for row in rows:
        if row["current_label"] == "AGENT" and row.get("source_agent_subtype") in weights:
            by_subtype[row["source_agent_subtype"]].append(int(row["final_decision"] != "AGENT"))
    total_weight = sum(weights.values())
    point = sum(weights[k] * (sum(v) / len(v)) for k, v in by_subtype.items()) / total_weight
    rng = random.Random(SEED)
    estimates = []
    for _ in range(iterations):
        estimate = 0.0
        for subtype, values in by_subtype.items():
            # Jeffreys-binomial simulation preserves uncertainty for all-zero/all-one strata.
            sampled = rng.betavariate(sum(values) + 0.5, len(values) - sum(values) + 0.5)
            estimate += weights[subtype] * sampled
        estimates.append(estimate / total_weight)
    estimates.sort()
    return {
        "rate": point,
        "ci95": [estimates[int(.025 * iterations)], estimates[int(.975 * iterations) - 1]],
        "population_mentions": total_weight,
        "affected_mention_estimate": round(point * total_weight),
        "method": "declared-subtype-weighted Jeffreys-binomial simulation",
    }


def kappa(left: list[str], right: list[str]) -> float | None:
    if not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    labels = set(left) | set(right)
    expected = sum((left.count(x) / len(left)) * (right.count(x) / len(right)) for x in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def rate_table(rows: list[dict]) -> dict:
    usable = [r for r in rows if r["final_decision"] in SEMANTIC]
    transitions = Counter(f"{r['current_label']}->{r['final_decision']}" for r in usable)
    by_current = {}
    for current in ("PERSON", "AGENT"):
        subset = [r for r in usable if r["current_label"] == current]
        by_current[current] = {
            "n": len(subset),
            "transitions": dict(Counter(r["final_decision"] for r in subset)),
            "error_rate": bootstrap_rate([int(r["final_decision"] != current) for r in subset]),
        }
    return {
        "sample_records": len(rows),
        "semantic_records": len(usable),
        "transitions": dict(transitions),
        "by_current_label": by_current,
        "ambiguous_or_insufficient": sum(r["final_decision"] in {"AMBIGUOUS_CONTEXT", "INSUFFICIENT_CONTEXT"} for r in rows),
        "boundary_errors": sum(r["final_decision"] == "BAD_BOUNDARY" or r["final_review"]["boundary_action"] != "NONE" for r in rows),
    }


def main() -> None:
    candidates = {}
    corpus_for = {}
    for corpus in ("synthetic", "pd1000"):
        for row in read_jsonl(ROOT / f"{corpus}_manual_audit_candidates.jsonl"):
            candidates[row["candidate_id"]] = row
            corpus_for[row["candidate_id"]] = corpus

    reviews = defaultdict(list)
    primary = {}
    overlap_secondary = {}
    for reviewer in ("reviewer_a", "reviewer_b", "reviewer_c"):
        for row in read_jsonl(OUT / f"{reviewer}.jsonl"):
            item = {"reviewer": reviewer, **row}
            reviews[row["record_id"]].append(item)
            if row["review_role"] == "PRIMARY":
                primary[row["record_id"]] = item
            else:
                overlap_secondary[row["record_id"]] = item
    followup = {r["record_id"]: {"reviewer": "followup_reviewer_d", **r} for r in read_jsonl(OUT / "followup_reviewer_d.jsonl")}
    expanded = {r["record_id"]: {"reviewer": "expanded_reviewer_e", **r} for r in read_jsonl(OUT / "expanded_reviewer_e.jsonl")}
    final_tie = {r["record_id"]: {"reviewer": "final_reviewer_h", **r} for r in read_jsonl(OUT / "final_reviewer_h.jsonl")}
    second_ids = {r["record_id"] for r in read_jsonl(ROOT / "followup_packets" / "second_review.jsonl")}
    tie_ids = {r["record_id"] for r in read_jsonl(ROOT / "followup_packets" / "tie_break.jsonl")}
    initial_expanded_ids = {r["record_id"] for r in read_jsonl(ROOT / "followup_packets" / "expanded_context_needed.jsonl")}

    frozen = []
    for record_id, candidate in candidates.items():
        p = primary[record_id]
        s = overlap_secondary.get(record_id)
        route = "PRIMARY_HIGH_CONFIDENCE"
        chosen = p
        if record_id in initial_expanded_ids:
            route, chosen = "EXPANDED_CONTEXT", expanded[record_id]
        elif record_id in tie_ids:
            route, chosen = "OVERLAP_TIEBREAK", followup[record_id]
        elif record_id in second_ids:
            f = followup[record_id]
            if p["decision"] == f["decision"]:
                route, chosen = "SECOND_REVIEW_CONSENSUS", f
            elif record_id in final_tie:
                route, chosen = "SECOND_REVIEW_FINAL_ADJUDICATION", final_tie[record_id]
            else:
                raise ValueError(f"missing final adjudication for {record_id}")
        elif s:
            route = "OVERLAP_CONSENSUS"
            chosen = p
        frozen.append({
            "record_id": record_id,
            "surface": candidate["surface"],
            "local_context": candidate["local_context"],
            "corpus_source": candidate["corpus_source"],
            "book_id": candidate.get("book_id"),
            "scene_id": candidate.get("scene_id"),
            "current_label": candidate["current_label"],
            "source_agent_subtype": candidate.get("source_agent_subtype"),
            "candidate_categories": candidate.get("candidate_categories", []),
            "final_decision": chosen["decision"],
            "final_confidence": chosen["confidence"],
            "final_review": chosen,
            "route": route,
            "review_chain": reviews[record_id] + ([followup[record_id]] if record_id in followup else []) + ([expanded[record_id]] if record_id in expanded else []) + ([final_tie[record_id]] if record_id in final_tie else []),
        })
    assert len(frozen) == 600
    synthetic = [r for r in frozen if corpus_for[r["record_id"]] == "synthetic"]
    pd1000 = [r for r in frozen if corpus_for[r["record_id"]] == "pd1000"]
    overlap = [r for r in frozen if r["record_id"] in overlap_secondary]
    write_jsonl(ROOT / "adjudicated_synthetic.jsonl", synthetic)
    write_jsonl(ROOT / "adjudicated_pd1000.jsonl", pd1000)
    write_jsonl(ROOT / "adjudicated_overlap.jsonl", overlap)

    # Agreement is measured before any sealed-label comparison influences decisions.
    agreement_rows = []
    for row in overlap:
        p = primary[row["record_id"]]
        s = overlap_secondary[row["record_id"]]
        agreement_rows.append((row, p, s))
    left = [p["decision"] for _, p, _ in agreement_rows]
    right = [s["decision"] for _, _, s in agreement_rows]
    agreement_by_corpus = {}
    for corpus in ("synthetic", "pd1000"):
        subset = [(p, s) for row, p, s in agreement_rows if corpus_for[row["record_id"]] == corpus]
        agreement_by_corpus[corpus] = {
            "n": len(subset),
            "raw_agreement": sum(p["decision"] == s["decision"] for p, s in subset) / len(subset),
            "cohen_kappa": kappa([p["decision"] for p, _ in subset], [s["decision"] for _, s in subset]),
        }
    subtype_agreement = {}
    for subtype in sorted({r.get("source_agent_subtype") for r, _, _ in agreement_rows if r.get("source_agent_subtype")}):
        subset = [(p, s) for row, p, s in agreement_rows if row.get("source_agent_subtype") == subtype]
        subtype_agreement[subtype] = {
            "n": len(subset),
            "raw_agreement": sum(p["decision"] == s["decision"] for p, s in subset) / len(subset),
            "cohen_kappa": kappa([p["decision"] for p, _ in subset], [s["decision"] for _, s in subset]),
        }
    agreement_report = {
        "overlap_records": len(overlap),
        "overlap_fraction": len(overlap) / len(frozen),
        "raw_agreement": sum(a == b for a, b in zip(left, right)) / len(left),
        "cohen_kappa": kappa(left, right),
        "agreement_by_corpus": agreement_by_corpus,
        "agreement_by_declared_synthetic_subtype": subtype_agreement,
        "disagreement_rate": sum(a != b for a, b in zip(left, right)) / len(left),
        "low_confidence_rate_all_initial_reviews": sum(r["confidence"] < .80 for values in reviews.values() for r in values) / sum(len(v) for v in reviews.values()),
        "tie_break_rate": (len(tie_ids) + len(final_tie)) / len(frozen),
        "expanded_context_rate": (len(initial_expanded_ids) + sum(r["review_role"] == "EXPANDED_CONTEXT" for r in final_tie.values())) / len(frozen),
        "boundary_error_rate_final": sum(r["final_decision"] == "BAD_BOUNDARY" or r["final_review"]["boundary_action"] != "NONE" for r in frozen) / len(frozen),
        "insufficient_context_rate_final": sum(r["final_decision"] == "INSUFFICIENT_CONTEXT" for r in frozen) / len(frozen),
    }
    (ROOT / "reviewer_agreement_report.json").write_text(json.dumps(agreement_report, indent=2, sort_keys=True) + "\n")

    weights = json.loads((ROOT / "synthetic_original_agent_subtype_weights.json").read_text())["declared_mentions"]
    subtype_rates = {}
    for subtype in sorted(weights):
        rows = [r for r in synthetic if r.get("source_agent_subtype") == subtype]
        subtype_rates[subtype] = {
            "sample_n": len(rows),
            "population_declared_mentions": weights[subtype],
            "final_decisions": dict(Counter(r["final_decision"] for r in rows)),
            "agent_to_person": bootstrap_rate([int(r["final_decision"] == "PERSON") for r in rows]),
            "agent_to_neither": bootstrap_rate([int(r["final_decision"] == "NEITHER") for r in rows]),
            "any_semantic_drift": bootstrap_rate([int(r["final_decision"] in SEMANTIC and r["final_decision"] != "AGENT") for r in rows]),
        }
    synthetic_weighted = weighted_synthetic_bootstrap(synthetic, weights)

    # Unbiased 100-mention PD1000 AGENT calibration supplements the risk-enriched 600-record slice.
    random_primary = {r["record_id"]: r for r in read_jsonl(OUT / "pd1000_random_reviewer_i.jsonl")}
    random_secondary = {r["record_id"]: r for r in read_jsonl(OUT / "pd1000_random_overlap_reviewer_j.jsonl")}
    random_final = {r["record_id"]: r for r in read_jsonl(OUT / "pd1000_random_final_reviewer_k.jsonl")}
    random_reviews = []
    for record_id, review in random_primary.items():
        secondary = random_secondary.get(record_id)
        if secondary and secondary["decision"] != review["decision"]:
            review = random_final[record_id]
        random_reviews.append(review)
    random_left = [random_primary[rid]["decision"] for rid in random_secondary]
    random_right = [random_secondary[rid]["decision"] for rid in random_secondary]
    random_pd = {
        "sample_n": len(random_reviews),
        "population_gold_agent_mentions": json.loads((ROOT / "pd1000_population_counts.json").read_text())["gold_agent_mentions"],
        "decisions": dict(Counter(r["decision"] for r in random_reviews)),
        "agent_to_person": bootstrap_rate([int(r["decision"] == "PERSON") for r in random_reviews if r["decision"] in SEMANTIC]),
        "agent_to_neither": bootstrap_rate([int(r["decision"] == "NEITHER") for r in random_reviews if r["decision"] in SEMANTIC]),
        "any_semantic_drift": bootstrap_rate([int(r["decision"] != "AGENT") for r in random_reviews if r["decision"] in SEMANTIC]),
        "overlap_agreement": {
            "n": len(random_secondary),
            "raw_agreement": sum(a == b for a, b in zip(random_left, random_right)) / len(random_left),
            "cohen_kappa": kappa(random_left, random_right),
            "final_tie_breaks": len(random_final),
        },
    }
    subtype_output = {
        "synthetic_declared_subtypes": subtype_rates,
        "synthetic_weighted_agent_drift": synthetic_weighted,
        "pd1000_risk_enriched_slice": rate_table(pd1000),
        "pd1000_random_agent_calibration": random_pd,
    }
    (ROOT / "subtype_error_rates.json").write_text(json.dumps(subtype_output, indent=2, sort_keys=True) + "\n")

    # Development auxiliary adjudication remains separate from model evidence.
    dev_reviews = {}
    for filename in ("development_reviewer_f.jsonl", "development_reviewer_g.jsonl"):
        dev_reviews.update({r["record_id"]: r for r in read_jsonl(OUT / filename)})
    dev_sealed = read_jsonl(ROOT / "development_auxiliary_sealed_lookup.jsonl")
    dev_candidates = {}
    for filename in ("pd1000_eval_ontology_candidates.jsonl", "modern_ontology_candidates.jsonl"):
        dev_candidates.update({r["candidate_id"]: r for r in read_jsonl(ROOT / filename)})
    dev_rows = []
    for sealed in dev_sealed:
        review = dev_reviews[sealed["record_id"]]
        source = dev_candidates[sealed["candidate_id"]]
        dev_rows.append({**sealed, "scene_id": source["scene_id"], "token_start": source.get("token_start"), "token_end": source.get("token_end"), "adjudicated_label": review["decision"], "review": review})
    dev_rates = {corpus: rate_table([{**r, "final_decision": r["adjudicated_label"], "final_review": r["review"]} for r in dev_rows if r["corpus"] == corpus]) for corpus in ("pd1000_eval", "modern")}

    disagreement = read_jsonl(ROOT / "development_model_disagreements.jsonl")
    audited_by_key = {
        (r["corpus"], r["scene_id"], r["token_start"], r["token_end"], r["current_label"]): r
        for r in dev_rows
    }
    reinterpreted = defaultdict(Counter)
    by_confusion = defaultdict(Counter)
    by_subtype = defaultdict(Counter)
    examples = []
    for row in disagreement:
        corpus = "modern" if row["dataset"] == "modern_development" else "pd1000_eval"
        key = (corpus, row["scene_id"], row["gold_start"], row["gold_end"], row["gold_label"])
        audit = audited_by_key.get(key)
        if not audit:
            continue
        new = audit["adjudicated_label"]
        if new in SEMANTIC and new == row["predicted_label"]:
            category = "evaluation_label_drift_explains_model_error"
        elif new in SEMANTIC and row["predicted_label"] == "O":
            category = "model_omission_after_adjudication"
        elif new in SEMANTIC:
            category = "model_type_confusion_after_adjudication"
        else:
            category = "audit_unresolved_or_boundary"
        reinterpreted[(row["model"], corpus)][category] += 1
        confusion_name = f"gold_{row['gold_label']}_predicted_{row['predicted_label']}"
        by_confusion[(row["model"], corpus, confusion_name)][category] += 1
        subtype = audit["review"].get("ontology_subtype", "unknown")
        by_subtype[(row["model"], corpus, subtype)][category] += 1
        if len(examples) < 100:
            examples.append({"model": row["model"], "corpus": corpus, "surface": row["surface"], "old_gold": row["gold_label"], "prediction": row["predicted_label"], "adjudicated": new, "category": category})
    confusion_output = {
        "development_adjudication_rates": dev_rates,
        "sampled_model_confusion_reinterpretation": {f"{m}:{c}": dict(v) for (m, c), v in reinterpreted.items()},
        "by_original_confusion": {f"{m}:{c}:{name}": dict(v) for (m, c, name), v in by_confusion.items()},
        "by_adjudicated_ontology_subtype": {f"{m}:{c}:{subtype}": dict(v) for (m, c, subtype), v in by_subtype.items()},
        "examples": examples,
        "warning": "Reinterpretation covers only audited development mentions present in the disagreement file; full matrices remain model behavior, not corpus-corruption proof.",
        "original_confusion_matrices": json.loads((ROOT / "development_confusion_matrices.json").read_text()),
    }
    (ROOT / "development_confusion_reinterpreted.json").write_text(json.dumps(confusion_output, indent=2, sort_keys=True) + "\n")

    # Conflicting surfaces are revealed only after each context decision and relation are frozen.
    conflict_reviews = {}
    for index in range(1, 4):
        conflict_reviews.update({r["record_id"]: r for r in read_jsonl(OUT / f"conflict_reviewer_{index}.jsonl")})
    conflict_sealed = {r["record_id"]: r for r in read_jsonl(ROOT / "conflicting_surface_sealed_lookup.jsonl")}
    conflict_output = []
    for record_id, review in sorted(conflict_reviews.items()):
        sealed = conflict_sealed[record_id]
        current = {r["context_id"]: r for r in sealed["contexts"]}
        joined = []
        for decision in review["context_decisions"]:
            joined.append({**decision, "current_label": current[decision["context_id"]]["current_label"], "candidate_id": current[decision["context_id"]]["candidate_id"]})
        conflict_output.append({
            "record_id": record_id,
            "corpus": sealed["corpus"],
            "normalized_surface": sealed["normalized_surface"],
            "relation": review["relation"],
            "relation_confidence": review["relation_confidence"],
            "relation_rationale": review["relation_rationale"],
            "contexts": joined,
        })
    write_jsonl(ROOT / "conflicting_surface_analysis.jsonl", conflict_output)

    syn_rates = rate_table(synthetic)
    pd_rates = rate_table(pd1000)
    risk_six = {k: subtype_rates[k] for k in ("artifact", "ai", "beast", "deity", "spirit", "ship")}
    conflict_relations = {corpus: dict(Counter(r["relation"] for r in conflict_output if r["corpus"] == corpus)) for corpus in ("synthetic", "pd1000", "pd1000_eval", "modern")}
    report = f"""# PERSON/AGENT Ontology Drift Audit

## Scope and method

- Frozen, blinded semantic adjudication covered 600 primary records: 300 synthetic and 300 PD1000.
- A 150-record overlap (25%) supplied independent agreement evidence. Low-confidence, disputed, and context-limited cases followed the predeclared second-review, tie-break, or expanded-context routes.
- An additional unbiased 100-record PD1000 AGENT sample supports corpus-level calibration because the mandated 300-record PD1000 slice was deliberately ontology-risk enriched.
- Development estimates use a separate blinded auxiliary sample of 300 PD1000EVAL and 216 Modern mentions. Model predictions were revealed only for subsequent reinterpretation.
- Labels in every source corpus remain unchanged.

## Reviewer reliability

- Raw overlap agreement: {agreement_report['raw_agreement']:.1%}; Cohen's kappa: {agreement_report['cohen_kappa']:.3f}.
- Disagreement rate: {agreement_report['disagreement_rate']:.1%}; tie-break rate: {agreement_report['tie_break_rate']:.1%}; expanded-context rate: {agreement_report['expanded_context_rate']:.1%}.
- Final boundary-error rate: {agreement_report['boundary_error_rate_final']:.1%}; final insufficient-context rate: {agreement_report['insufficient_context_rate_final']:.1%}.

## Synthetic

- The subtype-weighted estimate is {synthetic_weighted['rate']:.1%} affected AGENT mentions (95% stratified uncertainty interval {synthetic_weighted['ci95'][0]:.1%}-{synthetic_weighted['ci95'][1]:.1%}), approximately {synthetic_weighted['affected_mention_estimate']:,} of {synthetic_weighted['population_mentions']:,} declared AGENT mentions.
- AGENT->PERSON and AGENT->NEITHER are reported separately by subtype in `subtype_error_rates.json`; this matters because artifacts and nonsentient ships are ontology exclusions, while individuated AIs, beasts, deities, and spirits normally move to PERSON.
- The drift is systematic by declared subtype rather than evidence that every AGENT label is bad. Targeted correction is warranted; wholesale relabelling is not.

## PD1000

- The mandated risk-enriched slice has transitions {json.dumps(pd_rates['transitions'], sort_keys=True)} and must not be treated as a prevalence sample.
- The unbiased 100-AGENT calibration estimates {random_pd['any_semantic_drift']['rate']:.1%} semantic drift (95% Wilson interval {random_pd['any_semantic_drift']['ci95'][0]:.1%}-{random_pd['any_semantic_drift']['ci95'][1]:.1%}); AGENT->PERSON is {random_pd['agent_to_person']['rate']:.1%}, and AGENT->NEITHER is {random_pd['agent_to_neither']['rate']:.1%}.
- Its independent 30-record overlap achieved {random_pd['overlap_agreement']['raw_agreement']:.1%} raw agreement (kappa {random_pd['overlap_agreement']['cohen_kappa']:.3f}); both disagreements received fresh expanded-context tie-breaks.
- Recurring ontology patterns and all feasible repeated PERSON/AGENT surfaces are cataloged in `conflicting_surface_analysis.jsonl`.

## Development

- PD1000EVAL audit transitions: {json.dumps(dev_rates['pd1000_eval']['transitions'], sort_keys=True)}.
- Modern development audit transitions: {json.dumps(dev_rates['modern']['transitions'], sort_keys=True)}.
- These are DEVELOPMENT suites, not untouched final evidence. `development_confusion_reinterpreted.json` separates sampled evaluation-label drift from genuine model type confusion and O omission.

## Conflicting surfaces

- Relation counts by corpus: `{json.dumps(conflict_relations, sort_keys=True)}`.
- A shared surface is not automatically inconsistent: contextual collectives, namesakes, and homonyms are retained separately. Boundary and alias/canonicalization defects are explicit categories.

## Ontology examples to add

- Nonsentient named artifact or weapon: NEITHER, even when it activates or causes an event.
- Nonsentient named ship: NEITHER; a ship becomes PERSON only when the prose establishes an individuated sentient character.
- Individuated sentient AI, deity, spirit, beast, alien, or construct: PERSON.
- Species or people acting collectively: AGENT; an individually named member: PERSON.
- Named organization, church, government, council, guild, army, or faction acting institutionally: AGENT.
- Personified institution without an individuated sentient character remains AGENT, not PERSON.
- Titles, partial mentions, and stale canonical substitutions are boundary/metadata defects, not ontology evidence.

## Limitations

- Statistical intervals quantify binomial sampling variation, not correlated story/template effects or adjudicator model-family bias.
- The synthetic estimate assumes the declared subtype counts are accurate and each within-subtype reservoir is representative.
- Development PERSON samples are ontology-edge enriched; use class-conditional transition estimates, not unqualified whole-corpus prevalence.
"""
    (ROOT / "ontology_drift_report.md").write_text(report, encoding="utf-8")

    synthetic_action = "RELABEL_TARGETED_SUBTYPES" if synthetic_weighted["rate"] >= .02 else "NO_ACTION"
    pd_point = random_pd["any_semantic_drift"]["rate"]
    pd_action = "RELABEL_TARGETED_SUBTYPES" if pd_point >= .03 else ("MASK_RISKY_SUBTYPES" if pd_point >= .01 else "NO_ACTION")
    recommendation = f"""# Recommended Corpus Action

No source labels were modified by this audit.

## Synthetic — {synthetic_action}

- Estimated affected AGENT fraction: {synthetic_weighted['rate']:.1%} (95% stratified uncertainty interval {synthetic_weighted['ci95'][0]:.1%}-{synthetic_weighted['ci95'][1]:.1%}); approximately {synthetic_weighted['affected_mention_estimate']:,} declared mentions.
- Correct only demonstrated subtype routes: artifacts/nonsentient objects toward NEITHER or masking, and individuated AI/beast/deity/spirit cases toward PERSON. Preserve genuine faction/cult/empire/guild collectives.
- Expected impact: reduce systematic AGENT-vs-PERSON and false-AGENT supervision, likely helping AGENT precision and ontology consistency. It cannot by itself solve AGENT->O recall.
- Cost: low-to-moderate because subtype metadata already provides a bounded regeneration/review key. Regenerating affected synthetic subsets is preferable where span-level relabelling is uncertain.

## PD1000 — {pd_action}

- Unbiased AGENT calibration: {pd_point:.1%} affected (95% Wilson interval {random_pd['any_semantic_drift']['ci95'][0]:.1%}-{random_pd['any_semantic_drift']['ci95'][1]:.1%}), implying roughly {round(pd_point * random_pd['population_gold_agent_mentions']):,} of {random_pd['population_gold_agent_mentions']:,} AGENT mentions if the sample generalizes.
- Use targeted correction or masks for named individuated supernatural beings and nonsentient ships/artifacts; do not relabel all AGENTs.
- Expected impact: principally ontology consistency, with a smaller but plausible improvement in AGENT/PERSON precision. Preserve Claude-Opus PD1000 as the closeout set after targeted cleanup.
- Cost: moderate; context-aware review is required and surface-only global replacement is unsafe.

## PD1000EVAL / Modern DEVELOPMENT — RELABEL_TARGETED_SUBTYPES

- Correct audited and systematically discoverable ontology/boundary defects before using these suites for model selection, version the corrected development labels, and retain the old score lineage.
- Re-run both base and large confusion matrices after corrections. Do not use these suites as FINAL_HOLDOUT evidence.

## Training impact

- Until corrections are materialized and regression-tested, mask identified disputed spans rather than converting uncertainty to O.
- Do not perform wholesale AGENT relabelling. The audit supports targeted subtype/action routes, not a claim that the entire AGENT pool is corrupt.
"""
    (ROOT / "recommended_corpus_action.md").write_text(recommendation, encoding="utf-8")

    immutable_files = sorted(
        [p for p in ROOT.rglob("*") if p.is_file() and p.name != "immutable_audit_manifest.json"]
    )
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        revision = None
    manifest = {
        "audit_id": "person_agent_ontology_audit_20260712",
        "status": "COMPLETE_LABELS_UNMODIFIED",
        "ontology_version": "rune_ner_ontology.v1",
        "random_seeds": [20260712, 20260713, 20260714, SEED],
        "reviewer_ids": ["reviewer_a", "reviewer_b", "reviewer_c", "followup_reviewer_d", "expanded_reviewer_e", "final_reviewer_h", "development_reviewer_f", "development_reviewer_g", "pd1000_random_reviewer_i", "pd1000_random_overlap_reviewer_j", "pd1000_random_final_reviewer_k", "conflict_reviewer_1", "conflict_reviewer_2", "conflict_reviewer_3"],
        "prompt_versions": ["person_agent_ontology_review_v1", "conflicting_surface_review_v1"],
        "commands": [
            "python scripts/prepare_person_agent_llm_audit.py",
            "python scripts/route_person_agent_llm_audit.py",
            "python scripts/build_person_agent_expanded_context.py --ids ... --output ...",
            "python scripts/prepare_person_agent_auxiliary_reviews.py",
            "python scripts/finalize_person_agent_ontology_audit.py",
        ],
        "git_revision_at_finalize": revision,
        "files_sha256": {str(path): sha256(path) for path in immutable_files},
    }
    (ROOT / "immutable_audit_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "synthetic_weighted_drift": synthetic_weighted,
        "pd1000_random_agent_drift": random_pd["any_semantic_drift"],
        "agreement": agreement_report,
        "development_rates": dev_rates,
        "actions": {"synthetic": synthetic_action, "pd1000": pd_action},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
