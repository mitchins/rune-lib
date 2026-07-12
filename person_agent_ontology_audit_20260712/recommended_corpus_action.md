# Recommended Corpus Action

No source labels were modified by this audit.

## Synthetic — RELABEL_TARGETED_SUBTYPES

- Estimated affected AGENT fraction: 38.7% (95% stratified uncertainty interval 36.4%-43.6%); approximately 591,788 declared mentions.
- Correct only demonstrated subtype routes: artifacts/nonsentient objects toward NEITHER or masking, and individuated AI/beast/deity/spirit cases toward PERSON. Preserve genuine faction/cult/empire/guild collectives.
- Expected impact: reduce systematic AGENT-vs-PERSON and false-AGENT supervision, likely helping AGENT precision and ontology consistency. It cannot by itself solve AGENT->O recall.
- Cost: low-to-moderate because subtype metadata already provides a bounded regeneration/review key. Regenerating affected synthetic subsets is preferable where span-level relabelling is uncertain.

## PD1000 — RELABEL_TARGETED_SUBTYPES

- Unbiased AGENT calibration: 56.2% affected (95% Wilson interval 46.3%-65.7%), implying roughly 19,565 of 34,783 AGENT mentions if the sample generalizes.
- Use targeted correction or masks for named individuated supernatural beings and nonsentient ships/artifacts; do not relabel all AGENTs.
- Expected impact: principally ontology consistency, with a smaller but plausible improvement in AGENT/PERSON precision. Preserve Claude-Opus PD1000 as the closeout set after targeted cleanup.
- Cost: moderate; context-aware review is required and surface-only global replacement is unsafe.

## PD1000EVAL / Modern DEVELOPMENT — RELABEL_TARGETED_SUBTYPES

- Correct audited and systematically discoverable ontology/boundary defects before using these suites for model selection, version the corrected development labels, and retain the old score lineage.
- Re-run both base and large confusion matrices after corrections. Do not use these suites as FINAL_HOLDOUT evidence.

## Training impact

- Until corrections are materialized and regression-tested, mask identified disputed spans rather than converting uncertainty to O.
- Do not perform wholesale AGENT relabelling. The audit supports targeted subtype/action routes, not a claim that the entire AGENT pool is corrupt.
