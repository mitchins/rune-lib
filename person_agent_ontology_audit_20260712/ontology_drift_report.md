# PERSON/AGENT Ontology Drift Audit

## Scope and method

- Frozen, blinded semantic adjudication covered 600 primary records: 300 synthetic and 300 PD1000.
- A 150-record overlap (25%) supplied independent agreement evidence. Low-confidence, disputed, and context-limited cases followed the predeclared second-review, tie-break, or expanded-context routes.
- An additional unbiased 100-record PD1000 AGENT sample supports corpus-level calibration because the mandated 300-record PD1000 slice was deliberately ontology-risk enriched.
- Development estimates use a separate blinded auxiliary sample of 300 PD1000EVAL and 216 Modern mentions. Model predictions were revealed only for subsequent reinterpretation.
- Labels in every source corpus remain unchanged.

## Reviewer reliability

- Raw overlap agreement: 90.7%; Cohen's kappa: 0.855.
- Disagreement rate: 9.3%; tie-break rate: 2.2%; expanded-context rate: 4.7%.
- Final boundary-error rate: 2.2%; final insufficient-context rate: 0.0%.

## Synthetic

- The subtype-weighted estimate is 38.7% affected AGENT mentions (95% stratified uncertainty interval 36.4%-43.6%), approximately 591,788 of 1,527,557 declared AGENT mentions.
- AGENT->PERSON and AGENT->NEITHER are reported separately by subtype in `subtype_error_rates.json`; this matters because artifacts and nonsentient ships are ontology exclusions, while individuated AIs, beasts, deities, and spirits normally move to PERSON.
- The drift is systematic by declared subtype rather than evidence that every AGENT label is bad. Targeted correction is warranted; wholesale relabelling is not.

## PD1000

- The mandated risk-enriched slice has transitions {"AGENT->AGENT": 112, "AGENT->NEITHER": 32, "AGENT->PERSON": 52, "PERSON->NEITHER": 6, "PERSON->PERSON": 85} and must not be treated as a prevalence sample.
- The unbiased 100-AGENT calibration estimates 56.2% semantic drift (95% Wilson interval 46.3%-65.7%); AGENT->PERSON is 40.6%, and AGENT->NEITHER is 15.6%.
- Its independent 30-record overlap achieved 93.3% raw agreement (kappa 0.890); both disagreements received fresh expanded-context tie-breaks.
- Recurring ontology patterns and all feasible repeated PERSON/AGENT surfaces are cataloged in `conflicting_surface_analysis.jsonl`.

## Development

- PD1000EVAL audit transitions: {"AGENT->AGENT": 90, "AGENT->NEITHER": 50, "AGENT->PERSON": 47, "PERSON->NEITHER": 3, "PERSON->PERSON": 94}.
- Modern development audit transitions: {"AGENT->AGENT": 88, "AGENT->NEITHER": 5, "AGENT->PERSON": 17, "PERSON->PERSON": 98}.
- These are DEVELOPMENT suites, not untouched final evidence. `development_confusion_reinterpreted.json` separates sampled evaluation-label drift from genuine model type confusion and O omission.

## Conflicting surfaces

- Relation counts by corpus: `{"modern": {}, "pd1000": {"ALIAS_CANONICALIZATION_ERROR": 1, "BOUNDARY_ERROR": 18, "HOMONYM": 157, "INSUFFICIENT_EVIDENCE": 2, "LEGITIMATE_CONTEXT_DEPENDENT": 33, "ONTOLOGY_INCONSISTENCY": 113}, "pd1000_eval": {"BOUNDARY_ERROR": 9, "HOMONYM": 42, "LEGITIMATE_CONTEXT_DEPENDENT": 16, "ONTOLOGY_INCONSISTENCY": 57}, "synthetic": {"HOMONYM": 22, "INSUFFICIENT_EVIDENCE": 1, "ONTOLOGY_INCONSISTENCY": 11}}`.
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
