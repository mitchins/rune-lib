# Blinded Conflicting-Surface Review Prompt v1

Apply the ontology in `person_agent_ontology_review_v1.md` independently to
every supplied context without seeing its current label. Then classify the
cross-context relation as exactly LEGITIMATE_CONTEXT_DEPENDENT,
ONTOLOGY_INCONSISTENCY, ALIAS_CANONICALIZATION_ERROR, BOUNDARY_ERROR, HOMONYM,
or INSUFFICIENT_EVIDENCE. Return per-context decisions and one relation with
confidence and a one-sentence rationale.
