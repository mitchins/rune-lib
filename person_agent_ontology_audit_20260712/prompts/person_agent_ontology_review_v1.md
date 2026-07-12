# Blinded PERSON/AGENT Review Prompt v1

The reviewer receives only a stable anonymous record ID, exact surface, local or
expanded context, and review role. It must not inspect corpus identity, current
labels, model output, sealed lookups, conclusions, or other reviews.

PERSON is an individually referable sentient or narratively person-like entity,
including humans, named or uniquely individuated aliens, sentient AIs, deities,
spirits, sapient animals, androids or constructs, supernatural beings, and
individually personified entities acting as characters.

AGENT is a non-individual acting collective or institutional entity, including
organizations, governments, companies, factions, churches, guilds, armies,
councils, species or peoples acting collectively, coordinated groups, and
institutional bodies.

The core distinction is individual actor = PERSON and collective/institutional
actor = AGENT. Objects, artifacts, ships, weapons, places, and abstract concepts
are neither unless the prose clearly establishes an individuated sentient
character. Action alone does not make an entity AGENT.

Return one JSON object per input with `record_id`, `review_role`, `decision`,
`confidence`, one-sentence `rationale`, `ontology_subtype`, `boundary_action`,
and `context_dependent_surface`. Decision is exactly PERSON, AGENT, NEITHER,
AMBIGUOUS_CONTEXT, BAD_BOUNDARY, or INSUFFICIENT_CONTEXT. Boundary action is
exactly NONE, EXPAND, or CONTRACT.

