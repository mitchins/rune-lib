# PERSON / AGENT ontology consistency audit — interim

Status: **candidate construction and model confusion complete; blind semantic adjudication pending**  
Ontology: `rune_ner_ontology.v1`  
Corpora modified: **none**

## Frozen inputs

- Synthetic training phase: `production_modernbert_large_curriculum_v1/phase_a_synthetic_1024.jsonl`
- Original synthetic export used for subtype-balanced semantic sampling: `ner_training_agent_rich_training_ultra_short_agent_rich_20260703_143554.jsonl`
- PD1000 train: `/mnt/storage/gold-books-pd1000/splits/pd1000_train_windows_2048.jsonl`
- PD1000 development: `/mnt/storage/gold-books-pd1000/splits/pd1000_eval_windows.jsonl`
- Modern development: `ood_validation_ground_truth_extended.jsonl`

SHA-256 fingerprints are recorded in `input_sha256.txt`.

## Manual audit slices

- Synthetic: 200 AGENT mentions, balanced 20 each across source subtypes `ai`, `artifact`, `beast`, `cult`, `deity`, `empire`, `faction`, `guild`, `ship`, and `spirit`; plus 100 PERSON edge cases.
- PD1000 train: 200 AGENT mentions plus 100 PERSON edge cases.
- Repeated PERSON/AGENT surface conflicts: 34 synthetic surfaces and 324 PD1000-train surfaces, with bounded examples retained for each label.
- Development candidate pools: all 10,583 PD1000EVAL AGENT mentions and all 116 modern-development AGENT mentions, plus heuristic PERSON edge slices and repeated-label conflicts.

The 600 primary records have sealed current-label lookups and separate blind-review sheets. No drift percentage will be reported as manual fact until those sheets are completed.

## Strong pre-adjudication signal: synthetic subtype design

The original synthetic export contains 221,623 AGENT entity definitions and 1,527,557 declared AGENT mentions. Source subtype counts include:

| Source subtype | Entity definitions | Declared mentions | Ontology-v1 risk |
|---|---:|---:|---|
| artifact | 35,250 | 232,335 | Usually O; sentient individual artifacts should be PERSON |
| ai | 13,940 | 132,951 | Individuated sentient AIs should be PERSON; non-sentient systems often O |
| beast | 8,762 | 75,395 | Individuated sapient/narrative beasts should be PERSON |
| deity | 6,693 | 44,883 | Individual deities should be PERSON; deity collectives may be AGENT |
| spirit | 12,506 | 94,881 | Individual spirits should be PERSON; genuine collectives may be AGENT |
| ship | 5,821 | 37,686 | Object ships are O; metonymic crews AGENT; sentient ship-characters PERSON |

The six high-risk source subtypes account for 878,131 / 1,527,557 declared synthetic AGENT mentions (57.5%). This is a risk stratum, not a measured error rate. The artifact subtype alone contributes 15.2% of declared AGENT mentions and contains many clearly non-AGENT objects, establishing a substantial lower-bound concern before sampling inference.

## Development model confusion matrices

These are model behavior measurements, not proof that development gold is corrupt.

### ModernBERT-base

| Development suite | Gold PERSON→AGENT | Gold AGENT→PERSON | Gold AGENT→O | Gold PERSON→O |
|---|---:|---:|---:|---:|
| PD1000EVAL | 109 | 1,732 | 4,843 | 9,677 |
| Modern | 3 | 7 | 23 | 99 |

### ModernBERT-large

| Development suite | Gold PERSON→AGENT | Gold AGENT→PERSON | Gold AGENT→O | Gold PERSON→O |
|---|---:|---:|---:|---:|
| PD1000EVAL | 370 | 1,167 | 4,820 | 10,281 |
| Modern | 2 | 2 | 31 | 95 |

On PD1000EVAL, both models classify many current AGENT mentions as PERSON or O. That is consistent with either ontology drift, difficult rare-class behavior, or both; manual adjudication is required to separate them.

The modern suite contains six rows exceeding the 4096-token evaluation cap. The matrix explicitly excludes 375 PERSON and seven AGENT mentions as `TRUNCATED` rather than misreporting them as O. A later sliding-window matrix is required for full-scene coverage.

## Candidate ontology examples pending promotion

The audit specifically tests:

- `Allah`, `Coyote`, `Iku-Turso`, and individual `Ifrit` occurrences as PERSON;
- collective `Moirai` occurrences as possible AGENT;
- `CASSANDRA`, `MNEMONE`, and other individuated speaking AIs as PERSON;
- non-sentient systems such as a mere protocol or surveillance product as O;
- `The Chorus` as AGENT when it denotes a collective consciousness;
- `Duskbane` as PERSON when the sentient sword speaks as an individual;
- `USS Reliant` as AGENT when a transmission clearly denotes its crew/command, but O when merely observed as a hull;
- `Yūrei Maru` as O when it is only a derelict object;
- concrete coordinated groups such as guards or firefighters as AGENT, while generic professions remain O.

## Current authorization

- Corpus relabelling: NO-GO
- Masking/correction recommendation: pending blind adjudication
- 100-scene production run: NO-GO
- Bulk generation: NO-GO
