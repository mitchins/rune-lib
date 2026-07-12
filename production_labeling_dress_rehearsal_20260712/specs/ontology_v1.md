# PERSON / LOCATION / AGENT Ontology v1

Status: **FROZEN**  
Version: `rune_ner_ontology.v1`  
Effective date: 2026-07-12

This specification controls proposal review, audit, materialization, and evaluation reconciliation. Changes require a new version and must not silently alter existing labels.

## Primary distinction

- **Individual actor -> PERSON**
- **Collective or institutional actor -> AGENT**
- **Physical place -> LOCATION**
- Ambiguous cases remain unresolved for review; uncertainty never becomes `O` automatically.

The ability to act is not sufficient by itself to make something AGENT.

## PERSON

An individually referable sentient or narratively person-like entity.

Include when individually named or uniquely individuated:

- humans and individually named fictional characters;
- sentient AIs, androids and constructs;
- deities and supernatural beings acting as individuals;
- individually referable aliens;
- sapient or narratively person-like animals;
- a named ship or system only when the text clearly treats it as an individual sentient character.

Positive examples:

- `Elizabeth Bennet`, `Adama`, `Athena` when Athena is the individual deity;
- `Aslan`, `the Doctor` when that phrase is the character's stable individuator;
- `Jeeves` when Jeeves is an individually sentient AI;
- `Glimfeather` when the named owl is a sapient character;
- a sentient ship named `Justice of Toren` when the mention denotes the individual consciousness.

Do not label as PERSON:

- ordinary pronouns;
- an unindividuated role such as `the guard`, `a doctor`, or `the king`;
- a species or people acting collectively;
- a non-sentient vehicle merely described with action verbs;
- an organization named after a person when the mention denotes the organization.

## LOCATION

A specifically referable physical or geographic place.

Include:

- countries, cities, regions and planets when the mention denotes territory or place;
- buildings, rooms, stations, rivers, roads, forests and landmarks when physically denoted;
- fictional physical places.

Positive examples:

- `Boston`, `Shaker Heights`, `Earth`, `Cair Paravel`, `Missouri River`;
- `British Embassy` when the context clearly means the building;
- `Lsel Station` when the mention means the physical station.

Do not label as LOCATION:

- a government or state acting institutionally;
- an embassy staff or diplomatic institution acting collectively;
- an organization merely headquartered in a named building;
- a vehicle used only as an object.

## AGENT

A non-individual collective or institutional entity acting, deciding, coordinating, governing, commanding, employing, publishing, attacking, investigating, trading, worshipping institutionally, or otherwise behaving as a group.

Include:

- organizations, companies, guilds and institutions;
- governments, councils, courts and embassies acting institutionally;
- factions, armies, crews and coordinated teams;
- churches or religions acting as institutions;
- species, peoples, families or collectives acting as a group;
- unnamed coordinated groups when the local mention denotes a concrete acting group rather than a generic class;
- a non-sentient ship name used metonymically for its acting crew or command structure.

Positive examples:

- `UNICEF`, `the British Government`, `the Catholic Church`, `the Fellowship`;
- `the royal guards` when they act as a coordinated unit;
- `the firefighters` when it means the concrete responding group in the scene;
- `Teixcalaan` when it means the government/state requesting or deciding;
- `the First Born` when the people act collectively in context;
- `Galactica` only when the mention clearly stands for its acting crew or command.

Do not label as AGENT:

- an individual actor, even divine, artificial, alien or non-human;
- a group word used generically, such as `firefighters need training`;
- a species name used only taxonomically;
- a physical nation, embassy, court building or station with no institutional reading;
- a non-sentient ship or artifact used merely as an object;
- products, works, spells, pathogens, events or abstract traditions unless a distinct acting institution is denoted.

## Boundary cases

Each occurrence is resolved from its own context. A repeated surface may have different labels.

| Surface | Contextual reading | Label |
|---|---|---|
| `Athena` | individual deity speaks | PERSON |
| `Athena` | organization or ship crew named Athena acts | AGENT |
| `British Embassy` | diplomats issue a statement | AGENT |
| `British Embassy` | character enters the building | LOCATION |
| `Teixcalaan` | government requests identity proof | AGENT |
| `Teixcalaan` | travel across its territory | LOCATION |
| `the Friends` | Society of Friends acts institutionally | AGENT |
| `her friends` | informal collection without coordinated agency | O |
| `the firefighters` | concrete responders coordinate at the fire | AGENT |
| `firefighters` | profession discussed generically | O |
| `Hegemony` | polity governs or decides | AGENT |
| `Hegemony ship` | `Hegemony` is only an attributive origin descriptor | review; do not infer automatically |
| `Galactica` | crew/command acts metonymically | AGENT |
| `Galactica` | non-sentient hull is damaged or observed | O |
| sentient `Justice of Toren` | individual ship-mind speaks | PERSON |
| `the Richardsons` | family acts collectively | AGENT |
| `Richardson` | individually identified family member | PERSON |

## Span boundaries

- Materialize the literal occurrence, never an advisory canonical alias.
- Exclude possessive suffixes (`'s`, `’s`) when the base mention is independently present.
- Exclude ordinary honorifics from PERSON when the name alone is sufficient (`Dr. Smith` -> `Smith`).
- Retain a title when it is the stable unique character name (`the Doctor`).
- Include organizational modifiers required to identify the entity (`British Government`, `Catholic Church`).
- Do not expand abbreviations or shortened mentions into absent text.
- Do not apply one decision to every repeated surface without checking each context.

## Uncertainty

If individual-versus-collective, place-versus-institution, sentient-versus-object, or generic-versus-concrete coordination cannot be resolved from the mention context, the candidate transitions to final review or an unresolved mask. It must not be forced into PERSON, LOCATION, AGENT, or `O`.
