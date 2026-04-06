#!/usr/bin/env python3
"""
Add LOCATION and AGENT entities to OOD validation ground truth.

This script updates ood_validation_ground_truth_modern_heavy.jsonl to include:
1. Existing characters converted to PERSON entities
2. Curated LOCATION entities for significant places
3. Curated AGENT entities for AI/factions/organizations

The entities are manually curated for accuracy in validation data.
"""

import json
from pathlib import Path

# Curated entities for each story
# Format: story_id -> {"locations": [...], "agents": [...]}
CURATED_ENTITIES = {
    "pride_prejudice_sample": {
        "locations": [
            {"text": "Netherfield Park", "location_type": "estate"},
            {"text": "Netherfield", "location_type": "estate"},
            {"text": "England", "location_type": "country"},
        ],
        "agents": []
    },
    "colour_of_magic_extract": {
        "locations": [
            {"text": "Ankh-Morpork", "location_type": "city"},
            {"text": "Ankh", "location_type": "city"},  # Also mentioned separately as river/district
            {"text": "Morpork", "location_type": "city"},  # Also mentioned separately as district
            {"text": "Discworld", "location_type": "world"},
            {"text": "Wizards' Quarter", "location_type": "district"},
            {"text": "Merchants Street", "location_type": "street"},
            {"text": "Crimson Leech", "location_type": "tavern"},
            {"text": "Broken Drum", "location_type": "tavern"},
            {"text": "Pearl Dock", "location_type": "dock"},
            {"text": "Ymitury", "location_type": "place"},
        ],
        "agents": []
    },
    "the_ships_of_merior_extract": {
        "locations": [
            {"text": "Ostermere", "location_type": "city"},
            {"text": "Havish", "location_type": "kingdom"},
            {"text": "Telmandir", "location_type": "city"},
        ],
        "agents": [
            {"text": "Fellowship", "subtype": "faction", "role": "force"},
            {"text": "Mistwraith", "subtype": "entity", "role": "antagonist"},
            {"text": "Dharkaron Avenger", "subtype": "deity", "role": "mythical"},
        ]
    },
    "the_ghost_in_the_doll_extract": {
        "locations": [
            {"text": "New York", "location_type": "city"},
            {"text": "Japan", "location_type": "country"},
            {"text": "Topeka", "location_type": "city"},
            {"text": "Kansas", "location_type": "state"},
        ],
        "agents": [
            {"text": "Palladium", "subtype": "ai", "role": "force"},
            {"text": "NAPA", "subtype": "organization", "role": "law_enforcement"},
            {"text": "Watch", "subtype": "organization", "role": "law_enforcement"},
            {"text": "Yliaster", "subtype": "system", "role": "technology"},
        ]
    },
    "a_memory_called_empire_extract": {
        "locations": [
            {"text": "Teixcalaan", "location_type": "empire"},
            {"text": "Lsel Station", "location_type": "station"},
        ],
        "agents": [
            {"text": "Teixcalaanli Empire", "subtype": "faction", "role": "government"},
            {"text": "Information Ministry", "subtype": "organization", "role": "government"},
            {"text": "Lsel Council", "subtype": "organization", "role": "government"},
            {"text": "Imperium", "subtype": "faction", "role": "government"},
        ]
    },
    "the_call_of_the_wild_scene": {
        "locations": [
            {"text": "Skookum Benches", "location_type": "wilderness"},
            {"text": "Yukon", "location_type": "territory"},
        ],
        "agents": []
    },
    "jane_eyre_scene": {
        "locations": [
            {"text": "Thornfield", "location_type": "estate"},
        ],
        "agents": []
    },
    "moby_dick_scene": {
        "locations": [
            {"text": "Pequod", "location_type": "ship"},
        ],
        "agents": []
    },
    "jane_eyre_scene_2": {
        "locations": [
            {"text": "Thornfield", "location_type": "estate"},
            {"text": "Ingram Park", "location_type": "estate"},
        ],
        "agents": []
    },
    "sherlock_holmes_scandal_bohemia": {
        "locations": [
            {"text": "Baker Street", "location_type": "street"},
            {"text": "Briony Lodge", "location_type": "house"},
            {"text": "Serpentine Avenue", "location_type": "street"},
        ],
        "agents": []
    },
    "mistborn_final_empire_scene_001": {
        "locations": [
            {"text": "Luthadel", "location_type": "city"},
            {"text": "Final Empire", "location_type": "empire"},
        ],
        "agents": [
            {"text": "Steel Ministry", "subtype": "faction", "role": "antagonist"},
        ]
    },
    "assassins_apprentice_scene_002": {
        "locations": [
            {"text": "Buckkeep", "location_type": "castle"},
            {"text": "Six Duchies", "location_type": "kingdom"},
        ],
        "agents": []
    },
    "magician_scene_003": {
        "locations": [
            {"text": "Crydee", "location_type": "town"},
            {"text": "Midkemia", "location_type": "world"},
        ],
        "agents": []
    },
    "red_rising_scene_004": {
        "locations": [
            {"text": "Mars", "location_type": "planet"},
            {"text": "Institute", "location_type": "academy"},
        ],
        "agents": [
            {"text": "Society", "subtype": "faction", "role": "force"},
        ]
    },
    "dune_scene_005": {
        "locations": [
            {"text": "Arrakis", "location_type": "planet"},
            {"text": "Caladan", "location_type": "planet"},
        ],
        "agents": [
            {"text": "Guild", "subtype": "organization", "role": "force"},
            {"text": "Fremen", "subtype": "faction", "role": "force"},
            {"text": "Imperium", "subtype": "faction", "role": "government"},
        ]
    },
    "lies_of_locke_lamora_scene_006": {
        "locations": [
            {"text": "Camorr", "location_type": "city"},
        ],
        "agents": []
    },
    "the_fifth_season_scene_007": {
        "locations": [
            {"text": "Stillness", "location_type": "continent"},
            {"text": "Fulcrum", "location_type": "institution"},
        ],
        "agents": [
            {"text": "Fulcrum", "subtype": "organization", "role": "force"},
            {"text": "Leadership", "subtype": "organization", "role": "government"},
        ]
    },
    "prince_of_thorns_scene_008": {
        "locations": [
            {"text": "Ancrath", "location_type": "kingdom"},
        ],
        "agents": []
    },
    "storm_front_scene_009": {
        "locations": [
            {"text": "Chicago", "location_type": "city"},
        ],
        "agents": []
    },
    "a_little_hatred_scene_010": {
        "locations": [
            {"text": "Adua", "location_type": "city"},
            {"text": "Union", "location_type": "nation"},
        ],
        "agents": []
    },
    "a_little_hatred_scene_011": {
        "locations": [
            {"text": "Adua", "location_type": "city"},
        ],
        "agents": []
    },
    "hyperion_scene_012": {
        "locations": [
            {"text": "Hyperion", "location_type": "planet"},
        ],
        "agents": [
            {"text": "Hegemony", "subtype": "faction", "role": "government"},
            {"text": "Ousters", "subtype": "faction", "role": "force"},
            {"text": "Shrike", "subtype": "entity", "role": "deity"},
            {"text": "Templar", "subtype": "organization", "role": "religious"},
        ]
    },
    "hyperion_scene_013": {
        "locations": [
            {"text": "Hyperion", "location_type": "planet"},
            {"text": "Time Tombs", "location_type": "ruins"},
        ],
        "agents": [
            {"text": "Catholic Church", "subtype": "organization", "role": "religious"},
            {"text": "Hegemony", "subtype": "faction", "role": "government"},
        ]
    },
    "agent_to_the_stars_scene_014": {
        "locations": [
            {"text": "Los Angeles", "location_type": "city"},
            {"text": "Hollywood", "location_type": "district"},
        ],
        "agents": []
    },
    "gone_girl_scene_a": {
        "locations": [
            {"text": "Missouri", "location_type": "state"},
            {"text": "North Carthage", "location_type": "town"},
        ],
        "agents": []
    },
    "gone_girl_scene_b": {
        "locations": [
            {"text": "North Carthage", "location_type": "town"},
        ],
        "agents": []
    },
    "it_ends_with_us_scene_a": {
        "locations": [
            {"text": "Boston", "location_type": "city"},
        ],
        "agents": []
    },
    "it_ends_with_us_scene_b": {
        "locations": [
            {"text": "Boston", "location_type": "city"},
        ],
        "agents": []
    },
    "little_fires_scene_a": {
        "locations": [
            {"text": "Shaker Heights", "location_type": "suburb"},
            {"text": "Cleveland", "location_type": "city"},
        ],
        "agents": []
    },
    "little_fires_scene_b": {
        "locations": [
            {"text": "Shaker Heights", "location_type": "suburb"},
        ],
        "agents": []
    },
}


def convert_characters_to_entities(characters):
    """Convert characters list to PERSON entities."""
    entities = []
    for char in characters:
        entity = {
            "text": char["name"],
            "type": "PERSON",
            "role": char.get("role", "supporting").lower(),
        }
        if "aliases" in char:
            entity["aliases"] = char["aliases"]
        entities.append(entity)
    return entities


def add_extended_entities(story):
    """Add entities field with PERSON, LOCATION, and AGENT types."""
    story_id = story["story_id"]
    
    # Start with existing characters as PERSON entities
    entities = convert_characters_to_entities(story.get("characters", []))
    
    # Add curated LOCATION and AGENT entities
    if story_id in CURATED_ENTITIES:
        curated = CURATED_ENTITIES[story_id]
        
        for loc in curated.get("locations", []):
            entities.append({
                "text": loc["text"],
                "type": "LOCATION",
                "location_type": loc.get("location_type", "unknown"),
            })
        
        for agent in curated.get("agents", []):
            entities.append({
                "text": agent["text"],
                "type": "AGENT",
                "subtype": agent.get("subtype", "faction"),
                "role": agent.get("role", "force"),
            })
    
    story["entities"] = entities
    return story


def main():
    input_path = Path("ood_validation_ground_truth_modern_heavy.jsonl")
    output_path = Path("ood_validation_ground_truth_extended.jsonl")
    
    print(f"Reading {input_path}...")
    
    with open(input_path) as f:
        stories = [json.loads(line) for line in f]
    
    print(f"Processing {len(stories)} stories...")
    
    updated_stories = []
    stats = {"locations": 0, "agents": 0, "persons": 0}
    
    for story in stories:
        updated = add_extended_entities(story)
        updated_stories.append(updated)
        
        # Count entities
        for ent in updated.get("entities", []):
            if ent["type"] == "PERSON":
                stats["persons"] += 1
            elif ent["type"] == "LOCATION":
                stats["locations"] += 1
            elif ent["type"] == "AGENT":
                stats["agents"] += 1
    
    print(f"\nEntity counts:")
    print(f"  PERSON:   {stats['persons']}")
    print(f"  LOCATION: {stats['locations']}")
    print(f"  AGENT:    {stats['agents']}")
    
    # Write output
    with open(output_path, "w") as f:
        for story in updated_stories:
            f.write(json.dumps(story) + "\n")
    
    print(f"\n✅ Written to {output_path}")
    
    # Show stories without curated entities
    missing = []
    for story in stories:
        if story["story_id"] not in CURATED_ENTITIES:
            missing.append(story["story_id"])
    
    if missing:
        print(f"\n⚠️  Stories without curated LOCATION/AGENT entities:")
        for sid in missing:
            print(f"   - {sid}")


if __name__ == "__main__":
    main()
